#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import argparse
import os
import logging
import math

import datasets
from datasets import load_from_disk
import torch
from torch.optim.lr_scheduler import LRScheduler
from transformers import AutoConfig, AutoTokenizer, DataCollatorForLanguageModeling
from modeling.modeling_deepseek import DeepseekV2ForCausalLM, DeepseekV2MoE, DeepseekV2RotaryEmbedding
from modeling.modeling_deepseek_modifier import nnscaler_deepseek_init

from nnscaler.utils import set_default_logger_level
from nnscaler.cli.trainer import Trainer
from nnscaler.cli.trainer_args import (
    CheckpointConfig,
    DatasetConfig,
    HookMapConfig,
    ModelConfig,
    OptimizerConfig,
    TrainerArgs,
    DataloaderConfig,
    AggregatedOutputs,
    LogConfig,
    DatasetSamplerConfig,
    LRSchedulerConfig,
)
from nnscaler.parallel import ComputeConfig, BroadcastGenFilesStrategy
from nnscaler.runtime.f16_optimizer import MixedPrecisionAdamW
from nnscaler.cli.loggers.tensorboard import TensorBoardLogger

_logger = logging.getLogger(__name__)


IGNORE_IDX = -100


class WarmupScheduler(LRScheduler):

    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch + 1 >= self.warmup_steps:
            return self.base_lrs
        return [base_lr * (self.last_epoch + 1) / self.warmup_steps for base_lr in self.base_lrs]


def get_tokenizer(tokenizer_name_or_path,
                  model_max_length=None,
                  default_bos_token="<s>",
                  default_eos_token="</s>",
                  default_pad_token="[PAD]",
                  default_unk_token="<unk>"):

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = default_pad_token
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = default_eos_token
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = default_bos_token
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = default_unk_token

    tokenizer.add_special_tokens(special_tokens_dict)
    if model_max_length:
        tokenizer.model_max_length = model_max_length
    return tokenizer


class WrapperModel(torch.nn.Module):
    def __init__(self, model_id):
        super().__init__()
        self.model = DeepseekV2ForCausalLM.from_pretrained(model_id, attn_implementation='flash_attention_2')
        self.model.train()

        # post-process model for usibility
        # - merge small linear weights into large ones to use high performance kernel `grouped gemm`
        #   and avoid the overhead that merge them on the fly. Note that checkpoints of the wrapped model
        #   cannot be loaded directly to the transformers model. You need to split the weights with correct
        #   names.
        # - reset `max_seq_len_cached`` in rotary embeddings since in transformers source code, `cos_cached`
        #   and `sin_cached` are evaluated during runtime, which violates the assumption of concrete tracer.
        #   As a result, we reset `max_seq_len_cached` to make caches static.
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        for name, child in self.model.named_modules():
            if isinstance(child, DeepseekV2MoE):
                _logger.info(f'Merging experts in {name} with {type(child)}')
                # num_local_experts, intermediate_size, hidden_size
                gate_projs = torch.stack([expert.gate_proj.weight for expert in child.experts], dim=0)
                up_projs = torch.stack([expert.up_proj.weight for expert in child.experts], dim=0)
                down_projs = torch.stack([expert.down_proj.weight for expert in child.experts], dim=0)
                child.register_parameter('gate_projs', torch.nn.Parameter(gate_projs))
                child.register_parameter('up_projs', torch.nn.Parameter(up_projs))
                child.register_parameter('down_projs', torch.nn.Parameter(down_projs))
            elif isinstance(child, DeepseekV2RotaryEmbedding):
                child.max_seq_len_cached = config.max_position_embeddings

    def forward(self, samples):
        outputs = self.model(
            input_ids=samples['net_input']['src_tokens'],
            use_cache=False,
            return_dict=False,
        )
        logits = outputs[0].view(-1, outputs[0].size(-1))
        labels = samples['target'].view(-1)
        normalized_logits = torch.nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32)
        loss = torch.nn.functional.nll_loss(normalized_logits, labels, reduction='sum', ignore_index=IGNORE_IDX)
        return loss, loss.data, samples['ntokens'], samples['nsentences']


def aggregate_outputs_fn(loss_outputs, sync_group) -> AggregatedOutputs:
    losses, ntokens_info = [], []
    for _, loss, ntokens, _ in loss_outputs:
        losses.append(loss)
        ntokens_info.append(ntokens)

    loss_sum = torch.sum(torch.stack(losses), dtype=torch.float64)
    torch.distributed.all_reduce(loss_sum, group=sync_group)
    ntokens_sum = torch.sum(torch.tensor(ntokens_info, dtype=torch.float64, device=torch.cuda.current_device()))
    torch.distributed.all_reduce(ntokens_sum, group=sync_group)
    num_batches = torch.tensor(len(losses), device=torch.cuda.current_device())
    torch.distributed.all_reduce(num_batches, group=sync_group)

    return AggregatedOutputs(
        loss_sum=loss_sum.item() / ntokens_sum.item() / math.log(2),
        num_batches=num_batches.item(),
        num_tokens=ntokens_sum.item(),
    )


def main(args):

    if args.run_mode == 'run':
        broadcast_strategy = 'all'
    else:
        broadcast_strategy = 'none'

    set_default_logger_level('INFO')

    nnscaler_deepseek_init()

    ## Setup Dataset ##

    dataset = load_from_disk(args.dataset_path)
    tokenizer = get_tokenizer(args.model_id)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    def collate_fn(samples):
        if len(samples) == 0:
            return {}

        mini_batch = data_collator(samples)
        _mini_batch = {}

        src_tokens = mini_batch.pop('input_ids')
        seq_len = src_tokens.size(-1)
        _mini_batch['src_tokens'] = src_tokens

        shift_labels = mini_batch['labels'][..., 1:]
        _mini_batch['labels'] = torch.nn.functional.pad(shift_labels, (0, 1), 'constant', IGNORE_IDX).contiguous()

        return {
            "nsentences": len(samples),
            "ntokens": len(samples) * seq_len,
            "net_input": _mini_batch,
            "target": _mini_batch.pop('labels'),
        }

    ## Config Trainer ##

    if args.run_mode == 'compile':
        if args.runtime_ngpus is None:
            raise ValueError('runtime_ngpus must be specified in compile mode')
        runtime_ngpus = args.runtime_ngpus
    elif args.run_mode == 'run':
        world_size = int(os.getenv('WORLD_SIZE'))
        if args.runtime_ngpus is None:
            runtime_ngpus = world_size
        else:
            if args.runtime_ngpus != world_size:
                raise ValueError('runtime_ngpus must match the number of GPUs in run mode')
            runtime_ngpus = args.runtime_ngpus
    if runtime_ngpus % args.plan_ngpus != 0:
        raise ValueError('runtime_ngpus must be a multiple of plan_ngpus')

    compute_config = ComputeConfig(
        plan_ngpus=args.plan_ngpus,
        runtime_ngpus=runtime_ngpus,
        constant_folding=True,
        use_zero=True,
        use_end2end=True,
        # autodist config:
        # - memory constraint is set to 64GB
        pas_config={
            'mem_constraint': 64,
        },
    )

    model_config = ModelConfig(
        type=WrapperModel,
        args={
            'model_id': args.model_id,
        },
    )

    # optimizer hyperparameters are from YaRN
    lrscheduler_config = LRSchedulerConfig(
        type=WarmupScheduler,
        args={
            'warmup_steps': 10,
        },
        interval='step',
    )

    optimizer_config = OptimizerConfig(
        type=MixedPrecisionAdamW,
        args={'lr': 1e-5, 'betas': (0.9, 0.95), 'weight_decay': 0.0, 'fused': True},
        clip_gnorm=1.0,
        loss_reduction='sum',
        grad_reduction='per-token-mean',
        aggregate_outputs_fn=aggregate_outputs_fn,
    )

    dataset_config = DatasetConfig(
        type=(lambda split: dataset),
        train_args={'split': 'train'},
    )

    dataloader_config = DataloaderConfig(
        train_args={
            'collate_fn': collate_fn,
            'drop_last': True,
        },
    )

    sampler_config = DatasetSamplerConfig(
        train_args={
            'shuffle': True,
        },
    )

    checkpoint_config = CheckpointConfig(
        every_n_train_steps=1000,
        save_type='deduped',
        resume_from=(args.resume_path or 'last'),
    )

    log_config = LogConfig(
        type=TensorBoardLogger,
        args={
            'name': args.name,
            'root_dir': './runs',
        },
    )

    trainer_args = TrainerArgs(
        instance_name=args.name,
        run_mode=args.run_mode,
        compute_config=compute_config,
        pas_policy='autodist',
        model=model_config,
        optimizer=optimizer_config,
        dataset=dataset_config,
        dataloader=dataloader_config,
        checkpoint=checkpoint_config,
        precision='bf16',
        max_epochs=1,
        micro_batch_size=4,
        grad_accumulation_steps=8,
        log=[log_config],
        seed=0,
        broadcast_strategy=broadcast_strategy,
        dataset_sampler=sampler_config,
        lr_scheduler=lrscheduler_config,
    )

    trainer = Trainer(train_args=trainer_args)
    trainer.run()


if __name__ == '__main__':
    ## Parse Args ##

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--name',
        default='deepseek-coder-v2-lite-2k',
        type=str,
        help='name of the experiment',
    )
    parser.add_argument(
        '--run_mode',
        default='run',
        choices=['run', 'compile'],
        help='run or compile',
    )
    parser.add_argument(
        '--plan_ngpus',
        type=int,
        required=True,
        help='specify the scale unit size',
    )
    parser.add_argument(
        '--runtime_ngpus',
        type=int,
        required=True,
        help='specify the number of GPUs to use',
    )
    parser.add_argument(
        '--resume_path',
        default=None,
        type=str,
        help='path to dir of ckpts or the ckpt file to resume from',
    )
    parser.add_argument(
        '--dataset_path',
        default=None,
        type=str,
        help='path to the dataset',
    )
    parser.add_argument(
        '--model_id',
        default=None,
        type=str,
        help='transformers model id',
    )
    args = parser.parse_args()

    main(args)
