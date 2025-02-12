#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import argparse
import os

import datasets
from datasets import load_from_disk
import huggingface_hub
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from lm_models.utils import nnscaler_lm_init
from chunk_linear_cross_entropy import chunk_linear_cross_entropy

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
)
from nnscaler.parallel import ComputeConfig, BroadcastGenFilesStrategy
from nnscaler.runtime.f16_optimizer import MixedPrecisionAdamW
from nnscaler.cli.loggers.tensorboard import TensorBoardLogger


IGNORE_IDX = -100


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
    def __init__(self, model_id, enable_chunk_loss, attn_implementation='flash_attention_2'):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation=attn_implementation)
        self.enable_chunk_loss = enable_chunk_loss

    def forward(self, samples):
        if self.enable_chunk_loss:
            outputs = self.model.model(
                input_ids=samples['net_input']['src_tokens'],
                use_cache=False,
                return_dict=False,
            )
            hidden_states = outputs[0]
            losses = chunk_linear_cross_entropy(hidden_states, self.model.lm_head.weight, samples['target'], IGNORE_IDX, 1024)
            loss = torch.sum(losses)
        else:
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
        loss_sum=loss_sum.item() / ntokens_sum.item(),
        num_batches=num_batches.item(),
        num_tokens=ntokens_sum.item(),
    )


def main(args):

    if args.run_mode == 'run':
        broadcast_strategy = 'all'
    else:
        broadcast_strategy = 'none'

    set_default_logger_level('INFO')

    nnscaler_lm_init(args)

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

        # cast `nsentences` and `ntokens` to tensor since current pipeline parallelism can only transfer data in tensor format
        return {
            "nsentences": torch.tensor(len(samples), dtype=torch.long),
            "ntokens": torch.tensor(len(samples) * seq_len, dtype=torch.long),
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
        pas_config={
            'mem_constraint': args.gpu_mem_constraint,
            'pipeline_pivots': args.pipeline_pivots,
            'pipeline_nstages': args.pipeline_nstages,
            'recompute_modules': args.recompute_modules,
        },
        trace_strategy=args.trace_strategy,
    )

    model_config = ModelConfig(
        type=WrapperModel,
        args={
            'model_id': args.model_id,
            'enable_chunk_loss': args.enable_chunk_loss,
            'attn_implementation': args.attn_implementation,
        },
    )

    # optimizer hyperparameters are from YaRN
    optimizer_config = OptimizerConfig(
        type=MixedPrecisionAdamW,
        args={'lr': 2e-5, 'betas': (0.9, 0.95), 'weight_decay': 0.0, 'fused': True},
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
        max_epochs=2,
        grad_accumulation_steps=args.grad_accumulation_steps,
        max_train_steps=args.max_train_steps,
        log=[log_config],
        seed=0,
        broadcast_strategy=broadcast_strategy,
        dataset_sampler=sampler_config,
    )

    trainer = Trainer(train_args=trainer_args)
    trainer.run()


if __name__ == '__main__':
    ## Parse Args ##

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--name',
        default='llama',
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
    parser.add_argument(
        '--gpu_mem_constraint',
        default=64,
        type=int,
        help='the max memory usage constraint (GB) per GPU during nnscaler generating distribution plan, recommended to be 80 percent of GPU memory',
    )
    parser.add_argument(
        '--trace_strategy',
        default='reuse_cache',
        type=str,
        help='trace strategy control the function execution during tracing model graph, `cuda_run_cpu_offload` and `reuse_cache` are recommended, please read `docs/source/parallel_module.md` for more information',
    )
    parser.add_argument(
        '--enable_chunk_loss',
        action='store_true',
        help='enable chunk loss that exchanges the speed of training for the memory usage',
    )
    parser.add_argument(
        '--pipeline_pivots',
        default='',
        type=str,
        help='explore pipeline parallelism by specifying the pipeline pivots for autodist',
    )
    parser.add_argument(
        '--pipeline_nstages',
        default=1,
        type=str,
        help='specify the number of stages in the pipeline (use "1" to disable pipeline; use "auto" for autodist)',
    )
    parser.add_argument(
        '--recompute_modules',
        default='',
        type=str,
        help='specify the modules to recompute in autodist',
    )
    parser.add_argument(
        '--grad_accumulation_steps',
        default=4,
        type=int,
        help='number of gradient accumulation steps',
    )
    parser.add_argument(
        '--max_train_steps',
        default=None,
        type=int,
        help='max training steps',
    )
    parser.add_argument(
        '--attn_implementation',
        default='flash_attention_2',
        type=str,
        help='attn implementation, can be flash_attention_2, spda, eager',
    )
    parser.add_argument(
        '--enable_diff_attn',
        action='store_true',
        help='enable diff attention implementation, eager is normal diff attention, flash_attention_2 is diff flash attention, and spda diff attention is not currently supported',
    )
    parser.add_argument(
        '--enable_ring_attn',
        action='store_true',
        help='enable ring attention, currently only diff flash attention is supported',
    )
    args = parser.parse_args()
    if args.pipeline_nstages != 'auto':
        args.pipeline_nstages = int(args.pipeline_nstages)
        if args.pipeline_nstages > 1 and not args.pipeline_pivots:
            raise ValueError('pipeline_pivots must be specified when pipeline is enabled')

    if os.getenv('DETERMINISTIC'):  # reduce randomness for integration test
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.use_deterministic_algorithms(True)

    main(args)
