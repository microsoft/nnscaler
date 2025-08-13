#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import argparse
import os

import torch

from nnscaler.utils import set_default_logger_level
from nnscaler.cli.trainer import Trainer
from nnscaler.cli.trainer_args import (
    CheckpointConfig,
    DatasetConfig,
    ModelConfig,
    OptimizerConfig,
    TrainerArgs,
    DataloaderConfig,
    LogConfig,
    DatasetSamplerConfig,
    LRSchedulerConfig,
)
from nnscaler.parallel import ComputeConfig
from nnscaler.runtime.f16_optimizer import MixedPrecisionAdamW
from nnscaler.cli.loggers.tensorboard import TensorBoardLogger

from examples.transformers_utils import WrapperModel, aggregate_outputs_fn
from examples.longrope2.data.dataset import get_dataset
from examples.longrope2.rope_modifier import modify_rope_cls
from examples.warmup_schedular import WarmupCosineAnnealingLR


def main(args):

    if args.run_mode == 'run':
        broadcast_strategy = 'all'
    else:
        broadcast_strategy = 'none'

    set_default_logger_level('INFO')

    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
    modify_rope_cls(LlamaRotaryEmbedding)

    ## Setup Dataset ##

    dataset, collate_fn = get_dataset(args.dataset_path, args.model_id)

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
            'config': args.model_config,
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

    lrscheduler_config = LRSchedulerConfig(
        type=WarmupCosineAnnealingLR,
        args={
            'warmup_steps': args.warmup_steps,
            'T_max': args.max_train_steps,
        },
        interval='step',
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
        every_n_train_steps=200,
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
        lr_scheduler=lrscheduler_config,
        dataset=dataset_config,
        dataloader=dataloader_config,
        checkpoint=checkpoint_config,
        precision='bf16',
        max_epochs=None,
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
        '--model_config',
        default=None,
        type=str,
        help='transformers model config json path',
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
        '--enable-chunk-loss',
        action='store_true',
        help='enable chunk loss that exchanges the speed of training for the memory usage',
    )
    parser.add_argument(
        '--pipeline_pivots',
        default='',
        type=str,
        help='specify the pipeline pivots for autodist',
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
        default=1000,
        type=int,
        help='max training steps',
    )
    parser.add_argument(
        '--warmup_steps',
        default=40,
        type=int,
        help='warmup steps',
    )
    parser.add_argument(
        '--attn_implementation',
        default='flash_attention_2',
        type=str,
        help='attn implementation, can be flash_attention_2, spda, eager',
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
