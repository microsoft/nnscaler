#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""
example:

PYTHONPATH=.:$PYTHONPATH OMP_NUM_THREADS=4 torchrun  \
    --nproc_per_node=4  \
    examples/vision/swin/train.py --policy pp --pp_size 4  --gbs 16  --fp16

PYTHONPATH=.:$PYTHONPATH OMP_NUM_THREADS=4 torchrun  \
    --nproc_per_node=4  \
    examples/vision/swin/train.py --policy 1f1b --pp_size 4  --gbs 16  --fp16

PYTHONPATH=.:$PYTHONPATH OMP_NUM_THREADS=4 torchrun  \
    --nproc_per_node=4  \
    examples/vision/swin/train.py --policy megatron --tp_size 4  --gbs 16  --fp16

PYTHONPATH=.:$PYTHONPATH OMP_NUM_THREADS=4 torchrun  \
    --nproc_per_node=4  \
    examples/vision/swin/train.py --policy megatron --tp_size 2 --dp_size 2  --gbs 16  --fp16

PYTHONPATH=.:$PYTHONPATH OMP_NUM_THREADS=4 torchrun  \
    --nproc_per_node=4  \
    examples/vision/swin/train.py --policy mesh_shard --tp_size 4  --gbs 16  --fp16

PYTHONPATH=.:$PYTHONPATH OMP_NUM_THREADS=4 torchrun  \
   --nproc_per_node=4   \
   examples/vision/swin/train.py --policy autodist --tp_size 2 --dp_size 2  --gbs 16  --fp16
"""

import logging
import itertools

import torch
from functools import partial
from examples.utils import init_random
from examples.vision.swin.model import Config, SwinTransformer, dummy_data

import nnscaler
from nnscaler.profiler.timer import CudaTimer, print_each_rank
from nnscaler.profiler.memory import memory_summary

import examples.vision.swin.policy.gallery as gallery

import argparse

import nnscaler.utils


def src_hash():
    import hashlib
    from pathlib import Path
    h = hashlib.md5()

    nnscaler_dir = Path(nnscaler.__file__).parent
    example_dir = nnscaler_dir.with_name('examples')
    for f in itertools.chain(nnscaler_dir.glob('**/*.py'), example_dir.glob('**/*.py')):
        h.update(f.stat().st_mtime_ns.to_bytes(8, 'little'))
    return h.hexdigest()


def train(args, compute_config: nnscaler.ComputeConfig):
    nnscaler.utils.set_default_logger_level(logging.INFO)
    cfg = Config()
    model = SwinTransformer()
    model = model.half() if args.fp16 else model
    gen_data = partial(dummy_data, args.mbs, torch.float16 if args.fp16 else torch.float32, cfg)

    init_random()
    DATA_SIZE = 1024
    data = []
    for _ in range(DATA_SIZE):
        data.append(gen_data())

    num_replicas = compute_config.runtime_ngpus // compute_config.plan_ngpus
    rank = torch.distributed.get_rank() // compute_config.plan_ngpus
    data = [data[i] for i in range(rank, len(data), num_replicas)]
    chunk_size = args.gbs // args.mbs
    data = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

    # get policy
    prefix = 'pas_'
    policy_name = prefix + args.policy
    if policy_name in gallery.__dict__:
        policy = gallery.__dict__[policy_name]
    else:
        policy = args.policy # use the builtin policies

    model: nnscaler.ParallelModule = nnscaler.parallelize(
        model,
        dummy_forward_args={'x': gen_data()},
        pas_policy=policy,
        compute_config=compute_config,
        reuse='moo',
        instance_name=args.policy
    )
    model.cuda()

    optimizer = nnscaler.build_optimizer(model, torch.optim.Adam, lr=5e-4, betas=(0.9, 0.999))

    torch.distributed.barrier()

    print_each_rank('model weight consumpition:')
    memory_summary()
    nparams = 0
    for param in model.parameters():
        nparams += param.nelement()
    print_each_rank(f'model parameter: {nparams}')

    iter_num = 5
    for idx in range(iter_num):
        model.train()

        # collect data
        samples = data[idx]

        model.train_step(samples)
        optimizer.step()
        optimizer.zero_grad()

    memory_summary()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='swin Train')
    parser.add_argument('--policy', type=str, help='PAS policy choice')
    parser.add_argument('--fp16', action='store_true', default=False,
                        help='use fp16 for the training')
    parser.add_argument('--dp_size', type=int, default=1,
                        help='size of data parallelism')
    parser.add_argument('--pp_size', type=int, default=1,
                        help='size of pipeline parallelism')
    parser.add_argument('--tp_size', type=int, default=1,
                        help='size of tensor parallelism')
    parser.add_argument('--zero', action='store_true', default=False,
                        help='use zero1 for the training')
    parser.add_argument('--mbs', type=int, default=4, help='micro batch size')
    parser.add_argument('--gbs', type=int, default=4, help='global batch size')

    args = parser.parse_args()

    nnscaler.init()

    if torch.distributed.get_world_size() != args.dp_size * args.pp_size * args.tp_size:
        raise ValueError('world size should be equal to dp_size * pp_size * tp_size')
    if args.gbs % (args.mbs * args.dp_size) != 0:
        raise ValueError('global batch size should be divisible by micro batch size')

    compute_config=nnscaler.ComputeConfig(
        plan_ngpus=args.pp_size * args.tp_size,
        runtime_ngpus=torch.distributed.get_world_size(),
        use_zero=args.zero,
        use_end2end=True,
        constant_folding=True,

        pas_config={
            # customized settings that can affect code generation.
            '_pas_name': args.policy,
            '_gbs': args.gbs,
            '_pp_size': args.pp_size,
            '_tp_size': args.tp_size,
            '_dp_size': args.dp_size,
            # for autodist only
            'update_freq': args.gbs // args.mbs// args.dp_size,
            'use_fp16': args.fp16,
            'explore_pipeline': args.pp_size > 1,
            # for pp
            'pipeline_nmicros': args.gbs // args.mbs // args.dp_size,
            'pipeline_nstages': args.pp_size,
        },
        user_config={
            'mbs': args.mbs,
            'fp16': args.fp16,
            'src_hash': src_hash(),
        }
    )

    train(args, compute_config)
