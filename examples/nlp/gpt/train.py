# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
example:

PYTHONPATH=.:$PYTHONPATH OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=4 \
    examples/nlp/gpt/train.py --policy PASMegatronTP --fp16
"""


import torch
import logging
from functools import partial

from model import GPT, Config
from model import get_gpt_dummy_dataloader

import cube
from cube.profiler.timer import CudaTimer, print_each_rank
from cube.profiler.memory import memory_summary

import examples.nlp.gpt.policy.spmd as spmd
import examples.nlp.gpt.policy.mpmd as mpmd

from examples.utils import get_policy

import argparse

parser = argparse.ArgumentParser(description='GPT Train')

parser.add_argument('--policy', type=str, help='PAS policy choice, starting with PAS')
parser.add_argument('--fp16', action='store_true', default=False,
                    help='use fp16 for the training')
parser.add_argument('--mbs', type=int, default=8,
                    help='micro-batch size')
parser.add_argument('--gbs', type=int, default=8,
                    help='global batch size')
parser.add_argument('--dp', type=int, default=1, 
                    help='data parallel size, only for megatron')
parser.add_argument('--tp', type=int, default=1,
                    help='tensor parallel size, only for megatron')

# arch
parser.add_argument('--layers', type=int, default=4,
                    help='number of transformer layers')
parser.add_argument('--hidden', type=int, default=1024,
                    help='hidden size')
parser.add_argument('--heads', type=int, default=16,
                    help='number of attention heads')
parser.add_argument('--seqlen', type=int, default=1024,
                    help='sequence length')
args = parser.parse_args()


cube.init()
cube.set_logger_level(logging.WARN)
logging.getLogger('cube.compiler').setLevel(logging.INFO)

# get policy
policy = get_policy([spmd, mpmd], args.policy)
policy = partial(policy, 
    nmicros=args.gbs//args.mbs, 
    dp_size=args.dp,
    tp_size=args.tp
)


def train():

    config = Config(
        hidden=args.hidden,
        layers=args.layers,
        heads=args.heads,
        ffn_hidden_dim=4*args.hidden,
        num_embeddings=51200,
        seqlen=args.seqlen,   
    )
    model = GPT(config)
    model = model if not args.fp16 else model.half()
    dataloader = get_gpt_dummy_dataloader(args.mbs, Config)

    @cube.compile(model, dataloader, PAS=policy)
    def train_iter(model, dataloader):
        input_ids, position_ids = next(dataloader)
        loss = model(input_ids, position_ids)
        loss.backward()
    model = cube.utils.load_model()

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-05, betas=(0.9, 0.98))

    torch.distributed.barrier()
    print_each_rank('model weight consumpition:', rank_only=0)
    memory_summary()

    CudaTimer().warmup()
    dataloader = iter(dataloader)
    iter_num, warmup = 5, 2
    for step in range(iter_num):
        if step == warmup:
            CudaTimer(enable=True).start('e2e')

        train_iter(model, dataloader)
        optimizer.step()
        optimizer.zero_grad()

        if step == 0:
            print_each_rank('passed first iteration')
        if (step + 1) % 2 == 0:
            print_each_rank(f'iter [{step + 1}/{iter_num}]', rank_only=0)

    CudaTimer().stop('e2e')
    print_each_rank('e2e time (ms) per iteration: {} ms'.format(
          CudaTimer().duration(iter_num-warmup, field_name='e2e')))
    CudaTimer().print_all(times=iter_num-warmup)

    memory_summary()


if __name__ == '__main__':

    cube.init()
    train()