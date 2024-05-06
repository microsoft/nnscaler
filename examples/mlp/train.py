# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
PYTHONPATH=.:$PYTHONPATH torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    examples/mlp/train.py --policy PASMegatronTP
"""

import torch
from torch import nn
from functools import partial

import cube
from cube.profiler import CudaTimer
from cube.profiler.timer import print_each_rank
from cube.runtime.utils import create_dummy_dataloader

import examples.mlp.policy.gallery as gallery
from examples.utils import get_policy

import argparse

parser = argparse.ArgumentParser(description='MLP example')
parser.add_argument('--policy', type=str, help='policy choice, starting with "PAS"')
parser.add_argument('--dim', type=int, default=1024, help='model hidden size')
parser.add_argument('--layers', type=int, default=16, help='number of linear layers')
parser.add_argument('--gbs', type=int, default=64, help='global batch size')
parser.add_argument('--mbs', type=int, default=64, help='micro batch size')
parser.add_argument('--tp-size', type=int, default=2, help='tensor parallelism size only for Megatron policy')
args = parser.parse_args()

cube.init()

# get policy
policy = get_policy([gallery], args.policy)
policy = partial(policy, nmicros=args.gbs//args.mbs, tp_size=args.tp_size)

# =================== Semantic Model Description ====================

class MLP(nn.Module):
    def __init__(self, dim: int, nlayers: int):
        super().__init__()
        self.layers = torch.nn.ModuleList([])
        for _ in range(nlayers):
            self.layers.append(nn.Linear(dim, dim, bias=False))

    def forward(self, data):
        x = data
        for layer in self.layers:
            x = layer(x)
        loss = torch.sum(x)
        return loss


def train():

    model = MLP(dim=args.dim, nlayers=args.layers)
    dataloader = create_dummy_dataloader(
        torch.randn(args.dim, device=torch.cuda.current_device()),
        args.mbs,
    )

    # compile a training iteration
    @cube.compile(model, dataloader, PAS=policy)
    def train_iter(model, dataloader):
        data = next(dataloader)
        loss = model(data)
        loss.backward()
    # load generated model
    model = cube.utils.load_model()
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    CudaTimer(enable=False).warmup()
    dataloader = iter(dataloader)
    iter_num, warmup = 5, 2
    for step in range(iter_num):
        if step == warmup:
            CudaTimer(enable=True).start('e2e')
        train_iter(model, dataloader)
        optimizer.step()
        optimizer.zero_grad()
        if (step + 1) % 2 == 0:
            print_each_rank(f'iter [{step + 1}/{iter_num}]', rank_only=0)

    CudaTimer().stop('e2e')
    print_each_rank('e2e time (ms) per iteration: {} ms'.format(
          CudaTimer().duration(iter_num-warmup, field_name='e2e')))
    CudaTimer().print_all(times=iter_num-warmup)


if __name__ == '__main__':
    train()