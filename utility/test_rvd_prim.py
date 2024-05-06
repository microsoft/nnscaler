# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=8 \
    utility/test_rvd_prim.py --prims allreduce

OMP_NUM_THREADS=4 torchrun \
    --nnode=2 --node_rank=$NODE_RANK --master_addr=node-0 \
    --nproc_per_node=8 \
    utility/test_rvd_prim.py --prims all
"""

from typing import Callable
import cube
import torch
import time
import argparse
from cube.profiler.timer import CudaTimer, print_each_rank

from cube.runtime.adapter.collectives import all_reduce, all_gather, reduce_scatter, all_to_all
from cube.runtime.device import DeviceGroup


def prim_allreduce(itensor, ranks, dim0=None, dim1=None):
    return all_reduce(itensor, ranks)


def bw_allreduce(itensor: torch.Tensor, ranks, sec_per_call: float):
    msg_size = itensor.nelement() * 4 / 1e9
    ndevs = len(ranks)
    algo_bw = msg_size / sec_per_call
    bus_bw = algo_bw * 2 * (ndevs - 1) / ndevs
    return algo_bw, bus_bw


def prim_allgather(itensor, ranks, dim0=0, dim1=None):
    return all_gather(itensor, dim0, ranks)


def bw_allgather(itensor: torch.Tensor, ranks, sec_per_call: float):
    ndevs = len(ranks)
    msg_size = itensor.nelement() * 4 / 1e9 * ndevs
    algo_bw = msg_size / sec_per_call
    bus_bw = algo_bw * (ndevs - 1) / ndevs
    return algo_bw, bus_bw


def prim_reducescatter(itensor, ranks, dim0=0, dim1=None):
    return reduce_scatter(itensor, dim0, ranks)


def bw_reducescatter(itensor: torch.Tensor, ranks, sec_per_call: float):
    msg_size = itensor.nelement() * 4 / 1e9
    ndevs = len(ranks)
    algo_bw = msg_size / sec_per_call
    bus_bw = algo_bw * (ndevs - 1) / ndevs
    return algo_bw, bus_bw


def prim_alltoall(itensor, ranks, dim0=0, dim1=1):
    return all_to_all(itensor, dim0, dim1, ranks)


def bw_alltoall(itensor: torch.Tensor, ranks, sec_per_call: float):
    msg_size = itensor.nelement() * 4 / 1e9
    ndevs = len(ranks)
    algo_bw = msg_size / sec_per_call
    bus_bw = algo_bw * (ndevs - 1) / ndevs
    return algo_bw, bus_bw


def prim_bw(prim: Callable, bandwidth: Callable, ranks, size, warmup=100, profile=100):
    if 'allgather' in prim.__name__:
        size = size // len(ranks)
    tensor: torch.Tensor = torch.zeros(size, device=torch.cuda.current_device())
    tensor = tensor.view(256, -1).contiguous()
    torch.distributed.barrier()
    # warm up
    for _ in range(warmup):
        _ = prim(tensor, ranks)
    # profile
    torch.cuda.synchronize()
    torch.distributed.barrier()
    tic = time.perf_counter()
    for _ in range(profile):
        _ = prim(tensor, ranks)
    torch.cuda.synchronize()
    toc = time.perf_counter()

    span = (toc - tic) / profile # seconds
    msg_size = tensor.nelement() * 4 // 1024 // 1024 # MB
    if 'allgather' in prim.__name__:
        msg_size = len(ranks) * tensor.nelement() * 4 // 1024 // 1024 # MB
    algo_bw, bus_bw = bandwidth(tensor, ranks, span)
    print_each_rank(
        '{} msg {} MB | wall-time(ms) algo-bw(GB/s) bus-bw(GB/s) {:.2f} {:.2f} {:.2f}'.format(
            prim.__name__, msg_size, span*1000, algo_bw, bus_bw
        ), rank_only=0
    )


if __name__ == '__main__':

    cube.init()

    parser = argparse.ArgumentParser(description='comm primitive')
    parser.add_argument('--prims', type=str, nargs='+', action='append', 
                        help='prims: all, allreduce, reducescatter, allgather, alltoall')
    parser.add_argument('--begin', type=int, default=1,
                        help='start message size in MB')
    parser.add_argument('--end', type=int, default=256,
                        help='end message size in MB')
    args = parser.parse_args()
    args.prims = args.prims[0]

    prims, bws = [], []
    if 'allreduce' in args.prims or 'all' in args.prims:
        prims.append(prim_allreduce)
        bws.append(bw_allreduce)
    if 'allgather' in args.prims or 'all' in args.prims:
        prims.append(prim_allgather)
        bws.append(bw_allgather)
    if 'reducescatter' in args.prims or 'all' in args.prims:
        prims.append(prim_reducescatter)
        bws.append(bw_reducescatter)
    if 'alltoall' in args.prims or 'all' in args.prims:
        prims.append(prim_alltoall)
        bws.append(bw_alltoall)

    ranks = tuple(range(DeviceGroup().world_size))
    CudaTimer(enable=False)
    for prim, bw in zip(prims, bws):
        print_each_rank(f'====> test start {prim.__name__}', rank_only=0)
        size = args.begin
        while size <= args.end:
            prim_bw(prim, bw, ranks, size * 1024 * 1024 // 4)
            size *= 2
        print_each_rank(f'====> test finish {prim.__name__}', rank_only=0)
