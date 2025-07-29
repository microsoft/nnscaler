"""
This module offers autograd functions for communication
primitives. This is typically used in the training with tensor 
parallelism scenario.
"""

from typing import List, Tuple
import torch

from nnscaler.profiler.timer import CudaTimer
from nnscaler.runtime.device import DeviceGroup
from .collectives import (
    all_reduce,
    all_gather,
    reduce_scatter,
    all_to_all,
    all_to_all_single,
    chunk
)


class AllReduceIdentity(torch.autograd.Function):

    @staticmethod
    def forward(ctx, itensor: torch.Tensor, ranks: Tuple[int]):
        return all_reduce(itensor, ranks)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def allreduce_identity(tensor: torch.Tensor, ranks: List[int]):
    return AllReduceIdentity.apply(tensor, ranks)


class IdentityAllreduce(torch.autograd.Function):

    @staticmethod
    def forward(ctx, itensor: torch.Tensor, ranks: Tuple[int]):
        ctx._ranks = ranks
        return itensor

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        ranks = ctx._ranks
        grad = all_reduce(grad, ranks)
        return grad, None


def identity_allreduce(tensor: torch.Tensor, ranks: Tuple[int]) -> torch.Tensor:
    return IdentityAllreduce.apply(tensor, ranks)


class AllReduceAllReduce(torch.autograd.Function):

    @staticmethod
    def forward(ctx, itensor: torch.Tensor, ranks: Tuple[int]):
        ctx._ranks = ranks
        otensor = all_reduce(itensor, ranks)
        return otensor

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        ranks = ctx._ranks
        grad = all_reduce(grad, ranks)
        return grad, None


def allreduce_allreduce(tensor: torch.Tensor, ranks: Tuple[int]) -> torch.Tensor:
    return AllReduceAllReduce.apply(tensor, ranks)


class ReduceScatterAllGather(torch.autograd.Function):

    @staticmethod
    def forward(ctx, itensor: torch.Tensor, dim: int, ranks: Tuple[int]):
        ctx._ranks = ranks
        ctx._dim = dim
        return reduce_scatter(itensor, dim, ranks)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        ranks = ctx._ranks
        dim = ctx._dim
        grad = all_gather(grad, dim, ranks)
        return grad, None, None


def reducescatter_allgather(tensor: torch.Tensor, dim: int, ranks: List[int]):
    return ReduceScatterAllGather.apply(tensor, dim, ranks)


class AllGatherReduceScatter(torch.autograd.Function):

    @staticmethod
    def forward(ctx, itensor: torch.Tensor, dim: int, ranks: Tuple[int]):
        ctx._ranks = ranks
        ctx._dim = dim
        return all_gather(itensor, dim, ranks)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        ranks = ctx._ranks
        dim = ctx._dim
        grad = reduce_scatter(grad, dim, ranks)
        return grad, None, None


def allgather_reducescatter(tensor: torch.Tensor, dim: int, ranks: Tuple[int]) -> torch.Tensor:
    return AllGatherReduceScatter.apply(tensor, dim, ranks)


class AllGatherSplit(torch.autograd.Function):

    @staticmethod
    def forward(ctx, itensor: torch.Tensor, dim: int, ranks: Tuple[int]):
        ctx._ranks = ranks
        ctx._dim = dim
        return all_gather(itensor, dim, ranks)      

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        ranks = ctx._ranks
        dim = ctx._dim
        return chunk(grad, dim, ranks), None, None


def allgather_split(tensor: torch.Tensor, dim: int, ranks: Tuple[int]) -> torch.Tensor:
    return AllGatherSplit.apply(tensor, dim, ranks)


class SplitAllGather(torch.autograd.Function):

    @staticmethod
    def forward(ctx, itensor: torch.Tensor, dim: int, ranks: Tuple[int]):
        """
        ranks should be the global rank
        """
        ctx._ranks = ranks
        ctx._dim = dim
        return chunk(itensor, dim, ranks)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        ranks = ctx._ranks
        dim = ctx._dim
        grad = all_gather(grad, dim, ranks)
        return grad, None, None


def split_allgather(tensor, dim: int, ranks: Tuple[int]) -> torch.Tensor:
    return SplitAllGather.apply(tensor, dim, ranks)


class AllToAllAllToAll(torch.autograd.Function):

    @staticmethod
    def forward(ctx, itensor: torch.Tensor, idim: int, odim: int, ranks: Tuple[int]):
        ctx._ranks = ranks
        ctx._idim = idim
        ctx._odim = odim
        return all_to_all(itensor, idim, odim, ranks)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        ranks = ctx._ranks
        idim, odim = ctx._idim, ctx._odim
        grad = all_to_all(grad, odim, idim, ranks)
        return grad, None, None, None


class AllToAllAllToAllSingle(torch.autograd.Function):

    @staticmethod
    def forward(ctx, itensor: torch.Tensor, idim: int, odim: int, ranks: Tuple[int]):
        ctx._ranks = ranks
        ctx._idim = idim
        ctx._odim = odim
        return all_to_all_single(itensor, idim, odim, ranks)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        ranks = ctx._ranks
        idim, odim = ctx._idim, ctx._odim
        grad = all_to_all_single(grad, odim, idim, ranks)
        return grad, None, None, None


def alltoall_alltoall(itensor: torch.Tensor, idim: int, odim: int, ranks: Tuple[int]) -> torch.Tensor:
    return AllToAllAllToAllSingle.apply(itensor, idim, odim, ranks)


class ReduceBroadcast(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_: torch.Tensor, dst: int, ranks: List[int]):
        group = DeviceGroup().get_group(ranks)
        ctx._dst = dst
        ctx._group = group
        world_size = torch.distributed.get_world_size(group)
        if world_size == 1:
            return input_
        CudaTimer().start(field_name='comm', predefined=True)
        torch.distributed.reduce(input_, dst, group=group)
        CudaTimer().stop(field_name='comm', predefined=True)
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        src = ctx._dst
        group = ctx._group
        world_size = torch.distributed.get_world_size(group)
        if world_size == 1:
            return grad_output, None, None
        CudaTimer().start(field_name='comm', predefined=True)
        torch.distributed.broadcast(grad_output, src, group=group)
        CudaTimer().stop(field_name='comm', predefined=True)
        return grad_output, None, None


class BroadcastReduce(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_: torch.Tensor, src: int, ranks: List[int]):
        group = DeviceGroup().get_group(ranks)
        ctx._src = src
        ctx._group = group
        world_size = torch.distributed.get_world_size(group)
        if world_size == 1:
            return input_
        CudaTimer().start(field_name='comm', predefined=True)
        torch.distributed.broadcast(input_, src, group=group)
        CudaTimer().stop(field_name='comm', predefined=True)
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        dst = ctx._src
        group = ctx._group
        world_size = torch.distributed.get_world_size(group)
        if world_size == 1:
            return grad_output, None, None
        CudaTimer().start(field_name='comm', predefined=True)
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        torch.distributed.reduce(grad_output, dst, group=group)
        CudaTimer().stop(field_name='comm', predefined=True)
        return grad_output, None, None
