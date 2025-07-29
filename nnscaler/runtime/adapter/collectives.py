"""
This module offers the wrap of communication primitives
based on `torch.distributed`. The use of these primitives standalone is typically
for non-autograd (e.g., inference) scenarios.

Every collective is implemented using out-of-place semantics.
"""

from typing import List, Tuple, Optional
import torch

from nnscaler.runtime.device import DeviceGroup
from nnscaler.profiler.timer import CudaTimer

from nnscaler.runtime.executor import AsyncCommHandler


def move(tensor: Optional[torch.Tensor], shape: Tuple[int], dtype: torch.dtype, src: int, dst: int, async_op=False):
    """
    Move a tensor from source device to destination device.
    """
    if not async_op:
        CudaTimer().start(field_name='comm', predefined=True)
    rank = torch.distributed.get_rank()
    work = None
    if rank == src:
        tensor = tensor.contiguous() if not tensor.is_contiguous() else tensor
        assert torch.is_tensor(tensor)
        if async_op:
            work = torch.distributed.isend(tensor, dst)
            # NOTE: we don't add isend work item into handler
        else:
            torch.distributed.send(tensor, dst)
    else:
        assert rank == dst
        tensor = torch.empty(shape, dtype=dtype,
            device=torch.cuda.current_device()
        )
        if async_op:
            work = torch.distributed.irecv(tensor, src)
            AsyncCommHandler().submit(tensor, [work])
        else:
            torch.distributed.recv(tensor, src)
    if not async_op:
        CudaTimer().stop(field_name='comm', predefined=True)
    return tensor


def all_reduce(tensor: torch.Tensor,
               ranks: List[int], async_op=False) -> torch.Tensor:
    """Allreduce"""
    if not async_op:
        CudaTimer().start(field_name='comm', predefined=True)
    tensor = tensor.contiguous() if not tensor.is_contiguous() else tensor
    tensor = tensor.detach().clone()
    group = DeviceGroup().get_group(ranks)

    if async_op:
        work = torch.distributed.all_reduce(tensor, group=group, async_op=True)
        AsyncCommHandler().submit(tensor, [work])
    else:
        torch.distributed.all_reduce(tensor, group=group)
    if not async_op:
        CudaTimer().stop(field_name='comm', predefined=True)
    return tensor


def all_gather(tensor: torch.Tensor, dim: int,
               ranks: Tuple[int], async_op=False) -> torch.Tensor:
    """Allgather"""
    if not async_op:
        CudaTimer().start(field_name='comm', predefined=True)
    tensor = tensor.contiguous() if not tensor.is_contiguous() else tensor
    group = DeviceGroup().get_group(ranks)
    tensor_list = [torch.empty_like(tensor) for _ in ranks]
    tensor_list[torch.distributed.get_rank(group)] = tensor.data
    work = torch.distributed.all_gather(tensor_list, tensor, group=group, async_op=async_op)
    if work:
        allgather_callback = lambda t: torch.concat(tuple(tensor_list), dim=dim)
        AsyncCommHandler().submit(tensor, [work], allgather_callback)
        otensor = tensor
    else:
        otensor = torch.concat(tuple(tensor_list), dim=dim)
    if not async_op:
        CudaTimer().stop(field_name='comm', predefined=True)
    return otensor


def reduce_scatter(tensor: torch.Tensor, dim: int,
                   ranks: Tuple[int], async_op=False) -> torch.Tensor:
    """ReduceScatter"""
    if not async_op:
        CudaTimer().start(field_name='comm', predefined=True)
    itensors = list(tensor.chunk(len(ranks), dim))
    for idx, t in enumerate(itensors):
        itensors[idx] = t.contiguous() if not t.is_contiguous() else t
    group = DeviceGroup().get_group(ranks)
    otensor = torch.empty_like(itensors[0], requires_grad=False)
    work = torch.distributed.reduce_scatter(otensor, itensors, group=group, async_op=async_op)
    if work:
        AsyncCommHandler().submit(otensor, [work])
    if not async_op:
        CudaTimer().stop(field_name='comm', predefined=True)
    return otensor


def all_to_all(tensor: torch.Tensor, idim: int, odim: int,
               ranks: Tuple[int], async_op=False) -> torch.Tensor:
    """All-to-all"""
    if not async_op:
        CudaTimer().start(field_name='comm', predefined=True)
    itensors = list(tensor.chunk(len(ranks), dim=odim))
    for idx, itensor in enumerate(itensors):
        itensors[idx] = itensor.contiguous() if not itensor.is_contiguous() else itensor
    otensors = [torch.empty_like(t) for t in itensors]
    group = DeviceGroup().get_group(ranks)
    work = torch.distributed.all_to_all(otensors, itensors, group=group, async_op=async_op)
    if work:
        all2all_callback = lambda t: torch.concat(tuple(otensors), dim=idim)
        AsyncCommHandler().submit(tensor, [work], all2all_callback)
        otensor = tensor
    else:
        otensor = torch.concat(tuple(otensors), dim=idim)
    if not async_op:
        CudaTimer().stop(field_name='comm', predefined=True)
    return otensor


def all_to_all_single(tensor: torch.Tensor, idim: int, odim: int,
                      ranks: Tuple[int], async_op: bool = False) -> torch.Tensor:
    """All-to-all for single tensor"""
    if not async_op:
        CudaTimer().start(field_name='comm', predefined=True)
    tensor = tensor.transpose(0, odim) if odim != 0 else tensor
    tensor = tensor.contiguous() if not tensor.is_contiguous() else tensor
    group = DeviceGroup().get_group(ranks)
    otensor = torch.empty_like(tensor)
    work = torch.distributed.all_to_all_single(otensor, tensor, group=group, async_op=async_op)
    
    def all2all_callback(t):
        t = t.transpose(0, odim) if odim != 0 else t
        return torch.concat(tuple(t.chunk(len(ranks), dim=odim)), dim=idim)
    
    if work:
        AsyncCommHandler().submit(tensor, [work], all2all_callback)
    else:
        otensor = all2all_callback(otensor)

    if not async_op:
        CudaTimer().stop(field_name='comm', predefined=True)
    return otensor


def chunk(itensor: torch.Tensor, dim: int, ranks: Tuple[int], async_op=False) -> torch.Tensor:
    """
    split dimension in n chunks and take idx-th chunk

    ranks (Tuple[int]): the order of split tensor.
    """
    group = DeviceGroup().get_group(ranks)
    idx = torch.distributed.get_rank(group)
    with torch.no_grad():
        otensor = itensor.chunk(len(ranks), dim)[idx]
        otensor = otensor.detach()
    return otensor


def rdscatter(itensor: torch.Tensor, shape: Tuple[int], dtype: torch.dtype,
              dim: int, src: int, dsts: Tuple[int], async_op=False):
    """
    RDScatter: split itensor at rank `src` along dim into `len(dsts)` chunks,
    and then send each chunk to `dst` devices.
    """
    if async_op:
        CudaTimer().start(field_name='comm', predefined=True)
    rank = torch.distributed.get_rank()
    if rank == src:
        with torch.no_grad():
            otensors = itensor.chunk(len(dsts), dim)
            for dst, otensor in zip(dsts, otensors):
                otensor = otensor.contiguous() if not otensor.is_contiguous() else otensor
                if async_op:
                    torch.distributed.isend(otensor, dst)
                else:
                    torch.distributed.send(otensor, dst)
        otensor = itensor
    else:
        assert rank in dsts
        shape = list(shape)
        shape[dim] = shape[dim] // len(dsts)
        otensor = torch.empty(
            shape, requires_grad=False, dtype=dtype,
            device=torch.cuda.current_device()
        )
        if async_op:
            work = torch.distributed.irecv(otensor, src)
            AsyncCommHandler().submit(otensor, [work])
        else:
            torch.distributed.recv(otensor, src)
    if async_op:
        CudaTimer().stop(field_name='comm', predefined=True)
    return otensor


def rvscatter(itensor: torch.Tensor, shape: Tuple[int], dtype: torch.dtype,
              src: int, dsts: Tuple[int], async_op=False):
    """
    src: global rank
    """
    if not async_op:
        CudaTimer().start(field_name='comm', predefined=True)
    group = DeviceGroup().get_group((src,) + dsts)
    rank = torch.distributed.get_rank()
    tensor: torch.Tensor = itensor / len(dsts) if src == rank else \
        torch.empty(shape, dtype=dtype, requires_grad=False)
    tensor = tensor.contiguous() if not tensor.is_contiguous() else tensor
    work = torch.distributed.broadcast(tensor, src, group=group, async_op=async_op)
    if work:
        AsyncCommHandler().submit(tensor, [work])
    if not async_op:
        CudaTimer().stop(field_name='comm', predefined=True)
    return tensor


def rdgather(itensor: torch.Tensor, shape: Tuple[int], dtype: torch.dtype,
             dim: int, srcs: Tuple[int], dst: int, async_op=False):
    """
    @param srcs Tuple[int]: global rank of each source device
    @param dst int: global rank of destination device
    """
    if not async_op:
        CudaTimer().start(field_name='comm', predefined=True)
    rank = torch.distributed.get_rank()
    if rank == dst:
        recv_tensors, works = [], []
        for src in srcs:
            tensor = torch.empty(shape, dtype=dtype, device=torch.cuda.current_device())
            recv_tensors.append(tensor)
            if async_op:
                work = torch.distributed.irecv(tensor, src)
                works.append(work)
            else:
                work = torch.distributed.recv(tensor, src)

        if async_op:
            rdgather_callback = lambda t: torch.cat(tuple(recv_tensors), dim=dim)
            AsyncCommHandler().submit(itensor, works, rdgather_callback)
            otensor = itensor
        else:
            otensor = torch.cat(tuple(recv_tensors), dim=dim)
    else:
        assert rank in srcs
        otensor = itensor.contiguous() if not itensor.is_contiguous() else itensor
        if async_op:
            torch.distributed.isend(otensor, dst)
        else:
            torch.distributed.send(otensor, dst)
    if not async_op:
        CudaTimer().stop(field_name='comm', predefined=True)
    return otensor


def rvgather(itensor: torch.Tensor, shape: Tuple[int], dtype: torch.dtype,
             srcs: Tuple[int], dst: int, async_op=False):
    """
    @param srcs Tuple[int]: global rank of each source device
    @param dst int: global rank of destination device
    """
    if not async_op:
        CudaTimer().start(field_name='comm', predefined=True)
    rank = torch.distributed.get_rank()
    group = DeviceGroup().get_group(srcs + (dst,))
    tensor = torch.zeros(shape, dtype=dtype, requires_grad=False) if rank == dst else itensor
    work = torch.distributed.reduce(tensor, dst, group=group, async_op=async_op)
    if work and rank == dst:
        AsyncCommHandler().submit(tensor, [work])
    if not async_op:
        CudaTimer().stop(field_name='comm', predefined=True)
    return tensor


def broadcast(itensor: torch.Tensor, shape: Tuple[int], dtype: torch.dtype, src: int, ranks: List[int], async_op=False) -> torch.Tensor:
    """
    Broadcast
    @param src: the global rank that holds tensor for broadcasting
    """
    if not async_op:
        CudaTimer().start(field_name='comm', predefined=True)
    rank = torch.distributed.get_rank()
    group = DeviceGroup().get_group(ranks)
    if rank == src:
        tensor = itensor.contiguous() if not itensor.is_contiguous() else itensor
    else:
        assert rank in ranks
        tensor = torch.empty(shape,
            device=torch.cuda.current_device(), requires_grad=False, dtype=dtype)
    work = torch.distributed.broadcast(tensor, src, group=group, async_op=async_op)
    if work and rank != src:
        AsyncCommHandler().submit(tensor, [work])
    if not async_op:
        CudaTimer().stop(field_name='comm', predefined=True)
    return tensor
