#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

# Credits: This logger implementation is inspired by project https://github.com/zhuzilin/ring-flash-attention

from typing import Optional, Tuple
from functools import reduce
import operator

import torch
import torch.distributed as dist


# copy from megatron/core/utils.py
class GlobalMemoryBuffer:
    """Global buffer to avoid dynamic memory allocations.
    Caller should ensure that buffers of the same name
    are not used concurrently."""

    def __init__(self):
        self.buffer = {}

    def get_tensor(self, tensor_shape, dtype, name):
        required_len = reduce(operator.mul, tensor_shape, 1)
        if (
            self.buffer.get((name, dtype), None) is None
            or self.buffer[(name, dtype)].numel() < required_len
        ):
            self.buffer[(name, dtype)] = torch.empty(
                required_len, dtype=dtype, device=torch.cuda.current_device(), requires_grad=False
            )

        return self.buffer[(name, dtype)][0:required_len].view(*tensor_shape)


def update_out_and_lse(
    out: Optional[torch.Tensor],
    lse: Optional[torch.Tensor],
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
    slice_=None,
) -> Tuple[torch.Tensor, torch.Tensor]:

    @torch.jit.script
    def _update_out_and_lse(
        out: torch.Tensor,
        lse: torch.Tensor,
        block_out: torch.Tensor,
        block_lse: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        block_out = block_out.to(torch.float32)
        block_lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)

        new_lse = lse + torch.log(1 + torch.exp(block_lse - lse))

        out = torch.exp(lse - new_lse) * out + torch.exp(block_lse - new_lse) * block_out

        lse = new_lse
        return out, lse

    if out is None:
        if slice_ is not None:
            raise RuntimeError("first update_out_and_lse should not pass slice_ args")
        out = block_out.to(torch.float32)
        lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)
    elif slice_ is not None:
        slice_out, slice_lse = out[slice_], lse[slice_]
        slice_out, slice_lse = _update_out_and_lse(
            slice_out, slice_lse, block_out, block_lse
        )
        out[slice_], lse[slice_] = slice_out, slice_lse
    else:
        out, lse = _update_out_and_lse(out, lse, block_out, block_lse)
    return out, lse


class RingComm:
    def __init__(self, process_group: dist.ProcessGroup):
        self._process_group = process_group
        self._ops = []
        self.rank = dist.get_rank(self._process_group)
        self.world_size = dist.get_world_size(self._process_group)
        self._reqs = None

        parts = self.world_size // 2
        self.ring_list = []
        for i in range(parts):
            self.ring_list.extend([i, self.world_size - i - 1])

        self.revert_rank = self.ring_list.index(self.rank)

        offset = ((dist.get_rank() // self.world_size) * self.world_size)
        self.send_rank = self.ring_list[(self.revert_rank + 1) % self.world_size] + offset
        self.recv_rank = self.ring_list[(self.revert_rank - 1) % self.world_size] + offset

    def send_recv(
        self, to_send: torch.Tensor, recv_tensor: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if recv_tensor is None:
            res = torch.empty_like(to_send)
        else:
            res = recv_tensor

        send_op = dist.P2POp(
            dist.isend, to_send, self.send_rank, group=self._process_group
        )
        recv_op = dist.P2POp(dist.irecv, res, self.recv_rank, group=self._process_group)
        self._ops.append(send_op)
        self._ops.append(recv_op)
        return res

    def commit(self):
        if self._reqs is not None:
            raise RuntimeError("commit called twice")
        self._reqs = dist.batch_isend_irecv(self._ops)

    def wait(self):
        if self._reqs is None:
            raise RuntimeError("wait called before commit")
        for req in self._reqs:
            req.wait()
        self._reqs = None
        self._ops = []


def shuffle_input(to_send: torch.Tensor, 
                  process_group: dist.ProcessGroup = None):
    
    if not to_send.is_contiguous():
        to_send = to_send.contiguous()

    # We must use outplace, otherwise it will raise error at backward due to inplace operations.
    # We can not change to_send directly and create a new tensor to store the result.
    to_send_f = torch.zeros_like(to_send)

    # assume the input sequence length is 8, and computation runs on 4 GPUs
    # the seq is represented as [0 1 2 3 4 5 6 7], world size is 4
    # the input status before `shuffle_input` is
    # - gpu A: [0 1]
    # - gpu B: [2 3]
    # - gpu C: [4 5]
    # - gpu D: [6 7]
    # the value of `to_send_slice` is
    # - gpu A: [1]
    # - gpu B: [3]
    # - gpu C: [5]
    # - gpu D: [7]
    block_seq_len = to_send.shape[1] // 2
    to_send_slice = to_send[:, block_seq_len:].contiguous()

    rank = dist.get_rank(process_group)
    world_size = dist.get_world_size(process_group)

    res = torch.zeros_like(to_send_slice)
    
    _ops = []
    offset = ((dist.get_rank() // world_size) * world_size)
    # rank  src_rank
    # 0     3
    # 1     2
    # 2     1
    # 3     0
    src_rank = (world_size - rank - 1) % world_size + offset
    send_op = dist.P2POp(
        dist.isend, to_send_slice, src_rank, group=process_group
    )
    recv_op = dist.P2POp(
        dist.irecv, res, src_rank, group=process_group)
    
    _ops.append(send_op)
    _ops.append(recv_op)
    
    response = dist.batch_isend_irecv(_ops)
    for resp in response:
        resp.wait()

    if rank >= world_size // 2: # D: 6 7, -> 1 6
        to_send_f[:, block_seq_len:] = to_send[:, :block_seq_len]
        to_send_f[:, :block_seq_len, ...] = res
    else:                       # A: 0 1, -> 0 7
        to_send_f[:, :block_seq_len] = to_send[:, :block_seq_len]
        to_send_f[:, block_seq_len:, ...] = res
    # after shuffle, the status of `to_send_f`
    # GPU A: [0 7]
    # GPU B: [2 5]
    # GPU C: [3 4]
    # GPU D: [1 6]
    
    return to_send_f


def recover_output(to_send: torch.Tensor, 
                   process_group: dist.ProcessGroup = None):

    if not to_send.is_contiguous():
        to_send = to_send.contiguous()
        
    to_send_f = torch.zeros_like(to_send)
    
    block_seq_len = to_send.shape[1] // 2
    
    rank = dist.get_rank(process_group)
    world_size = dist.get_world_size(process_group)
    
    if rank >= world_size // 2:
        to_send_slice = to_send[:, :block_seq_len, ...].contiguous()
    else:
        to_send_slice = to_send[:, block_seq_len:, ...].contiguous()
    res = torch.zeros_like(to_send_slice)
    
    assert to_send_slice.is_contiguous()
    assert res.is_contiguous()
    
    _ops = []
    offset = ((dist.get_rank() // world_size) * world_size)
    src_rank = (world_size - rank - 1) % world_size + offset
    send_op = dist.P2POp(
        dist.isend, to_send_slice, src_rank, group=process_group
    )
    recv_op = dist.P2POp(
        dist.irecv, res, src_rank, group=process_group)
    
    _ops.append(send_op)
    _ops.append(recv_op)
    
    response = dist.batch_isend_irecv(_ops)
    for resp in response:
        resp.wait()
            
    if rank >= world_size // 2:
        to_send_f[:, :block_seq_len] = to_send[:, block_seq_len:, ...]
        to_send_f[:, block_seq_len:] = res
    else:
        to_send_f[:, :block_seq_len] = to_send[:, :block_seq_len, ...]
        to_send_f[:, block_seq_len:] = res
        
    return to_send_f.contiguous()
