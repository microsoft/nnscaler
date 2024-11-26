#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from typing import List

import nnscaler
import torch
import pytest

from ..launch_torchrun import launch_torchrun, clone_to_cpu


def _init_distributed(expected_devices=2, backend=None):
    if torch.cuda.is_available() and torch.cuda.device_count() >= expected_devices:
        torch.distributed.init_process_group(backend='nccl')
        rank = torch.distributed.get_rank()
        torch.cuda.set_device(rank)
        torch.set_default_device(f'cuda:{rank}')
    else:
        torch.distributed.init_process_group(backend='gloo')
        torch.set_default_device('cpu')


def _get_tensor(shape: List[int], device=None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    tensor = torch.randn(shape, dtype=dtype, device=device or torch.cuda.current_device())
    return tensor


def _move_worker(async_op: bool):
    shape = [128, 256]

    tensor = _get_tensor(shape)
    tensor = nnscaler.runtime.adapter.move(tensor, shape, torch.float32, 0, 1, async_op=async_op)

    if async_op:
        tensor = nnscaler.runtime.executor.AsyncCommHandler().wait(tensor)
    return clone_to_cpu(tensor)


def _allreduce_worker(async_op: bool):
    shape = [128, 256]

    tensor = _get_tensor(shape)
    sum_tensor = tensor.clone().detach()
    sum_tensor = nnscaler.runtime.adapter.all_reduce(sum_tensor, [0, 1], async_op=async_op)

    if async_op:
        sum_tensor = nnscaler.runtime.executor.AsyncCommHandler().wait(sum_tensor)
    return (clone_to_cpu(tensor), clone_to_cpu(sum_tensor))


def _allgather_worker(async_op: bool):
    shape = [128, 256]

    tensor = _get_tensor(shape)
    otensor = nnscaler.runtime.adapter.all_gather(tensor, 0, [0, 1], async_op=async_op)

    if async_op:
        otensor = nnscaler.runtime.executor.AsyncCommHandler().wait(otensor)
    return (clone_to_cpu(tensor), clone_to_cpu(otensor))


def _reduce_scatter_worker(async_op: bool):
    shape = [128, 256]

    tensor = _get_tensor(shape)
    otensor = nnscaler.runtime.adapter.reduce_scatter(tensor, 0, [0, 1], async_op=async_op)

    if async_op:
        otensor = nnscaler.runtime.executor.AsyncCommHandler().wait(otensor)

    return (clone_to_cpu(tensor), clone_to_cpu(otensor))


def _all2all_worker(async_op):
    shape = [128, 256]

    tensor = _get_tensor(shape)

    # # synchronize
    otensor = nnscaler.runtime.adapter.all_to_all(tensor, 0, 1, [0, 1], async_op=async_op)

    if async_op:
        otensor = nnscaler.runtime.executor.AsyncCommHandler().wait(otensor)

    return (clone_to_cpu(tensor), clone_to_cpu(otensor))


def _2gpu_worker():
    _init_distributed(2)
    result = {}
    result['move'] = _move_worker(False)
    result['move_async'] = _move_worker(True)
    result['allreduce'] = _allreduce_worker(False)
    result['allreduce_async'] = _allreduce_worker(True)
    result['allgather'] = _allgather_worker(False)
    result['allgather_async'] = _allgather_worker(True)
    result['reduce_scatter'] = _reduce_scatter_worker(False)
    result['reduce_scatter_async'] = _reduce_scatter_worker(True)
    result['all2all'] = _all2all_worker(False)
    result['all2all_async'] = _all2all_worker(True)

    return result

@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason='lack of gpu devices')
def test_2gpu():
    results = launch_torchrun(2, _2gpu_worker)

    for op in ['', '_async']:
        # check move
        outputs = results[0][f'move{op}'], results[1][f'move{op}']
        assert torch.equal(outputs[0], outputs[1])

        # check allreduce
        outputs = results[0][f'allreduce{op}'], results[1][f'allreduce{op}']
        assert torch.equal(outputs[0][1], outputs[1][1])
        assert torch.equal(outputs[0][0] + outputs[1][0], outputs[0][1])

        # check allgather
        outputs = results[0][f'allgather{op}'], results[1][f'allgather{op}']
        result = torch.concat([outputs[0][0], outputs[1][0]], dim=0)
        assert torch.equal(outputs[0][1], outputs[1][1])
        assert torch.equal(result, outputs[0][1])

        # check reduce_scatter
        outputs = results[0][f'reduce_scatter{op}'], results[1][f'reduce_scatter{op}']
        result = (outputs[0][0] + outputs[1][0]).chunk(2, dim=0)
        assert torch.equal(outputs[0][1], result[0])
        assert torch.equal(outputs[1][1], result[1])

        # check all2all
        outputs = results[0][f'all2all{op}'], results[1][f'all2all{op}']
        in0 = outputs[0][0].chunk(2, dim=1)
        in1 = outputs[1][0].chunk(2, dim=1)
        out0 = torch.concat([in0[0], in1[0]], dim=0)
        out1 = torch.concat([in0[1], in1[1]], dim=0)
        assert torch.equal(outputs[0][1], out0)
        assert torch.equal(outputs[1][1], out1)


def _rdscatter_worker(async_op):
    shape = [128, 256]

    tensor = _get_tensor(shape)
    otensor = nnscaler.runtime.adapter.rdscatter(
        tensor, shape, torch.float32, dim=0, src=0, dsts=[1,2], async_op=async_op)

    if async_op:
        otensor = nnscaler.runtime.executor.AsyncCommHandler().wait(otensor)

    return (clone_to_cpu(tensor), clone_to_cpu(otensor))


def _rdgather_worker(async_op):
    shape = [128, 256]

    tensor = _get_tensor(shape)
    otensor = nnscaler.runtime.adapter.rdgather(
        tensor, shape, torch.float32, dim=0, srcs=[1,2], dst=0)

    if async_op:
        otensor = nnscaler.runtime.executor.AsyncCommHandler().wait(otensor)

    return (clone_to_cpu(tensor), clone_to_cpu(otensor))


def _broadcast_worker(async_op):
    shape = [128, 256]

    tensor = _get_tensor(shape)

    # synchronize
    otensor = nnscaler.runtime.adapter.broadcast(
        tensor, shape, torch.float32, src=0, ranks=[0,1,2])
    if async_op:
        otensor = nnscaler.runtime.executor.AsyncCommHandler().wait(otensor)

    return (clone_to_cpu(tensor), clone_to_cpu(otensor))


def _3gpu_worker():
    _init_distributed(3)
    result = {}
    result['rdscatter'] = _rdscatter_worker(False)
    result['rdscatter_async'] = _rdscatter_worker(True)
    result['rdgather'] = _rdgather_worker(False)
    result['rdgather_async'] = _rdgather_worker(True)
    result['broadcast'] = _broadcast_worker(False)
    result['broadcast_async'] = _broadcast_worker(True)
    return result


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 3, reason='lack of gpu devices')
def test_3gpu():
    results = launch_torchrun(3, _3gpu_worker)

    for op in ['', '_async']:
        # check rdscatter
        outputs = results[0][f'rdscatter{op}'], results[1][f'rdscatter{op}'], results[2][f'rdscatter{op}']
        result = outputs[0][0].chunk(2, dim=0)
        assert torch.equal(outputs[1][1], result[0])
        assert torch.equal(outputs[2][1], result[1])

        # check rdgather
        outputs = results[0][f'rdgather{op}'], results[1][f'rdgather{op}'], results[2][f'rdgather{op}']
        result = torch.cat((outputs[1][0], outputs[2][0]), dim=0)
        assert torch.equal(outputs[0][1], result)

        # check broadcast
        outputs = results[0][f'broadcast{op}'], results[1][f'broadcast{op}'], results[2][f'broadcast{op}']
        assert torch.equal(outputs[0][0], outputs[0][1])
        assert torch.equal(outputs[0][0], outputs[1][1])
        assert torch.equal(outputs[0][0], outputs[2][1])
