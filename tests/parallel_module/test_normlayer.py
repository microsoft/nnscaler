#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import uuid
import torch.distributed as dist
import tempfile
import torch
import pytest
import random
from unittest.mock import patch

import nnscaler
from nnscaler.graph.graph import IRGraph
from nnscaler.ir.operator import IRFwOperation
from nnscaler.runtime.device import DeviceGroup
from tests.parallel_module.test_gencode import _gencode_contains
from nnscaler.graph.function.wrapnn import convert_to_wrapnn, wrapnn, NnScalerBatchNorm2d, undo_convert_to_wrapnn, _ORIGINAL_MODULE_ATTR
from nnscaler.parallel import parallelize, ComputeConfig
from torch.nn.parallel import DistributedDataParallel as DDP

from tests.utils import retry, init_random
from .common import init_distributed
from ..launch_torchrun import launch_torchrun
from torch.distributed.run import elastic_launch, LaunchConfig
from torch.distributed.elastic.multiprocessing.errors import ChildFailedError


def policy(graph: IRGraph, resource: ComputeConfig, dim: int) -> IRGraph:
    ngpus = resource.plan_ngpus
    partitioned = False
    for idx, node in enumerate(graph.select(ntype=IRFwOperation)):
        if (
            not partitioned
            and node.signature == "nnscaler.graph.function.wrapnn.wrap_batchnorm2d_func"
        ):
            print("Partitioned node: ", node)
            sub_nodes = graph.partition(
                node, node.algorithm("dim"), idx=0, dim=dim, num=ngpus
            )
            partitioned = True
        elif (
            not partitioned
            and node.signature
            == "nnscaler.graph.function.wrapnn.wrap_instancenorm2d_func"
        ):
            print("Partitioned node: ", node)
            sub_nodes = graph.partition(
                node, node.algorithm("dim"), idx=0, dim=0, num=ngpus
            )
            partitioned = True
        else:
            sub_nodes = graph.replicate(node, times=ngpus)
        for idx, sub_node in enumerate(sub_nodes):
            graph.assign(sub_node, idx)
    assert partitioned, f"expect instancenorm / batchnorm in graph, but not found."
    return graph


def compute_error(tensor1, tensor2):
    mean_abs_error = torch.abs(tensor1 - tensor2).mean().item()
    max_abs_error = torch.abs(tensor1 - tensor2).max().item()
    return mean_abs_error, max_abs_error


def generate_parallel_data(size, device, dtype):
    shared_data = [torch.randn(size, device=device, dtype=dtype) for _ in range(2)]
    return shared_data


class BatchNorm2dModule(torch.nn.Module):
    def __init__(self):
        super(BatchNorm2dModule, self).__init__()
        self.bn = torch.nn.BatchNorm2d(8)

    def forward(self, x):
        return self.bn(x)


def _gencode_batchnorm2d_function(tempdir, config, pas_policy):
    init_distributed()
    m = BatchNorm2dModule().cuda()
    x = torch.randn(8, 8, 32, 32).cuda()

    with patch("nnscaler.graph.function.wrapnn.undo_convert_to_wrapnn", side_effect=undo_convert_to_wrapnn) as px:
        m_new = parallelize(
            m,
            {"x": x},
            pas_policy,
            config,
            gen_savedir=tempdir,
            load_module=True,
            reuse="override",
        )
    px.assert_called()
    # bn should be restored after parallelize
    assert isinstance(m.bn, torch.nn.BatchNorm2d)
    assert not hasattr(m.bn, _ORIGINAL_MODULE_ATTR)

    assert m_new is not None
    m_new.train()
    output = m_new(x)
    assert output is not None

    bn = BatchNorm2dModule().cuda()
    bn.train()
    ref_output = bn(x)
    assert torch.equal(
        [y for x, y in m_new.named_buffers() if x.startswith('bn_running_mean_')][0],
        bn.bn.running_mean
    ), "Custom output does not match PyTorch output"

    assert torch.equal(
        [y for x, y in m_new.named_buffers() if x.startswith('bn_running_var_')][0],
        bn.bn.running_var
    ), "Custom output does not match PyTorch output"

    assert torch.equal(
        output, ref_output
    ), "Custom output does not match PyTorch output"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="lack of GPU devices")
def test_codegen_batchnorm2d_1_1():
    with tempfile.TemporaryDirectory() as tempdir:
        launch_torchrun(
            1, _gencode_batchnorm2d_function, tempdir, ComputeConfig(1, 1), "dp"
        )


def _gencode_batchnorm2d_function_2(tempdir, config, pas_policy):
    nnscaler.init()
    rank_id = dist.get_rank()
    dtype = torch.bfloat16
    init_random()
    device = torch.device(f"cuda:{rank_id}")

    m = BatchNorm2dModule().to(device)
    shared_data = generate_parallel_data((8, 8, 32, 32), device, dtype)
    x_part = shared_data[rank_id]

    m_new = parallelize(
        m,
        {"x": x_part},
        pas_policy,
        config,
        gen_savedir=tempdir,
        load_module=True,
        reuse="override",
    )
    m_new.to(device)
    assert m_new is not None
    m_new.train()
    output = m_new(x_part)
    assert output is not None

    gather_output = [torch.empty_like(output) for _ in range(2)]
    dist.all_gather(gather_output, output)
    y_output = torch.cat(gather_output, dim=0)

    bn = BatchNorm2dModule().to(device)
    s_bn = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
        bn, process_group=dist.new_group([0, 1])
    )
    s = DDP(s_bn, device_ids=[rank_id])
    s.train()
    s_output = s(x_part)
    s_gather_output = [torch.empty_like(s_output) for _ in range(2)]
    dist.all_gather(s_gather_output, s_output)
    sync_output = torch.cat(s_gather_output, dim=0)

    assert torch.equal(
        y_output, sync_output
    ), "Custom output does not match PyTorch output"

    y = torch.cat(shared_data, dim=0)
    model = BatchNorm2dModule().cuda()
    model.train()
    output = model(y)
    current_mean_error, current_max_error = compute_error(output, y_output)
    mean_error, max_error = compute_error(sync_output, output)
    assert (current_mean_error - mean_error) == 0 and (
        current_max_error - max_error
    ) == 0, "Custom output is not the same as PyTorch output error"


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Need at least 2 GPU devices")
def test_codegen_batchnorm2d_1_2():
    with tempfile.TemporaryDirectory() as tempdir:
        launch_torchrun(
            2, _gencode_batchnorm2d_function_2, tempdir, ComputeConfig(1, 2), "dp"
        )


def _gencode_batchnorm2d_function_4(tempdir, config, pas_policy, dim):
    nnscaler.init()
    rank_id = dist.get_rank()
    dtype = torch.bfloat16
    init_random()
    device = torch.device(f"cuda:{rank_id}")

    m = BatchNorm2dModule().to(device)

    x_list = generate_parallel_data((8, 8, 32, 32), device, dtype)
    x = x_list[rank_id // 2]

    m_new = parallelize(
        m,
        {"x": x},
        lambda graph, resource: pas_policy(graph, resource, dim),
        config,
        gen_savedir=tempdir,
        load_module=True,
        reuse="override",
    )
    m_new.to(device)
    assert m_new is not None
    m_new.train()
    output = m_new(x)
    assert output is not None

    gather_output = [torch.empty_like(output) for _ in range(4)]
    dist.all_gather(gather_output, output)
    y_output = torch.cat([gather_output[0], gather_output[2]], dim=0)

    y = torch.cat([x_list[0], x_list[1]], dim=0)
    bn = BatchNorm2dModule().cuda()
    bn.train()
    ref_output = bn(y)
    current_mean_error, current_max_error = compute_error(y_output, ref_output)
    assert (
        current_mean_error
    ) < 1e-6, "Custom output is not the same as PyTorch output error"

    x = torch.chunk(x_list[rank_id // 2], 2, dim=0)[rank_id % 2]

    bn = BatchNorm2dModule().to(device)
    s_bn = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
        bn, process_group=dist.new_group([0, 1, 2, 3])
    )
    s = DDP(s_bn, device_ids=[rank_id])
    s.train()
    s_output = s(x)
    s_gather_output = [torch.empty_like(s_output) for _ in range(4)]
    dist.all_gather(s_gather_output, s_output)
    sync_output = torch.cat([s_gather_output[0], s_gather_output[1]], dim=0)
    sync_output_all = torch.cat(
        [
            s_gather_output[0],
            s_gather_output[1],
            s_gather_output[2],
            s_gather_output[3],
        ],
        dim=0,
    )

    assert torch.equal(
        gather_output[0], sync_output
    ), "Custom output does not match PyTorch SyncBatchNorm output"
    assert torch.equal(
        y_output, sync_output_all
    ), "Custom output does not match PyTorch SyncBatchNorm output"


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="Need at least 4 GPU devices")
@pytest.mark.parametrize("dim", [0, 1])
def test_codegen_batchnorm2d_2_4(dim):
    with tempfile.TemporaryDirectory() as tempdir:
        launch_torchrun(
            4,
            _gencode_batchnorm2d_function_4,
            tempdir,
            ComputeConfig(2, 4),
            policy,
            dim,
        )


def _gencode_batchnorm2d_function_eval(tempdir, config, pas_policy):
    init_distributed()
    m = BatchNorm2dModule().cuda()
    x = torch.randn(8, 8, 32, 32).cuda()
    m_new = parallelize(
        m,
        {"x": x},
        pas_policy,
        config,
        gen_savedir=tempdir,
        load_module=True,
        reuse="override",
    )
    assert m_new is not None
    m_new.eval()
    output = m_new(x)
    assert output is not None
    bn = BatchNorm2dModule().cuda()
    bn.eval()
    ref_output = bn(x)

    assert torch.equal(
        [y for x, y in m_new.named_buffers() if x.startswith('bn_running_mean_')][0],
        bn.bn.running_mean
    ), "Custom output does not match PyTorch output"

    assert torch.equal(
        [y for x, y in m_new.named_buffers() if x.startswith('bn_running_var_')][0],
        bn.bn.running_var
    ), "Custom output does not match PyTorch output"


    assert torch.equal(
        output, ref_output
    ), "Custom output does not match PyTorch output"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="lack of GPU devices")
def test_codegen_batchnorm2d_eval_1_1():
    with tempfile.TemporaryDirectory() as tempdir:
        launch_torchrun(
            1, _gencode_batchnorm2d_function_eval, tempdir, ComputeConfig(1, 1), "dp"
        )


def _gencode_batchnorm2d_function_eval_2(tempdir, config, pas_policy):
    nnscaler.init()
    rank_id = dist.get_rank()
    dtype = torch.bfloat16
    init_random()
    device = torch.device(f"cuda:{rank_id}")

    m = BatchNorm2dModule().to(device)
    shared_data = generate_parallel_data((4, 8, 32, 32), device, dtype)
    x_part = shared_data[rank_id]

    m_new = parallelize(
        m,
        {"x": x_part},
        pas_policy,
        config,
        gen_savedir=tempdir,
        load_module=True,
        reuse="override",
    )
    m_new.to(device)
    assert m_new is not None
    m_new.eval()
    output = m_new(x_part)
    assert output is not None

    gather_output = [torch.empty_like(output) for _ in range(2)]
    dist.all_gather(gather_output, output)
    y_output = torch.cat(gather_output, dim=0)

    bn = BatchNorm2dModule().to(device)
    s_bn = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
        bn, process_group=dist.new_group([0, 1])
    )

    s = DDP(s_bn, device_ids=[rank_id])
    s.eval()
    s_output = s(x_part)
    s_gather_output = [torch.empty_like(s_output) for _ in range(2)]
    dist.all_gather(s_gather_output, s_output)
    sync_output = torch.cat(s_gather_output, dim=0)

    assert torch.equal(
        y_output, sync_output
    ), "Custom output does not match PyTorch output"

    y = torch.cat(shared_data, dim=0)
    model = BatchNorm2dModule().cuda()
    model.eval()
    output = model(y)
    current_mean_error, current_max_error = compute_error(output, y_output)
    ref_mean_error, ref_max_error = compute_error(sync_output, output)
    assert (
        abs(current_mean_error - ref_mean_error) == 0
        and abs(current_max_error - ref_max_error) == 0
    ), "Custom output is not the same as PyTorch output error"
    assert torch.allclose(
        output, y_output, atol=1e-6
    ), "Custom output does not match PyTorch output"


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Need at least 2 GPU devices")
def test_codegen_batchnorm2d_eval_1_2():
    with tempfile.TemporaryDirectory() as tempdir:
        launch_torchrun(
            2, _gencode_batchnorm2d_function_eval_2, tempdir, ComputeConfig(1, 2), "dp"
        )


def _gencode_batchnorm2d_function_eval_4(tempdir, config, pas_policy, dim):
    nnscaler.init()
    rank_id = dist.get_rank()
    world_size = dist.get_world_size()
    dtype = torch.bfloat16
    init_random()
    device = torch.device(f"cuda:{rank_id}")

    m = BatchNorm2dModule().to(device)

    x_list = generate_parallel_data((8, 8, 32, 32), device, dtype)
    x = x_list[rank_id // 2]

    m_new = parallelize(
        m,
        {"x": x},
        lambda graph, resource: pas_policy(graph, resource, dim),
        config,
        gen_savedir=tempdir,
        load_module=True,
        reuse="override",
    )
    m_new.to(device)
    assert m_new is not None
    m_new.eval()
    output = m_new(x)
    assert output is not None

    gather_output = [torch.empty_like(output) for _ in range(4)]
    dist.all_gather(gather_output, output)
    y_output = torch.cat([gather_output[0], gather_output[2]], dim=0)

    y = torch.cat([x_list[0], x_list[1]], dim=0)
    bn = BatchNorm2dModule().cuda()
    bn.eval()
    ref_output = bn(y)
    current_mean_error, current_max_error = compute_error(y_output, ref_output)
    assert (
        current_mean_error
    ) < 1e-6, "Custom output is not the same as PyTorch output error"
    assert torch.allclose(
        y_output, ref_output, 1e-6
    ), "Custom output does not match PyTorch output"

    x = torch.chunk(x_list[rank_id // 2], 2, dim=0)[rank_id % 2]

    bn = BatchNorm2dModule().to(device)
    s_bn = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
        bn, process_group=dist.new_group([0, 1, 2, 3])
    )
    s = DDP(s_bn, device_ids=[rank_id])
    s.eval()
    s_output = s(x)
    s_gather_output = [torch.empty_like(s_output) for _ in range(4)]
    dist.all_gather(s_gather_output, s_output)
    sync_output = torch.cat([s_gather_output[0], s_gather_output[1]], dim=0)
    sync_output_all = torch.cat(
        [
            s_gather_output[0],
            s_gather_output[1],
            s_gather_output[2],
            s_gather_output[3],
        ],
        dim=0,
    )

    assert torch.equal(
        gather_output[0], sync_output
    ), "Custom output does not match PyTorch SyncBatchNorm output"
    assert torch.equal(
        y_output, sync_output_all
    ), "Custom output does not match PyTorch SyncBatchNorm output"


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="Need at least 4 GPU devices")
@pytest.mark.parametrize("dim", [0, 1])
def test_codegen_batchnorm2d_eval_2_4(dim):
    with tempfile.TemporaryDirectory() as tempdir:
        launch_torchrun(
            4,
            _gencode_batchnorm2d_function_eval_4,
            tempdir,
            ComputeConfig(2, 4),
            policy,
            dim,
        )


class InstanceNorm2dModule(torch.nn.Module):
    def __init__(self):
        super(InstanceNorm2dModule, self).__init__()
        self.inorm = torch.nn.InstanceNorm2d(4)
        self.inorm.running_mean = torch.zeros(4)
        self.inorm.running_var = torch.ones(4)

    def forward(self, x):
        return self.inorm(x)


def _gencode_instancenorm2d_function(tempdir, config, pas_policy):
    init_distributed()
    m = InstanceNorm2dModule().cuda()
    m_new = parallelize(
        m,
        {"x": torch.randn(4, 4, 32, 32).cuda()},
        lambda graph, resource: pas_policy(graph, resource, dim=0),
        config,
        gen_savedir=tempdir,
        load_module=True,
        reuse="override",
    )
    assert m_new is not None
    x = torch.randn(4, 4, 32, 32).cuda()
    m_new.train()
    output = m_new(x)
    assert output is not None
    bn = torch.nn.InstanceNorm2d(4).cuda()
    bn.running_mean = torch.zeros(4)
    bn.running_var = torch.ones(4)
    bn.train()
    ref_output = bn(x)
    assert torch.equal(
        output, ref_output
    ), "Custom output does not match PyTorch output in training mode"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="lack of GPU devices")
def test_codegen_instancenorm2d_1_1():
    with tempfile.TemporaryDirectory() as tempdir:
        launch_torchrun(
            1, _gencode_instancenorm2d_function, tempdir, ComputeConfig(1, 1), policy
        )


def _gencode_instancenorm2d_function_2(tempdir, config, pas_policy):
    init_distributed()
    rank_id = dist.get_rank()
    dtype = torch.bfloat16
    init_random()
    device = torch.device(f"cuda:{rank_id}")
    m = InstanceNorm2dModule().cuda()

    shared_data = generate_parallel_data((2, 4, 32, 32), device, dtype)
    x_part = shared_data[rank_id]

    m_new = parallelize(
        m,
        {"x": x_part},
        lambda graph, resource: pas_policy(graph, resource, dim=0),
        config,
        gen_savedir=tempdir,
        load_module=True,
        reuse="override",
    )
    assert m_new is not None
    m_new.train()
    output = m_new(x_part)
    assert output is not None

    gather_output = [torch.empty_like(output) for _ in range(2)]
    dist.all_gather(gather_output, output)
    y_output = torch.cat(gather_output, dim=0)

    bn = torch.nn.InstanceNorm2d(4).to(device)
    bn.running_mean = torch.zeros(4, device=device)
    bn.running_var = torch.ones(4, device=device)
    y = torch.cat(shared_data, dim=0)
    bn.train()
    ref_output = bn(y)
    current_mean_error, current_max_error = compute_error(y_output, ref_output)
    assert (
        abs(current_mean_error) < 1e-6
    ), "Custom output is not the same as PyTorch output error"
    assert torch.allclose(
        y_output, ref_output, atol=1e-6
    ), "Custom output does not match PyTorch output"


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Need at least 2 GPU devices")
def test_codegen_instancenorm2d_1_2():
    with tempfile.TemporaryDirectory() as tempdir:
        launch_torchrun(
            2, _gencode_instancenorm2d_function_2, tempdir, ComputeConfig(1, 2), policy
        )


def _gencode_instancenorm2d_function_4(tempdir, config, pas_policy):
    init_distributed()
    rank_id = dist.get_rank()
    dtype = torch.bfloat16
    init_random()
    device = torch.device(f"cuda:{rank_id}")
    m = InstanceNorm2dModule().cuda()

    x_list = generate_parallel_data((2, 4, 32, 32), device, dtype)
    x = x_list[rank_id // 2]

    m_new = parallelize(
        m,
        {"x": x},
        lambda graph, resource: pas_policy(graph, resource, dim=0),
        config,
        gen_savedir=tempdir,
        load_module=True,
        reuse="override",
    )
    assert m_new is not None
    m_new.train()
    output = m_new(x)
    assert output is not None

    gather_output = [torch.empty_like(output) for _ in range(4)]
    dist.all_gather(gather_output, output)
    y_output = torch.cat([gather_output[0], gather_output[2]], dim=0)

    bn = torch.nn.InstanceNorm2d(4).to(device)
    bn.running_mean = torch.zeros(4, device=device)
    bn.running_var = torch.ones(4, device=device)
    y = torch.cat([x_list[0], x_list[1]], dim=0)
    bn.train()
    ref_output = bn(y)
    current_mean_error, current_max_error = compute_error(y_output, ref_output)
    assert (
        abs(current_mean_error) < 1e-6
    ), "Custom output is not the same as PyTorch output error"
    assert torch.allclose(
        y_output, ref_output, atol=1e-6
    ), "Custom output does not match PyTorch output"


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="Need at least 4 GPU devices")
def test_codegen_instancenorm2d_2_4():
    with tempfile.TemporaryDirectory() as tempdir:
        launch_torchrun(
            4, _gencode_instancenorm2d_function_4, tempdir, ComputeConfig(2, 4), policy
        )


def _gencode_instancenorm2d_function_eval(tempdir, config, pas_policy):
    init_distributed()
    m = InstanceNorm2dModule().cuda()
    m.eval()
    m_new = parallelize(
        m,
        {"x": torch.randn(4, 4, 32, 32).cuda()},
        lambda graph, resource: pas_policy(graph, resource, dim=0),
        config,
        gen_savedir=tempdir,
        load_module=True,
        reuse="override",
    )
    assert m_new is not None
    x = torch.randn(4, 4, 32, 32).cuda()
    output = m_new(x)
    assert output is not None

    bn = torch.nn.InstanceNorm2d(4).cuda()
    bn.running_mean = torch.zeros(4)
    bn.running_var = torch.ones(4)
    bn.eval()
    ref_output = bn(x)
    assert torch.equal(
        output, ref_output
    ), "Custom output does not match PyTorch output in evaluation mode"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="lack of GPU devices")
def test_codegen_instancenorm2d_1_1_eval():
    with tempfile.TemporaryDirectory() as tempdir:
        launch_torchrun(
            1,
            _gencode_instancenorm2d_function_eval,
            tempdir,
            ComputeConfig(1, 1),
            policy,
        )


def _gencode_instancenorm2d_function_eval_2(tempdir, config, pas_policy):
    init_distributed()
    rank_id = dist.get_rank()
    dtype = torch.bfloat16
    init_random()
    device = torch.device(f"cuda:{rank_id}")
    m = InstanceNorm2dModule().cuda()

    shared_data = generate_parallel_data((2, 4, 32, 32), device, dtype)
    x_part = shared_data[rank_id]

    m_new = parallelize(
        m,
        {"x": x_part},
        lambda graph, resource: pas_policy(graph, resource, dim=0),
        config,
        gen_savedir=tempdir,
        load_module=True,
        reuse="override",
    )
    assert m_new is not None
    m_new.eval()
    output = m_new(x_part)
    assert output is not None

    gather_output = [torch.empty_like(output) for _ in range(2)]
    dist.all_gather(gather_output, output)
    y_output = torch.cat(gather_output, dim=0)

    bn = torch.nn.InstanceNorm2d(4).to(device)
    bn.running_mean = torch.zeros(4, device=device)
    bn.running_var = torch.ones(4, device=device)
    y = torch.cat(shared_data, dim=0)
    bn.eval()
    ref_output = bn(y)
    current_mean_error, current_max_error = compute_error(y_output, ref_output)
    assert (
        abs(current_mean_error) < 1e-6
    ), "Custom output is not the same as PyTorch output error"
    assert torch.allclose(
        y_output, ref_output, atol=1e-6
    ), "Custom output does not match PyTorch output"


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Need at least 2 GPU devices")
def test_codegen_instancenorm2d_1_2_eval():
    with tempfile.TemporaryDirectory() as tempdir:
        launch_torchrun(
            2,
            _gencode_instancenorm2d_function_eval_2,
            tempdir,
            ComputeConfig(1, 2),
            policy,
        )


def _gencode_instancenorm2d_function_eval_4(tempdir, config, pas_policy):
    init_distributed()
    rank_id = dist.get_rank()
    dtype = torch.bfloat16
    init_random()
    device = torch.device(f"cuda:{rank_id}")
    m = InstanceNorm2dModule().cuda()

    x_list = generate_parallel_data((2, 4, 32, 32), device, dtype)
    x = x_list[rank_id // 2]

    m_new = parallelize(
        m,
        {"x": x},
        lambda graph, resource: pas_policy(graph, resource, dim=0),
        config,
        gen_savedir=tempdir,
        load_module=True,
        reuse="override",
    )
    assert m_new is not None
    m_new.eval()
    output = m_new(x)
    assert output is not None

    gather_output = [torch.empty_like(output) for _ in range(4)]
    dist.all_gather(gather_output, output)
    y_output = torch.cat([gather_output[0], gather_output[2]], dim=0)

    bn = torch.nn.InstanceNorm2d(4).to(device)
    bn.running_mean = torch.zeros(4, device=device)
    bn.running_var = torch.ones(4, device=device)
    y = torch.cat([x_list[0], x_list[1]], dim=0)
    bn.eval()
    ref_output = bn(y)
    current_mean_error, current_max_error = compute_error(y_output, ref_output)
    assert (
        abs(current_mean_error) < 1e-6
    ), "Custom output is not the same as PyTorch output error"
    assert torch.allclose(
        y_output, ref_output, atol=1e-6
    ), "Custom output does not match PyTorch output"


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="Need at least 4 GPU devices")
def test_codegen_instancenorm2d_2_4_eval():
    with tempfile.TemporaryDirectory() as tempdir:
        launch_torchrun(
            4,
            _gencode_instancenorm2d_function_eval_4,
            tempdir,
            ComputeConfig(2, 4),
            policy,
        )


class NestedBatchNorm2dModule(torch.nn.Module):
    def __init__(self):
        super(NestedBatchNorm2dModule, self).__init__()
        self.nested = BatchNorm2dModule()
        self.linear = torch.nn.Linear(8, 8)

    def forward(self, x):
        # doesn't care about forward
        pass


def test_convert_to_wrapnn():
    m = NestedBatchNorm2dModule()

    def check_converted(mc):
        assert len(list(mc.children())) == 2
        assert isinstance(mc.nested, BatchNorm2dModule)
        assert len(list(mc.nested.children())) == 1
        assert isinstance(mc.nested.bn, NnScalerBatchNorm2d)
        assert len(list(mc.nested.bn.children())) == 0
        assert id(m.linear) == id(mc.linear)
        assert len(list(m.modules())) == len(list(mc.modules()))

        assert isinstance(getattr(mc.nested.bn, _ORIGINAL_MODULE_ATTR), torch.nn.BatchNorm2d)
        assert not hasattr(mc.linear, _ORIGINAL_MODULE_ATTR)
        assert not hasattr(mc, _ORIGINAL_MODULE_ATTR)
        assert not hasattr(mc.nested, _ORIGINAL_MODULE_ATTR)

    def check_undo_converted(mcc):
        assert len(list(mcc.children())) == 2
        assert isinstance(mcc.nested, BatchNorm2dModule)
        assert len(list(mcc.nested.children())) == 1
        assert not isinstance(mcc.nested.bn, NnScalerBatchNorm2d)
        assert len(list(mcc.nested.bn.children())) == 0
        assert id(m.linear) == id(mcc.linear)
        assert not hasattr(mc.linear, _ORIGINAL_MODULE_ATTR)
        assert len(list(m.modules())) == len(list(mcc.modules()))

        assert not hasattr(mcc.nested.bn, _ORIGINAL_MODULE_ATTR)
        assert not hasattr(mcc.linear, _ORIGINAL_MODULE_ATTR)
        assert not hasattr(mcc, _ORIGINAL_MODULE_ATTR)
        assert not hasattr(mcc.nested, _ORIGINAL_MODULE_ATTR)

    mc = convert_to_wrapnn(m)
    check_converted(mc)
    mcc = undo_convert_to_wrapnn(mc)
    check_undo_converted(mcc)

    with wrapnn(m) as mc:
        check_converted(mc)
    check_undo_converted(m)

    with wrapnn(m, restore=False) as mc:
        check_converted(mc)
    check_converted(m)
