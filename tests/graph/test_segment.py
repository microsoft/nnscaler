#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import torch
import pytest
import torch.nn as nn
import tempfile
import shutil
import contextlib
from pathlib import Path


import nnscaler
import nnscaler.graph.function.function as F
from nnscaler.ir.tensor import IRFullTensor
from nnscaler.graph import IRGraph
from nnscaler.ir.adapter import IRAdapter
from nnscaler.parallel import ComputeConfig, parallelize, build_optimizer
from nnscaler.ir.operator import IRFwOperation, IRDataOperation
from tests.parallel_module.test_gencode import _gencode_contains
from ..utils import replace_all_device_with, clear_dir_on_rank0, init_random
from ..launch_torchrun import torchrun


def _tensor(shape, requires_grad=True):
    return IRFullTensor(shape, requires_grad=requires_grad).tosub()


def test_create_segment_loss_func():
    data = _tensor([256, 256], False)
    w1 = _tensor([256, 256])
    out1 = _tensor([256, 256])
    matmul_1 = F.Linear(data, w1)
    matmul_1.set_output(0, out1)
    w2 = _tensor([256, 256])
    out2 = _tensor([256, 256])
    matmul_2 = F.Linear(out1, w2)
    matmul_2.set_output(0, out2)
    loss = _tensor([1])
    sum = F.Sum(out2)
    sum.set_output(0, loss)
    d = _tensor([1], False)
    get = F.GetAttr(loss, 'data', 'getattr')
    get.set_output(0, d)
    nodes = [matmul_1, matmul_2, sum, get]
    graph = IRGraph(nodes, [data], [loss, d], 'genmodel')
    graph.backward(loss)
    segment = graph.create_segment([matmul_2, sum, get])
    print(segment.extra_repr())
    assert len(segment.outputs()) == 2
    assert segment.output(0) == loss
    assert segment.output(1) == d


def test_create_segment_loss_adapter():
    data = _tensor([256, 256], False)
    w1 = _tensor([256, 256])
    out1 = _tensor([256, 256])
    matmul_1 = F.Linear(data, w1)
    matmul_1.set_output(0, out1)
    w2 = _tensor([256, 256])
    out2 = _tensor([256, 256])
    matmul_2 = F.Linear(out1, w2)
    matmul_2.set_output(0, out2)
    loss = _tensor([1])
    sum = F.Sum(out2)
    sum.set_output(0, loss)
    sum.device = 0
    adapter = IRAdapter([sum.output(0)], [sum.output(0)])
    nodes = [matmul_1, matmul_2, sum, adapter]
    graph = IRGraph(nodes, [data], [loss], 'genmodel')
    graph.backward(loss)
    segment = graph.create_segment([matmul_2, sum, adapter])
    print(segment.extra_repr())
    assert len(segment.outputs()) == 1
    assert segment.output(0) == loss


class ModelA(nn.Module):

    def __init__(self):
        super(ModelA, self).__init__()
        self.fc = nn.Linear(8, 8)

    def forward(self, q):
        q = self.fc(q)
        q = q.reshape(q.size(0), q.size(1) * q.size(2), -1)
        q = q.transpose(0, 1)
        l = q.sum()
        return l, l.data
 
 
def policy_transpose(graph: IRGraph, resource: ComputeConfig) -> IRGraph:
    ngpus = resource.plan_ngpus
    for _, node in enumerate(graph.select(ntype=IRFwOperation)):
        print(node.signature)
        if node.signature in ["torch.transpose"]:
            sub_nodes = graph.partition(
                node, node.algorithms('dim'), idx=0, dim=0, num=ngpus)
        else:
            sub_nodes = graph.replicate(node, times=ngpus)
 
        for idx, sub_node in enumerate(sub_nodes):
            graph.assign(sub_node, idx)
    for node in graph.select(ntype=IRDataOperation):
        sub_nodes = graph.replicate(node, times=ngpus)
        for idx, sub_node in enumerate(sub_nodes):
            graph.assign(sub_node, idx)
 
    return graph


def worker_a():
    nnscaler.init()
    init_random()
    m = ModelA()
    m.train()
    trace_data = torch.randn([2, 2, 2, 8], dtype=torch.float32, device=torch.cuda.current_device())
    data = torch.randn([2, 2, 2, 8], dtype=torch.float32, device=torch.cuda.current_device())
    with clear_dir_on_rank0(Path(tempfile.gettempdir()) / 'test_infer_grad_pyfunc') as tempdir:
        pm = parallelize(
            m,
            {'q': trace_data,},
            policy_transpose,
            ComputeConfig(2, 2, use_end2end=True),
            gen_savedir=tempdir,
            reuse='override',
        )
        pm.to('cuda')
        ret = pm.train_step((data,))


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason='lack of gpu devices')
def test_infer_grad_pyfunc():
    torchrun(2, worker_a)
    # should not raise any exception
    assert True


def func(x: torch.Tensor) -> torch.Tensor:
    return x.detach().clone()

nnscaler.register_op('* -> *')(func)

class ModelB(nn.Module):

    def __init__(self):
        super(ModelB, self).__init__()
        self.fc1 = nn.Linear(8, 8, bias=False)

    def forward(self, q):
        q = self.fc1(q)
        k = func(q)
        l = q.sum() + k.sum()
        return l, l


def policy_nograd(graph: IRGraph, cfg: ComputeConfig) -> IRGraph:
    ngpus = cfg.plan_ngpus
    # print(graph.nodes())
    if cfg.use_end2end: 
        fc1_node = graph.nodes()[1]
        func_node = graph.nodes()[2]
    else:
        fc1_node = graph.nodes()[0]
        func_node = graph.nodes()[1]
    assert fc1_node.inputs()[0].requires_grad and fc1_node.inputs()[0].grad
    assert fc1_node.inputs()[1].requires_grad and fc1_node.inputs()[1].grad
    assert fc1_node.outputs()[0].requires_grad and fc1_node.outputs()[0].grad
    assert func_node.inputs()[0].requires_grad and not func_node.inputs()[0].grad
    assert not func_node.outputs()[0].requires_grad and not func_node.outputs()[0].grad
    # add multiref since consumers of fc1's output may in different partition states
    # without it generated adapters are wrong
    graph.multiref(fc1_node.outputs()[0].parent)

    for _, node in enumerate(graph.select(ntype=IRFwOperation)):
        # print(node.signature)
        if node.signature == 'torch.nn.functional.linear':
            sub_nodes = graph.partition(
                node, node.algorithms('dim'), idx=0, dim=0, num=ngpus)
        elif node.signature == 'torch.sum':
            sub_nodes = graph.partition(
                node, node.algorithms('dim'), idx=0, dim=1, num=ngpus)
        elif 'func' in node.signature:
            sub_nodes = graph.partition(
                node, node.algorithms('dim'), idx=0, dim=0, num=ngpus)
        else:
            sub_nodes = graph.replicate(node, times=ngpus)
 
        for idx, sub_node in enumerate(sub_nodes):
            graph.assign(sub_node, idx)

    for node in graph.select(ntype=IRDataOperation):
        sub_nodes = graph.replicate(node, times=ngpus)
        for idx, sub_node in enumerate(sub_nodes):
            graph.assign(sub_node, idx)
    return graph


def worker_b(use_end2end):
    nnscaler.init()
    m = ModelB()
    m.train()
    init_random()
    trace_data = torch.randn([2, 2, 2, 8], dtype=torch.float32, device=torch.cuda.current_device())
    data = torch.randn([2, 2, 2, 8], dtype=torch.float32, device=torch.cuda.current_device())
    with clear_dir_on_rank0(Path(tempfile.gettempdir()) / 'test_infer_grad_no_grad') as tempdir:
        pm = parallelize(
            m,
            {'q': trace_data,},
            policy_nograd,
            ComputeConfig(2, 2, use_end2end=use_end2end),
            gen_savedir=tempdir,
            reuse='override',
        )
        # adapter between q to q.sum()
        assert len(_gencode_contains(tempdir, ModelB, pm.rank, 'nnscaler.runtime.adapter.nn.alltoall_alltoall')) == 1
        # adapter between q to func(q)
        assert len(_gencode_contains(tempdir, ModelB, pm.rank, 'nnscaler.runtime.adapter.all_to_all')) == 1
        # adapter between q.sum() to add
        assert len(_gencode_contains(tempdir, ModelB, pm.rank, 'nnscaler.runtime.adapter.nn.allreduce_identity')) == 1
        # adapter between k.sum() to add
        assert len(_gencode_contains(tempdir, ModelB, pm.rank, 'nnscaler.runtime.adapter.all_reduce')) == 1

        pm.to('cuda')
        if use_end2end:
            ret = pm.train_step((data,))
        else:
            ret = pm.forward(data)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason='lack of gpu devices')
@pytest.mark.parametrize('use_end2end', [True, False])
def test_infer_grad_no_grad(use_end2end):
    torchrun(2, worker_b, use_end2end)
    # should not raise any exception
    assert True
