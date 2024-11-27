#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import torch
import torch.nn as nn
import tempfile
import shutil
import contextlib
import pytest
from pathlib import Path


import nnscaler
import nnscaler.graph.function.function as F
from nnscaler.ir.tensor import IRFullTensor
from nnscaler.graph import IRGraph
from nnscaler.ir.adapter import IRAdapter
from nnscaler.parallel import ComputeConfig, parallelize, build_optimizer
from nnscaler.ir.operator import IRFwOperation, IRDataOperation
from nnscaler.graph.segment import IRSegment
from nnscaler.graph.schedule.predefined import PredefinedSched
from tests.utils import clear_dir_on_rank0, init_random
from tests.launch_torchrun import torchrun
from tests.parallel_module.test_gencode import _gencode_contains


def get_mem():
    return torch.cuda.max_memory_allocated() // 1024 // 1024


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = torch.nn.Linear(4096, 4096, bias=False)
        self.fc2 = torch.nn.Linear(4096, 4096, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x.sum()


class Model2(torch.nn.Module):
    def __init__(self):
        super(Model2, self).__init__()
        self.fc1 = torch.nn.Linear(4096, 4096, bias=False)
        self.fc2 = torch.nn.Linear(4096, 4096, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        l = x.sum()
        return l, l.data


def policy_pp(graph, cfg):
    data_loader, fc1, fc2, loss = graph.nodes()[:4]
    graph.staging([fc1, fc2])
    stages = graph.select(ntype=IRSegment, flatten=False)
    stages = [s for s in stages if s.isfw()]

    sub_nodes = graph.replicate(data_loader, 4)
    for i, sub_node in enumerate(sub_nodes):
        graph.assign(sub_node, i)

    sub_nodes = graph.partition(fc1, fc1.algorithms('dim'), idx=0, dim=0, num=2)
    graph.assign(sub_nodes[0], 0)
    graph.assign(sub_nodes[1], 1)

    identity = stages[1].nodes()[0]
    sub_nodes = graph.replicate(identity, 2)
    graph.assign(sub_nodes[0], 2)
    graph.assign(sub_nodes[1], 3)

    sub_nodes = graph.partition(fc2, fc2.algorithms('dim'), idx=0, dim=0, num=2)
    graph.assign(sub_nodes[0], 2)
    graph.assign(sub_nodes[1], 3)

    sub_nodes = graph.partition(loss, loss.algorithms('dim'), idx=0, dim=0, num=2)
    graph.assign(sub_nodes[0], 2)
    graph.assign(sub_nodes[1], 3)

    PredefinedSched.sched_1f1b(graph, 4, len(stages))

    return graph


def worker_pipeline_2x2(model_cls):
    nnscaler.init()
    m = model_cls()
    m.train()
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
    trace_data = torch.randn([2048, 4096], dtype=torch.float32, device=torch.cuda.current_device())

    with clear_dir_on_rank0(Path(tempfile.gettempdir()) / 'test_detach_loss_pp_2x2') as tempdir:
        pm = parallelize(
                m,
                {'x': trace_data},
                policy_pp,
                ComputeConfig(4, 4, use_end2end=True),
                reuse='override',
                gen_savedir=tempdir,
            )

        if pm.rank in [2, 3]:
            assert len(_gencode_contains(tempdir, model_cls, pm.rank, 'detach\(\)')) == 4

        samples = [torch.randn([2048, 4096], dtype=torch.float32, device=torch.cuda.current_device()) for _ in range(4)]
        ret = pm.train_step(samples)
        mem0 = get_mem()
        ret = pm.train_step(samples)
        mem1 = get_mem()
        ret = pm.train_step(samples)
        mem2 = get_mem()
        ret = pm.train_step(samples)
        mem3 = get_mem()
        # print(mem0, mem1, mem2, mem3)
        assert mem0 == mem1 == mem2 == mem3


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4, reason='lack of gpu devices')
@pytest.mark.parametrize('model_cls', [Model, Model2])
def test_detach_loss_pipeline_hard(model_cls):
    torchrun(4, worker_pipeline_2x2, model_cls)
    # should not raise any exception
    assert True


def policy_easy(graph, cfg):
    data_loader, fc1, fc2, loss = graph.nodes()[:4]
    graph.staging([fc1, fc2])
    stages = graph.select(ntype=IRSegment, flatten=False)
    stages = [s for s in stages if s.isfw()]

    sub_nodes = graph.replicate(data_loader, 2)
    for i, sub_node in enumerate(sub_nodes):
        graph.assign(sub_node, i)

    graph.assign(fc1, 0)

    identity = stages[1].nodes()[0]
    graph.assign(identity, 1)
    graph.assign(fc2, 1)
    graph.assign(loss, 1)

    PredefinedSched.sched_1f1b(graph, 4, len(stages))

    return graph


def worker_pipeline_2(model_cls):
    nnscaler.init()
    m = model_cls()
    m.train()
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
    trace_data = torch.randn([2048, 4096], dtype=torch.float32, device=torch.cuda.current_device())

    with clear_dir_on_rank0(Path(tempfile.gettempdir()) / 'test_detach_loss_pp_2') as tempdir:
        pm = parallelize(
                m,
                {'x': trace_data},
                policy_easy,
                ComputeConfig(2, 2, use_end2end=True),
                reuse='override',
                gen_savedir=tempdir,
            )
        pm.to(torch.cuda.current_device())

        if pm.rank == 1:
            assert len(_gencode_contains(tempdir, model_cls, pm.rank, 'detach\(\)')) == 4
        samples = [torch.randn([2048, 4096], dtype=torch.float32, device=torch.cuda.current_device()) for _ in range(4)]
        ret = pm.train_step(samples)
        mem0 = get_mem()
        ret = pm.train_step(samples)
        mem1 = get_mem()
        ret = pm.train_step(samples)
        mem2 = get_mem()
        ret = pm.train_step(samples)
        mem3 = get_mem()
        assert mem0 == mem1 == mem2 == mem3


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason='lack of gpu devices')
@pytest.mark.parametrize('model_cls', [Model, Model2])
def test_detach_loss_pipeline_easy(model_cls):
    torchrun(2, worker_pipeline_2, model_cls)
    # should not raise any exception
    assert True
