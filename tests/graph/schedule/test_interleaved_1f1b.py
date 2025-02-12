#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import torch
import torch.distributed
import torch.nn as nn
import tempfile
import shutil
import contextlib
import pytest
from pathlib import Path


import nnscaler
from nnscaler import parallelize, ComputeConfig, ParallelModule
from nnscaler.parallel import build_optimizer, sync_grad_when, merge_state_dicts
from nnscaler.ir.tensor import IRFullTensor
from nnscaler.graph import IRGraph
from nnscaler.ir.adapter import IRAdapter
from nnscaler.parallel import ComputeConfig, parallelize, build_optimizer
from nnscaler.ir.operator import IRFwOperation, IRDataOperation
from nnscaler.graph.segment import IRSegment
from nnscaler.graph.schedule.predefined import PredefinedSched
from tests.utils import clear_dir_on_rank0, init_random
from tests.launch_torchrun import torchrun
from tests.parallel_module.common import assert_equal
from tests.parallel_module.test_gencode import _gencode_contains
from tests.launch_torchrun import launch_torchrun, clone_to_cpu_recursively


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = torch.nn.Linear(32, 32, bias=False)
        self.fc2 = torch.nn.Linear(32, 32, bias=False)
        self.fc3 = torch.nn.Linear(32, 32, bias=False)
        self.fc4 = torch.nn.Linear(32, 32, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x.sum()


def policy_1f1b(graph, cfg):
    data_loader, fc1, fc2, fc3, fc4, loss = graph.nodes()[:6]
    graph.staging([fc1, fc3,])
    stages = graph.select(ntype=IRSegment, flatten=False)
    stages = [s for s in stages if s.isfw()]

    sub_nodes = graph.replicate(data_loader, 2)
    for i, sub_node in enumerate(sub_nodes):
        graph.assign(sub_node, i)

    graph.assign(fc1, 0)
    graph.assign(fc2, 0)

    identity = stages[1].nodes()[0]
    graph.assign(identity, 1)
    graph.assign(fc3, 1)
    graph.assign(fc4, 1)
    graph.assign(loss, 1)

    PredefinedSched.sched_1f1b(graph, cfg.pas_config['n_micro_batches'], len(stages))

    return graph


def policy_1f1b_interleaved(graph, cfg):
    data_loader, fc1, fc2, fc3, fc4, loss = graph.nodes()[:6]
    graph.staging([fc1, fc2, fc3, fc4])
    stages = graph.select(ntype=IRSegment, flatten=False)
    stages = [s for s in stages if s.isfw()]

    sub_nodes = graph.replicate(data_loader, 2)
    for i, sub_node in enumerate(sub_nodes):
        graph.assign(sub_node, i)

    graph.assign(fc1, 0)

    identity = stages[1].nodes()[0]
    graph.assign(identity, 1)
    graph.assign(fc2, 1)

    identity = stages[2].nodes()[0]
    graph.assign(identity, 0)
    graph.assign(fc3, 0)

    identity = stages[3].nodes()[0]
    graph.assign(identity, 1)
    graph.assign(fc4, 1)
    graph.assign(loss, 1)

    PredefinedSched.sched_1f1b_interleaved(graph, cfg.pas_config['n_micro_batches'], len(stages))

    return graph


def _train_pp(model: ParallelModule, num_replicas, rank):
    mbs = model.nmicros_per_scheduler_step
    assert model.use_scheduler
    init_random()
    optimizer = build_optimizer(model, torch.optim.Adam, lr=0.1)
    data = []
    DATA_SIZE = mbs * 4
    for _ in range(DATA_SIZE):
        data.append(
            torch.randn((2, 32), device='cuda', dtype=torch.float32)
        )
    data = [data[i] for i in range(rank, DATA_SIZE, num_replicas)]
    chunks = [data[i:i + mbs] for i in range(0, len(data), mbs)]
    results = []
    for _, x in enumerate(chunks):
        model.train()
        _ = model.train_step(x)
        optimizer.step()
        optimizer.zero_grad()
        results.append(clone_to_cpu_recursively(model.state_dict()))
    return results


def worker_pipeline_2(n_micro_batches):
    nnscaler.init()
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
    m = Model()
    m.train()
    trace_data = torch.randn([2, 32], dtype=torch.float32, device=torch.cuda.current_device())
    cfg = ComputeConfig(2, 2, use_end2end=True, pas_config=dict(n_micro_batches=n_micro_batches))

    with clear_dir_on_rank0(Path(tempfile.gettempdir()) / 'test_1f1b_interleaved') as tempdir:
        pm_1f1b = parallelize(
                m,
                {'x': trace_data},
                policy_1f1b,
                cfg,
                reuse='override',
                gen_savedir=tempdir,
                instance_name='1f1b',
        ).cuda()
        pm_1f1b_interleaved = parallelize(
                m,
                {'x': trace_data},
                policy_1f1b_interleaved,
                cfg,
                reuse='override',
                gen_savedir=tempdir,
                instance_name='1f1b_interleaved',
        ).cuda()

    results_1f1b = _train_pp(pm_1f1b, 1, 0)
    results_1f1b_interleaved = _train_pp(pm_1f1b_interleaved, 1, 0)
    return (results_1f1b, results_1f1b_interleaved)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason='lack of gpu devices')
@pytest.mark.parametrize('n_micro_batches', [2, 4, 6])
def test_interleaved_1f1b(n_micro_batches):
    results = launch_torchrun(2, worker_pipeline_2, n_micro_batches)
    results_1f1b0, results_1f1b_interleaved0 = results[0]
    results_1f1b1, results_1f1b_interleaved1 = results[1]

    assert len(results_1f1b0) == len(results_1f1b_interleaved0)

    for i in range(len(results_1f1b0)):
        assert_equal(
            merge_state_dicts([results_1f1b0[i], results_1f1b1[i]]),
            merge_state_dicts([results_1f1b_interleaved0[i], results_1f1b_interleaved1[i]])
        )
