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
        self.fc3 = torch.nn.Linear(4096, 4096, bias=False)
        self.fc4 = torch.nn.Linear(4096, 4096, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x.sum()

def policy_pp(graph, cfg):
    data_loader, fc1, fc2, fc3, fc4, loss = graph.nodes()[:6]
    graph.staging([fc1, fc2, fc3, fc4])
    stages = graph.select(ntype=IRSegment, flatten=False)
    stages = [s for s in stages if s.isfw()]

    sub_nodes = graph.replicate(data_loader, 4)
    for i, sub_node in enumerate(sub_nodes):
        graph.assign(sub_node, i)

    graph.assign(fc1, 0)

    identity = stages[1].nodes()[0]
    graph.assign(identity, 1)
    graph.assign(fc2, 1)

    identity = stages[2].nodes()[0]
    graph.assign(identity, 2)
    graph.assign(fc3, 2)

    identity = stages[3].nodes()[0]
    graph.assign(identity, 3)
    graph.assign(fc4, 3)

    graph.assign(loss, 3)

    PredefinedSched.sched_1f1b(graph, 4, len(stages))
    return graph


def worker_async_dp2_tp1_pp4(model_cls):
    nnscaler.init()
    m = model_cls()
    m.train()
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
    trace_data = torch.randn([2048, 4096], dtype=torch.float32, device=torch.cuda.current_device())

    with clear_dir_on_rank0(Path(tempfile.gettempdir()) / 'test_async_pp') as tempdir:
        pm = parallelize(
                m,
                {'x': trace_data},
                policy_pp,
                ComputeConfig(4, 8, use_end2end=True, use_async_reducer=True),
                reuse='override',
                gen_savedir=tempdir,
            )

        samples = [torch.randn([2048, 4096], dtype=torch.float32, device=torch.cuda.current_device()) for _ in range(4)]
        for i in range(4):
            ret = pm.train_step(samples)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 8, reason='lack of gpu devices')
@pytest.mark.parametrize('model_cls', [Model])
def test_async_pipeline(model_cls):
    torchrun(8, worker_async_dp2_tp1_pp4, model_cls)
    # should not raise any exception
    assert True
