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


class ModelA(torch.nn.Module):

    def __init__(self):
        super(ModelA, self).__init__()
        self.fc = torch.nn.Linear(10, 10)

    def forward(self, x):
        x = self.fc(x)
        l = x.sum()
        return l, l.data


@pytest.mark.skipif(not torch.cuda.is_available(), reason='lack of gpu devices')
def test_loss_multiref():
    m = ModelA()
    m.train()
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
    trace_data = torch.randn([2, 10], dtype=torch.float32, device=torch.cuda.current_device())

    with tempfile.TemporaryDirectory() as tempdir:
        pas_cfg = {
            'parallel_profile': False
        }
        parallelize(
                m,
                {'x': trace_data},
                'autodist',
                ComputeConfig(1, 1, use_end2end=True, pas_config=pas_cfg),
                reuse='override',
                gen_savedir=tempdir,
                load_module=False,
        )

        assert len(_gencode_contains(tempdir, ModelA, 0, '\.multiref')) == 1


class ModelB(torch.nn.Module):

    def __init__(self):
        super(ModelB, self).__init__()
        self.fc = torch.nn.Linear(10, 10, bias=False)
        self.fc1 = torch.nn.Linear(10, 10, bias=False)
        self.fc2 = torch.nn.Linear(10, 10, bias=False)

    def forward(self, x):
        x = self.fc(x)
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        y = x1 + x2
        l = y.sum()
        return l


@pytest.mark.skipif(not torch.cuda.is_available(), reason='lack of gpu devices')
def test_same_partition():
    m = ModelB()
    m.train()
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
    trace_data = torch.randn([2, 10], dtype=torch.float32, device=torch.cuda.current_device())

    with tempfile.TemporaryDirectory() as tempdir:
        pas_cfg = {
            'parallel_profile': False
        }
        parallelize(
                m,
                {'x': trace_data},
                'autodist',
                ComputeConfig(1, 1, use_end2end=True, pas_config=pas_cfg),
                reuse='override',
                gen_savedir=tempdir,
                load_module=False,
        )

        # this multiref is generated by `local_consumer_multiref` in `IRAdapterGener`
        assert len(_gencode_contains(tempdir, ModelB, 0, '\.multiref')) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason='lack of gpu devices')
def test_diff_partition_1():
    m = ModelB()
    m.train()
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
    trace_data = torch.randn([2, 10], dtype=torch.float32, device=torch.cuda.current_device())

    with tempfile.TemporaryDirectory() as tempdir:
        pas_cfg = {
            'load_plan_path': Path(__file__).parent / 'multiref_plan1.json',
        }
        parallelize(
                m,
                {'x': trace_data},
                'autodist',
                ComputeConfig(2, 2, use_end2end=True, pas_config=pas_cfg),
                reuse='override',
                gen_savedir=tempdir,
                load_module=False,
        )

        # this multiref is generated by `local_consumer_multiref` in `IRAdapterGener`
        assert len(_gencode_contains(tempdir, ModelB, 0, '\.multiref')) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason='lack of gpu devices')
def test_diff_partition_2():
    m = ModelB()
    m.train()
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
    trace_data = torch.randn([2, 10], dtype=torch.float32, device=torch.cuda.current_device())

    with tempfile.TemporaryDirectory() as tempdir:
        pas_cfg = {
            'load_plan_path': Path(__file__).parent / 'multiref_plan2.json',
        }
        parallelize(
                m,
                {'x': trace_data},
                'autodist',
                ComputeConfig(2, 2, use_end2end=True, pas_config=pas_cfg),
                reuse='override',
                gen_savedir=tempdir,
                load_module=False,
        )

        # this multiref is generated by `local_consumer_multiref` in `IRAdapterGener`
        assert len(_gencode_contains(tempdir, ModelB, 0, '\.multiref')) == 1
        # generate code like, should be only one identity_allreduce
        # linear_34 = torch.nn.functional.linear(x_42, self.fc_weight_33, bias=None)
        # del x_42
        # linear_34 = nnscaler.runtime.adapter.nn.identity_allreduce(linear_34, ranks=[0, 1])
        # linear_105, linear_109 = nnscaler.runtime.function.multiref(linear_34, times=2)
        assert len(_gencode_contains(tempdir, ModelB, 0, 'nnscaler.runtime.adapter.nn.identity_allreduce')) == 1
