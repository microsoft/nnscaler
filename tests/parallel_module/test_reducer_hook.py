#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import tempfile
from pathlib import Path
from collections import defaultdict

import pytest
import torch
from torch import nn

from nnscaler.parallel import ComputeConfig, parallelize, build_optimizer
from nnscaler.runtime.module import ParallelModule

from .common import CubeLinear, init_random, init_distributed
from ..launch_torchrun import launch_torchrun, clone_to_cpu_recursively
from ..utils import clear_dir_on_rank0


class FcRelu(nn.Module):
    def __init__(self, in_features=4, out_features=4, bias=True):
        super().__init__()
        self.fc1 = CubeLinear(in_features, in_features, bias=bias)
        self.relu1 = nn.ReLU()
        self.fc2 = CubeLinear(in_features, out_features, bias=bias)
        self.relu2 = nn.ReLU()
        self.fc3 = CubeLinear(out_features, out_features, bias=bias)
        self.relu3 = nn.ReLU()


    def forward(self, x):
        return self.relu3(self.fc3(self.relu2(self.fc2(self.relu1(self.fc1(x))))))


def _to_cube_model(module, pas, compute_config, cube_savedir, instance_name):
    return parallelize(
        module,
        {'x': torch.tensor([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])},
        pas,
        compute_config,
        gen_savedir=cube_savedir,
        instance_name=instance_name
    )

def _create_module(pas, compute_config, cube_savedir):
    class CompiledModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc_relu1 = _to_cube_model(FcRelu(), pas, compute_config, cube_savedir, 'fc_relu1')
            self.fc_relu2 = _to_cube_model(FcRelu(), pas, compute_config, cube_savedir, 'fc_relu2')
            self.linear3 = nn.Linear(4, 1)
            self.sigmoid = nn.Sigmoid()
        def forward(self, x):
            x = self.fc_relu1(x)
            x = self.fc_relu2(x)
            x = self.linear3(x)
            x = self.sigmoid(x)
            return x
    init_random()
    compiled_module = CompiledModule().cuda()
    return compiled_module


def _train(model):
    init_random()

    pre_called = defaultdict(int)
    post_called = defaultdict(int)
    def pre_hook(reducer, grad):
        pre_called[reducer] += 1

    def post_hook(reducer, grad):
        post_called[reducer] += 1

    loss_fn = nn.BCELoss()

    optimizer = build_optimizer(model, torch.optim.Adam, lr=0.1)
    optimizer.register_reducer_pre_hook(pre_hook)
    optimizer.register_reducer_post_hook(post_hook)

    reducers = []
    for m in model.modules():
        if isinstance(m, ParallelModule):
            reducers.extend(m.reducers)
    if optimizer._non_parallel_module_reducer:
        reducers.append(optimizer._non_parallel_module_reducer)

    if not reducers:
        print('No reducer found, skip test_hook')
        return

    data = []
    DATA_SIZE = 20
    UPDATE_FREQ = 2
    for _ in range(DATA_SIZE):
        data.append((
            torch.randn((2, 4), device='cuda', dtype=torch.float32),
            torch.rand((2, 1), device='cuda', dtype=torch.float32),
        ))
    results = []
    for i, (x, y) in enumerate(data):
        model.train()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        if i % UPDATE_FREQ == UPDATE_FREQ - 1:
            optimizer.step()
            grads = {n: p.grad for n, p in model.named_parameters()}
            results.append(clone_to_cpu_recursively([y_pred, loss, grads]))
            optimizer.zero_grad()
            weights = {n: p.data for n, p in model.named_parameters()}
            results[-1].append(clone_to_cpu_recursively(weights))
            assert pre_called == post_called
            assert set(pre_called.keys()) == set(reducers)
            assert all(v == (i + 1) // UPDATE_FREQ for v in pre_called.values())
    return results


def _gpu_worker(pas, plan_ngpus, runtime_ngpus=None):
    init_distributed()
    with clear_dir_on_rank0(Path(tempfile.gettempdir()) / 'cube_test_hook') as tempdir:
        compiled_module = _create_module(pas, ComputeConfig(plan_ngpus, runtime_ngpus or plan_ngpus), tempdir)
        _train(compiled_module)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='lack of gpu devices')
def test_hook_tp_gpu1():
    launch_torchrun(1, _gpu_worker, 'tp', 1)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason='lack of gpu devices')
def test_hook_tp_gpu2():
    launch_torchrun(2, _gpu_worker, 'tp', 2)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4, reason='lack of gpu devices')
def test_hook_tp_gpu4():
    launch_torchrun(4, _gpu_worker, 'tp', 2, 4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='lack of gpu devices')
def test_hook_dp_gpu1():
    launch_torchrun(1, _gpu_worker, 'dp', 1)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason='lack of gpu devices')
def test_hook_dp_gpu2():
    launch_torchrun(2, _gpu_worker, 'data', 2)
