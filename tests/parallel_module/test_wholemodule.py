#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import tempfile
import itertools
import re
from pathlib import Path
import shutil

import pytest
import torch
from torch import nn
import numpy as np

from nnscaler.parallel import ComputeConfig, parallelize, build_optimizer
from nnscaler.runtime.module import ParallelModule

from .common import CubeLinear, init_random, init_distributed
from ..launch_torchrun import launch_torchrun, clone_to_cpu_recursively
from ..utils import clear_dir_on_rank0


class FcRelu(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.fc1 = CubeLinear(in_features, in_features, bias=bias)
        self.relu1 = nn.ReLU()
        self.fc2 = CubeLinear(in_features, out_features, bias=bias)
        self.relu2 = nn.ReLU()
        self.fc3 = CubeLinear(out_features, out_features, bias=bias)
        self.relu3 = nn.ReLU()


    def forward(self, x):
        return self.relu3(self.fc3(self.relu2(self.fc2(self.relu1(self.fc1(x))))))


class FcRelu_4_4(FcRelu):
    def __init__(self):
        super().__init__(4, 4)


def _to_cube_model(module, pas, compute_config, cube_savedir, instance_name):
    return parallelize(
        module,
        {'x': torch.tensor([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])},
        pas,
        compute_config,
        gen_savedir=cube_savedir,
        instance_name=instance_name
    )


def _create_modules(pas, compute_config, cube_savedir):
    class OrigModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc_relu1 = FcRelu_4_4()
            self.fc_relu2 = FcRelu_4_4()
            self.linear3 = nn.Linear(4, 1)
            self.sigmoid = nn.Sigmoid()
        def forward(self, x):
            x = self.fc_relu1(x)
            x = self.fc_relu2(x)
            x = self.linear3(x)
            x = self.sigmoid(x)
            return x
    init_random()
    orig_module = OrigModule().cuda()
    init_random()
    compiled_module = _to_cube_model(OrigModule(), pas, compute_config, cube_savedir, 'orig_module_whole').cuda()
    return orig_module, compiled_module


def _train(model, is_cube):
    init_random()

    loss_fn = nn.BCELoss()
    if is_cube:
        optimizer = build_optimizer(model, torch.optim.Adam, lr=0.1)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    data = []
    DATA_SIZE = 20
    UPDATE_FREQ = 1  # TODO: update_freq support
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
    return results


def _gpu_worker(pas, ngpus):
    init_distributed()
    with clear_dir_on_rank0(Path(tempfile.gettempdir()) / 'cube_test') as tempdir:
        orig_module, compiled_module = _create_modules(pas, ComputeConfig(ngpus, ngpus), tempdir)
        orig_results = _train(orig_module, False)
        compiled_results = _train(compiled_module, True)
        return (
            orig_results,
            compiled_results,
            compiled_module.fullmap,
            compiled_module.dist_param_map,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason='lack of gpu devices')
def test_module_tp_gpu1():
    results = launch_torchrun(1, _gpu_worker, 'tp', 1)
    orig_results, compiled_results, _, _ = results[0]
    for orig, compiled in zip(orig_results, compiled_results):
        assert torch.allclose(orig[0], compiled[0], rtol=1e-6, atol=1e-6)  # pred
        assert torch.allclose(orig[1], compiled[1], rtol=1e-6, atol=1e-6)  # loss

        # grad
        compiled_cleaned = {re.sub(r"_[0-9]+", '', k).replace('.', '_'): v for k, v in compiled[2].items()}
        assert len(orig[2]) == len(compiled_cleaned)
        for k in orig[2].keys():
            assert torch.allclose(orig[2][k], compiled_cleaned[k.replace('.', '_')], rtol=1e-6, atol=1e-6)

        # weights
        compiled_cleaned = {re.sub(r"_[0-9]+", '', k).replace('.', '_'): v for k, v in compiled[3].items()}
        assert len(orig[3]) == len(compiled_cleaned)
        for k in orig[3].keys():
            assert torch.allclose(orig[3][k], compiled_cleaned[k.replace('.', '_')], rtol=1e-6, atol=1e-6)


def _compare_weights(orig0, orig1, compiled0, compiled1, module_fullmap, module_dist_param_map):
    cube_state = [(compiled0, {'state':{}}, module_dist_param_map[0], module_fullmap[0]), (compiled1, {'state':{}}, module_dist_param_map[1], module_fullmap[1])]
    merged_state, _ = ParallelModule.merge_partial_states(cube_state)
    assert len(compiled1) == len(compiled0) == len(orig0)
    for k, v in merged_state.items():
        assert torch.allclose(v.cpu(), orig0[k].cpu(), rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason='lack of gpu devices')
def test_module_tp_gpu2():
    results = launch_torchrun(2, _gpu_worker, 'tp', 2)
    results0, results1 = results[0], results[1]
    eps = 1e-4

    module_fullmap = results0[2], results1[2]
    module_dist_param_map = results0[3], results1[3]

    for orig0, compiled0, orig1, compiled1 in zip(results0[0], results0[1], results1[0], results1[1]):
        assert torch.allclose(orig0[0], orig1[0], rtol=eps, atol=eps)  # pred
        assert torch.allclose(orig0[0], compiled0[0], rtol=eps, atol=eps)  # pred
        assert torch.allclose(orig1[0], compiled1[0], rtol=eps, atol=eps)  # pred

        assert torch.allclose(orig0[1], orig1[1], rtol=eps, atol=eps)  # loss
        assert torch.allclose(orig0[1], compiled0[1], rtol=eps, atol=eps)  # loss
        assert torch.allclose(orig1[1], compiled1[1], rtol=eps, atol=eps)  # loss

        # grad
        for k in orig0[2].keys():
            assert torch.allclose(orig0[2][k], orig1[2][k], rtol=eps, atol=eps)
        _compare_weights(orig0[2], orig1[2], compiled0[2], compiled1[2], module_fullmap, module_dist_param_map)

        # weights
        for k in orig0[3].keys():
            assert torch.allclose(orig0[3][k], orig1[3][k], rtol=eps, atol=eps)
        _compare_weights(orig0[3], orig1[3], compiled0[3], compiled1[3], module_fullmap, module_dist_param_map)
