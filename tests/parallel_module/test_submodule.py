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
    class CompiledModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc_relu1 = _to_cube_model(FcRelu_4_4(), pas, compute_config, cube_savedir, 'fc_relu1')
            self.fc_relu2 = _to_cube_model(FcRelu_4_4(), pas, compute_config, cube_savedir, 'fc_relu2')
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
    compiled_module = CompiledModule().cuda()
    return orig_module, compiled_module


def _train(model, update_freq, is_cube):
    init_random()

    loss_fn = nn.BCELoss()
    if is_cube:
        optimizer = build_optimizer(model, torch.optim.Adam, lr=0.1)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    data = []
    DATA_SIZE = 20
    UPDATE_FREQ = update_freq
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


def _gpu_worker(pas, ngpus, update_freq):
    init_distributed()
    with clear_dir_on_rank0(Path(tempfile.gettempdir()) / 'cube_test') as tempdir:
        orig_module, compiled_module = _create_modules(pas, ComputeConfig(ngpus, ngpus), tempdir)
        orig_results = _train(orig_module, update_freq, False)
        compiled_results = _train(compiled_module, update_freq, True)
        return (
            orig_results,
            compiled_results,
            compiled_module.fc_relu1.fullmap,
            compiled_module.fc_relu1.dist_param_map,
            compiled_module.fc_relu2.fullmap,
            compiled_module.fc_relu2.dist_param_map,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason='lack of gpu devices')
@pytest.mark.parametrize('update_freq', [1, 2, 4])
def test_submodules_tp_gpu1(update_freq):
    results = launch_torchrun(1, _gpu_worker, 'tp', 1, update_freq)
    orig_results, compiled_results, _, _, _, _ = results[0]
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


def _get_fc_weights(state_dict: dict, prefix):
    result = {}
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            result[k[len(prefix):]] = v
        else:
            new_state_dict[k] = v
    state_dict.clear()
    state_dict.update(new_state_dict)
    return result


def _compare_weights(orig0, orig1, compiled0, compiled1, fc1_fullmap, fc2_fullmap, fc1_dist_param_map, fc2_dist_param_map):
    fc1_weights0 = _get_fc_weights(compiled0, 'fc_relu1.')
    fc2_weights0 = _get_fc_weights(compiled0, 'fc_relu2.')
    fc1_weights1 = _get_fc_weights(compiled1, 'fc_relu1.')
    fc2_weights1 = _get_fc_weights(compiled1, 'fc_relu2.')

    cube_state_fc1 = [(fc1_weights0, {'state':{}}, fc1_dist_param_map[0], fc1_fullmap[0]), (fc1_weights1, {'state':{}}, fc1_dist_param_map[1], fc1_fullmap[1])]
    cube_state_fc2 = [(fc2_weights0, {'state':{}}, fc2_dist_param_map[0], fc2_fullmap[0]), (fc2_weights1, {'state':{}}, fc2_dist_param_map[1], fc2_fullmap[1])]
    merged_fc1, _ = ParallelModule.merge_partial_states(cube_state_fc1)
    merged_fc1_fixed = {}
    for k, v in merged_fc1.items():
        merged_fc1_fixed['fc_relu1.' + k] = v
    merged_fc2, _ = ParallelModule.merge_partial_states(cube_state_fc2)
    merged_fc2_fixed = {}
    for k, v in merged_fc2.items():
        merged_fc2_fixed['fc_relu2.' + k] = v
    assert len(merged_fc1_fixed) + len(merged_fc2_fixed) + len(compiled0) == len(orig0)
    assert len(compiled1) == len(compiled0)
    for k, v in compiled0.items():
        assert torch.allclose(compiled0[k].cpu(), compiled1[k].cpu(), rtol=1e-4, atol=1e-4)
    for k, v in itertools.chain(merged_fc1_fixed.items(), merged_fc2_fixed.items(), compiled0.items()):
        assert torch.allclose(v.cpu(), orig0[k].cpu(), rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason='lack of gpu devices')
@pytest.mark.parametrize('update_freq', [1, 2, 4])
def test_submodules_tp_gpu2(update_freq):
    results = launch_torchrun(2, _gpu_worker, 'tp', 2, update_freq)
    results0, results1 = results[0], results[1]
    eps = 1e-4

    fc1_fullmap = results0[2], results1[2]
    fc1_dist_param_map = results0[3], results1[3]

    fc2_fullmap = results0[4], results1[4]
    fc2_dist_param_map = results0[5],results1[5]

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
        _compare_weights(orig0[2], orig1[2], compiled0[2], compiled1[2], fc1_fullmap, fc2_fullmap, fc1_dist_param_map, fc2_dist_param_map)

        # weights
        for k in orig0[3].keys():
            assert torch.allclose(orig0[3][k], orig1[3][k], rtol=eps, atol=eps)
        _compare_weights(orig0[3], orig1[3], compiled0[3], compiled1[3], fc1_fullmap, fc2_fullmap, fc1_dist_param_map, fc2_dist_param_map)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='cuda is not available')
@pytest.mark.parametrize('update_freq', [1, 2, 4])
def test_submodules_dp_gpu1(update_freq):
    results = launch_torchrun(1, _gpu_worker, 'dp', 1, update_freq)
    orig_results, compiled_results, _, _, _, _ = results[0]
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


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason='lack of gpu devices')
@pytest.mark.parametrize('update_freq', [1, 2, 4])
def test_submodules_dp_gpu2(update_freq):
    eps = 1e-4
    results = launch_torchrun(2, _gpu_worker, 'data', 2, update_freq)
    for r in results.values():
        orig_results, compiled_results, _, _, _, _ = r
        for orig, compiled in zip(orig_results, compiled_results):
            assert torch.allclose(orig[0], compiled[0], rtol=eps, atol=eps)  # pred
            assert torch.allclose(orig[1], compiled[1], rtol=eps, atol=eps)  # loss

            # grad
            compiled_cleaned = {re.sub(r"_[0-9]+", '', k).replace('.', '_'): v for k, v in compiled[2].items()}
            assert len(orig[2]) == len(compiled_cleaned)
            for k in orig[2].keys():
                assert torch.allclose(orig[2][k], compiled_cleaned[k.replace('.', '_')], rtol=eps, atol=eps)

            # weights
            compiled_cleaned = {re.sub(r"_[0-9]+", '', k).replace('.', '_'): v for k, v in compiled[3].items()}
            assert len(orig[3]) == len(compiled_cleaned)
            for k in orig[3].keys():
                assert torch.allclose(orig[3][k], compiled_cleaned[k.replace('.', '_')], rtol=eps, atol=eps)
