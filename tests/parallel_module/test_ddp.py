#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import tempfile
import itertools
import re
from pathlib import Path
import shutil
import pytest
from typing import Dict, Tuple, List
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

import numpy as np

from nnscaler.parallel import ComputeConfig, parallelize, build_optimizer
from nnscaler.runtime.module import ParallelModule
from nnscaler.runtime.gnorm import calcuate_gnorm

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


    def forward(self, x):
        return self.relu2(self.fc2(self.relu1(self.fc1(x))))


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

class OrigModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4, 4)
        self.fc_relu1 = FcRelu_4_4()
        self.linear2 = nn.Linear(4, 4)
        self.fc_relu2 = FcRelu_4_4()
        self.linear3 = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.linear1(x)
        x = self.fc_relu1(x)
        x = self.linear2(x)
        x = self.fc_relu2(x)
        x = self.linear3(x)
        x = self.sigmoid(x)
        return x


def _create_torch_module():
    init_random()
    return OrigModule().cuda()


def _create_cube_module(pas, compute_config, cube_savedir):
    class CompiledModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(4, 4)
            self.fc_relu1 = _to_cube_model(FcRelu_4_4(), pas, compute_config, cube_savedir, 'fc_relu1')
            self.linear2 = nn.Linear(4, 4)
            self.fc_relu2 = _to_cube_model(FcRelu_4_4(), pas, compute_config, cube_savedir, 'fc_relu2')
            self.linear3 = nn.Linear(4, 1)
            self.sigmoid = nn.Sigmoid()
        def forward(self, x):
            x = self.linear1(x)
            x = self.fc_relu1(x)
            x = self.linear2(x)
            x = self.fc_relu2(x)
            x = self.linear3(x)
            x = self.sigmoid(x)
            return x
    init_random()
    compiled_module = CompiledModule().cuda()
    return compiled_module

DATA_SIZE = 32

@dataclass
class StepResult:
    pred: torch.Tensor
    loss: torch.Tensor
    grads: Dict[str, torch.Tensor]
    gnorm: torch.Tensor
    weights: Dict[str, torch.Tensor]


def _train_ddp(model, update_freq, num_replicas, rank):
    from torch.nn.parallel import DistributedDataParallel as DDP
    init_random()

    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model = DDP(model, device_ids=[rank])

    data = []
    UPDATE_FREQ = update_freq
    init_random()
    for _ in range(DATA_SIZE):
        data.append((
            torch.randn((2, 4), device='cuda', dtype=torch.float32),
            torch.rand((2, 1), device='cuda', dtype=torch.float32),
        ))
    data = [data[i] for i in range(rank, DATA_SIZE, num_replicas)]
    results = []
    for i, (x, y) in enumerate(data):
        model.train()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        if i % UPDATE_FREQ == UPDATE_FREQ - 1:
            optimizer.step()
            # remove leadding `module.` prefix
            prefix_len = len('module.')
            grads = {n[prefix_len:]: p.grad for n, p in model.named_parameters()}
            gnorm = calcuate_gnorm(list(model.parameters()))[0]
            results.append(clone_to_cpu_recursively([y_pred, loss, grads, gnorm]))
            optimizer.zero_grad()
            weights = {n[prefix_len:]: p.data for n, p in model.named_parameters()}
            results[-1].append(clone_to_cpu_recursively(weights))
            results[-1] = StepResult(*results[-1])
    return results


def _train(model, is_cube, update_freq, num_replicas, rank):
    init_random()

    loss_fn = nn.BCELoss()
    if is_cube:
        optimizer = build_optimizer(model, torch.optim.Adam, lr=0.01)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    data = []
    UPDATE_FREQ = update_freq
    init_random()
    for _ in range(DATA_SIZE):
        data.append((
            torch.randn((2, 4), device='cuda', dtype=torch.float32),
            torch.rand((2, 1), device='cuda', dtype=torch.float32),
        ))
    data = [data[i] for i in range(rank, DATA_SIZE, num_replicas)]
    results = []
    for i, (x, y) in enumerate(data):
        model.train()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        if i % UPDATE_FREQ == UPDATE_FREQ - 1:
            optimizer.step()
            grads = {n: p.grad for n, p in model.named_parameters()}
            if is_cube:
                gnorm = optimizer.clip_gnorm()
            else:
                gnorm = calcuate_gnorm(list(model.parameters()))[0]
            results.append(clone_to_cpu_recursively([y_pred, loss, grads, gnorm]))
            optimizer.zero_grad()
            weights = {n: p.data for n, p in model.named_parameters()}
            results[-1].append(clone_to_cpu_recursively(weights))
            results[-1] = StepResult(*results[-1])
    return results


def _gpu_worker_ga(update_freq):
    init_distributed()
    orig_module = _create_torch_module()
    # update_freq *2 to simulate ddp = 2
    orig_results = _train(orig_module, False, update_freq*2, 1, 0)
    return (
        orig_results,
    )


def _gpu_worker_ddp(update_freq):
    init_distributed()
    orig_module = _create_torch_module()
    orig_results = _train_ddp(orig_module, update_freq, 2, torch.distributed.get_rank())
    return (
        orig_results,
    )

def _gpu_worker_cube(pas, plan_ngpus, runtime_ngpus, update_freq, use_zero):
    init_distributed()
    with clear_dir_on_rank0(Path(tempfile.gettempdir()) / 'cube_test') as tempdir:
        compiled_module = _create_cube_module(pas,
            ComputeConfig(plan_ngpus, runtime_ngpus, use_zero=use_zero),
            tempdir
        )
        compiled_results = _train(
            compiled_module, True, update_freq,
            runtime_ngpus // plan_ngpus,
            torch.distributed.get_rank() // plan_ngpus
        )
        return (
            compiled_results,
            compiled_module.fc_relu1.fullmap,
            compiled_module.fc_relu1.dist_param_map,
            compiled_module.fc_relu2.fullmap,
            compiled_module.fc_relu2.dist_param_map,
        )

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


def _compare_weights(orig0, compiled0, compiled1, fc1_fullmap, fc2_fullmap, fc1_dist_param_map, fc2_dist_param_map):
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
        assert torch.allclose(compiled0[k], compiled1[k], rtol=1e-4, atol=1e-4)
    for k, v in itertools.chain(merged_fc1_fixed.items(), merged_fc2_fixed.items(), compiled0.items()):
        # print(f'key: {k}, max diff: {torch.max(torch.abs(orig0[k] - v))}')
        assert torch.allclose(v, orig0[k], rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4, reason='lack of gpu devices')
@pytest.mark.parametrize('update_freq', [1, 2, 4])
def test_tp_ddp(update_freq):
    orig_results: Dict[int, tuple] = launch_torchrun(2, _gpu_worker_ddp, update_freq)
    orig_results2: Dict[int, tuple] = launch_torchrun(1, _gpu_worker_ga, update_freq)

    # check equavalence of ddp and gradient accumulation
    ddp_worker_result0, ddp_worker_result1 = orig_results[0], orig_results[1]
    ddp_result0, ddp_result1 = ddp_worker_result0[0], ddp_worker_result1[0]
    for i in range(len(ddp_result0)):
        for k in ddp_result0[i].grads.keys(): # grad
            ddp_result0[i].grads[k] += ddp_result1[i].grads[k]
        for k in ddp_result0[i].weights.keys():  # weights
            assert torch.equal(ddp_result0[i].weights[k], ddp_result1[i].weights[k])

    ga_simulated_result0: List[StepResult] = orig_results2[0][0]
    assert len(ddp_result0) == len(ga_simulated_result0)
    assert len(ddp_result1) == len(ga_simulated_result0)
    for i in range(len(ddp_result0)):
        a0, b = ddp_result0[i], ga_simulated_result0[i]
        for k in b.grads.keys(): # grad
            # print('grad: ', k, torch.max(torch.abs(a0[2][k] - b[2][k])))
            assert torch.allclose(a0.grads[k], b.grads[k], atol=1e-2, rtol=1e-2)  # grad
        for k in b.weights.keys():  # weights
            # ddp will prefix `module.` to the key
            # print('weight: ', k, torch.max(torch.abs(a0[3][k]- b[3][k])))
            assert torch.allclose(a0.weights[k], b.weights[k], atol=1e-2, rtol=1e-2)  # weights

    cube_results = launch_torchrun(4, _gpu_worker_cube, 'tp', 2, 4, update_freq, False)
    zcube_results = launch_torchrun(4, _gpu_worker_cube, 'tp', 2, 4, update_freq, True)
    worker_results0, worker_results1,  worker_results2, worker_results3 = cube_results[0], cube_results[1], cube_results[2], cube_results[3]
    results0: List[StepResult] = worker_results0[0]
    results1: List[StepResult] = worker_results1[0]
    results2: List[StepResult] = worker_results2[0]
    results3: List[StepResult] = worker_results3[0]
    zworker_results0, zworker_results1,  zworker_results2, zworker_results3 = zcube_results[0], zcube_results[1], zcube_results[2], zcube_results[3]
    zresults0: List[StepResult] = zworker_results0[0]
    zresults1: List[StepResult] = zworker_results1[0]
    zresults2: List[StepResult] = zworker_results2[0]
    zresults3: List[StepResult] = zworker_results3[0]

    fc1_fullmap = worker_results0[1], worker_results1[1]
    assert fc1_fullmap == (worker_results2[1], worker_results3[1])
    fc1_dist_param_map = (worker_results0[2], worker_results1[2])
    assert fc1_dist_param_map == (worker_results2[2], worker_results3[2])

    fc2_fullmap = worker_results0[3], worker_results1[3]
    assert fc2_fullmap == (worker_results2[3], worker_results3[3])
    fc2_dist_param_map = worker_results0[4],worker_results1[4]
    assert fc2_dist_param_map == (worker_results2[4], worker_results3[4])

    fc1_fullmap = zworker_results0[1], zworker_results1[1]
    assert fc1_fullmap == (zworker_results2[1], zworker_results3[1])
    fc1_dist_param_map = (zworker_results0[2], zworker_results1[2])
    assert fc1_dist_param_map == (zworker_results2[2], zworker_results3[2])

    fc2_fullmap = zworker_results0[3], zworker_results1[3]
    assert fc2_fullmap == (zworker_results2[3], zworker_results3[3])
    fc2_dist_param_map = zworker_results0[4], zworker_results1[4]
    assert fc2_dist_param_map == (zworker_results2[4], zworker_results3[4])

    # pred, loss
    for r0, r1 in [(results0, results1), (results2, results3),
                   (zresults0, zresults1), (zresults2, zresults3),
                   (results0, zresults0), (results2, zresults2)
        ]:
        # have the same input
        assert len(r0) == len(r1)  # iteration count
        for i in range(len(r0)):
            a, b = r0[i], r1[i]
            assert torch.equal(a.pred, b.pred)  # pred
            assert torch.equal(a.loss, b.loss)  # loss
            assert torch.equal(a.gnorm, b.gnorm)  # gnorm

    # grad, weights
    for r0, r1 in [(results0, results2), (results1, results3),
                   (zresults0, zresults2), (zresults1, zresults3),
                   (results0, zresults0), (results1, zresults1)
        ]:
        # in the same shard, grads and weights are the same
        assert len(r0) == len(r1)
        for i in range(len(r0)):
            a, b = r0[i], r1[i]
            assert torch.equal(a.gnorm, b.gnorm)  # gnorm
            for k in a.grads.keys(): # grad
                assert torch.equal(a.grads[k], b.grads[k])
            for k in a.weights.keys():  # weights
                assert torch.equal(a.weights[k], b.weights[k])

    assert len(ga_simulated_result0) == len(results0)
    for i in range(len(ddp_result0)):
        orig0, compiled0, compiled1 = ga_simulated_result0[i], results0[i], results1[i]
        assert torch.allclose(orig0.gnorm, compiled0.gnorm, atol=1e-6, rtol=1e-6)  # gnorm
        # grad
        _compare_weights(orig0.grads, compiled0.grads, compiled1.grads, fc1_fullmap, fc2_fullmap, fc1_dist_param_map, fc2_dist_param_map)
        # weights
        _compare_weights(orig0.weights, compiled0.weights, compiled1.weights, fc1_fullmap, fc2_fullmap, fc1_dist_param_map, fc2_dist_param_map)
