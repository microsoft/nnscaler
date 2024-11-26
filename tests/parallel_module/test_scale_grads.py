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
from nnscaler.runtime.module import ParallelModule, ExtraState
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

DATA_SIZE = 64

@dataclass
class StepResult:
    pred: torch.Tensor
    loss: torch.Tensor
    grads: Dict[str, torch.Tensor]
    gnorm: torch.Tensor
    weights: Dict[str, torch.Tensor]


def _train(model: torch.nn.Module, num_replicas, rank, scale_grads: bool):
    NUM_SCALE_UNITS = 2
    NUM_SAMPLES_PER_UPDATE = 2
    init_random()

    loss_fn = nn.BCELoss()
    optimizer = build_optimizer(model, torch.optim.Adam, lr=0.01)
    if scale_grads:
        # before reduce
        optimizer.register_reducer_pre_hook(lambda reducer, grad: grad.div_(NUM_SCALE_UNITS))
        # after reduce
        optimizer.register_reducer_post_hook(lambda reducer, grad: grad.mul_(NUM_SCALE_UNITS))
    data = []
    init_random()
    for _ in range(DATA_SIZE):
        data.append((
            torch.randn((2, 4), device='cuda', dtype=torch.float32),
            torch.rand((2, 1), device='cuda', dtype=torch.float32),
        ))
    data = [data[i] for i in range(rank, len(data), num_replicas)]
    results = []
    for i, (x, y) in enumerate(data):
        model.train()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.sync_shard_grad()
        optimizer.scale_grads(1/NUM_SAMPLES_PER_UPDATE)
        optimizer.step()
        grads = {n: p.grad for n, p in model.named_parameters()}
        gnorm = optimizer.clip_gnorm()
        results.append(clone_to_cpu_recursively([y_pred, loss, grads, gnorm]))
        optimizer.zero_grad()
        weights = {n: p.data for n, p in model.named_parameters()}
        results[-1].append(clone_to_cpu_recursively(weights))
        results[-1] = StepResult(*results[-1])

    return results


def _gpu_worker(pas, plan_ngpus, runtime_ngpus, scale_grads: bool):
    init_distributed()
    with clear_dir_on_rank0(Path(tempfile.gettempdir()) / 'cube_test_scale_grads') as tempdir:
        compiled_module = _create_cube_module(pas,
            ComputeConfig(plan_ngpus, runtime_ngpus, use_zero=True),
            tempdir
        )
        return _train(
            compiled_module,
            runtime_ngpus // plan_ngpus,
            torch.distributed.get_rank() // plan_ngpus,
            scale_grads
        )

@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4, reason='lack of gpu devices')
def test_scale_grads():
    cube_results = launch_torchrun(4, _gpu_worker, 'tp', 2, 4, True)
    rcube_results = launch_torchrun(4, _gpu_worker, 'tp', 2, 4, False)

    results0, results1,  results2, results3 = cube_results[0], cube_results[1], cube_results[2], cube_results[3]
    rresults0, rresults1,  rresults2, rresults3 = rcube_results[0], rcube_results[1], rcube_results[2], rcube_results[3]

    # pred, loss, gnorm
    for r0, r1 in [(results0, results1), (results2, results3),
                   (rresults0, rresults1), (rresults2, rresults3),
                   (results0, rresults0), (results2, rresults2)
        ]:
        assert len(r0) == len(r1)  # iteration count
        for i in range(len(r0)):
            a, b = r0[i], r1[i]
            assert torch.equal(a.pred, b.pred)  # pred
            assert torch.equal(a.loss, b.loss)  # loss
            assert torch.equal(a.gnorm, b.gnorm)  # gnorm

    # grad, weights
    for r0, r1 in [(results0, results2), (results1, results3),
                   (rresults0, rresults2), (rresults1, rresults3),
                   (results0, rresults0), (results1, rresults1)
        ]:
        assert len(r0) == len(r1)
        for i in range(len(r0)):
            a, b = r0[i], r1[i]
            # for grads, as we have scale_grads,
            # grads in `parameters_for_optimizer` are scaled, but the rest are not
            # so they can be the same or scaled by 2 or divided by 2
            for k in a.grads.keys(): # grad
                assert torch.equal(a.grads[k], b.grads[k]) \
                    or torch.equal(a.grads[k], b.grads[k] * 2) \
                    or torch.equal(a.grads[k], b.grads[k] / 2)
            # in the same shard, weights are the same
            for k in a.weights.keys():  # weights
                assert torch.equal(a.weights[k], b.weights[k])
