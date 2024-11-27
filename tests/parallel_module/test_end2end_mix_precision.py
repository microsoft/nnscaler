#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""
PYTHONPATH=.:$PYTHONPATH torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    examples/mlp/train.py --policy PASMegatronTP
"""

from pathlib import Path
import tempfile
from typing import Dict, TypedDict
import pytest
import torch
from torch import nn
import torch.distributed

import nnscaler
from nnscaler.runtime.gnorm import calcuate_gnorm
from nnscaler.runtime.utils import microbatches
from nnscaler.runtime.module import ParallelModule
from nnscaler.parallel import ComputeConfig, build_optimizer, parallelize, merge_state_dicts
from .common import assert_equal, init_distributed, PASMegatron, init_random
from ..launch_torchrun import clone_to_cpu_recursively, launch_torchrun

from .test_checkpoint import End2EndMLP
from .test_end2end import allclose, merge_cube_result
from ..utils import init_parameter, clear_dir_on_rank0


DATA_SIZE = 16


class MPModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w0 = torch.nn.Parameter(torch.empty(8, 8, dtype=torch.float32))
        self.b0 = torch.nn.Parameter(torch.empty(8, dtype=torch.float32))

        self.w1 = torch.nn.Parameter(torch.empty(8, 8, dtype=torch.float64))
        self.b1 = torch.nn.Parameter(torch.empty(8, dtype=torch.float64))

        self.w2 = torch.nn.Parameter(torch.empty(8, 8, dtype=torch.float32))
        self.b2 = torch.nn.Parameter(torch.empty(8, dtype=torch.float64))
        self.loss_fn = nn.BCELoss()

        self.reset_parameters(self.w0, self.b0)
        self.reset_parameters(self.w1, self.b1)
        self.reset_parameters(self.w2, self.b2)

    def reset_parameters(self, w, b) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        import math
        torch.nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        if b is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(w)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(b, -bound, bound)

    def forward(self, data: dict):
        x = data['data']
        x = self.w0 @ x + self.b0
        x = x.to(torch.float64)
        x = self.w1 @ x + self.b1
        x = self.w2 @ x.float()
        x = x.to(torch.float64) + self.b2
        x = torch.sigmoid(x.float())
        loss = self.loss_fn(x, data['target'])
        return loss


def dummy_data():
    return {
        'data': torch.randn(
            8, 8, device=torch.cuda.current_device()),
        'target': torch.rand(
            8, 8, device=torch.cuda.current_device())
    }


def _train_cube(model: ParallelModule, mbs, num_replicas, rank):
    init_random()
    optimizer = build_optimizer(model, torch.optim.Adam, lr=0.01)
    data = []
    init_random()
    for _ in range(DATA_SIZE):
        data.append(dummy_data())
    data = [data[i] for i in range(rank, DATA_SIZE, num_replicas)]
    chunks = [data[i:i + mbs] for i in range(0, len(data), mbs)]
    results = []
    for _, x in enumerate(chunks):
        model.train()
        losses = model.train_step(x)
        print(f'loss {_}: {losses}')
        optimizer.step()
        # gnorm = optimizer.clip_gnorm()
        grads = {n: p.grad for n, p in model.named_parameters()}
        model._add_extra_state(grads, '')
        weights = {n: p.data for n, p in model.named_parameters()}
        model._add_extra_state(weights, '')
        # gnorm calculation doesn't support float64, so let's skip it
        results.append(clone_to_cpu_recursively([grads, weights, torch.tensor(0.0)]))
        optimizer.zero_grad()
    return results


def _train_ga(model, update_freq, data_size=DATA_SIZE):
    init_random()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    data = []
    init_random()
    for _ in range(data_size):
        data.append(dummy_data())
    results = []
    for i, x in enumerate(data):
        model.train()
        loss = model(x)
        print(f'loss {i}: {loss}')
        loss.backward()
        if i % update_freq == update_freq - 1:
            optimizer.step()
            grads = {n: p.grad for n, p in model.named_parameters()}
            weights = {n: p.data for n, p in model.named_parameters()}
            # gnorm calculation doesn't support float64, so let's skip it
            results.append(clone_to_cpu_recursively([grads, weights, torch.tensor(0.0)]))
            optimizer.zero_grad()
    return results


def gpu_worker_cube(use_zero=False, async_reducer=False, use_bucket=False):
    init_distributed()
    init_random()
    plan_ngpus = 2
    runtime_ngpus = 4
    nmicros = plan_ngpus
    with clear_dir_on_rank0(Path(tempfile.gettempdir()) / 'cube_test_end2end_mp') as tempdir:
        init_random()
        model = MPModule()
        model = parallelize(
            model,
            {'data': dummy_data()},
            pas_policy='tp',
            compute_config= ComputeConfig(
                plan_ngpus, runtime_ngpus,
                use_end2end=True,
                use_zero=use_zero,
                use_async_reducer=async_reducer,
                reducer_bucket_cap_mb=1e-6 if use_bucket else 0, # 1e-6 to make sure one parameter per bucket
            ),
            gen_savedir=tempdir
        )
        # (intra + inter) * (float32 + float64)
        assert len(model.reducers) == 4
        model.cuda()
        train_result = _train_cube(model, nmicros, runtime_ngpus // plan_ngpus, torch.distributed.get_rank() // plan_ngpus)

        with torch.inference_mode():
            model.eval()
            init_random()
            infer_data = []
            for _ in range(nmicros):
                infer_data.append(dummy_data())
            infer_result = clone_to_cpu_recursively(model.infer_step(infer_data))

        return train_result, infer_result, clone_to_cpu_recursively(infer_data)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4, reason='lack of gpu devices')
def test_mixed_precision():
    torch.cuda.set_device(0)
    torch.set_default_device(f'cuda:0')
    init_random()
    model = MPModule()
    torch.save(model.state_dict(), 'model.pth')
    ga4_result = _train_ga(model, 4)  # micro_batch_size = 4
    assert len(ga4_result) == 4

    cube2_results_non_pipeline = {}
    for use_async_reducer in [False, True]:
        for use_zero in [False, True]:
            for use_bucket in [False, True]:
                cube2_results_non_pipeline[(use_zero, use_async_reducer, use_bucket)] = launch_torchrun(
                    4, gpu_worker_cube,
                    use_zero, use_async_reducer, use_bucket
                )

    for r in cube2_results_non_pipeline.values():
        for _, v in r.items():
            # all losses should be scalar tensor
            assert all(i.shape == () for i in v[1])

    cube2_result_non_pipeline = {kk: merge_cube_result({k: v[0] for k, v in vv.items()}) for kk, vv in cube2_results_non_pipeline.items()}

    for r in cube2_result_non_pipeline.values():
        assert len(r) == 4

    for use_async_reducer in [False, True]:
        for use_zero in [False, True]:
            for use_bucket in [False, True]:
                allclose(cube2_result_non_pipeline[(use_zero, use_async_reducer, use_bucket)], ga4_result, atol=1e-5, rtol=1e-5) # looks tp introduces more error

    for use_zero in [False, True]:
        # when use_bucket, it should be the same for both async and non-async
        assert_equal(cube2_result_non_pipeline[(use_zero, use_async_reducer, True)],
                     cube2_result_non_pipeline[(use_zero, not use_async_reducer, True)])

    infer_results = {k: v[1] for k, v in cube2_results_non_pipeline[(False, False, False)].items()}
    infer_datas = {k: v[2] for k, v in cube2_results_non_pipeline[(False, False, False)].items()}
    assert len(infer_results) == 4
    assert len(infer_datas) == 4
    infer_result = infer_results[0]
    infer_data = infer_datas[0]
    for k in infer_results:
        assert_equal(infer_results[k], infer_result)
    for k in infer_datas:
        assert_equal(infer_datas[k], infer_data)

    for i, data in enumerate(infer_data):
        with torch.inference_mode():
            model.eval()
            loss = model({key: v.cuda() for key, v in data.items()})
            assert torch.allclose(loss.cpu(), infer_result[i].cpu(), atol=1e-6, rtol=1e-6)
