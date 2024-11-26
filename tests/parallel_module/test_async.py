#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from pathlib import Path
import tempfile
import pytest
import torch
from torch import nn

from nnscaler import parallelize, ComputeConfig, ParallelModule

from nnscaler.parallel import build_optimizer, sync_grad_when, merge_state_dicts
from tests.launch_torchrun import launch_torchrun
from tests.launch_torchrun import clone_to_cpu_recursively
from tests.parallel_module.common import assert_equal, init_distributed
from tests.utils import clear_dir_on_rank0, init_random
from .test_wholemodule import FcRelu_4_4


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


def _to_cube_model(module, pas, compute_config, cube_savedir, instance_name):
    return parallelize(
        module,
        {'x': torch.tensor([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])},
        pas,
        compute_config,
        gen_savedir=cube_savedir,
        instance_name=instance_name
    )


def _create_modules(pas, compute_config, cube_savedir, name_prefix=''):
    init_random()
    whole_module = _to_cube_model(
        OrigModule(), pas, compute_config, cube_savedir, f'{name_prefix}whole'
    ).cuda()
    init_random()
    sub_module = OrigModule().cuda()
    sub_module.fc_relu1 = _to_cube_model(
        sub_module.fc_relu1, pas, compute_config, cube_savedir, f'{name_prefix}fc_relu1'
    ).cuda()
    sub_module.fc_relu2 = _to_cube_model(
        sub_module.fc_relu2, pas, compute_config, cube_savedir, f'{name_prefix}fc_relu2'
    ).cuda()
    return whole_module, sub_module


def _train(model: ParallelModule, update_freq):
    init_random()

    loss_fn = nn.BCELoss()
    optimizer = build_optimizer(model, torch.optim.Adam, lr=0.1)
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
        with sync_grad_when(i % UPDATE_FREQ == UPDATE_FREQ - 1):
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
        if i % UPDATE_FREQ == UPDATE_FREQ - 1:
            optimizer.step()
            optimizer.zero_grad()
            results.append(clone_to_cpu_recursively([y_pred, model.state_dict()]))
    return results


def _gpu_worker(pas, ngpus, update_freq):
    init_distributed()
    with clear_dir_on_rank0(Path(tempfile.gettempdir()) / 'cube_test_async') as tempdir:
        whole_module_async, sub_module_async = _create_modules(
            pas, ComputeConfig(
                1, ngpus, use_async_reducer=True,
                reducer_bucket_cap_mb=1e-6
            ),
            tempdir,
            'async_',
        )
        whole_module_sync, sub_module_sync = _create_modules(
            pas, ComputeConfig(
                1, ngpus, use_async_reducer=False,
                reducer_bucket_cap_mb=100
            ),
            tempdir,
            'sync_',
        )
        whole_async_results = _train(whole_module_async, update_freq)
        whole_sync_results = _train(whole_module_sync, update_freq)
        sub_async_results = _train(sub_module_async, update_freq)
        sub_sync_results = _train(sub_module_sync, update_freq)
        return (
            whole_async_results,
            whole_sync_results,
            sub_async_results,
            sub_sync_results
        )


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason='lack of gpu devices')
@pytest.mark.parametrize('update_freq', [1, 4])
def test_dp2(update_freq):
    results = launch_torchrun(2, _gpu_worker, 'dp', 2, update_freq)
    whole_async0, whole_sync0, sub_async0, sub_sync0 = results[0]
    whole_async1, whole_sync1, sub_async1, sub_sync1 = results[1]

    assert len(whole_async0) == len(whole_sync0) == len(sub_async0) == len(sub_sync0)

    for iter in range(len(whole_async0)): # for each iteration
        iter_whole_async0 = whole_async0[iter]
        iter_whole_sync0 = whole_sync0[iter]
        iter_sub_async0 = sub_async0[iter]
        iter_sub_sync0 = sub_sync0[iter]

        iter_whole_async1 = whole_async1[iter]
        iter_whole_sync1 = whole_sync1[iter]
        iter_sub_async1 = sub_async1[iter]
        iter_sub_sync1 = sub_sync1[iter]

        # pred
        assert torch.equal(iter_whole_async0[0], iter_whole_async1[0])
        assert torch.equal(iter_sub_async0[0], iter_sub_async1[0])
        assert torch.equal(iter_whole_sync0[0], iter_whole_sync1[0])
        assert torch.equal(iter_sub_sync0[0], iter_sub_sync1[0])

        assert torch.equal(iter_whole_async0[0], iter_whole_sync0[0])
        assert torch.equal(iter_sub_async0[0], iter_sub_sync0[0])
        assert torch.equal(iter_whole_async0[0], iter_sub_async0[0])

        # weights
        whole_async_weights, _ = merge_state_dicts([iter_whole_async0[1], iter_whole_async1[1]])
        whole_sync_weights, _ = merge_state_dicts([iter_whole_sync0[1], iter_whole_sync1[1]])
        sub_async_weights, _ = merge_state_dicts([iter_sub_async0[1], iter_sub_async1[1]])
        sub_sync_weights, _ = merge_state_dicts([iter_sub_sync0[1], iter_sub_sync1[1]])

        assert_equal(whole_async_weights, whole_sync_weights)
        assert_equal(sub_async_weights, sub_sync_weights)

        assert set(whole_async_weights.keys()) == set(sub_async_weights.keys())

        for key in whole_async_weights.keys():
            assert torch.equal(whole_async_weights[key], sub_async_weights[key])
