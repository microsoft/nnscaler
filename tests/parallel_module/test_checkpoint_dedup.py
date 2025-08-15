#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import tempfile
from pathlib import Path
import pytest
from typing import Dict, Tuple, List, Any

import torch
from torch import nn

from nnscaler.parallel import ComputeConfig, parallelize, build_optimizer, \
    merge_state_dicts, load_merged_state_dict, \
    deduped_state_dict, load_deduped_state_dict
from nnscaler.runtime.module import ParallelModule

from .common import PASMegatron, CubeLinear, init_random, init_distributed, assert_equal
from ..launch_torchrun import launch_torchrun
from .test_checkpoint import gendata, train_step, End2EndMLP, End2EndMLPWithUnusedAndShared
from ..utils import clear_dir_on_rank0


class FcRelu(nn.Module):
    def __init__(self, in_features, bias=True):
        super().__init__()
        init_random()
        self.fc1 = CubeLinear(in_features, in_features, bias=bias)
        self.relu1 = nn.ReLU()
        self.fc2 = CubeLinear(in_features, in_features, bias=bias)
        self.relu2 = nn.ReLU()


    def forward(self, x):
        return self.relu2(self.fc2(self.relu1(self.fc1(x))))


class FcRelu4(FcRelu):
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


def _create_cube_module(pas, compute_config1, compute_config2, cube_savedir):
    init_random()
    class ParallelModule0(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(4, 4)
            self.fc_relu1 = _to_cube_model(
                FcRelu4(), pas,
                compute_config1, cube_savedir, f'fc_relu1'
            )
            self.linear2 = nn.Linear(4, 4)
            self.fc_relu2 = _to_cube_model(
                FcRelu4(), pas,
                compute_config2, cube_savedir, f'fc_relu2'
            )
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
    return ParallelModule0().cuda()


DATA_SIZE = 256
CKPT_FILE_NAME_TEMPLATE = '{}.pth'


def _train(model: torch.nn.Module, ckpt_dir):
    CKPT_FILE_NAME = CKPT_FILE_NAME_TEMPLATE.format(torch.distributed.get_rank())

    DATA = gendata(model, DATA_SIZE, 0, DATA_SIZE, 0, 1)
    optimizer = build_optimizer(model, torch.optim.Adam, lr=0.01)
    for i, (x, y) in enumerate(DATA):
        train_step(model, x, y, optimizer)
        optimizer.zero_grad()
    deduped_model_state_dict, deduped_opt_state_dict = deduped_state_dict(model, optimizer)
    torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'model-dedup': deduped_model_state_dict,
            'optimizer-dedup': deduped_opt_state_dict
        }, ckpt_dir / CKPT_FILE_NAME)


def _check_deduped(model: torch.nn.Module, ckpt_dir):
    rank = torch.distributed.get_rank()
    ckpt_files = [
        ckpt_dir / CKPT_FILE_NAME_TEMPLATE.format(i)
        for i in range(torch.distributed.get_world_size())
    ]
    ckpt_state_dicts = [torch.load(f, weights_only=False) for f in ckpt_files]
    model_state_dicts = [ckpt['model'] for ckpt in ckpt_state_dicts]
    optimizer_state_dicts = [ckpt['optimizer'] for ckpt in ckpt_state_dicts]
    dedupped_model_state_dicts = [ckpt['model-dedup'] for ckpt in ckpt_state_dicts]
    dedupped_optimizer_state_dicts = [ckpt['optimizer-dedup'] for ckpt in ckpt_state_dicts]

    parallel_modules = [m for m in model.modules() if isinstance(m, ParallelModule)]
    # assert len(parallel_modules) == 2

    module_dedup_group_size = [m.module_dedup_group_size for m in parallel_modules]
    opt_dedup_group_size = [m.optimizer_dedup_group_size for m in parallel_modules]
    assert all(s1 >= s2 for s1, s2 in zip(opt_dedup_group_size, module_dedup_group_size))
    assert all(s1 % s2 == 0 for s1, s2 in zip(opt_dedup_group_size, module_dedup_group_size))

    # check deduped state dicts are correct
    for i, (
        model_state_dict,
        optimizer_state_dict,
        dedupped_model_state_dict,
        dedupped_optimizer_state_dict
    ) in enumerate(zip(model_state_dicts, optimizer_state_dicts, dedupped_model_state_dicts, dedupped_optimizer_state_dicts)):
        if i == 0:
            assert_equal(model_state_dict, dedupped_model_state_dict)
        elif i >= max(module_dedup_group_size):
            # only EXTRA_STATEs are kept
            assert len(dedupped_model_state_dict) == len(parallel_modules)
            assert all(k.endswith(ParallelModule.EXTRA_STATE_KEY) for k in dedupped_model_state_dict.keys())
        else:
            if not isinstance(model, ParallelModule):
                # in this case, non parallel module is removed, so it should have less keys
                assert len(parallel_modules) < len(dedupped_model_state_dict) < len(model_state_dict)
            else:
                assert len(dedupped_model_state_dict) == len(model_state_dict)
            for k, v in dedupped_model_state_dict.items():
                assert_equal(v, model_state_dict[k])

        # we keep param_groups in all ranks.
        assert_equal(dedupped_optimizer_state_dict['param_groups'], optimizer_state_dict['param_groups'])
        if i == 0:
            assert_equal(optimizer_state_dict, dedupped_optimizer_state_dict)
        elif i >= max(opt_dedup_group_size):
            # only EXTRA_STATEs and param_groups are kept
            assert not dedupped_optimizer_state_dict['state']  # should have empty state
        else:
            if not isinstance(model, ParallelModule):
                # in this case, non parallel module is removed, so it should have less keys
                assert 0 < len(dedupped_optimizer_state_dict['state'])  < len(optimizer_state_dict['state'])
            else:
                assert len(dedupped_optimizer_state_dict['state']) == len(optimizer_state_dict['state'])
            for k, v in dedupped_optimizer_state_dict['state'].items():
                assert_equal(v, optimizer_state_dict['state'][k])

    # check deduped state dicts can be merged and output exactly the same state dict
    merged_model_state_dicts, merged_optimizer_state_dict = \
        merge_state_dicts(model_state_dicts, optimizer_state_dicts)
    merged_model_state_dicts_dedup, merged_optimizer_state_dict_dedup = \
        merge_state_dicts(dedupped_model_state_dicts, dedupped_optimizer_state_dicts)

    assert_equal(merged_model_state_dicts, merged_model_state_dicts_dedup)
    assert_equal(merged_optimizer_state_dict, merged_optimizer_state_dict_dedup)

    # check deduped state dicts can be loaded to the model
    # which should output the same state dict as the original model
    optimizer = build_optimizer(model, torch.optim.Adam, lr=0.01)
    load_deduped_state_dict(model, dedupped_model_state_dicts[rank],
        optimizer, dedupped_optimizer_state_dicts[rank]
    )
    assert_equal(model.state_dict(), model_state_dicts[rank])
    assert_equal(optimizer.state_dict(), optimizer_state_dicts[rank])


def _gpu_worker(pas, cc1, cc2):
    init_distributed()
    with clear_dir_on_rank0(Path(tempfile.gettempdir()) / 'cube_test_ckpt_compact') as tempdir:
        _train(_create_cube_module(pas, cc1, cc2, tempdir), tempdir)
        torch.distributed.barrier()
        _check_deduped(
            _create_cube_module(pas, cc1, cc2, tempdir),
            tempdir
        )

@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4, reason='lack of gpu devices')
@pytest.mark.parametrize('use_zero', [True, False])
def test_checkpoint_compact(use_zero):
    cc1 = ComputeConfig(1, 4, use_zero=use_zero, zero_ngroups=2 if use_zero else 1)
    cc2 = ComputeConfig(1, 4, use_zero=use_zero, zero_ngroups=4 if use_zero else 1)
    launch_torchrun(4, _gpu_worker, 'tp', cc1, cc2)

    # mixed zero and non-zero
    cc1 = ComputeConfig(2, 4, use_zero=not use_zero, zero_ngroups=2 if not use_zero else 1)
    cc2 = ComputeConfig(2, 4, use_zero=use_zero, zero_ngroups=1)
    launch_torchrun(4, _gpu_worker, 'tp', cc1, cc2)


def _gpu_worker_pipeline(cc):
    init_distributed()
    with clear_dir_on_rank0(Path(tempfile.gettempdir()) / 'cube_test_ckpt_compact_pipeline') as tempdir:
        for model_cls in [End2EndMLP, End2EndMLPWithUnusedAndShared]:
            pipeline_moule_cls = model_cls.to_pipeline_module(cc, tempdir)
            _train(pipeline_moule_cls().cuda(), tempdir)
            torch.distributed.barrier()
            _check_deduped(
                pipeline_moule_cls().cuda(),
                tempdir
            )


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4, reason='lack of gpu devices')
def test_checkpoint_compact_pipeline():
    cc1 = ComputeConfig(2, 4, use_zero=False)
    launch_torchrun(4, _gpu_worker_pipeline, cc1)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4, reason='lack of gpu devices')
def test_checkpoint_compact_pipeline_use_zero():
    cc1 = ComputeConfig(2, 4, use_zero=True, zero_ngroups=1)
    cc2 = ComputeConfig(2, 4, use_zero=True, zero_ngroups=2)
    launch_torchrun(4, _gpu_worker_pipeline, cc1)
    launch_torchrun(4, _gpu_worker_pipeline, cc2)
