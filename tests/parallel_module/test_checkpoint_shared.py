#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import tempfile
from pathlib import Path
import pytest
from typing import Dict, Tuple, List, Any

import torch
from torch import nn

from nnscaler.parallel import ComputeConfig, parallelize, build_optimizer, merge_state_dicts, load_merged_state_dict

from .common import CubeLinear, init_random, init_distributed
from ..launch_torchrun import launch_torchrun
from .test_checkpoint import End2EndMLP, train_step, gendata
from ..utils import clear_dir_on_rank0


class FcReluWithShared(nn.Module):
    def __init__(self, in_features, bias=True):
        super().__init__()
        init_random()
        self.unused_fc1 = CubeLinear(in_features, in_features, bias=bias)
        self.fc1 = CubeLinear(in_features, in_features, bias=bias)
        self.relu1 = nn.ReLU()
        self.fc2 = CubeLinear(in_features, in_features, bias=bias)
        self.fc2.fc.weight = self.fc1.fc.weight # share the weights
        self.relu2 = nn.ReLU()


    def forward(self, x):
        return self.relu2(self.fc2(self.relu1(self.fc1(x))))


class FcRelu_4_WithShared(FcReluWithShared):
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


def _create_cube_module(pas, compute_config, cube_savedir, module_type='sub/raw'):
    init_random()
    if module_type == 'sub/raw':
        class RawModuleWithShared(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(4, 4)
                self.fc_relu1 = FcRelu_4_WithShared()
                self.unused_linear1 = nn.Linear(4, 4)
                self.linear2 = nn.Linear(4, 4)
                self.linear2.weight = self.linear1.weight # share the weights
                self.linear3 = nn.Linear(4, 1)
                self.sigmoid = nn.Sigmoid()
            def forward(self, x):
                x = self.linear1(x)
                x = self.fc_relu1(x)
                x = self.linear2(x)
                x = self.linear3(x)
                x = self.sigmoid(x)
                return x
        init_random()
        return RawModuleWithShared().cuda()
    elif module_type == 'sub/cube':
        class ParallelModuleWithShared(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(4, 4)
                self.fc_relu1 = _to_cube_model(
                    FcRelu_4_WithShared(), pas,
                    compute_config, cube_savedir, 'fc_relu1'
                )
                self.unused_linear1 = nn.Linear(4, 4)
                self.linear2 = nn.Linear(4, 4)
                self.linear2.weight = self.linear1.weight # share the weights
                self.linear3 = nn.Linear(4, 1)
                self.sigmoid = nn.Sigmoid()
            def forward(self, x):
                x = self.linear1(x)
                x = self.fc_relu1(x)
                x = self.linear2(x)
                x = self.linear3(x)
                x = self.sigmoid(x)
                return x
        init_random()
        return ParallelModuleWithShared().cuda()
    elif module_type.startswith('pipeline/'):
        class RawModuleWithUnused(End2EndMLP):
            def __init__(self):
                super().__init__()
                self.linear0_unused = nn.Linear(4, 4)  # unused weights
                self.layers[2].weight = self.layers[0].weight  # shared weights in same stage
        init_random()
        if module_type.endswith('/raw'):
            return RawModuleWithUnused().cuda()
        else:
            return RawModuleWithUnused.to_pipeline_module(compute_config, cube_savedir, 'pipeline')().cuda()
    elif module_type.startswith('pipeline2/'):
        class RawModuleWithUnused(End2EndMLP):
            def __init__(self):
                super().__init__()
                self.linear0_unused = nn.Linear(4, 4)  # unused weights
                self.layers[5].weight = self.layers[0].weight  # shared weights across stages
        init_random()
        if module_type.endswith('/raw'):
            return RawModuleWithUnused().cuda()
        else:
            return RawModuleWithUnused.to_pipeline_module(compute_config, cube_savedir, 'pipeline')().cuda()


DATA_SIZE = 256
RAW_CKPT_FILE_NAME = 'raw.pth'


def _train_raw(model: torch.nn.Module, ckpt_dir):
    DATA = gendata(model, DATA_SIZE, 0, DATA_SIZE, 0, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for i, (x, y) in enumerate(DATA):
        y_pred, loss = train_step(model, x, y, optimizer)
        optimizer.zero_grad()
    torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, ckpt_dir / RAW_CKPT_FILE_NAME)


def _load_merged(parallel_model: torch.nn.Module, ckpt_dir):
    raw_ckpt_dict = torch.load(ckpt_dir / RAW_CKPT_FILE_NAME)
    raw_model_state_dict: Dict[str, Any] = raw_ckpt_dict['model']
    raw_opt_state_dict = raw_ckpt_dict['optimizer']
    optimizer = build_optimizer(parallel_model, torch.optim.Adam, lr=0.01)
    load_merged_state_dict(
        parallel_model, raw_model_state_dict,
        optimizer, raw_opt_state_dict,
    )

    ckpt_file_template = 'ckpt_{rank}.pth'
    ckpt_merged_file = ckpt_dir / 'ckpt_merged.pth'
    ckpt_file = ckpt_dir / ckpt_file_template.format(
        rank=torch.distributed.get_rank()
    )
    model_state_dict = parallel_model.state_dict()
    optimizer_state_dict = optimizer.state_dict()
    torch.save({
        'model': model_state_dict,
        'optimizer': optimizer_state_dict
    }, ckpt_file)

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        ckpt_files = [ckpt_dir / ckpt_file_template.format(rank=i) for i in range(torch.distributed.get_world_size())]
        ckpt_state_dicts = [torch.load(f, weights_only=False) for f in ckpt_files]
        model_state_dicts = [ckpt['model'] for ckpt in ckpt_state_dicts]
        optimizer_state_dicts = [ckpt['optimizer'] for ckpt in ckpt_state_dicts]
        merged_model_state_dicts, merged_optimizer_state_dict = merge_state_dicts(model_state_dicts, optimizer_state_dicts)
        torch.save({
            'model': merged_model_state_dicts,
            'optimizer': merged_optimizer_state_dict
        }, ckpt_merged_file)
         # only key that contains `unused`` and not start with `unused` will be removed
        raw_model_state_dict = {
            key: value
            for key, value in raw_model_state_dict.items()
            if not ('unused' in key and not key.startswith('unused'))
        }
        assert set(merged_model_state_dicts.keys()) == set(raw_model_state_dict.keys())
        for index in merged_model_state_dicts.keys():
            assert torch.equal(merged_model_state_dicts[index].cuda(), raw_model_state_dict[index].cuda())

        assert set(merged_optimizer_state_dict.keys()) == set(raw_opt_state_dict.keys())
        assert merged_optimizer_state_dict['param_groups'] == raw_opt_state_dict['param_groups']
        assert set(merged_optimizer_state_dict['state']) == set(raw_opt_state_dict['state'])
        for index in merged_optimizer_state_dict['state']:
            for key in ('step', 'exp_avg', 'exp_avg_sq'):
                assert torch.equal(merged_optimizer_state_dict['state'][index][key].cuda(), raw_opt_state_dict['state'][index][key].cuda())


def _gpu_worker(module_type, use_zero, pas, plan_ngpus, runtime_ngpus):
    # Basic logic:
    #   a. first train the original model, get a full state dict
    #   b. then use parallel model to load the full state dict as a merged state dict
    #   c. then parallel model save their own state dicts, and merge them to get a merged state dict.
    #   d. compare the full state dict in step a and the merged state dict in step c. They should be the same.
    init_distributed()
    compute_config = ComputeConfig(plan_ngpus, runtime_ngpus, use_zero=use_zero)
    with clear_dir_on_rank0(Path(tempfile.gettempdir()) / 'cube_test_ckpt') as tempdir:
        if torch.distributed.get_rank() == 0:
            tempdir.mkdir(parents=True, exist_ok=True)
            _train_raw(_create_cube_module(pas, compute_config, tempdir, f'{module_type}/raw'), tempdir)
        torch.distributed.barrier()
        _load_merged(
            _create_cube_module(pas, compute_config, tempdir, f'{module_type}/cube'),
            tempdir
        )

@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4, reason='lack of gpu devices')
@pytest.mark.parametrize('use_zero', [True, False])
@pytest.mark.parametrize('module_type', ['sub', 'pipeline', 'pipeline2'])
def test_checkpoint_load_from_raw_checkpoint(module_type, use_zero):
    """
    Test when the checkpoint is generated from raw module and need to be loaded to parallel module.
    """
    plan_ngpus = 2
    runtime_ngpus = 4
    launch_torchrun(4, _gpu_worker, module_type, use_zero, 'tp', plan_ngpus, runtime_ngpus)
