#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import tempfile
import itertools
import re
from pathlib import Path
import shutil
import pytest
from typing import Dict, Tuple, List, Any
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

import numpy as np

from nnscaler.parallel import ComputeConfig, parallelize, build_optimizer, merge_state_dicts, load_merged_state_dict
from nnscaler.runtime.module import ParallelModule, ExtraState
from nnscaler.runtime.gnorm import calcuate_gnorm

from .common import CubeLinear, init_random, init_distributed
from ..launch_torchrun import launch_torchrun, clone_to_cpu_recursively
from .test_checkpoint_shared import _train_raw, _load_merged
from ..utils import clear_dir_on_rank0


class FcReluWithUnused(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.unused_fc0 = CubeLinear(out_features, out_features, bias=bias)
        self.fc1 = CubeLinear(in_features, in_features, bias=bias)
        self.relu1 = nn.ReLU()
        self.unused_fc1 = CubeLinear(out_features, out_features, bias=bias)
        self.fc2 = CubeLinear(in_features, out_features, bias=bias)
        self.unused_fc2 = CubeLinear(out_features, out_features, bias=bias)
        self.relu2 = nn.ReLU()


    def forward(self, x):
        return self.relu2(self.fc2(self.relu1(self.fc1(x))))


class FcRelu_4_4_WithUnused(FcReluWithUnused):
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


def _create_cube_module(pas, compute_config, cube_savedir, module_type='raw'):
    init_random()
    if module_type == 'raw':
        class RawModuleWithUnused(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.unused_linear0 = nn.Linear(4, 4)
                self.linear1 = nn.Linear(4, 4)
                self.fc_relu1 = FcRelu_4_4_WithUnused()
                self.unused_linear1 = nn.Linear(4, 4)
                self.linear3 = nn.Linear(4, 1)
                self.unused_linear2 = nn.Linear(4, 4)
                self.sigmoid = nn.Sigmoid()
            def forward(self, x):
                x = self.linear1(x)
                x = self.fc_relu1(x)
                x = self.linear3(x)
                x = self.sigmoid(x)
                return x
        init_random()
        return RawModuleWithUnused().cuda()
    else:
        class ParallelModuleWithUnused(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.unused_linear0 = nn.Linear(4, 4)
                self.linear1 = nn.Linear(4, 4)
                self.fc_relu1 = _to_cube_model(
                    FcRelu_4_4_WithUnused(), pas,
                    compute_config, cube_savedir, 'fc_relu1'
                )
                self.unused_linear1 = nn.Linear(4, 4)
                self.linear3 = nn.Linear(4, 1)
                self.unused_linear2 = nn.Linear(4, 4)
                self.sigmoid = nn.Sigmoid()
            def forward(self, x):
                x = self.linear1(x)
                x = self.fc_relu1(x)
                x = self.linear3(x)
                x = self.sigmoid(x)
                return x
        init_random()
        return ParallelModuleWithUnused().cuda()


def _gpu_worker(use_zero, pas, plan_ngpus, runtime_ngpus):
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
            _train_raw(_create_cube_module(pas, compute_config, tempdir, 'raw'), tempdir)
        torch.distributed.barrier()
        _load_merged(
            _create_cube_module(pas, compute_config, tempdir, 'cube'),
            tempdir
        )

@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4, reason='lack of gpu devices')
@pytest.mark.parametrize('use_zero', [True, False])
def test_checkpoint_load_from_raw_checkpoint(use_zero):
    """
    Test when the checkpoint is generated from raw module and need to be loaded to parallel module.
    """
    plan_ngpus = 2
    runtime_ngpus = 4
    launch_torchrun(4, _gpu_worker, use_zero, 'tp', plan_ngpus, runtime_ngpus)
