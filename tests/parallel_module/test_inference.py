#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from pathlib import Path
import shutil
import tempfile

import pytest
import torch
from torch import nn

from nnscaler.parallel import ComputeConfig, parallelize

from .common import CubeLinear, init_distributed, init_random
from ..launch_torchrun import torchrun
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


class Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_relu1 = FcRelu(4, 4)
        self.fc_relu2 = FcRelu(4, 4)
        self.dropout = nn.Dropout(0.5)
        self.linear3 = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.fc_relu1(x)
        x = self.fc_relu2(x)
        x = self.dropout(x)
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

def _inference_worker(ngpus, inference_only):
    init_distributed()
    init_random()

    with clear_dir_on_rank0(Path(tempfile.gettempdir()) / 'cube_inference_test') as tempdir:
        model = Module()
        model.eval()

        cube_model = _to_cube_model(model, 'tp',
            ComputeConfig(ngpus, ngpus, inference_only=inference_only),
            tempdir, 'test_inference'
        )

        data = torch.tensor([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])
        assert not model.training
        assert not cube_model.training
        model.cuda()

        with torch.inference_mode():
            result = model(data)
            cube_result = cube_model(data)
            assert torch.allclose(result, cube_result, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='lack of gpu devices')
def test_inference1():
    torchrun(1, _inference_worker, 1, True)
    torchrun(1, _inference_worker, 1, False)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason='lack of gpu devices')
def test_inference2():
    torchrun(2, _inference_worker, 2, True)
    torchrun(2, _inference_worker, 2, False)
