#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from pathlib import Path
import tempfile
import torch

import pytest
import torch.distributed

from nnscaler.parallel import parallelize, ComputeConfig
from nnscaler.flags import CompileFlag

from .common import init_distributed
from ..launch_torchrun import launch_torchrun
from ..utils import catch_stdout, clear_dir_on_rank0


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(64, 64)

    # x with shape [128, 64]
    def forward(self, x):
        return self.fc(x)


def _to_cube_model(module, compute_config, cube_savedir, instance_name, input_shape, init_module_params=True):
    return parallelize(
        module,
        {'x': torch.randn(input_shape)},
        'tp',
        compute_config,
        gen_savedir=cube_savedir,
        instance_name=instance_name,
        init_module_params=init_module_params
    )


def _gpu_worker():
    init_distributed()
    compute_config = ComputeConfig(1, 1, use_zero=False)
    try:
        CompileFlag.line_timer = True
        with clear_dir_on_rank0(Path(tempfile.gettempdir()) / 'cube_test_line_timer') as tempdir:
            net = _to_cube_model(Net(), compute_config, tempdir, 'net', (128, 64))
            x = torch.randn(128, 64).cuda()

            with catch_stdout() as log_stream:
                net(x)
                logs = log_stream.getvalue()
                assert 'line timer: 0' in logs
    finally:
        CompileFlag.line_timer = False


@pytest.mark.skipif(not torch.cuda.is_available(), reason='lack of gpu devices')
def test_line_timer():
    launch_torchrun(1, _gpu_worker)
