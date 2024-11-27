#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import tempfile
from typing import *

import pytest
import torch
import torch.nn as nn

from nnscaler.parallel import ComputeConfig, parallelize

from .utils import init_random

MBS = 2
DIM = 16
LAYERS = 16

class MLP(nn.Module):
    def __init__(self, dim: int = DIM, nlayers: int = LAYERS):
        init_random()
        super().__init__()
        self.layers = torch.nn.ModuleList([])
        for _ in range(nlayers):
            self.layers.append(nn.Linear(dim, dim, bias=False))
        self.loss_fn = nn.BCELoss()

    def forward(self, data: Dict[str, torch.Tensor]):
        x = data['data']
        for layer in self.layers:
            x = layer(x)
        x = torch.sigmoid(x)
        loss = self.loss_fn(x, data['target'])
        return loss


def dummy_data():
    return {
        'data': torch.randn(
            MBS, DIM, device=torch.cuda.current_device()),
        'target': torch.rand(
            MBS, DIM, device=torch.cuda.current_device())
    }


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4, reason='lack of gpu devices')
def test_autodist():
    with tempfile.TemporaryDirectory() as tempdir:
        m_new = parallelize(
            MLP(),
            {'data': dummy_data()},
            'autodist',
            ComputeConfig(2, 4, pas_config={
                    'update_freq': 1,
                    'task_name': 'test_autodist',
            }),
            gen_savedir=tempdir,
            load_module=False
        )
        assert m_new is None
