#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import tempfile
import torch
from nnscaler.parallel import parallelize, ComputeConfig

from .helper import cus_add, cus_sub
from ...utils import replace_all_device_with


class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a, b):
        return cus_add(a, b) + cus_sub(a, b)


@replace_all_device_with('cpu')
def test_script_func():
    with tempfile.TemporaryDirectory() as tempdir:
        m_new = parallelize(
            Model(),
            {'a': torch.rand(10), 'b': torch.rand(10)},
            'tp',
            ComputeConfig(2, 2),
            gen_savedir=tempdir,
            load_module=False
        )
