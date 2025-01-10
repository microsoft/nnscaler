#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import tempfile
import torch
from nnscaler.parallel import parallelize, ComputeConfig

from ...utils import replace_all_device_with


class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        k = list(x.keys())[0]
        v = x[k]
        y = list(x.values())[0]
        z = list(x.items())[0][1]
        return torch.sum(v + y + z)


@replace_all_device_with('cpu')
def test_script_func():
    with tempfile.TemporaryDirectory() as tempdir:
        m_new = parallelize(
            Model(),
            {'x': {'a': torch.rand(10)}},
            'tp',
            ComputeConfig(2, 2),
            gen_savedir=tempdir,
            load_module=False
        )
        assert True
