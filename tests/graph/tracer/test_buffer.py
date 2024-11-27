#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import tempfile

import torch

from nnscaler import parallelize, ComputeConfig
from tests.utils import replace_all_device_with

from tests.parallel_module.test_gencode import _gencode_contains, print_gencode
from .test_ctxt_manager import TestModule


class BufferModuleNested(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("sub_buffer0_u", torch.tensor(1.0), persistent=False)
        self.register_buffer("sub_buffer0_p", torch.tensor([2.0]), persistent=True)

    def forward(self, x):
        return x + self.sub_buffer0_u + self.sub_buffer0_p


class BufferModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.test_module = TestModule()
        self.buffer_module = BufferModuleNested()
        self.register_buffer("root_buffer0_u", torch.tensor([1.0]), persistent=False)
        self.register_buffer("root_buffer0_p", torch.tensor(2.0), persistent=True)

    def forward(self, x, position_ids):
        x = self.test_module(x, position_ids)
        x = self.buffer_module(x)
        return x + self.root_buffer0_u + self.root_buffer0_p



@replace_all_device_with('cpu')
def test_buffer():
    with tempfile.TemporaryDirectory() as tempdir:
        model = BufferModule()
        dummy_input = {'x': torch.rand(1, 100, 128), 'position_ids': torch.arange(0, 100, dtype=torch.int64).reshape(1, 100)}
        parallelize(model, dummy_input, 'dp', ComputeConfig(1, 1), gen_savedir=tempdir, load_module=False)
        # code will look like:
        # self.register_buffer('test_module_rotary_emb_inv_freq_94', torch.empty((64,), dtype=torch.float32), persistent=False)
        # self.register_buffer('buffer_module_sub_buffer0_u_114', torch.empty((), dtype=torch.float32), persistent=False)
        # self.register_buffer('buffer_module_sub_buffer0_p_116', torch.empty((1,), dtype=torch.float32), persistent=True)
        # self.register_buffer('root_buffer0_u_118', torch.empty((1,), dtype=torch.float32), persistent=False)
        # self.register_buffer('root_buffer0_p_120', torch.empty((), dtype=torch.float32), persistent=True)

        assert _gencode_contains(tempdir, BufferModule, 0,
            r'self.register_buffer\(\'test_module_rotary_emb_inv_freq_\d+\', torch.empty\(\(64,\), dtype=torch.float32\), persistent=False\)'
        )
        assert _gencode_contains(tempdir, BufferModule, 0,
            r'self.register_buffer\(\'buffer_module_sub_buffer0_u_\d+\', torch.empty\(\(\), dtype=torch.float32\), persistent=False\)'
        )
        assert _gencode_contains(tempdir, BufferModule, 0,
            r'self.register_buffer\(\'buffer_module_sub_buffer0_p_\d+\', torch.empty\(\(1,\), dtype=torch.float32\), persistent=True\)'
        )
        assert _gencode_contains(tempdir, BufferModule, 0,
            r'self.register_buffer\(\'root_buffer0_u_\d+\', torch.empty\(\(1,\), dtype=torch.float32\), persistent=False\)'
        )
        assert _gencode_contains(tempdir, BufferModule, 0,
            r'self.register_buffer\(\'root_buffer0_p_\d+\', torch.empty\(\(\), dtype=torch.float32\), persistent=True\)'
        )
