#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import pytest
import tempfile
import math
import torch

from nnscaler.graph.parser.converter import convert_model

from ...utils import replace_all_device_with


@replace_all_device_with('cpu')
def test_input_broadcast_constant_attr():
    class SimpleModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = torch.nn.Linear(10, 5)

        def forward(self, sample):
            res = sample['y'] + 1
            res = res - 1
            res = res * 1
            res = res / 1
            res = res // 1
            res = res % 1
            res = res ** 1
            res = res - 1
            res = -res
            res = math.exp(res)
            res = math.sqrt(res)
            return self.fc(sample['x']), res

    with tempfile.TemporaryDirectory() as tempdir:
        cube_graph = convert_model(SimpleModel(), {'sample': {'x': torch.rand(4, 10), 'y': 10}}, tempdir, constant_folding=False)
        # check input is not constant
        assert not cube_graph.input(0).value['y'].is_constant
        for i, name in enumerate(['add', 'sub', 'mul', 'truediv', 'floordiv', 'mod', 'pow', 'sub', 'neg', 'exp', 'sqrt']):
            op_node = cube_graph.nodes()[i + 1]
            assert op_node.signature.split(".")[-1] == name and not op_node.output(0).is_constant
