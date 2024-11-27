#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import torch

from nnscaler.graph.parser.converter import to_fx_graph

from ...utils import replace_all_device_with


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.nn.functional.dropout(x, 0.1, self.training)


@replace_all_device_with('cpu')
def test_getattr_from_root():
    model = SimpleModel()
    dummy_input = {'x': torch.rand(10)}
    traced_graph = to_fx_graph(model, dummy_input)
    traced_graph(**dummy_input)
