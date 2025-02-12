#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from collections import namedtuple

import torch
from nnscaler.graph.parser.converter import to_fx_graph

from ...utils import replace_all_device_with


def func_with_type_hint(x: list[torch.Tensor]) -> torch.Tensor:
    return x[0] + x[1]


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 5)
        self.fc2 = torch.nn.Linear(10, 5)

    def forward(self, data: dict[str, torch.Tensor]):
        return func_with_type_hint([self.fc1(data['x']), self.fc2(data['x'])])


@replace_all_device_with('cpu')
def test_type_hint():
    model = SimpleModel()
    dummy_input = {'data': {'x': torch.rand(10)}}
    traced_graph = to_fx_graph(model, dummy_input)

    # just check if we can trace a model contains original type hint
    assert True
