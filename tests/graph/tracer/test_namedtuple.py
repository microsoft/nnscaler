#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from collections import namedtuple

import torch
from nnscaler.graph.parser.converter import to_fx_graph

from ...utils import replace_all_device_with


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 5)
        self.fc2 = torch.nn.Linear(10, 5)

    def forward(self, x):
        Result = namedtuple('Result', ['r1', 'r2'])
        return Result(self.fc1(x), self.fc2(x))


@replace_all_device_with('cpu')
def test_namedtuple():
    model = SimpleModel()
    dummy_input = {'x': torch.rand(10)}
    traced_graph = to_fx_graph(model, dummy_input)

    # just check if we can trace a model contains namedtuple
    assert True
