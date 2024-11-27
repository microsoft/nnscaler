#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import torch
from nnscaler.graph.parser.converter import to_fx_graph

from ...utils import replace_all_device_with


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 5)
    
    def forward(self, x, y):
        loss = torch.nn.CrossEntropyLoss()
        return loss(self.fc1(x), y)


@replace_all_device_with('cpu')
def test_module_jit_init():
    model = SimpleModel()
    dummy_input = {'x': torch.rand(2, 10), 'y': torch.tensor([0,1])}
    traced_graph = to_fx_graph(model, dummy_input)

    cross_entropy_node = list(traced_graph.graph.nodes)[5]
    assert cross_entropy_node.name == 'cross_entropy', f'{cross_entropy_node.name}'
    assert '_module_constants.0' in cross_entropy_node.meta['nn_module_stack'], f"{cross_entropy_node.meta['nn_module_stack']}"
