#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import pytest
import torch

from nnscaler.graph.parser.converter import to_fx_graph

from ...utils import replace_all_device_with


@replace_all_device_with('cpu')
def test_no_grad():
    class SimpleModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = torch.nn.Linear(10, 10)
            self.fc2 = torch.nn.Linear(10, 10)

        def forward(self, x):
            with torch.no_grad():
                x = self.fc1(x)
            x = self.fc2(x)
            return x

    traced_graph = to_fx_graph(SimpleModel(), {'x': torch.rand(4, 10)})

    # The traced graph:
    #
    # def forward(self, x):
    #     fc1_weight = self.fc1.weight
    #     fc1_bias = self.fc1.bias
    #     linear = torch._C._nn.linear(x, fc1_weight, fc1_bias);  x = fc1_weight = fc1_bias = None
    #     fc2_weight = self.fc2.weight
    #     fc2_bias = self.fc2.bias
    #     linear_1 = torch._C._nn.linear(linear, fc2_weight, fc2_bias);  linear = fc2_weight = fc2_bias = None
    #     return linear_1

    assume_no_requires_grad_nodes = set(['x', 'linear'])
    actual_no_requires_grad_nodes = set()
    for node in traced_graph.graph.nodes:
        if node.meta['tensor_meta'].requires_grad == False:
            actual_no_requires_grad_nodes.add(node.name)
    assert assume_no_requires_grad_nodes == actual_no_requires_grad_nodes, \
        f'assume no require grad node: {assume_no_requires_grad_nodes}, actual no require grad node: {actual_no_requires_grad_nodes}'
