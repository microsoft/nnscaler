#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import torch
import pytest

from nnscaler.graph.parser.converter import to_fx_graph

from ...utils import replace_all_device_with


@replace_all_device_with('cpu')
def test_cls_wrapper():
    class MyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 5)

        def forward(self, x):
            # if we don't wrap tuple or float, the trace will raise error here
            x_value = float(tuple(x)[0])
            x = torch.fill(torch.empty(1, 3), x_value)
            return self.linear(x)

    dummy_input = {'x': torch.tensor([1.1, 1.2, 1.3])}
    module = MyModule()
    fx_graph = to_fx_graph(module, dummy_input)

    # just check there is no error raised
    assert True

    # The traced graph is:
    #
    # def forward(self, x):
    #     tuple_1 = tuple(x);  x = None
    #     getitem = tuple_1[0];  tuple_1 = None
    #     float_1 = float(getitem);  getitem = None
    #     empty = torch.empty(1, 3)
    #     fill = torch.fill(empty, float_1);  empty = float_1 = None
    #     linear_weight = self.linear.weight
    #     linear_bias = self.linear.bias
    #     linear = torch._C._nn.linear(fill, linear_weight, linear_bias);  fill = linear_weight = linear_bias = None
    #     return linear
