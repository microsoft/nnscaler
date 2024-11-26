#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import pytest
import torch

from nnscaler.graph.parser.converter import to_fx_graph

from ...utils import replace_all_device_with


@replace_all_device_with('cpu')
def test_dce_useless_next():
    class SimpleModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.m_dict = torch.nn.ModuleDict({'test': torch.nn.Linear(10, 8)})
            self.m_keys = ['test', 'default']

        def forward(self, x):
            for key in self.m_keys:
                # TODO: (ning) This kind of code style will call instruction 'BINARY_SUBSCR'
                # And we should not pass a proxy to the 'BINARY_SUBSCR', or it will trigger a type check error.
                # This has already fixed in concrete_proxy.py, but it is hard to add a test for this, add it when we get a idea.
                if key in self.m_dict.keys():
                    x = self.m_dict[key](x)
            return x

    traced_graph = to_fx_graph(SimpleModel(), {'x': torch.rand(4, 10)})
    for node in traced_graph.graph.nodes:
        assert node.target is not next and node.target is not iter
