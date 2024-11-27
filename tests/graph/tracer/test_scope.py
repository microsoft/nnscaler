#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import torch

from nnscaler.graph.parser.converter import to_fx_graph
from ...utils import replace_all_device_with


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(10, 5)
        self.m = SimpleModel2()
    
    def forward(self, x):
        # node add_2
        return self.fc(x) + self.m.forward(x)

class SimpleModel2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m2 = SimpleModel3()
        self.ffn = torch.nn.Linear(10, 5)

    def forward(self, x):
        # node add_1
        return self.m2.forward(x) + self.ffn(x)
    
class SimpleModel3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 5)
        self.fc2 = torch.nn.Linear(10, 5)
    
    def forward(self, x):
        # node add
        return self.fc1(x) + self.fc2(x)


@replace_all_device_with('cpu')
def test_scope():
    model = SimpleModel()
    dummy_input = {'x': torch.rand(10)}
    traced_graph = to_fx_graph(model, dummy_input)
    traced_graph(**dummy_input)

    name_map = {
        'add': 'm.m2',
        'add_1': 'm',
        # 'add_2': None  # add_2 is at root module, so it will have an empty stack
    }

    viewed_nodes = set()
    for node in traced_graph.graph.nodes:
        if node.name in name_map:
            viewed_nodes.add(node.name)
            module_path = list(node.meta['nn_module_stack'])[-1]
            assert module_path == name_map[node.name], f'{module_path} == {name_map[node.name]}'

    assert viewed_nodes == set(name_map.keys())
