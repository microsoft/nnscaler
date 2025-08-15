#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from nnscaler.graph.tracer import concrete_trace, wrap_utils
from nnscaler.graph.tracer.metadata import DICT_KEYS_TYPE, DICT_VALUES_TYPE, DICT_ITEMS_TYPE
import torch


def wrap_as_dict(x):
    return {'x': x}

def dict_keys_as_input(x_keys):
    return x_keys

def dict_values_as_input(x_values):
    return x_values

def dict_items_as_input(x_items):
    return x_items


class TestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.randn(5, 10))

    def forward(self, x):
        x_dict = wrap_as_dict(x)
        x_keys = [_ for _ in dict_keys_as_input(x_dict.keys())]
        x_values = [_ for _ in dict_values_as_input(x_dict.values())]
        x_items = [_ for _ in dict_items_as_input(x_dict.items())]

        x = self.param + x_dict[x_keys[0]] + x_values[0] + x_items[0][1]
        return x


def test_dict_iter_metadata():
    graph = concrete_trace(TestModule(),
                           {'x': torch.randn(5, 10)},
                            autowrap_leaf_function={
                                wrap_as_dict: wrap_utils.LeafWrapInfo([], True, None),
                                dict_keys_as_input: wrap_utils.LeafWrapInfo([], True, None),
                                dict_values_as_input: wrap_utils.LeafWrapInfo([], True, None),
                                dict_items_as_input: wrap_utils.LeafWrapInfo([], True, None)
                            },
                            strategy='cpu')
    nodes = list(graph.graph.nodes)
    dict_keys_as_input_node = nodes[2]
    assert isinstance(dict_keys_as_input_node.meta['tensor_meta'], DICT_KEYS_TYPE)
    dict_valus_as_input_node = nodes[6]
    assert isinstance(dict_valus_as_input_node.meta['tensor_meta'], DICT_VALUES_TYPE)
    dict_items_as_input_node = nodes[10]
    assert isinstance(dict_items_as_input_node.meta['tensor_meta'], DICT_ITEMS_TYPE)
