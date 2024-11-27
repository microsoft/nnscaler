#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import tempfile
import importlib
from pathlib import Path

import torch
import pytest

from nnscaler.graph.parser.converter import to_fx_graph, to_ir_graph
from nnscaler.graph.parser import FxModuleParser
from nnscaler.ir.cten import IRObject, IRTensor

from ...utils import replace_all_device_with


@replace_all_device_with('cpu')
def test_to_graph():
    class MyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 5)

        def forward(self, x, **kwargs):
            return self.linear(x)
    dummy_input = {'x': torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])}
    module = MyModule()
    fx_graph = to_fx_graph(module, dummy_input)
    assert fx_graph is not None
    nodes = list(fx_graph.graph.nodes)

    # starts with placeholder, and ends with output
    assert nodes[0].op == 'placeholder'
    assert nodes[0].target == 'x'
    assert nodes[1].op == 'placeholder'
    assert nodes[1].target == '**kwargs'  # should keep the double stars
    assert nodes[-1].op == 'output'
    assert nodes[-1].target == 'output'

    # should have linear.weight, linear.bias, and linear(x)
    assert any(node.op == 'get_attr' and node.target == 'linear.weight' for node in nodes)
    assert any(node.op == 'get_attr' and node.target == 'linear.bias' for node in nodes)
    assert any(node.op == 'call_function' and node.target == torch.nn.functional.linear for node in nodes)

    with tempfile.TemporaryDirectory() as tempdir:
        to_ir_graph(fx_graph, dummy_input, attr_savedir=tempdir, constant_folding=False)
        ir_graph = to_ir_graph(fx_graph, dummy_input, attr_savedir=tempdir, constant_folding=False)
        assert ir_graph is not None
        assert (Path(tempdir) / FxModuleParser.ATTR_MAP_FILE).exists()
        assert (Path(tempdir) / FxModuleParser.ATTR_CONTENT_FILE_0).exists()
        assert ir_graph.name == 'MyModule'
        inputs = ir_graph.inputs()
        assert len(inputs) == 2
        assert inputs[0].name == nodes[0].name
        assert isinstance(inputs[0], IRTensor)
        assert inputs[1].name == nodes[1].name
        assert isinstance(inputs[1], IRObject)

        outputs = ir_graph.outputs()
        assert len(outputs) == 1

        nodes = list(ir_graph.nodes())
        assert any(node.signature == 'torch.nn.functional.linear' for node in nodes)


@replace_all_device_with('cpu')
def test_record_codeline():
    class MyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 5)

        def forward(self, x, *args):
            return self.linear(x)
    dummy_input = {'x': torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])}
    module = MyModule()
    fx_graph = to_fx_graph(module, dummy_input)

    cube_path = str(Path(importlib.util.find_spec('nnscaler').origin).parent) + '/'

    for node in fx_graph.graph.nodes:
        if 'frame_record' in node.meta and cube_path in str(node.meta['frame_record']):
            err_msg = f"Cube root path should not in node comment {node.meta['frame_record']}"
            raise RuntimeError(err_msg)


@replace_all_device_with('cpu')
def test_record_metadata():
    class MyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 5)

        def forward(self, x):
            return self.linear(x)
    dummy_input = {'x': torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])}
    module = MyModule()
    fx_graph = to_fx_graph(module, dummy_input)

    from nnscaler.graph.tracer.concrete_proxy import ConcreteProxy
    from nnscaler.graph.tracer import TensorMetadata

    for node in fx_graph.graph.nodes:
        # this assert is only for this simple model, all node should have TensorMetadata type 'tensor_meta'
        # other complex model nodes may not have 'tensor_meta' or a TensorMetadata type 'tensor_meta'
        assert 'tensor_meta' in node.meta and isinstance(node.meta['tensor_meta'], TensorMetadata)
        tm = node.meta['tensor_meta']
        assert not isinstance(tm.shape, ConcreteProxy)
        assert not isinstance(tm.dtype, ConcreteProxy)
        assert not isinstance(tm.requires_grad, ConcreteProxy)
        assert not isinstance(tm.stride, ConcreteProxy)
        assert not isinstance(tm.memory_format, ConcreteProxy)
