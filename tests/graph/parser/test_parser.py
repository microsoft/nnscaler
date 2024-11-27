#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import tempfile
import pytest
import torch

import nnscaler
from nnscaler.ir.cten import IRObject, IRTensor
from nnscaler.graph.parser.converter import to_fx_graph, to_ir_graph

from ...utils import replace_all_device_with


@replace_all_device_with('cpu')
def test_multi_consume():

    class MyModule(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.param1 = torch.nn.Parameter(torch.empty(4, 4))
            self.param2 = torch.nn.Parameter(torch.empty(4, 4))

        def forward(self, x):
            shortcut = x
            x = torch.matmul(x, self.param1)
            x = x + self.param2
            x = x + shortcut
            x = x + self.param1
            return torch.sum(x)

    dummy_input = {'x': torch.randn(4, 4)}
    module = MyModule()
    fx_graph = to_fx_graph(module, dummy_input)

    print(fx_graph.graph)

    with tempfile.TemporaryDirectory() as tempdir:
        ir_graph = to_ir_graph(fx_graph, dummy_input, attr_savedir=tempdir, constant_folding=False)
    assert ir_graph is not None
    assert len(ir_graph.attributes()) == 2  # param1 and param2
    assert len(ir_graph.full_tensors()) == 8


@replace_all_device_with('cpu')
def test_parser_nested_inputs():

    class MyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.param1 = torch.nn.Parameter(torch.empty(4, 4))
            self.param2 = torch.nn.Parameter(torch.empty(4, 4))

        def forward(self, x: dict):
            shortcut = x['data']
            x = torch.matmul(x['data'], self.param1)
            x = x + self.param2
            x = x + shortcut
            x = x + self.param1
            return {'loss': torch.sum(x)}

    dummy_input = {'x': {'data': torch.randn(4, 4)}}
    module = MyModule()
    fx_graph = to_fx_graph(module, dummy_input)

    print(fx_graph.graph)
    with tempfile.TemporaryDirectory() as tempdir:
        ir_graph = to_ir_graph(fx_graph, dummy_input, attr_savedir=tempdir, constant_folding=False)
        print(ir_graph.extra_repr())

    assert len(ir_graph.inputs()) == 1
    assert isinstance(ir_graph.input(0), IRObject)
    assert isinstance(ir_graph.input(0).value, dict)
    assert isinstance(ir_graph.input(0).value['data'], IRTensor)
    assert len(ir_graph.outputs()) == 1
    assert isinstance(ir_graph.output(0), dict)
    assert isinstance(ir_graph.output(0)['loss'], IRTensor)


@replace_all_device_with('cpu')
def test_max():

    class MyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.max(x, dim=1, keepdim=True)[0]

    dummy_input = {'x': torch.randn(4, 1024)}
    module = MyModule()
    fx_graph = to_fx_graph(module, dummy_input)

    print(fx_graph.graph)
    with tempfile.TemporaryDirectory() as tempdir:
        ir_graph = to_ir_graph(fx_graph, dummy_input, attr_savedir=tempdir, constant_folding=False)
        print(ir_graph.extra_repr())

    assert isinstance(ir_graph.output(0), IRTensor)
    assert ir_graph.output(0).shape == (4, 1)


@replace_all_device_with('cpu')
def test_min():

    class MyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.min(x, dim=1, keepdim=True)[0]

    dummy_input = {'x': torch.randn(10, 256)}
    module = MyModule()
    fx_graph = to_fx_graph(module, dummy_input)

    print(fx_graph.graph)
    with tempfile.TemporaryDirectory() as tempdir:
        ir_graph = to_ir_graph(fx_graph, dummy_input, attr_savedir=tempdir, constant_folding=False)
        print(ir_graph.extra_repr())

    assert isinstance(ir_graph.output(0), IRTensor)
    assert ir_graph.output(0).shape == (10, 1)


@nnscaler.register_op('m n -> m n, m n, ?')
def func_multi_outputs(x):
    return x, x, 3


@nnscaler.register_op('m n -> ?')
def func_output_list(x, factor=1):
    x = x * factor
    return [x, x]


@nnscaler.register_op('m n -> m n, m n')
def func_output_list2(x, factor=1):
    x = x * factor
    return [x, x]


@replace_all_device_with('cpu')
@pytest.mark.parametrize('output_list', [True, False])
def test_num_outputs(tmp_path, output_list):
    class MyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            out = func_multi_outputs(x)
            y, _, scalar = out
            (sz, _) = y.shape
            sz = sz + scalar
            if output_list:
                return func_output_list(y, factor=sz)
            else:
                return func_output_list2(y, factor=sz)

    dummy_input = {'x': torch.randn(4, 4)}
    module = MyModule()
    fx_graph = to_fx_graph(module, dummy_input)

    print(fx_graph.graph)
    ir_graph = to_ir_graph(fx_graph, dummy_input, attr_savedir=tmp_path, constant_folding=False)
    print(ir_graph.extra_repr())

    assert len(ir_graph.nodes()) == 5
    assert len(ir_graph.nodes()[0].outputs()) == 3
    assert len(ir_graph.outputs()) == 1
    assert isinstance(ir_graph.output(0), list)
    if output_list:
        assert len(ir_graph.nodes()[-1].outputs()) == 1
    else:
        assert len(ir_graph.nodes()[-1].outputs()) == 2
