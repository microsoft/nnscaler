#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import operator

import torch
import nnscaler
from nnscaler.graph.parser.converter import to_fx_graph
from tests.utils import replace_all_device_with


class ContextConstantFoldingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ma = torch.nn.Parameter(torch.rand(3, 3))
        self.mb = torch.nn.Parameter(torch.rand(3, 3))
        self.mc = torch.nn.Parameter(torch.rand(3, 3))
        self.md = torch.nn.Parameter(torch.rand(3, 3))
        self.me = torch.nn.Parameter(torch.rand(3, 3))
        self.mf = torch.nn.Parameter(torch.rand(3, 3))
        self.mg = torch.nn.Parameter(torch.rand(3, 3))

    def forward(self, a: torch.Tensor):
        b = self.ma * a
        with nnscaler.constant_folding():

            c = self.mb * b
            with nnscaler.no_constant_folding():
                d = self.mc * c
                with nnscaler.constant_folding():
                    e = self.md * d

                f = self.me * e

            g = self.mf * f

        h = self.mg * g
        return h


@replace_all_device_with('cpu')
def test_context_folding():
    model = ContextConstantFoldingModule()
    dummy_input = {'a': torch.rand(3, 3)}
    traced_graph = to_fx_graph(model, dummy_input)
    nodes = list(traced_graph.graph.nodes)

    assert nodes[0].name == 'a'
    assert nodes[0].op == 'placeholder'
    assert nodes[0].meta['op_context']['constant_folding'] is None

    assert nodes[-1].name == 'output'
    assert nodes[-1].op == 'output'
    assert nodes[-1].meta['op_context']['constant_folding'] is None

    # b = self.ma * a
    # h = self.mg * g
    assert nodes[1].name == 'ma'
    assert nodes[1].op == 'get_attr'
    assert nodes[1].meta['op_context']['constant_folding'] is None
    assert nodes[2].name == 'mul'
    assert nodes[2].op == 'call_function'
    assert nodes[2].target == operator.mul
    assert nodes[2].meta['op_context']['constant_folding'] is None

    assert nodes[13].name == 'mg'
    assert nodes[13].op == 'get_attr'
    assert nodes[13].meta['op_context']['constant_folding'] is None
    assert nodes[14].name.startswith('mul_')
    assert nodes[14].op == 'call_function'
    assert nodes[14].target == operator.mul
    assert nodes[14].meta['op_context']['constant_folding'] is None

    # with nnscaler.constant_folding():
    #     c = self.mb * b
    #     g = self.mf * f
    assert nodes[3].name == 'mb'
    assert nodes[3].op == 'get_attr'
    assert nodes[3].meta['op_context']['constant_folding'] is True
    assert nodes[4].name.startswith('mul')
    assert nodes[4].op == 'call_function'
    assert nodes[4].target == operator.mul
    assert nodes[4].meta['op_context']['constant_folding'] is True

    assert nodes[11].name == 'mf'
    assert nodes[11].op == 'get_attr'
    assert nodes[11].meta['op_context']['constant_folding'] is True
    assert nodes[12].name.startswith('mul')
    assert nodes[12].op == 'call_function'
    assert nodes[12].target == operator.mul
    assert nodes[12].meta['op_context']['constant_folding'] is True

    # d = self.mc * c
    # f = self.me * e
    assert nodes[5].name == 'mc'
    assert nodes[5].op == 'get_attr'
    assert nodes[5].meta['op_context']['constant_folding'] is False
    assert nodes[6].name.startswith('mul')
    assert nodes[6].op == 'call_function'
    assert nodes[6].target == operator.mul
    assert nodes[6].meta['op_context']['constant_folding'] is False

    assert nodes[9].name == 'me'
    assert nodes[9].op == 'get_attr'
    assert nodes[9].meta['op_context']['constant_folding'] is False
    assert nodes[10].name.startswith('mul')
    assert nodes[10].op == 'call_function'
    assert nodes[10].target == operator.mul
    assert nodes[10].meta['op_context']['constant_folding'] is False

    # e = self.md * d
    assert nodes[7].name == 'md'
    assert nodes[7].op == 'get_attr'
    assert nodes[7].meta['op_context']['constant_folding'] is True
    assert nodes[8].name.startswith('mul')
    assert nodes[8].op == 'call_function'
    assert nodes[8].target == operator.mul
    assert nodes[8].meta['op_context']['constant_folding'] is True


class CFModule1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mb = torch.nn.Parameter(torch.rand(3, 3))

    @nnscaler.constant_folding()
    def forward(self, a: torch.Tensor):
        return self.mb * a


class CFModule2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mc = torch.nn.Parameter(torch.rand(3, 3))

    @nnscaler.constant_folding(False)
    def forward(self, a: torch.Tensor):
        return self.mc * a


class CFModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ma = torch.nn.Parameter(torch.rand(3, 3))
        self.mb = CFModule1()
        self.mc = CFModule2()
        self.md = torch.nn.Parameter(torch.rand(3, 3))
        self.me = torch.nn.Parameter(torch.rand(3, 3))

    def forward(self, a: torch.Tensor):
        x = self.ma * a
        x = self.mb(x)
        with nnscaler.constant_folding():
            x = self.mc(x)
            x = self.md * x
        x = self.me * x
        return x


@replace_all_device_with('cpu')
def test_context_folding_module():
    model = CFModule()
    dummy_input = {'a': torch.rand(3, 3)}
    traced_graph = to_fx_graph(model, dummy_input)
    nodes = list(traced_graph.graph.nodes)

    assert nodes[0].name == 'a'
    assert nodes[0].op == 'placeholder'
    assert nodes[0].meta['op_context']['constant_folding'] is None

    assert nodes[-1].name == 'output'
    assert nodes[-1].op == 'output'
    assert nodes[-1].meta['op_context']['constant_folding'] is None

    # x = self.ma * a
    # x = self.me * x
    assert nodes[1].name == 'ma'
    assert nodes[1].op == 'get_attr'
    assert nodes[1].meta['op_context']['constant_folding'] is None
    assert nodes[2].name == 'mul'
    assert nodes[2].op == 'call_function'
    assert nodes[2].target == operator.mul
    assert nodes[2].meta['op_context']['constant_folding'] is None

    assert nodes[9].name == 'me'
    assert nodes[9].op == 'get_attr'
    assert nodes[9].meta['op_context']['constant_folding'] is None
    assert nodes[10].name.startswith('mul_')
    assert nodes[10].op == 'call_function'
    assert nodes[10].target == operator.mul
    assert nodes[10].meta['op_context']['constant_folding'] is None

    # x = self.mb(x)
    assert nodes[3].name == 'mb_mb'
    assert nodes[3].op == 'get_attr'
    assert nodes[3].meta['op_context']['constant_folding'] is True
    assert nodes[4].name.startswith('mul')
    assert nodes[4].op == 'call_function'
    assert nodes[4].target == operator.mul
    assert nodes[4].meta['op_context']['constant_folding'] is True

    # x = self.mc(x)
    assert nodes[5].name == 'mc_mc'
    assert nodes[5].op == 'get_attr'
    assert nodes[5].meta['op_context']['constant_folding'] is False
    assert nodes[6].name.startswith('mul')
    assert nodes[6].op == 'call_function'
    assert nodes[6].target == operator.mul
    assert nodes[6].meta['op_context']['constant_folding'] is False

    # x = self.md * x
    assert nodes[7].name == 'md'
    assert nodes[7].op == 'get_attr'
    assert nodes[7].meta['op_context']['constant_folding'] is True
    assert nodes[8].name.startswith('mul')
    assert nodes[8].op == 'call_function'
    assert nodes[8].target == operator.mul
    assert nodes[8].meta['op_context']['constant_folding'] is True


class OpReorderModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ma = torch.nn.Parameter(torch.rand(3, 3))

    def forward(self, a: torch.Tensor):
        shape = a.shape
        with nnscaler.constant_folding():
            b = self.ma.abs() + shape[0]
        return b


@replace_all_device_with('cpu')
def test_context_folding_reordered_and_dce():
    """
    Test getattr will not be reordered (old lazy-style implementation will reorder it)
    """
    model = OpReorderModule()
    dummy_input = {'a': torch.rand(3, 3)}
    traced_graph = to_fx_graph(model, dummy_input)
    nodes = list(traced_graph.graph.nodes)
    assert nodes[0].name == 'a'
    assert nodes[0].op == 'placeholder'
    assert nodes[0].meta['op_context']['constant_folding'] is None
    assert nodes[-1].name == 'output'
    assert nodes[-1].op == 'output'
    assert nodes[-1].meta['op_context']['constant_folding'] is None

    # a.shape
    assert nodes[1].name.startswith('getattr')
    assert nodes[1].op == 'call_function'
    assert nodes[1].target == getattr
    nodes[1].args[1] == 'shape'

    #b = self.ma.abs() + shape[0]
    assert nodes[2].name == 'ma'
    assert nodes[2].op == 'get_attr'
    assert nodes[2].meta['op_context']['constant_folding'] is True

    # the ma.abs getattr will be eliminated by dce.
    assert nodes[3].name.startswith('abs')
    assert nodes[3].op == 'call_method'
    assert nodes[3].target == 'abs'
    assert nodes[3].meta['op_context']['constant_folding'] is True

    assert nodes[4].name == 'getitem'
    assert nodes[4].op == 'call_function'
    assert nodes[4].target == operator.getitem
    assert nodes[4].meta['op_context']['constant_folding'] is True

    assert nodes[5].name == 'add'
    assert nodes[5].op == 'call_function'
    assert nodes[5].target == operator.add
    assert nodes[5].meta['op_context']['constant_folding'] is True
