#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import pytest

import tempfile
import torch
import math
import os
from pathlib import Path
from nnscaler.parallel import _gen_graph
from nnscaler.policies import _tp, _replica
from nnscaler.graph.gener.gen import IRAdapterGener
from nnscaler.execplan import ExecutionPlan
from nnscaler.execplan.planpass.fusion import DiffFusion
from nnscaler.execplan.planpass.grouping import Grouping
from nnscaler.ir.adapter.prim import AllReduceIdentityPrim, AllToAllAllToAllPrim, AllGatherSplitPrim, AllReducePrim
from nnscaler.codegen.emit import FuncEmission
from ..utils import replace_all_device_with


# in this test, we check following assumptions when partition the loss
# - the output of a parallel module should be same with the input module
# - the adapter should be AllReduceIdentityPrim

class ModuleA(torch.nn.Module):
    def __init__(self):
        super(ModuleA, self).__init__()
        self.fc = torch.nn.Linear(10, 10)

    def forward(self, x):
        x = self.fc(x)
        x = x.sum()
        return x


def pas_partition_loss_simple(graph):
    dataloader = graph.nodes()[0]
    linear = graph.nodes()[1]
    loss = graph.nodes()[2]
    _replica(graph, dataloader, [0, 1])
    _tp(graph, linear, [0, 1], 0, 0)
    _tp(graph, loss, [0, 1], 0, 0)
    return graph


class ModuleB(torch.nn.Module):
    def __init__(self):
        super(ModuleB, self).__init__()
        self.fc = torch.nn.Linear(10, 10)

    def forward(self, x):
        x = self.fc(x)
        x = x.sum()
        y = x.data
        return x, y


def pas_partition_loss_hard(graph):
    dataloader = graph.nodes()[0]
    linear = graph.nodes()[1]
    loss = graph.nodes()[2]
    get_attr = graph.nodes()[3]
    graph.multiref(loss.outputs()[0].parent)
    _replica(graph, dataloader, [0, 1])
    _tp(graph, linear, [0, 1], 0, 0)
    _tp(graph, loss, [0, 1], 0, 0)
    # .data is automatically replicated since it is a IRPyFunc
    return graph

class ModuleC(torch.nn.Module):
    def __init__(self):
        super(ModuleC, self).__init__()
        self.fc = torch.nn.Linear(10, 10)

    def forward(self, x):
        x = self.fc(x)
        y = x + x
        return x, y


def pas_parallel_module(graph):
    linear = graph.nodes()[0]
    add = graph.nodes()[1]
    _tp(graph, linear, [0, 1], 0, 0)
    _tp(graph, add, [0, 1], 0, 1)
    return graph


def mini_compile_and_check(model_type, pas, checker, end2end_mode):
    dummy_input = {'x': torch.randn(2, 10)}
    model = model_type()
    model.train()

    with tempfile.TemporaryDirectory() as tempdir:
        init_graph, _ = _gen_graph(model, dummy_input, tempdir, constant_folding=True, end2end_mode=end2end_mode)
        partitioned_graph = pas(init_graph)
        adapter_graph = IRAdapterGener.gen(partitioned_graph, cost_fn=None)
        execplan = ExecutionPlan.from_graph(adapter_graph)
        execplan = DiffFusion.apply(execplan)
        execplan = Grouping.apply(execplan)
        checker(init_graph, partitioned_graph, adapter_graph, execplan)


@replace_all_device_with('cpu')
def test_loss_partition_simple():
    from nnscaler.ir.unique import IDGenerator
    IDGenerator().clear()

    def checker(init_graph, partitioned_graph, adapter_graph, execplan):
        fw_graph = execplan.seq(0)[1]
        bw_graph = execplan.seq(0)[2]
        adapter = fw_graph.nodes()[-1]
        assert len(adapter.prims) == 1
        assert isinstance(adapter.prims[0], AllReduceIdentityPrim)
        assert fw_graph.outputs() == init_graph.outputs()
        emit = FuncEmission()
        input_tensors, output_tensors, output_grads, input_grads = \
            emit.get_backward_callsite_io_tensors(bw_graph)
        assert len(output_tensors) == 1
        assert output_tensors[0] == fw_graph.outputs()[0]

    mini_compile_and_check(ModuleA, pas_partition_loss_simple, checker, True)


@replace_all_device_with('cpu')
def test_loss_partition_hard():
    from nnscaler.ir.unique import IDGenerator
    IDGenerator().clear()

    def checker(init_graph, partitioned_graph, adapter_graph, execplan):
        fw_graph = execplan.seq(0)[1]
        bw_graph = execplan.seq(0)[2]
        adapter1 = fw_graph.nodes()[-4]
        adapter2 = fw_graph.nodes()[-2]
        assert len(adapter1.prims) == 1
        assert isinstance(adapter1.prims[0], AllReduceIdentityPrim)
        assert len(adapter2.prims) == 1
        assert isinstance(adapter2.prims[0], AllReducePrim)
        assert fw_graph.outputs() == init_graph.outputs()
        emit = FuncEmission()
        input_tensors, output_tensors, output_grads, input_grads = \
            emit.get_backward_callsite_io_tensors(bw_graph)
        assert len(output_tensors) == 1
        assert output_tensors[0] == fw_graph.outputs()[0]

    mini_compile_and_check(ModuleB, pas_partition_loss_hard, checker, True)


@replace_all_device_with('cpu')
def test_segment_parallel_module():
    from nnscaler.ir.unique import IDGenerator
    IDGenerator().clear()

    def checker(init_graph, partitioned_graph, adapter_graph, execplan):
        # print(adapter_graph.nodes())
        fw_graph = execplan.seq(0)[0]
        bw_graph = execplan.seq(0)[1]
        # print(fw_graph.nodes())
        # print(bw_graph.nodes())
        adapter0 = fw_graph.nodes()[2]
        adapter1 = fw_graph.nodes()[3]
        adapter2 = fw_graph.nodes()[5]
        assert(len(adapter0.prims) == 1)
        assert(isinstance(adapter0.prims[0], AllToAllAllToAllPrim))
        assert(len(adapter1.prims) == 1)
        assert(isinstance(adapter1.prims[0], AllGatherSplitPrim))
        assert(len(adapter2.prims) == 1)
        assert(isinstance(adapter2.prims[0], AllGatherSplitPrim))
        assert fw_graph.outputs() == init_graph.outputs()

    mini_compile_and_check(ModuleC, pas_parallel_module, checker, False)
