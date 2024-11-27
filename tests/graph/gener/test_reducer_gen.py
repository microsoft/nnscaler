#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import pytest
from pathlib import Path
from nnscaler.graph.gener.gen import IRAdapterGener

from nnscaler.graph import IRGraph
from nnscaler.graph.segment import IRSegment
from nnscaler.graph.parser.converter import convert_model
from nnscaler.ir.operator import IRFwOperation
from nnscaler.ir.tensor import IRFullTensor
from nnscaler.ir.adapter import IRWeightReducer
from nnscaler.parallel import ComputeConfig, _load_parallel_module_class, parallelize
from ...utils import new_empty

import torch
import tempfile
import importlib

from ...utils import replace_all_device_with


def make_param(shape, dtype) -> IRFullTensor:
    param = IRFullTensor(shape=shape, dtype=dtype, requires_grad=True)
    param.as_param()
    return param


class ReducerModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.param1 = torch.nn.Parameter(torch.zeros([128, 128], dtype=torch.float32))
        self.param2 = torch.nn.Parameter(torch.zeros([128, 128], dtype=torch.float32))

    def forward(self, x):
        x = torch.matmul(x, self.param1)
        x = torch.matmul(x, self.param2)
        x = x + self.param1
        x = torch.sum(x)
        return x


def build_graph():
    # build graph
    model = ReducerModule()
    with tempfile.TemporaryDirectory() as tempdir:
        graph = convert_model(
            model,
            {'x': torch.randn([128, 128], dtype=torch.float32)},
            attr_savedir=tempdir,
            constant_folding=True
        )
    graph.backward(graph.output(0))
    return graph


@pytest.mark.skipif(not torch.cuda.is_available(), reason='lack of gpu devices')
def test_cross_segment_weight_reducer():

    graph = build_graph()
    [matmul1, matmul2, add, sum] = graph.select(ntype=IRFwOperation)
    graph.group([matmul1, matmul2])
    graph.group([add, sum])

    for idx, segment in enumerate(graph.select(ntype=IRSegment, flatten=False)):
        if not segment.isfw():
            continue
        for node in segment.nodes():
            graph.assign(node, idx)

    print(graph.extra_repr())

    # build reducer
    graph = IRAdapterGener.gen_weight(graph)
    print(graph.extra_repr())
    reducers = graph.select(ntype=IRWeightReducer)
    assert len(reducers) == 1
    assert len(reducers[0].inputs()) == 1
    assert reducers[0].input(0) == matmul1.input(1)
    assert reducers[0].device == (0, 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='lack of gpu devices')
def test_replicate_shared_param():

    graph = build_graph()
    for node in graph.select(ntype=IRFwOperation):
        sn1, sn2 = graph.replicate(node, 2)
        graph.assign(sn1, 0)
        graph.assign(sn2, 1)

    graph = IRAdapterGener.gen_weight(graph)
    print(graph.extra_repr())

    reducers = graph.select(ntype=IRWeightReducer)
    assert len(reducers) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='lack of gpu devices')
def test_reducer_partially_shared_part():
    graph = build_graph()
    [matmul1, matmul2, add, sum] = graph.select(ntype=IRFwOperation)

    m1, m2 = graph.partition(matmul1, matmul1.algorithms('dim'), idx=0, dim=1, num=2)
    graph.assign(m1, 0)
    graph.assign(m2, 1)

    add1, add2 = graph.partition(add, add.algorithms('dim'), idx=0, dim=1, num=2)
    graph.assign(add1, 0)
    graph.assign(add2, 1)

    for node in [matmul2, sum]:
        sn1, sn2 = graph.replicate(node, 2)
        graph.assign(sn1, 0)
        graph.assign(sn2, 1)

    print(graph.extra_repr())

    with pytest.raises(RuntimeError):
        graph = IRAdapterGener.gen_weight(graph)
    print(graph.extra_repr())


def pas_intra_reducer(graph: IRGraph, config: ComputeConfig):
    dataloader = graph.nodes()[0]
    sn0, sn1 = graph.replicate(dataloader, 2)
    graph.assign(sn0, 0)
    graph.assign(sn1, 1)

    fw_nodes = graph.select(ntype=IRFwOperation)

    for i, node in enumerate(fw_nodes):
        if i == 1:
            sn0, sn1 = graph.partition(node, node.algorithms('dim'), idx=1, dim=0, num=2)
        else:
            sn0, sn1 = graph.replicate(node, 2)
        graph.assign(sn0, 0)
        graph.assign(sn1, 1)
    return graph


@replace_all_device_with('cpu')
def test_intra_scale_unit_reducers():
    compute_config = ComputeConfig(
        plan_ngpus=2,
        runtime_ngpus=4,
        constant_folding=True,
        use_zero=True,
        use_end2end=True,
    )
    model = ReducerModule()
    with tempfile.TemporaryDirectory() as tempdir:
        parallelize(
            model,
            {'x': torch.randn([128, 128], dtype=torch.float32)},
            pas_intra_reducer,
            compute_config,
            gen_savedir=tempdir,
            reuse='match',
            load_module=False,
        )
        for i in range(4):
            module_class = _load_parallel_module_class(ReducerModule, gen_savedir=Path(tempdir), rank=i)
            m = new_empty(module_class)
            assert len(m.reducers) == 2
            reducer0, reducer1 = m.reducers
            assert len(reducer0.params) == 1
            assert reducer0.params[0].shape == torch.Size([128, 128])
            assert len(reducer1.params) == 1
            assert reducer1.params[0].shape == torch.Size([64, 128])
