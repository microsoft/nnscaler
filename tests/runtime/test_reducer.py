#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""
pytest unit_tests/runtime/test_reducer.py
"""
import torch
import logging
from functools import partial

import nnscaler
from nnscaler.compiler import compile
from nnscaler.utils import load_model
from nnscaler.graph import IRGraph
from nnscaler.ir.operator import IRFwOperation
from nnscaler.flags import CompileFlag
from nnscaler.runtime.adapter.reducer import Reducer
from ..launch_torchrun import torchrun
from ..utils import init_parameter, assert_parity, mock_reducer_env


class MLP(torch.nn.Module):
    def __init__(self, dim=512, nlayers=4):
        super().__init__()
        self.layers = torch.nn.ModuleList([])
        for _ in range(nlayers):
            self.layers.append(torch.nn.Linear(dim, dim, bias=False))

    def forward(self, data):
        x = data
        for layer in self.layers:
            x = layer(x)
        loss = torch.sum(x)
        return loss


def get_dummy_data(batch_size: int = 256):
    torch.random.manual_seed(0)
    return torch.randn(
        [batch_size, 512], dtype=torch.float32,
        device=torch.cuda.current_device())


def baseline():

    model = MLP()
    init_parameter(model)
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    losses = []
    for _ in range(3):
        x = get_dummy_data()
        loss = model(x)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss = loss.item()
        while abs(loss) > 10.0:
            loss /= 10.0
        losses.append(loss)

    return losses


def reducer(use_zero: bool, async_reducer: bool):

    CompileFlag.use_zero = use_zero
    CompileFlag.async_reducer = async_reducer

    model = MLP()
    init_parameter(model)

    def policy(graph: IRGraph, resource):

        def tensor_parallelism(node, idx, dim, num):
            sub_nodes = graph.partition(
                node, node.algorithm('dim'), idx=idx, dim=dim, num=num)
            for idx, sub_node in enumerate(sub_nodes):
                graph.assign(sub_node, idx)
            return sub_nodes

        l1, l2, l3, l4 = graph.select(name='linear')

        # l1 data parallelism
        tensor_parallelism(l1, idx=0, dim=0, num=resource.ngpus)
        # l2 data parallelism
        tensor_parallelism(l2, idx=0, dim=0, num=resource.ngpus)
        # l3, l4 replicate

        for node in graph.select(ntype=IRFwOperation):
            if len(node.device) == 0:
                sub_nodes = graph.replicate(node, times=resource.ngpus)
                for idx, sub_node in enumerate(sub_nodes):
                    graph.assign(sub_node, idx)
        return graph

    x = get_dummy_data()

    @compile(model, x, PAS=policy)
    def train_iter(model, x):
        loss = model(x)
        loss.backward()
        return loss

    model = load_model()
    optimizer = torch.optim.Adam(model.parameters_for_optimizer(), lr=0.01)

    losses = []
    for _ in range(3):
        x = get_dummy_data()
        loss = train_iter(model, x)
        optimizer.step()
        optimizer.zero_grad()
        ## === neccessary for zero ===
        model.gather_params()
        ## ===========================
        loss = loss.item()
        while abs(loss) > 10.0:
            loss /= 10.0
        losses.append(loss)

    return losses


def reducer_test():
    nnscaler.init()
    CompileFlag.disable_code_line_info = True  # speedup parse
    print('starting zero=True, async=True')
    assert_parity(baseline, partial(reducer, True, True))
    print('starting zero=True, async=False')
    assert_parity(baseline, partial(reducer, True, False))
    print('starting zero=False, async=True')
    assert_parity(baseline, partial(reducer, False, True))
    print('starting zero=False, async=False')
    assert_parity(baseline, partial(reducer, False, False))

test_reducer_2gpu = partial(torchrun, 2, reducer_test)


@mock_reducer_env(0, 2)
def test_reducer_build():
    reducer = Reducer([0, 1], max_bucket_size_bytes=48)  # 24 bytes means 12 float32
    reducer.add_param(torch.nn.Parameter(torch.randn(1, 2)))  # 4 floats # small at first <bucket 0>
    reducer.add_param(torch.nn.Parameter(torch.randn(1, 14))) # 16 floats # bigger than max_bucket_size_bytes <bucket 1>
    reducer.add_param(torch.nn.Parameter(torch.randn(1, 5)))  # 8 floats # small again <bucket 2>
    reducer.add_param(torch.nn.Parameter(torch.randn(1, 5)))  # 8 floats # small again <bucket 3>
    reducer.add_param(torch.nn.Parameter(torch.randn(1, 1)))  # 4 floats small again <bucket 3>
    reducer.add_param(torch.nn.Parameter(torch.randn(1, 1)))  # 4 floats small again <bucket 4>
    reducer.build_buckets()
    assert len(reducer.buckets) == 5
    buckets = list(reversed(reducer.buckets))
    assert buckets[0].numel == 2
    assert buckets[0]._aligned_numel == 4
    assert buckets[1].numel == 14
    assert buckets[1]._aligned_numel == 16
    assert buckets[2].numel == 5
    assert buckets[2]._aligned_numel == 8
    assert buckets[3].numel == 6
    assert buckets[3]._aligned_numel == 12
    assert buckets[4].numel == 1
    assert buckets[4]._aligned_numel == 4