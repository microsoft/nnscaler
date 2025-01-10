#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""
pytest unit_tests/graph/test_multiref.py
"""

import torch
import logging
from functools import partial

import nnscaler
from nnscaler.compiler import compile
from nnscaler.utils import set_default_logger_level, load_model
from nnscaler.graph import IRGraph
from nnscaler.ir.operator import IRFwOperation
from ..launch_torchrun import torchrun
from ..utils import init_parameter, assert_parity


def _param(shape, dtype=torch.float32):
    return torch.nn.Parameter(torch.empty(shape, dtype=dtype))


class OpModule(torch.nn.Module):
    def __init__(self, shape=[256, 512]):
        super().__init__()
        self.param = _param(shape)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """
        residual on x and self.param
        """
        residual = x
        x = residual * self.param
        y = residual + y
        y = y * self.param
        loss = torch.sum(y)
        return loss


def get_dummy_data(batch_size: int = 256):
    torch.random.manual_seed(0)
    return (
        torch.rand([batch_size, 512], dtype=torch.float32, device=torch.cuda.current_device()),
        torch.rand([batch_size, 512], dtype=torch.float32, device=torch.cuda.current_device()),
    )


def baseline():

    model = OpModule()
    init_parameter(model)
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    losses = []
    for _ in range(3):
        x, y = get_dummy_data()
        loss = model(x, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())

    return losses


def multiref():

    model = OpModule()
    init_parameter(model)
    x, y = get_dummy_data()

    def policy(graph: IRGraph, resource):

        first_mul = graph.select('mul')[0]
        first_add = graph.select('add')[0]

        sub_muls = graph.partition(
            first_mul, first_mul.algorithm('dim'),
            idx=0, dim=0, num=resource.ngpus
        )
        for idx, sub_node in enumerate(sub_muls):
            graph.assign(sub_node, idx)

        sub_adds = graph.partition(
            first_add, first_add.algorithm('dim'),
            idx=0, dim=0, num=resource.ngpus
        )
        for idx, sub_node in enumerate(sub_adds):
            graph.assign(sub_node, idx)

        for node in graph.select(ntype=IRFwOperation):
            if len(node.device) == 0:
                sub_nodes = graph.replicate(node, times=resource.ngpus)
                for idx, sub_node in enumerate(sub_nodes):
                    graph.assign(sub_node, idx)
        return graph

    x, y = get_dummy_data()

    @compile(model, x, y, PAS=policy)
    def train_iter(model, x, y):
        loss = model(x, y)
        loss.backward()
        return loss

    model = load_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    losses = []
    for _ in range(3):
        x, y = get_dummy_data()
        loss = train_iter(model, x, y)
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())

    return losses


def multiref_test():
    nnscaler.init()
    set_default_logger_level(logging.INFO)
    assert_parity(baseline, multiref)


test_multiref_1gpu = partial(torchrun, 1, multiref_test)
test_multiref_2gpu = partial(torchrun, 2, multiref_test)
