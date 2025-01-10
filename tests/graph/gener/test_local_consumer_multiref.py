#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import torch
import torch.nn as nn
import tempfile
import pytest

import nnscaler.graph.function.function as F
from nnscaler.parallel import ComputeConfig, parallelize
from tests.parallel_module.test_gencode import _gencode_contains, print_gencode


class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.fc0 = torch.nn.Linear(2, 2, bias=False)
        self.fc1 = torch.nn.Linear(2, 2, bias=False)
        self.fc2 = torch.nn.Linear(2, 2, bias=False)

    def forward(self, x):
        x = self.fc0(x)
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x = x1 + x2
        loss = torch.sum(x)
        return loss


def pas(graph, compute_config):
    fc0, fc1, fc2, add, loss = graph.nodes()[:5]
    sub_nodes = graph.partition(fc0, fc0.algorithm('dim'), idx=0, dim=0, num=2)
    graph.assign(sub_nodes[0], 0)
    graph.assign(sub_nodes[1], 1)

    sub_nodes = graph.partition(fc1, fc1.algorithm('dim'), idx=1, dim=0, num=2)
    graph.assign(sub_nodes[0], 0)
    graph.assign(sub_nodes[1], 1)

    sub_nodes = graph.partition(fc2, fc2.algorithm('dim'), idx=1, dim=0, num=2)
    graph.assign(sub_nodes[0], 0)
    graph.assign(sub_nodes[1], 1)

    sub_nodes = graph.partition(add, add.algorithm('dim'), idx=0, dim=0, num=2)
    graph.assign(sub_nodes[0], 0)
    graph.assign(sub_nodes[1], 1)

    sub_nodes = graph.partition(loss, loss.algorithm('dim'), idx=0, dim=0, num=2)
    graph.assign(sub_nodes[0], 0)
    graph.assign(sub_nodes[1], 1)

    return graph


def test_local_consumer_multiref():
    m = Model()
    m.train()
    torch.manual_seed(0)
    trace_data = torch.randn([2, 2])

    with tempfile.TemporaryDirectory() as tempdir:
        parallelize(
            m,
            {'x': trace_data},
            pas,
            ComputeConfig(2, 2, use_end2end=False, trace_strategy='cpu',),
            reuse='override',
            gen_savedir=tempdir,
            load_module=False,
        )
        # print_gencode(tempdir, Model, 0)
        # The generated code should be like:
        # linear_56 = torch.nn.functional.linear(x_54, self.fc0_weight_33, bias=None)
        # del x_54
        # linear_34 = nnscaler.runtime.adapter.nn.allgather_reducescatter(linear_56, dim=0, ranks=[0, 1])
        # del linear_56
        # linear_125, linear_129 = nnscaler.runtime.function.multiref(linear_34, times=2)
        # del linear_34
        # linear_1_70 = torch.nn.functional.linear(linear_125, self.fc1_weight_68, bias=None)
        # del linear_125
        # linear_2_84 = torch.nn.functional.linear(linear_129, self.fc2_weight_82, bias=None)
        # del linear_129
        # linear_1_96 = nnscaler.runtime.adapter.nn.alltoall_alltoall(linear_1_70, idim=1, odim=0, ranks=[0, 1])
        # del linear_1_70
        # linear_2_98 = nnscaler.runtime.adapter.nn.alltoall_alltoall(linear_2_84, idim=1, odim=0, ranks=[0, 1])
        # del linear_2_84
        # add_100 = torch.add(linear_1_96, linear_2_98, alpha=1)
        for i in range(2):
            # output of fc0 is used two times in each device, so local multiref will be added in each device
            assert len(_gencode_contains(tempdir, Model, i, 'nnscaler.runtime.function.multiref')) == 1
            assert len(_gencode_contains(tempdir, Model, i, 'nnscaler.runtime.adapter.nn.allgather_reducescatter')) == 1
            assert len(_gencode_contains(tempdir, Model, i, 'nnscaler.runtime.adapter.nn.alltoall_alltoall')) == 2
