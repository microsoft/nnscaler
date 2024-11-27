#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""
pytest unit_tests/compiler/test_compile.py
"""
import torch
from functools import partial
import more_itertools as mitr

import pytest

import nnscaler
from nnscaler.ir.tensor import IRSubTensor
from nnscaler.utils import load_model
from nnscaler.compiler import compile
from nnscaler.runtime.utils import microbatches
from nnscaler.graph import IRGraph
from nnscaler.graph.segment import IRSegment
from nnscaler.ir.operator import IRFwOperation, IRDataOperation
from nnscaler.flags import CompileFlag
from ..launch_torchrun import torchrun
from ..utils import init_parameter, assert_parity, replace_all_device_with


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


def get_dummy_data(batch_size: int = 512):
    torch.random.manual_seed(0)
    return torch.randn(
        [128, 512], dtype=torch.float32,
        device=torch.cuda.current_device()).repeat([batch_size // 128, 1])


def baseline():

    model = MLP()
    init_parameter(model)
    model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    losses = []
    for _ in range(3):
        x = get_dummy_data()
        loss = model(x)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss = loss.item()
        while abs(loss) > 10.0:
            loss /= 10.0  # scale for comparison
        losses.append(loss)

    return losses


# ================================== cube functionality ========================================

def pipe_policy(graph: IRGraph, resource, ngpus_per_unit: int):

    ngpus = min(ngpus_per_unit, resource.ngpus)
    fnodes = graph.select(ntype=IRFwOperation)

    stages = mitr.divide(ngpus, fnodes)
    stages = [list(s) for s in stages]
    lead_nodes = [s[0] for s in stages]
    graph.staging(lead_nodes)

    for dl in graph.select(ntype=IRDataOperation):
        graph.assign(dl, 0)

    stages = graph.select(ntype=IRSegment, flatten=False)
    stages = [s for s in stages if s.isfw()]
    for idx, stage in enumerate(stages):
        graph.assign(stage, idx)
    return graph


def tp_policy(graph: IRGraph, resource, ngpus_per_unit: int):

    ngpus = min(ngpus_per_unit, resource.ngpus)

    def tensor_parallelism(node, idx, dim, num):
        sub_nodes = graph.partition(
            node, node.algorithms('dim'), idx=idx, dim=dim, num=num)
        for idx, sub_node in enumerate(sub_nodes):
            graph.assign(sub_node, idx)
        return sub_nodes

    # loss partition
    for loss in graph.select(name='sum'):
        tensor_parallelism(loss, idx=0, dim=0, num=ngpus)

    l1, l2, l3, l4 = graph.select(name='linear')

    # l1 tensor parallelism
    tensor_parallelism(l1, idx=1, dim=0, num=ngpus)
    # l2 data parallelism
    tensor_parallelism(l2, idx=0, dim=0, num=ngpus)
    # l3 tensor parallelism
    tensor_parallelism(l3, idx=1, dim=1, num=ngpus)
    # l4 replicate

    for node in graph.select(ntype=(IRFwOperation, IRDataOperation)):
        if len(node.device) == 0:
            sub_nodes = graph.replicate(node, times=ngpus)
            for idx, sub_node in enumerate(sub_nodes):
                graph.assign(sub_node, idx)
    return graph


def cube_run(ngpus_per_unit: int, policy):

    nnscaler.init()
    CompileFlag.disable_code_line_info = True  # speedup parse

    model = MLP()
    init_parameter(model)

    ngpus_per_unit = min(ngpus_per_unit, torch.distributed.get_world_size())
    nreplicas = torch.distributed.get_world_size() // ngpus_per_unit
    batch_size = 512 // nreplicas
    print('>> set batch size to', batch_size)
    x = get_dummy_data(batch_size=batch_size)

    dl = microbatches([x,])

    policy = partial(policy, ngpus_per_unit=ngpus_per_unit)

    @compile(model, dl, PAS=policy, scale=True)
    def train_iter(model, dataloader):
        x = next(iter(dataloader))
        loss = model(x)
        loss.backward()
        return loss

    model = load_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    losses = []
    for _ in range(3):
        x = get_dummy_data(batch_size=batch_size)
        dl = microbatches([x,])
        loss = train_iter(model, dl)
        loss = loss * nreplicas
        optimizer.step()
        optimizer.zero_grad()
        loss = loss.item()
        while abs(loss) > 10.0:
            loss /= 10.0  # scale for comparison
        losses.append(loss)

    return losses

# single-gpu test
test_single = partial(torchrun, 1, assert_parity,
    baseline,
    partial(cube_run, 1, tp_policy)
)

# scale test
test_scale2 = partial(torchrun, 2, assert_parity,
    baseline,
    partial(cube_run, 1, tp_policy)
)

# tensor parallelism test
test_tp2 = partial(torchrun, 2, assert_parity,
    baseline,
    partial(cube_run, 2, tp_policy)
)

# tensor parallelism + scale test
test_tp2scale2 = partial(torchrun, 4, assert_parity,
    baseline,
    partial(cube_run, 2, tp_policy)
)

# pipeline parallelism test
test_pipe2 = partial(torchrun, 2, assert_parity,
    baseline,
    partial(cube_run, 2, pipe_policy)
)

# pipeline parallelism + scale test
test_pipe2scale2 = partial(torchrun, 4, assert_parity,
    baseline,
    partial(cube_run, 2, pipe_policy)
)


class TupleReturnModule2(torch.nn.Module):
    def __init__(self, return_type=0):
        super().__init__()
        self.return_type = return_type
        self.linear = torch.nn.Linear(3, 5)

    def forward(self, x):
        if self.return_type == 0:
            return self.linear(x),
        else:
            return [[self.linear(x)]]


def tuple_return_run(return_type):
    from nnscaler.policies import pas_dp
    from nnscaler import ComputeConfig
    from contextlib import nullcontext

    model = TupleReturnModule2(return_type)
    data = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    dl = microbatches([data,])

    def policy(graph, *args, **kwargs):
        return pas_dp(graph, ComputeConfig(1, 1))

    context = nullcontext() if return_type != 0 else pytest.raises(RuntimeError, match='Single tuple outputs.*')
    with context:
        @compile(model, dl, PAS=policy, scale=False)
        def train_iter(model, dataloader):
            x = next(iter(dataloader))
            loss = model(x)
            assert len(loss) == 1 and len(loss[0]) == 1 and isinstance(loss[0][0], IRSubTensor)
            return loss



test_tuple_return0 = partial(torchrun, 1, tuple_return_run, 0)
test_tuple_return1 = partial(torchrun, 1, tuple_return_run, 1)
