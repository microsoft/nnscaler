#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from datetime import datetime
import math
import random
import shutil
from typing import Any, Dict, List, Optional, Tuple
import more_itertools as mitr
import contextlib

import torch
from torch import nn
import numpy as np

from nnscaler.graph.schedule.predefined import PredefinedSched
from nnscaler.parallel import ComputeConfig
from nnscaler.graph.function.anchor import IRGraphAnchor
from nnscaler.graph.function.dimops import IRDimops
from nnscaler.graph.graph import IRGraph
from nnscaler.graph.segment import IRSegment
from nnscaler.ir.operator import IRDataOperation, IRFwOperation
from nnscaler.policies import _tp, _replica

from ..utils import init_random


def create_mesh(ngpus: int, group_num: Tuple[int]) -> Tuple[Tuple[Tuple[int]]]:
    """
    Create hybrid (nested) groups given the each group number.

    The product of group_num should be same with total devices.

    e.g., 6 device to 2 x 3 mesh will results [dim][group_id] = tuple[int]:
        (
            ( (0,1,2), (3,4,5) ),
            ( (0,3), (2,5), (3,6) ),
        )
    """
    group_num = np.array(group_num)
    cnt = np.prod(group_num)
    assert cnt == ngpus, 'total device not match'
    grid = np.arange(cnt).reshape(tuple(group_num))
    dims = list(range(len(group_num)))
    outputs = []
    for dim, num in enumerate(group_num):
        remain = ngpus // num
        order = tuple(dims[:dim] + dims[dim+1:] + [dim])
        grid_dim = np.transpose(grid, order).reshape((remain,num))
        grid_dim = grid_dim.tolist()
        outputs.append(tuple(tuple(ranks) for ranks in grid_dim))
    assert len(outputs) == len(group_num)
    return tuple(outputs)


def PASMegatron(graph: IRGraph, config: ComputeConfig):
    num_stages = config.pas_config['pipeline_nstages']
    nmicros = config.pas_config['pipeline_nmicros']
    scheduler = config.pas_config.get('pipeline_scheduler', '1f1b')
    tp_size = config.plan_ngpus // num_stages
    _, tp_mesh = create_mesh(config.plan_ngpus, (num_stages, tp_size))

    # group to sub-graphs
    linears = graph.select(name='linear')
    stage_start_nodes = linears[::len(linears) // num_stages][:num_stages]
    graph.staging(stage_start_nodes)

    segments = graph.select(ntype=IRSegment, flatten=False)
    fsegs = [seg for seg in segments if seg.isfw()]

    for sid, segment in enumerate(fsegs):
        # get tensor parallel group
        tp_group = tp_mesh[sid]
        for idx, node in enumerate(segment.nodes()):
            if node.name == 'linear':
                _tp(graph, node, idx=1, dim=idx%2, devs=tp_group)
            else:
                _replica(graph, node, devs=tp_group)

    for dl in graph.select(ntype=IRDataOperation):
        _replica(graph, dl, devs=list(range(config.plan_ngpus)))
    config.apply_pipeline_scheduler(graph, num_stages, nmicros, scheduler)
    return graph


class CubeLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=False)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.fc.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.fc.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        x = self.fc(x)
        if self.bias is not None:
            x = x + self.bias
        return x


def init_distributed():
    torch.distributed.init_process_group(backend='nccl')
    rank = torch.distributed.get_rank()
    torch.cuda.set_device(rank)
    torch.set_default_device(f'cuda:{rank}')


def assert_equal(a: Any, b: Any):
    assert type(a) == type(b)
    if isinstance(a, torch.Tensor):
        assert torch.equal(a.cpu(), b.cpu())
    elif isinstance(a, dict):
        assert len(a) == len(b)
        for k in a.keys():
            assert_equal(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for i in range(len(a)):
            assert_equal(a[i], b[i])
    else:
        assert a == b


def assert_close(a: Any, b: Any, atol=1e-6, rtol=1e-6):
    assert type(a) == type(b)
    if isinstance(a, torch.Tensor):
        assert torch.allclose(a.cpu(), b.cpu(), atol=atol, rtol=rtol)
    elif isinstance(a, dict):
        assert len(a) == len(b)
        for k in a.keys():
            assert_close(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for i in range(len(a)):
            assert_close(a[i], b[i])
    else:
        raise ValueError(f'unsupported type {type(a)}')