#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import tempfile
from pathlib import Path
import pytest
from typing import Dict, Tuple, List, Any

import torch
from torch import nn

from nnscaler.parallel import ComputeConfig, parallelize, build_optimizer, \
    merge_state_dicts, load_merged_state_dict, \
    deduped_state_dict, load_deduped_state_dict
from nnscaler.runtime.module import ParallelModule
from nnscaler.graph.graph import IRGraph
from nnscaler.ir.operator import IRFwOperation
from nnscaler.policies import _tp, _replica
from nnscaler.runtime.module import dedup_attrs

from .common import init_distributed, assert_equal
from ..launch_torchrun import launch_torchrun
from ..utils import clear_dir_on_rank0

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(4, 4, bias=False)
        self.fc2 = torch.nn.Linear(4, 4, bias=False)
        self.fc3 = torch.nn.Linear(4, 4, bias=False)
        # register a buffer
        self.register_buffer('buffer', torch.zeros(4))

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.buffer + x
        return x

def pas(graph: IRGraph, config: ComputeConfig):
    fw_nodes = graph.select(ntype=IRFwOperation)
    assert len(fw_nodes) == 4
    devs = list(range(config.plan_ngpus))
    # partition the batch dim, weight is replicated
    _tp(graph, fw_nodes[0], idx=0, dim=0, devs=devs)
    # partition the weight, input is replicated
    _tp(graph, fw_nodes[1], idx=1, dim=0, devs=devs)
    _replica(graph, fw_nodes[2], devs=devs)
    _replica(graph, fw_nodes[3], devs=devs)
    return graph

def _gpu_worker_spmd(cc: ComputeConfig):
    init_distributed()
    with clear_dir_on_rank0(Path(tempfile.gettempdir()) / 'nnscaler_test_dedup_attr') as tempdir:
        module = parallelize(
            Net(),
            {'x': torch.tensor([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])},
            pas,
            cc,
            gen_savedir=tempdir,
            instance_name='attr_dedup'
        )
        print(module.fullmap)
        world_size = torch.distributed.get_world_size()
        attr_area_maps = [None for _ in range(world_size)]
        curr_rank = torch.distributed.get_rank()
        torch.distributed.all_gather_object(attr_area_maps, module.fullmap)
        rank2attr_area_map = {}
        for i, attr_area_map in enumerate(attr_area_maps):
            rank2attr_area_map[i] = attr_area_map
        torch.distributed.barrier()
        dedup_meta_info = dedup_attrs(rank2attr_area_map)
        dedup_area_map = list(dedup_meta_info[curr_rank].items())
        if curr_rank == 0:
            assert len(dedup_area_map) == 4
            assert dedup_area_map[0][1].orig_name == 'fc1.weight'
            assert dedup_area_map[0][1].slicers == (slice(0, 4, None), slice(0, 4, None))
            assert dedup_area_map[1][1].orig_name == 'fc2.weight'
            assert dedup_area_map[1][1].slicers == (slice(0, 2, None), slice(0, 4, None))
            assert dedup_area_map[2][1].orig_name == 'fc3.weight'
            assert dedup_area_map[2][1].slicers == (slice(0, 4, None), slice(0, 4, None))
            assert dedup_area_map[3][1].orig_name == 'buffer'
            assert dedup_area_map[3][1].slicers == (slice(0, 4, None),)
        elif curr_rank == 1:
            assert len(dedup_area_map) == 1
            assert dedup_area_map[0][1].orig_name == 'fc2.weight'
            assert dedup_area_map[0][1].slicers == (slice(2, 4, None), slice(0, 4, None))
        else:
            raise RuntimeError(f'Unexpected rank {curr_rank}')


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason='lack of gpu devices')
def test_attr_dedup():
    cc = ComputeConfig(2, 2, use_zero=False)
    launch_torchrun(2, _gpu_worker_spmd, cc)
