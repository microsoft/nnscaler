#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import pytest

import tempfile
import torch
import torch.nn.functional as F
import os
from pathlib import Path
from nnscaler.graph.parser.converter import to_fx_graph, to_ir_graph
from nnscaler.autodist.model_graph import ModelGraph
from nnscaler.autodist.autodist_config import AutoDistConfig
from nnscaler.autodist.spmd_solver import SPMDSolver


class Model(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            x = x.sum()
        return x


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_autocast():
    bsz, seq_len, hidden_dim = 2, 16, 16

    dummy_input = {'x': torch.randn(bsz, seq_len, hidden_dim)}
    model = Model(hidden_dim)
    model.train()
    fx_graph = to_fx_graph(model, dummy_input)

    with tempfile.TemporaryDirectory() as tempdir:
        ir_graph = to_ir_graph(fx_graph,
                               dummy_input,
                               attr_savedir=tempdir,
                               constant_folding=True)

        cfg = AutoDistConfig(mesh_col=2, re_profile=True, parallel_profile=False)
        model_graph = ModelGraph(ir_graph, cfg)

        spmd_solver = SPMDSolver(
            graph=model_graph,
            mesh_desc=cfg.mesh_desc,
            autodist_config=cfg,
            stage_num=1,
            micro_batch_num=cfg.update_freq,
        )

        partition_counts = [
            spmd_solver.get_op_partition_count(i)
            for i in range(model_graph.op_num)
        ]
        assert partition_counts == [4, 4, 4, 4]
