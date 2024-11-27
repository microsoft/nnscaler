#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import pytest

import tempfile
import torch
import os
from pathlib import Path
from nnscaler.graph.parser.converter import to_fx_graph, to_ir_graph
from nnscaler.autodist.model_graph import ModelGraph
from nnscaler.autodist.autodist_config import AutoDistConfig
from nnscaler.autodist.spmd_solver import SPMDSolver


class Attention(torch.nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.q_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        score = torch.matmul(q, k.transpose(-2, -1))
        score = torch.nn.functional.softmax(score, dim=-1)
        out = torch.matmul(score, v)
        out = self.out_proj(out)
        return out


class FFN(torch.nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        return x


class Decoder(torch.nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = Attention(hidden_dim)
        self.ffn = FFN(hidden_dim)

    def forward(self, x):
        x = self.attn(x)
        x = self.ffn(x)
        x = x.sum()
        return x


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_partition_constraint():
    bsz, seq_len, hidden_dim = 2, 128, 768

    dummy_input = {'x': torch.randn(bsz, seq_len, hidden_dim)}
    model = Decoder(hidden_dim)
    model.train()
    fx_graph = to_fx_graph(model, dummy_input)

    with tempfile.TemporaryDirectory() as tempdir:
        ir_graph = to_ir_graph(fx_graph,
                               dummy_input,
                               attr_savedir=tempdir,
                               constant_folding=True)

        pc_path = Path(os.path.dirname(
            os.path.realpath(__file__))) / 'test_pc.yaml'
        profile_dir = Path(os.path.dirname(os.path.realpath(__file__))) / 'test_partition_constraint_profile'
        cfg = AutoDistConfig(partition_constraints_path=pc_path, mesh_col=2, profile_dir=profile_dir)
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
        '''
        q_proj: 1
        k_proj: 1
        v_proj: 1
        transpose: 4
        matmul: 1
        softmax: 3
        matmul: 1
        out_proj: 1
        fc1: 2
        relu: 4
        fc2: 2
        sum: 4
        '''
        assert partition_counts == [1, 1, 1, 4, 1, 3, 1, 1, 2, 4, 2, 4]
