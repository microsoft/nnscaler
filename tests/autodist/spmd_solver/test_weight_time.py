#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import pytest
import sys

import tempfile
import torch
import math
import os
from pathlib import Path
from nnscaler.cli.trainer_args import TrainerArgs
from nnscaler.graph.parser.converter import to_fx_graph, to_ir_graph
from nnscaler.autodist.model_graph import ModelGraph
from nnscaler.autodist.autodist_config import AutoDistConfig
from nnscaler.autodist.spmd_solver import SPMDSolver
from nnscaler.parallel import ComputeConfig

class Attention(torch.nn.Module):

    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x):
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(
            self.head_dim)
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = torch.nn.functional.dropout(attn_weights,
                                                   p=0.0,
                                                   training=self.training)
        attn_out = torch.matmul(attn_weights, v)

        attn_out = attn_out.transpose(1, 2).contiguous().reshape(
            bsz, seq_len, self.hidden_dim)
        attn_out = self.o_proj(attn_out)
        return attn_out

class AttentionModel(torch.nn.Module):

    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.attention = Attention(hidden_dim, num_heads)

    def forward(self, x):
        return self.attention(x).sum()

@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_follow_MLP():
    from nnscaler.ir.unique import IDGenerator
    IDGenerator().clear()

  
    bsz, seq_len, hidden_dim, num_heads = 1024, 128, 512, 8
    dummy_input = {
        'x': torch.rand(bsz, seq_len, hidden_dim),
    }
    model = AttentionModel(hidden_dim, num_heads)
    model.train()

     
    fx_graph = to_fx_graph(model, dummy_input)
    with tempfile.TemporaryDirectory() as tempdir:
        ir_graph = to_ir_graph(fx_graph,
                               dummy_input,
                               attr_savedir=tempdir,
                               constant_folding=True)
        print(ir_graph.nodes())

    pc_path = Path(os.path.dirname(__file__)) / 'test_attention_follow.yaml'
    profile_dir = Path(os.path.dirname(__file__)) / './test_follow_attention_profile'
    cfg = AutoDistConfig(partition_constraints_path=pc_path, mesh_col=4, memory_granularity=1024)
    model_graph = ModelGraph(ir_graph, cfg)
    spmd_solver = SPMDSolver(
        graph=model_graph,
        mesh_desc=cfg.mesh_desc,
        autodist_config=cfg,
        stage_num=1,
        micro_batch_num=cfg.update_freq,
    )

    is_correct = False
    for p in spmd_solver.partition_info:
        for i in p:
            if(i.weight_update_time != 0):
                is_correct = True
                break
    assert is_correct == True
    