#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import pytest
import os
from pathlib import Path

import tempfile
import torch
from nnscaler.graph.parser.converter import to_fx_graph, to_ir_graph
from nnscaler.autodist.model_graph import ModelGraph
from nnscaler.autodist.autodist_config import AutoDistConfig
from nnscaler.autodist.spmd_solver import SPMDSolver
import nnscaler
from tests.utils import raises_with_cause


# this is a wrong annotation
@nnscaler.register_op('l^ hq dim^, l^ hkv dim^, l^ hkv dim^ -> l^ hq dim^')
def mock_attention(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
    return x + y + z


class Model(torch.nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.q_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x):
        bsz, seq_len, hidden_dim = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = q.view(bsz * seq_len, 2, hidden_dim // 2)
        k = k.view(bsz * seq_len, 2, hidden_dim // 2)
        v = v.view(bsz * seq_len, 2, hidden_dim // 2)
        x = mock_attention(q, k, v)
        x = x.reshape(bsz, seq_len, 2, hidden_dim // 2)
        return x.sum()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_trigger_follow_error():
    bsz, seq_len, hidden_dim = 2, 16, 16

    dummy_input = {'x': torch.randn(bsz, seq_len, hidden_dim)}
    model = Model(hidden_dim)
    model.train()
    fx_graph = to_fx_graph(model, dummy_input)
    profile_dir = Path(os.path.dirname(__file__)) / './test_trigger_follow_error'

    with tempfile.TemporaryDirectory() as tempdir:
        ir_graph = to_ir_graph(fx_graph,
                               dummy_input,
                               attr_savedir=tempdir,
                               constant_folding=True)

        cfg = AutoDistConfig(mesh_col=2, parallel_profile=False, profile_dir=profile_dir)
        model_graph = ModelGraph(ir_graph, cfg)

        with raises_with_cause(AssertionError, match='find multiple p_fathers'):
            spmd_solver = SPMDSolver(
                graph=model_graph,
                mesh_desc=cfg.mesh_desc,
                autodist_config=cfg,
                stage_num=1,
                micro_batch_num=cfg.update_freq,
            )
