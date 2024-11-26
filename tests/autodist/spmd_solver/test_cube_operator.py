#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import tempfile
import os
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

import nnscaler
from nnscaler.graph.parser.converter import to_fx_graph, to_ir_graph
from nnscaler.autodist.model_graph import ModelGraph
from nnscaler.autodist.autodist_config import AutoDistConfig
from nnscaler.autodist.spmd_solver import SPMDSolver


@nnscaler.register_op(
    '(1 h) l^ d^, (1 h) l^ d^, (1 h) l^ d^ -> (1 h) l^ d^', 'mock_attention')
def mock_attention(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
    return x + y + z


class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, y, z):
        t = mock_attention(x, y, z)
        return t.sum()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cube_operator():
    bsz, head_num, seq_len, head_dim = 1, 8, 128, 64
    data = torch.randn((bsz * head_num, seq_len, head_dim))

    dummy_input = {'x': data, 'y': data, 'z': data}
    model = Model()
    model.train()
    fx_graph = to_fx_graph(model, dummy_input)

    with tempfile.TemporaryDirectory() as tempdir:
        ir_graph = to_ir_graph(fx_graph,
                               dummy_input,
                               attr_savedir=tempdir,
                               constant_folding=True)
        cfg = AutoDistConfig(mesh_col=2)
        model_graph = ModelGraph(ir_graph, cfg)
        mock_attention_op = model_graph.operator_list[0]
        assert mock_attention_op.pos2dim_id((0, 0)) == 'h'
        assert mock_attention_op.dim_id2pos('h') == (0, 0)


class CVModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.out_channel = 32
        self.kernel_size = 3

    def forward(self, input):
        batch, in_channel, height, width = input.shape

        input = input.view(1, batch * in_channel, height, width)
        weight = torch.randn(batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size)
        out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
        _, _, height, width = out.shape
        out = out.view(batch, self.out_channel, height, width)
        return out


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cube_operator_conv_transpose2d():
    """
    ConvTranspose2D and ConvTranspose1D oC dim can't be split
    """
    batch, in_channel, height, width = 2, 16, 32, 32
    input = torch.randn((batch, in_channel, height, width))

    dummy_input = {'input': input}
    model = CVModel()
    model.train()
    fx_graph = to_fx_graph(model, dummy_input)

    with tempfile.TemporaryDirectory() as tempdir:
        ir_graph = to_ir_graph(fx_graph,
                               dummy_input,
                               attr_savedir=tempdir)
        cfg = AutoDistConfig(mesh_col=2)
        model_graph = ModelGraph(ir_graph, cfg)
        spmd_solver = SPMDSolver(
            graph=model_graph,
            mesh_desc=cfg.mesh_desc,
            autodist_config=cfg,
        )

        partition_counts = [
            spmd_solver.get_op_partition_count(i)
            for i in range(model_graph.op_num)
        ]
        assert partition_counts == [4, 1, 2, 2]
