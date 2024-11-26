#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import pytest

import tempfile
import torch
from nnscaler.graph.parser.converter import to_fx_graph, to_ir_graph
from nnscaler.ir.operator import IRFwOperation
from nnscaler.autodist.model_graph import calc_flops


class Model(torch.nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, t_1d, t_2d, t_3d):
        x = self.fc1(t_3d)
        y = self.fc2(t_3d)
        z1 = torch.bmm(x, y)
        z2 = torch.matmul(t_1d, t_1d)
        z3 = torch.matmul(t_2d, t_2d)
        z4 = torch.matmul(t_1d, t_2d)
        z5 = torch.matmul(t_2d, t_1d)
        z6 = torch.matmul(t_2d, t_3d)
        z7 = torch.matmul(t_3d, t_2d)
        return x.sum() + y.sum() + z1.sum() + z2.sum() + z3.sum() + z4.sum(
        ) + z5.sum() + z6.sum() + z7.sum()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_calc_flops():
    batch_size, hidden_dim = 2, 1024
    dummy_input = {
        't_1d': torch.randn(hidden_dim),
        't_2d': torch.randn(hidden_dim, hidden_dim),
        't_3d': torch.randn(batch_size, hidden_dim, hidden_dim)
    }
    model = Model(hidden_dim)
    fx_graph = to_fx_graph(model, dummy_input)

    with tempfile.TemporaryDirectory() as tempdir:
        ir_graph = to_ir_graph(fx_graph,
                               dummy_input,
                               attr_savedir=tempdir,
                               constant_folding=False)
    nodes = ir_graph.select(ntype=IRFwOperation)
    assert calc_flops(
        nodes[0]) == 2 * batch_size * hidden_dim * hidden_dim * hidden_dim
    assert calc_flops(
        nodes[1]) == 2 * batch_size * hidden_dim * hidden_dim * hidden_dim
    assert calc_flops(
        nodes[2]) == 2 * batch_size * hidden_dim * hidden_dim * hidden_dim
    assert calc_flops(nodes[3]) == 2 * hidden_dim
    assert calc_flops(nodes[4]) == 2 * hidden_dim * hidden_dim * hidden_dim
    assert calc_flops(nodes[5]) == 2 * hidden_dim * hidden_dim
    assert calc_flops(nodes[6]) == 2 * hidden_dim * hidden_dim
    assert calc_flops(
        nodes[7]) == 2 * batch_size * hidden_dim * hidden_dim * hidden_dim
    assert calc_flops(
        nodes[8]) == 2 * batch_size * hidden_dim * hidden_dim * hidden_dim
