#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import pytest

import tempfile
import torch
from nnscaler.graph.parser.converter import to_fx_graph, to_ir_graph
from nnscaler.autodist.model_graph import ModelGraph
from nnscaler.autodist.autodist_config import AutoDistConfig
from nnscaler.autodist.spmd_solver import SPMDSolver


class Model(torch.nn.Module):

    def __init__(self, dict_size, hidden_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(dict_size, hidden_dim)
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.fc2.weight = self.fc1.weight
        self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.fc = torch.nn.Linear(hidden_dim, dict_size, bias=False)
        self.fc.weight = self.embedding.weight

    def forward(self, x):
        x = self.embedding(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc(x)
        return x.sum()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_shared_param():
    bsz, seq_len, hidden_dim, dict_size = 2, 128, 768, 1024

    dummy_input = {'x': torch.randint(0, dict_size, (bsz, seq_len))}
    model = Model(dict_size, hidden_dim)
    model.train()
    fx_graph = to_fx_graph(model, dummy_input)

    with tempfile.TemporaryDirectory() as tempdir:
        ir_graph = to_ir_graph(fx_graph,
                               dummy_input,
                               attr_savedir=tempdir,
                               constant_folding=True)

        cfg = AutoDistConfig(mesh_col=4)
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
        # batch size cannot be partitioned on 4 devices
        # each operator can be replicated and partitioned on the sequence length dim
        # for fc3, the out_feature and in_feature dims can be partitioned
        # for sum, the hidden dim can be partitioned
        assert partition_counts == [2, 2, 2, 4, 2, 3]
