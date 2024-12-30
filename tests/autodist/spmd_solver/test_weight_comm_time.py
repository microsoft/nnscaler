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
from tests.autodist.spmd_solver.test_follow import AttentionModel

@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_weight_comm_time():
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

    pc_path = Path(os.path.dirname(__file__)) / 'test_weight_comm_time.yaml'
    profile_dir = Path(os.path.dirname(__file__)) / './test_follow_attention_profile'
    cfg = AutoDistConfig(partition_constraints_path = pc_path, mesh_col = 4, memory_granularity = 1024, profile_dir = profile_dir)
    model_graph = ModelGraph(ir_graph, cfg)

    spmd_solver = SPMDSolver(
        graph=model_graph,
        mesh_desc=cfg.mesh_desc,
        autodist_config=cfg,
        stage_num=1,
        micro_batch_num=cfg.update_freq,
    )

    cost_info = spmd_solver.partition_info

    for i in range(spmd_solver.graph.op_num):

        cur_op = spmd_solver.get_operator(i)
        weight_comm_times = [desc.weight_update_time for desc in cost_info[i]]
        if 'linear' in cur_op.op_name:
            assert any(comm > 0 for comm in weight_comm_times)
        else:
            assert all(comm == 0 for comm in weight_comm_times)

