#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import pytest

import tempfile
import torch
import math
import os
from pathlib import Path
from nnscaler.graph.parser.converter import to_fx_graph, to_ir_graph
from nnscaler.autodist.model_graph import ModelGraph
from nnscaler.autodist.autodist_config import AutoDistConfig
from nnscaler.autodist.spmd_solver import SPMDSolver


def rotate_half(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=0):
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Model(torch.nn.Module):

    def __init__(self, head_num, hidden_dim):
        super().__init__()
        self.q_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.head_num = head_num
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // head_num

    def forward(self, x, cos, sin, position_ids):
        bsz, seq_len, hidden_dim = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        q = q.view(bsz, seq_len, self.head_num, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.head_num, self.head_dim).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        out = q + k
        return out.sum()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_follow_rope():
    from nnscaler.ir.unique import IDGenerator
    IDGenerator().clear()
    bsz, seq_len, head_num, hidden_dim = 2, 128, 8, 512
    head_dim = hidden_dim // head_num
    dummy_input = {
        'x': torch.rand(bsz, seq_len, hidden_dim),
        'cos': torch.rand(seq_len, head_dim),
        'sin': torch.rand(seq_len, head_dim),
        'position_ids': torch.arange(seq_len, dtype=torch.long),
    }
    model = Model(head_num, hidden_dim)
    model.train()
    fx_graph = to_fx_graph(model, dummy_input)

    with tempfile.TemporaryDirectory() as tempdir:
        ir_graph = to_ir_graph(fx_graph,
                               dummy_input,
                               attr_savedir=tempdir,
                               constant_folding=True)
        '''
        the computation graph is as follows:
        q_proj      fullslice                    fullslice                          k_proj
          |            |                            |                                |
        view        unsqueeze                    unsqueeze                          view
          |            |  \                         |                                |
        transpose      |    -------------------------------------------------mul----transpose
          |            |                            |                         |
          ------------mul   fullsclie fullslice     |   fullsclie fullslice   |
                       |       \       |            |      \       |          |
                       |        \     neg           |       \     neg         |
                       |         \     |            |        \     |          |
                       |          concat            |         concat          |
                       |             |              |            |            |
                      add-----------mul-------------------------mul----------add
                       |                                                      |
                       |                                                      |
                       ---------------------------add--------------------------
                                                   |
                                                  sum
        currently, the following chain is only composed of unary ops
        there are 2 chains in total:
        1. view -> transpose -> fullslice -> fullslice -> neg -> concat
        2. view -> transpose -> fullslice -> fullslice -> neg -> concat
        3. fullslice -> unsqueeze
        4. fullslice -> unsqueeze
        5. add -> sum
        in future, we may add follow chains for binary ops, like mul, add, etc.
        '''

        profile_dir = Path(os.path.dirname(__file__)) / './test_follow_rope_profile'
        cfg = AutoDistConfig(mesh_col=2, profile_dir=profile_dir)
        model_graph = ModelGraph(ir_graph, cfg)

        spmd_solver = SPMDSolver(
            graph=model_graph,
            mesh_desc=cfg.mesh_desc,
            autodist_config=cfg,
            stage_num=1,
            micro_batch_num=cfg.update_freq,
        )

        assert spmd_solver.follow_ids == [
            0, 1, 2, 2, 4, 4, 6, 6, 8, 8, 10, 3, 3, 12, 11, 15, 16, 17, 5, 5, 19, 18, 22, 23, 24, 24
        ]
        partition_counts = [
            spmd_solver.get_op_partition_count(i)
            for i in range(model_graph.op_num)
        ]
        chains = [[2, 3, 11, 12, 13, 14], [4, 5, 18, 19, 20, 21], [2, 3], [4, 5], [24, 25]]
        for chain in chains:
            assert all(partition_counts[i] == partition_counts[chain[0]] for i in chain)


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
def test_follow_attention():
    from nnscaler.ir.unique import IDGenerator
    IDGenerator().clear()
    bsz, seq_len, hidden_dim, num_heads = 2, 128, 512, 8
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
        '''
        the computation graph is as follows:
        2linear    3linear    4linear
           |         |         |
          5view      7view     9view
           |         |         |
        6transpose 8transpose 10transpose
            \        |         |
             |    11transpose  |
             \      /          |
              12matmul         |
                |              |
               13div           |
                |              |
15training   14softmax         |
    |           |              |
      \ ----  16dropout        |
                 \            /
                  \          /
                    17matmul
                       |
                  18transpose
                       |
                  19contiguous
                       |
                    20reshape
                       |
                    21linear
                       |
                      22sum

        the follow chain is as follows:
        1. view -> transpose
        2. view -> transpose -> transpose
        3. view -> transpose
        4. div -> softmax -> dropout
        5. transpose -> contiguous -> reshape
        '''

        pc_path = Path(os.path.dirname(__file__)) / 'test_attention_follow.yaml'
        profile_dir = Path(os.path.dirname(__file__)) / './test_follow_attention_profile'
        cfg = AutoDistConfig(partition_constraints_path=pc_path, mesh_col=2, profile_dir=profile_dir, memory_granularity=1024)
        model_graph = ModelGraph(ir_graph, cfg)

        spmd_solver = SPMDSolver(
            graph=model_graph,
            mesh_desc=cfg.mesh_desc,
            autodist_config=cfg,
            stage_num=1,
            micro_batch_num=cfg.update_freq,
        )

        assert spmd_solver.follow_ids == [
            0, 1, 2, 3, 3, 5, 5, 7, 7, 6, 10, 11, 11, 13, 12, 15, 16, 16, 17, 19, 20
        ]
        partition_counts = [
            spmd_solver.get_op_partition_count(i)
            for i in range(model_graph.op_num)
        ]
        assert partition_counts == [
            2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 1, 2, 2, 4, 4, 4, 2, 4
        ]
        # under the current partition constraints, the solver should generate
        # a Megatron-LM plan
        expected_out = [
            # partition out feature for q_proj
            (2, (((1, 0), 2),)),
            # partition out feature for k_proj
            (3, (((1, 0), 2),)),
            # partition out feature for v_proj
            (4, (((1, 0), 2),)),
            # partition hidden dim for q's view
            (5, (((0, 2), 2),)),
            # partition the head dim for q's transpose
            (6, (((0, 2), 2),)),
            # partition the hidden dim for k's view
            (7, (((0, 2), 2),)),
            # partition the head dim for k's transpose
            (8, (((0, 2), 2),)),
            # partition the hidden dim for v's view
            (9, (((0, 2), 2),)),
            # partition the head dim for v's transpose
            (10, (((0, 2), 2),)),
            # partition the head dim for k's 2nd transpose
            (11, (((0, 1), 2),)),
            # partition the head dim for matmul(q, k)
            (12, (((0, 1), 2),)),
            # partition the head dim div
            (13, (((0, 1), 2),)),
            # partition the head dim for softmax
            (14, (((0, 1), 2),)),
            # replicate `training`
            (15, (((-1, -1), 2),)),
            # partition the head dim for dropout
            (16, (((0, 1), 2),)),
            # partition the head dim for matmul(attn_weights, v)
            (17, (((0, 1), 2),)),
            # partition the head dim for attn_out.transpose
            (18, (((0, 1), 2),)),
            # partition the head dim for contiguous
            (19, (((0, 2), 2),)),
            # partition the head dim for reshape
            (20, (((0, 2), 2),)),
            # partition the input feature for o_proj
            (21, (((0, 2), 2),)),
            # replicate the sum
            (22, (((-1, -1), 2),))
        ]

        def helper(search_out):
            return search_out[0][0].to_json()['desc']['partition_descs']

        dp_spmd_outs = spmd_solver.do_dp([(0, model_graph.op_num - 1)], 1)
        ilp_spmd_outs = spmd_solver.do_ilp([(0, model_graph.op_num - 1)], 1)
        assert helper(dp_spmd_outs) == expected_out
        assert helper(ilp_spmd_outs) == expected_out

@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_solver_data_parallel():
    from nnscaler.ir.unique import IDGenerator
    IDGenerator().clear()
    bsz, seq_len, hidden_dim, num_heads = 2, 2048, 512, 8
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

        profile_dir = Path(os.path.dirname(__file__)) / './test_solver_data_parallel'
        cfg = AutoDistConfig(mesh_col=2, profile_dir=profile_dir, memory_granularity=1024)
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
        print(partition_counts)
        assert partition_counts == [
            5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 6, 4, 4, 1, 4, 6, 4, 4, 4, 5, 4
        ]
        # should generate a pure data parallel plan, e.g., partition the batch dim
        expected_out = [
            (2, (((0, 0), 2),)),
            (3, (((0, 0), 2),)),
            (4, (((0, 0), 2),)),
            (5, (((0, 0), 2),)),
            (6, (((0, 0), 2),)),
            (7, (((0, 0), 2),)),
            (8, (((0, 0), 2),)),
            (9, (((0, 0), 2),)),
            (10, (((0, 0), 2),)),
            (11, (((0, 0), 2),)),
            (12, (((0, 0), 2),)),
            (13, (((0, 0), 2),)),
            (14, (((0, 0), 2),)),
            (15, (((-1, -1), 2),)),
            (16, (((0, 0), 2),)),
            (17, (((0, 0), 2),)),
            (18, (((0, 0), 2),)),
            (19, (((0, 0), 2),)),
            (20, (((0, 0), 2),)),
            (21, (((0, 0), 2),)),
            (22, (((0, 0), 2),))
        ]

        def helper(search_out):
            return search_out[0][0].to_json()['desc']['partition_descs']

        dp_spmd_outs = spmd_solver.do_dp([(0, model_graph.op_num - 1)], 1)
        ilp_spmd_outs = spmd_solver.do_ilp([(0, model_graph.op_num - 1)], 1)
        assert helper(dp_spmd_outs) == expected_out
        assert helper(ilp_spmd_outs) == expected_out
