#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import nnscaler.autodist.dp_solver as dp_solver

# use a naive ffn to test the dynamic programming solver
# the ffn has 3 layers
# - linear layer
# - relu layer
# - linear layer
# each operator has 2 partition options

def test_dp_solver():
    solver = dp_solver.DPSolver(True, 80 * 1024, 1)
    solver.add_interval(0, 2)

    solver.add_node(0, 0, [0], [], 2, False, False, False)
    solver.add_partition(0, 0, 1, 1, 1, 1, 1, 1, 0, [[]])
    solver.add_partition(0, 1, 2, 2, 2, 2, 2, 2, 1, [[]])

    solver.add_node(1, 1, [1], [0], 2, False, False, False)
    solver.add_partition(1, 0, 0.5, 1, 1, 1, 1, 1, 0, [[0.1, 1]])
    solver.add_partition(1, 1, 1, 2, 2, 2, 2, 2, 1, [[1, 0]])

    solver.add_node(2, 2, [2], [1], 2, False, False, False)
    solver.add_partition(2, 0, 1, 1, 1, 1, 1, 1, 0, [[0.2, 1]])
    solver.add_partition(2, 1, 2, 2, 2, 2, 2, 2, 1, [[1, 0]])

    solver.solve()

    ans = solver.get_results(0, 2)

    best = ans[0]

    # optimal all time 1 + 0.5 + 0.1 + 1 + 0.2 = 2.8
    assert best.all_time == 2.8
    # the optimal plan is each operator's first partition
    assert best.path == [(0, 0), (1, 0), (2, 0)]

def test_dp_solver_mem():
    solver = dp_solver.DPSolver(True, 100, 1)
    solver.add_interval(0, 4)

    solver.add_node(0, 0, [0], [], 1, True, True, False)
    solver.add_partition(0, 0, 0.1, 10, 1, 1, 1, 1, 0, [[]])

    solver.add_node(1, 1, [1], [0], 1, True, False, True)
    solver.add_partition(1, 0, 0.2, 10, 2, 2, 2, 2, 0, [[0]])

    solver.add_node(2, 2, [2], [1], 1, True, True, False)
    solver.add_partition(2, 0, 0.3, 10, 3, 3, 3, 3, 0, [[0]])

    solver.add_node(3, 3, [3], [2], 1, True, True, False)
    solver.add_partition(3, 0, 0.4, 10, 4, 4, 4, 4, 0, [[0]])

    solver.add_node(4, 4, [4], [3], 1, True, False, True)
    solver.add_partition(4, 0, 0.5, 10, 5, 5, 5, 5, 0, [[0]])

    # the total memory cost should be
    # param: 10 - 1 + 10 - 2 + 10 - 3 + 10 - 4 + 10 - 5 = 35
    # buffer: 5 + 4 = 9
    # activation: 1 + 3 + 4 = 8
    # opt_transient_mem: 1 + 2 + 3 + 4 + 5 = 15
    # recompute: max(1 + 2, 3 + 4 + 5) = 12
    # in all: 35 + 9 + max(8, 15) + 12 = 71

    solver.solve()

    ans = solver.get_results(0, 4)

    best = ans[0]
    assert best.all_time == 1.5
    assert best.path == [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]
    assert best.memory == 71

def test_dp_solver_build_in_edges():
    # mock following code
    # dropout_rate = self.attention_dropout if self.training else 0.0
    # attn_output = nnscaler_flash_attention_forward(
    #     query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate, causal=causal
    # )
    # 3 nodes will be generated, there are no following chains and tensors between them
    # 1. self_getattr
    # 2. ifexpr
    # 3. nnscaler_flash_attention_forward
    solver = dp_solver.DPSolver(True, 100, 1)
    solver.add_interval(0, 2)

    solver.add_node(0, 0, [0], [], 1, False, False, False)
    solver.add_partition(0, 0, 0, 0, 0, 0, 0, 0, 0, [[]])

    solver.add_node(1, 1, [1], [], 1, False, False, False)
    solver.add_partition(1, 0, 0, 0, 0, 0, 0, 0, 0, [[]])

    solver.add_node(2, 2, [2], [], 1, False, False, False)
    solver.add_partition(2, 0, 1, 0, 0, 0, 0, 0, 0, [[]])

    solver.solve()

    ans = solver.get_results(0, 2)

    best = ans[0]
    assert best.path == [(0, 0), (1, 0), (2, 0)]

def test_dp_solver_mem_bound():
    solver = dp_solver.DPSolver(True, 10, 1)
    solver.add_interval(0, 2)

    solver.add_node(0, 0, [0], [], 1, False, False, False)
    solver.add_partition(0, 0, 0, 8, 0, 0, 0, 0, 0, [[]])

    solver.add_node(1, 1, [1], [], 1, False, False, False)
    solver.add_partition(1, 0, 0, 5, 0, 0, 0, 0, 0, [[]])

    solver.add_node(2, 2, [2], [], 1, False, False, False)
    solver.add_partition(2, 0, 1, 11, 0, 0, 0, 0, 0, [[]])

    solver.solve()

    ans = solver.get_results(0, 2)
    assert len(ans) == 0
