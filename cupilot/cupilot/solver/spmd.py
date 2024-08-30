# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
SPMD solver similar to intra_op of Alpa
"""
from typing import List, Dict, Optional, Tuple
import logging
import more_itertools
import multiprocessing
import numpy as np
import warnings
import time

from cube.ir.operator import IRFwOperation

from .block import IRBlock
from ..estimator.cost_model import CostModel
from ..plan.plan import StageSpec

# ILP solver
import pulp
from pulp import LpVariable, LpProblem, LpMinimize, LpStatus, lpSum, lpDot, LpStatus


_logger = logging.getLogger(__name__)


Nodes = Tuple[IRBlock]


class SpmdSolver:

    def __init__(self, cost_model: CostModel, recompute: bool,
                 memory_saving: bool = True):
        """SPMD Solver for searching the best spmd parallelism configuration.

        Note the operators that are assigned to devices will not be searched.

        Args:
            cost_model (CostModel): cost model for communication and computation
            recompute (bool): whether to apply recompute
            memory_saving (bool): True to remove replication of nodes if there are other
                partitioning choices for each node.
        """
        self.cost_model: CostModel = cost_model
        self.recompute: bool = recompute
        self.memory_saving: bool = memory_saving
        # device idx -> param memory limit
        self.device_mem_limit_bytes: Dict[int, int] = {}

        # (blocks, dp_size, tp_size) -> StageSpec
        self._cache: Dict[Tuple[Nodes, int, int], StageSpec] = {}

    def add_device_mem_limit(self, device: int, limit_gb: float):
        self.device_mem_limit_bytes[device] = int(limit_gb * 1024 * 1024 * 1024)

    def clear(self):
        self._cache = {}

    def solve(self, blocks: List[IRBlock],
              devices: Tuple[int],
              inflights: int,
              memory_limit: int,
              init_mem: int = 0,
              init_comp: float = 0.,
              min_dp: int = 1, max_dp: int = 32,
              min_tp: int = 1, max_tp: int = 32,) -> Optional[StageSpec]:

        device_constraints = set()
        for blk in blocks:
            if blk.devices is not None:
                device_constraints.add(blk.devices)
        # at least two blocks have conflicts on device constraints,
        # therefore cannot be in one stage
        if len(device_constraints) > 1:
            return None
        # The stage device doesn't satisfy constaints of blocks
        if len(device_constraints) == 1:
            if devices not in device_constraints:
                return None

        # setup device memory constraint
        for devid in devices:
            if devid in self.device_mem_limit_bytes:
                memory_limit = min(memory_limit, self.device_mem_limit_bytes[devid])
        
        min_dp = max(min_dp, max(blk.min_dp for blk in blocks))
        max_dp = min(max_dp, min(blk.max_dp for blk in blocks))
        min_tp = max(min_tp, max(blk.min_tp for blk in blocks))
        max_tp = min(max_tp, min(blk.max_tp for blk in blocks))

        if min_dp > max_dp or min_tp > max_tp:
            return None

        # True for 1, 2, 4, 8, 16,...
        is_of_power2 = lambda n: (n & (n-1) == 0) and n != 0

        best_stage_spec = None
        min_latency = None

        no_solution = False
        for dp in range(min_dp, min(len(devices) + 1, max_dp + 1)):
            # constraints: only search for gpu# of power of 2
            if not is_of_power2(dp): continue
            # get tp size
            if len(devices) % dp != 0: continue
            tp = len(devices) // dp

            if not (min_tp <= tp <= max_tp): continue
            # constraints: only search for gpu# of power of 2
            if not is_of_power2(tp): continue

            # this means a larger tp size is already infeasible to
            # satisfy memory limit
            if no_solution:
                self._cache[self.get_key(blocks, dp, tp)] = None
                continue

            spec = self._solve(blocks, dp, tp, inflights, memory_limit,
                               init_mem, init_comp)
            # no solution -- the later smaller tp will also be infeasible
            if spec is None:
                no_solution = True
                continue

            if min_latency is None or spec.est_latency < min_latency:
                best_stage_spec = spec
                min_latency = spec.est_latency
        
        return best_stage_spec

    def get_key(self, blocks: List[IRBlock], dp_size: int, tp_size: int):
        """Get the key of the solved problem"""
        return (tuple(blocks), dp_size, tp_size)

    def _solve(self,
               blocks: List[IRBlock],
               dp_size: int,
               tp_size: int,
               inflights: int,
               memory_limit: int,
               init_mem: int,
               init_comp: float) -> Optional[StageSpec]:
        """
        Search for the best spmd parallelism configuration given parallelism size.
        The search is only suitable for training.

        Args:
            blocks (List[IRBlock])
            dp_size (int): data parallel size
            tp_size (int): tensor parallel size
            inflights (int): maximal inflight micro-batches
            memory_limit (int): memory upper bound
            
        Returns:
            spec (StageSpec | None): operator transformation configuration
                None indicates no solution given by memory limit.
        """
        key = self.get_key(blocks, dp_size, tp_size)
        if key in self._cache:
            return self._cache[key]

        tic = time.time()

        fnodes: List[IRFwOperation] = list(more_itertools.flatten(blk.nodes for blk in blocks))
        fnodes = self.cost_model.get_search_nodes(fnodes)

        # create variables (nodes)
        s, d, c = {}, {}, {}  # partition index, computation cost, communication cost
        e, r = [], []  # inter-node resharding cost

        num_nodes = 0
        for fnode in fnodes:
            cid = fnode.cid
            algos = self.cost_model.partition_algos[fnode.cid]
            npartitions = len(algos)
            s[cid] = LpVariable.matrix(f's[{num_nodes}]', (range(npartitions),), cat='Binary')
            d[cid] = self.cost_model.get_comp_cost(fnode, tp_size).flatten() / dp_size
            c[cid] = self.cost_model.get_comm_cost(fnode, tp_size).flatten() / dp_size
            # setup initial value
            for pidx, strategy in enumerate(algos):
                if strategy is None: continue
                idx, dim = strategy
                identifier = fnode.anno.input(idx)[dim].identifiers[0]
                # we constrain a node that can only be evenly partitioned
                if fnode.anno.getlen(identifier) % (tp_size * dp_size) != 0:
                    s[cid][pidx].setInitialValue(False)
                    s[cid][pidx].fixValue()
                    npartitions -= 1
            # remove replicate choice if we have other choices to
            # partition nodes to save memory
            if self.memory_saving and npartitions > 1 and algos[0] is None:
                s[cid][0].setInitialValue(False)
                s[cid][0].fixValue()
                npartitions -= 1
            if npartitions <= 0:
                raise RuntimeError(
                    f"Infeasible problem: cannot find a partition choice for node: {fnode.name}[{cid}] "
                    f"in problem tp={tp_size}, dp={dp_size}")
            num_nodes += 1

        edges = self.cost_model.get_edges(fnodes)
        num_edges = 0
        for src, dsts in edges.items():
            for dst in dsts:
                nsrc = len(self.cost_model.partition_algos[src.cid])
                ndst = len(self.cost_model.partition_algos[dst.cid])
                e.append(LpVariable.matrix(f"e[{src.cid}, {dst.cid}]",
                                           (range(nsrc * ndst),),
                                           cat='Binary'))
                r.append(self.cost_model.get_pair_reshard_cost(src, dst, tp_size).flatten() / dp_size)
                num_edges += 1

        # initial value: --skip

        # objective
        prob = LpProblem('spmd', LpMinimize)
        # computation cost
        obj = 0
        for fnode in fnodes:
            cid = fnode.cid
            obj += lpDot(s[cid], c[cid]) + lpDot(s[cid], d[cid])
        # communication cost
        for i in range(num_edges):
            obj += lpDot(e[i], r[i])

        prob += obj

        # constraints

        # a) only one partition can be selected
        for fnode in fnodes:
            prob += lpSum(s[fnode.cid]) == 1
        for i in range(num_edges):
            prob += lpSum(e[i]) == 1

        # e_src_dst[i][j] = 1 => s_src[i] == 1 and s_dst[j] == 1
        eidx = 0
        for src, dsts in edges.items():
            for dst in dsts:
                for row in range(len(s[src.cid])):
                    C = len(s[dst.cid])
                    prob += lpSum(
                        e[eidx][row * C + col] for col in range(0, C)) <= s[src.cid][row]
                for col in range(len(s[dst.cid])):
                    R = len(s[src.cid])
                    C = len(s[dst.cid])
                    prob += lpSum(
                        e[eidx][row * C + col] for row in range(0, R)) <= s[dst.cid][col]
                eidx += 1

        # b) memory constraint --skip

        assert "PULP_CBC_CMD" in pulp.listSolvers(onlyAvailable=True), (
            "Please install ILP solvers by 'sudo apt install coinor-cbc' or 'pip install pulp'")

        time_limit = 600
        solver = pulp.PULP_CBC_CMD(
            mip=True, msg=0, 
            timeLimit=time_limit, 
            threads=multiprocessing.cpu_count())
        prob.solve(solver)

        status = prob.status
        if status == pulp.LpStatusInfeasible:
            raise RuntimeError(
                f"infeasible problem: {len(blocks)} blocks, tp={tp_size}, dp={dp_size}. "
                f"Please report the bug")
        elif status == pulp.LpStatusUndefined:
            raise RuntimeError(
                f"Cannot find solution of the problem within time limit: "
                f"{len(blocks)} blocks, tp={tp_size}, dp={dp_size}"
            )

        objective = pulp.value(prob.objective)
        objective = float(objective) if objective is not None else -1.0
        # print(f"ILP Status: {LpStatus[status]}\tObjective: {objective}")
        # print(f"#nodes: {num_nodes},  #edges: {num_edges}")
        # print(f'ILP search time: {time.time() - tic:.2f} seconds')

        # reshard_cost = 0
        # for i in range(num_edges):
        #     reshard_cost += lpDot(e[i], r[i])
        # reshard_cost = pulp.value(reshard_cost)
        # print(f'debug info: reshard cost: {reshard_cost}')

        def get_non_zero_index(binary_vector):
            """Get the index of non-zero item in a vector."""
            ct = 0
            ret = None
            for i, elem in enumerate(binary_vector):
                if pulp.value(elem):
                    ret = i
                    ct += 1

            assert ct == 1
            return ret

        tp_spec: Dict[int, int] = {}
        for fnode in fnodes:
            index = get_non_zero_index(s[fnode.cid])
            tp_spec[fnode.cid] = index

        # check results
        e_val = np.full((num_edges,), -1, dtype=np.int32)
        eidx = 0
        for (src, dsts) in edges.items():
            for dst in dsts:
                e_val[eidx] = get_non_zero_index(e[eidx])
                src_spec_index = e_val[eidx] // len(s[dst.cid])
                dst_spec_index = e_val[eidx] % len(s[dst.cid])
                assert src_spec_index == tp_spec[src.cid]
                assert dst_spec_index == tp_spec[dst.cid]
                eidx += 1

        if objective > 1e13:
            warnings.warn("Detect unexpected behaviors in the auto-sharding pass.")

        # get tensor parallelism spec
        stage_tp_spec = {}
        names = {}
        for fnode in fnodes:
            split = None if tp_size == 1 else \
                self.cost_model.partition_algos[fnode.cid][tp_spec[fnode.cid]]
            stage_tp_spec[fnode.cid] = split
            names[fnode.cid] = fnode.name

        # estimate memory
        splits = []
        for fnode in fnodes:
            split = stage_tp_spec[fnode.cid]
            split = None if split is None else (split[0], split[1], tp_size)
            splits.append(split)
        # assume adam optimizer
        span, mem_cost = self.cost_model.estimator(fnodes, splits, inflights)
        mem_cost += init_mem
        if mem_cost > memory_limit:
            mem_gb = round(mem_cost/1024/1024/1024, 2)
            _logger.debug(f'results of {len(tp_spec)} nodes: tp={tp_size}, dp={dp_size}: no solution (memory: {mem_gb} GB)')
            stage = None
        else:
            objective = objective + init_comp
            stage = StageSpec(
                est_latency=objective / 3 * 4 if self.recompute else objective,
                est_memory=mem_cost,
                tp_size=tp_size,
                dp_size=dp_size,
                tp_spec=stage_tp_spec,
                names=names,
            )
            _logger.debug(f'results of {len(stage_tp_spec)} nodes: tp={tp_size}, dp={dp_size} '
                          f'lat={round(objective, 2)} ms, mem={round(mem_cost/1024/1024/1024, 2)} GB')
        self._cache[key] = stage
        return stage
