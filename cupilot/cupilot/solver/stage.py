# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Staging policy
"""
from typing import List, Tuple, Dict, Optional
import time
import logging

from .spmd import SpmdSolver
from .block import IRBlock
from ..plan.plan import ParallelSpec, StageSpec


_logger = logging.getLogger(__name__)


Nodes = Tuple[IRBlock]


class StageSolver:

    def __init__(self, spmd_solver: SpmdSolver,
                 max_d: int,
                 max_t: Optional[int] = None,
                 max_p: Optional[int] = None):
        self.spmd_solver = spmd_solver
        self.max_d = max_d
        self.max_t: Optional[int] = max_t
        self.max_p: Optional[int] = max_p

        # search caches: (nodes, ndevs, nstages) -> ParallelSpec
        self._cache: Dict[Tuple[Nodes, int, int], Optional[ParallelSpec]] = dict()

    def solve(self, 
              nodes: Tuple[IRBlock],
              ndevs: int,
              memory_limit_bytes: int = None,
              init_mem_cost: Tuple[float] = None, 
              init_comp_cost: Tuple[float] = None) -> Optional[ParallelSpec]:
        """Solve the placement problem using DP algorithm.

        Args:
            nodes (Tuple[IRBlock]): sub-graph
            ndevs (int): number of devices
            memory_limit_bytes (int): memory limit in bytes
            init_mem_cost (Tuple[float]): initial memory cost for each device
            init_comp_cost (Tuple[float]): initial computation cost for each device
        
        Returns:
            ParallelSpec or None: placement plan
        """
        # clear cache
        self._cache = {}
        self.spmd_solver.clear()

        if not all(isinstance(n, IRBlock) for n in nodes):
            raise ValueError(f"nodes must be IRBlock")
        nodes: Tuple[IRBlock] = tuple(nodes)

        init_mem_cost = (0.,) * ndevs if init_mem_cost is None else tuple(init_mem_cost)
        if len(init_mem_cost) != ndevs:
            raise ValueError(f"init_mem_cost length {len(init_mem_cost)} != ndevs {ndevs}")
        init_comp_cost = (0.,) * ndevs if init_comp_cost is None else tuple(init_comp_cost)
        if len(init_comp_cost) != ndevs:
            raise ValueError(f"init_comp_cost length {len(init_comp_cost)} != ndevs {ndevs}")

        tic = time.time()

        # setup constraints
        self.max_d = min(self.max_d, ndevs)
        self.max_t = ndevs if self.max_t is None else min(ndevs, self.max_t)
        self.max_p = ndevs if self.max_p is None else min(ndevs, self.max_p)
        self.max_p = min(len(nodes), self.max_p)

        _logger.info(f'constructing dp tables of {len(nodes)} blocks...')
        min_cost, best_spec = None, None
        devices = tuple(range(ndevs))
        for nstages in range(1, self.max_p+1):
            spec = self._DP(nodes,
                            devices,
                            nstages,
                            memory_limit_bytes,
                            init_mem_cost,
                            init_comp_cost)
            if spec is None:
                continue
            if min_cost is None or spec.est_latency < min_cost:
                min_cost, best_spec = spec.est_latency, spec
        assert best_spec is not None, f"no solution"
        toc = time.time()
        span = toc - tic
        _logger.info(f'placement searching time: {round(span, 3)} s')
        _logger.info(f'estimated latency per microbatch {round(min_cost, 3)} ms')
        assert all(isinstance(stage, StageSpec) for stage in best_spec.stages)
        return best_spec

    @staticmethod
    def iter_subgraph(nodes: Tuple[IRBlock], nstages: int) -> Tuple[List[IRBlock], List[IRBlock]]:
        """
        Iterate sub-graphs of the nodes

        Args:
            nodes (Tuple[IRBlock]): sub-graph.
            nstages (Tuple[int]): number of stages

        Yields:
            Tuple[Tuple[IRBlock], Tuple[int]]: (nodes1, devs1)
            Tuple[Tuple[IRBlock], Tuple[int]]: (nodes2, devs2)
        """
        assert nstages > 0
        if nstages == 1:
            # remaining 1 stage, take all
            yield nodes, ()
        else:
            assert len(nodes) >= nstages
            for idx in range(len(nodes)):
                nodes1, nodes2 = nodes[:idx+1], nodes[idx+1:]
                # each stage at least has one block
                if len(nodes2) < nstages - 1: break
                yield nodes1, nodes2

    def _DP(self, nodes: Tuple[IRBlock], devices: Tuple[int], s: int,
            mem_limit: int,
            init_mem_cost: Tuple[int],
            init_comp_cost: Tuple[int]) -> Optional[ParallelSpec]:
        """
        DP algorithm to search for balanced pipeline stage divisions by considering
        tensor parallelism and pipeline parallelism.

        cost[D][k][s] = min_{D' \in D} min_{t, d where t*d<=k} max( 
            TPS(D\D',t,d,s), cost[D'][k-d*t][s-1] )

        D: subgraph
        k: number of devices
        t: tensor parallelism size
        d: data parallelism size
        s: number of pipeline stages

        Args:
            nodes (Tuple[IRBlock]): sub-graph
            devices (Tuple[int]): device list (in order)
            s (int): number of pipeline stages
            init_mem_cost: Tuple[int]: initial memory cost for each device
            init_comp_cost: Tuple[int]: initial computation cost for each device

        Returns:
            ParallelSpec or None: the best parallelization plan
        """
        nodes = nodes if isinstance(nodes, tuple) else tuple(nodes)
        k = len(devices)
        key = (nodes, k, s)
        if key in self._cache:
            return self._cache[key]

        # dp tatble boundary
        if len(nodes) == 0:
            best_spec = ParallelSpec(stages=[], est_latency=0.)
            self._cache[key] = best_spec
            return best_spec

        assert not (k == 0 or s == 0), \
            f"illegal configuration: nodes: {len(nodes)} k={k}, s={s}: device number (k) cannot be smaller than pipeline stages (s)"
        assert k >= s, f"expected k >= s but got k={k}, s={s}"

        # True for 1,2,4,8,16,...
        is_of_power2 = lambda n: (n & (n-1) == 0) and n != 0

        # construct dynamic programming table
        best_spec: Optional[ParallelSpec] = None
        min_cost = None

        for sub1, sub2 in StageSolver.iter_subgraph(nodes, s):
            # iterate devices
            for ndevs1 in range(1, k+1):
                # device number can be out of constraint boundary
                if ndevs1 > self.max_t * self.max_d: continue
                # constraints: only search for gpu# of power of 2
                if not is_of_power2(ndevs1): continue
                # constraints: all devices must be used
                if s == 1 and ndevs1 != k: continue
                if s >= 2 and len(sub2) == 0: continue
                # guarantee sub-problem searchable: each stage should
                # have at least 1 distinct device for execution
                ndevs2 = k - ndevs1
                if ndevs2 < s - 1: continue
                # early discard as at least one stage will use more devices
                # that are out of constraint boundary
                if s - 1 > 0:
                    if (ndevs2 + s - 2) // (s - 1) > self.max_d * self.max_t: continue

                devs1, devs2 = devices[:ndevs1], devices[ndevs1:]

                # spmd solver results
                init_mem = max(init_mem_cost[:ndevs1])
                init_lat = max(init_comp_cost[:ndevs1])
                stage_spec = self.spmd_solver.solve(sub1, 
                                                    devs1, 
                                                    s, 
                                                    mem_limit,
                                                    init_mem=init_mem,
                                                    init_comp=init_lat,
                                                    max_dp=self.max_d,
                                                    max_tp=self.max_t)
                if stage_spec is None: continue

                # sub problem (stage 1->s) cost
                sub2_spec = self._DP(sub2, 
                                     devs2, 
                                     s-1,
                                     mem_limit,
                                     init_mem_cost[ndevs1:],
                                     init_comp_cost[ndevs1:])
                if sub2_spec is None: continue

                # update cost
                cost = max(stage_spec.est_latency, sub2_spec.est_latency)
                if min_cost is None or cost < min_cost:
                    stages = [stage_spec] + sub2_spec.stages
                    best_spec = ParallelSpec(stages=stages, est_latency=cost)
                    min_cost = cost

        self._cache[key] = best_spec
        return best_spec
