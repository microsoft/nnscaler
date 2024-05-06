# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
SPMD solver
"""
from typing import List, Optional, Tuple
import logging
from dataclasses import dataclass

from cube.ir.cten import IRTensor

from .block import IRBlock
from ..estimator import Estimator

_logger = logging.getLogger(__name__)


Nodes = Tuple[IRBlock]

@dataclass
class StageSpec:
    blocks: Tuple[IRBlock]
    dp_size: int
    tp_size: int
    latency: float
    memory: float


class SpmdSolver:

    def __init__(self, estimator: Estimator,
                 recompute: bool,
                 train: bool,
                 param_limit: Optional[int] = None):
        """
        Args:
            estimator (Estimator)
            recompute (bool): whether to apply recompute
            train (bool): whether the graph is for training
            param_limit (int | None): parameter limit
        """
        self.recompute: bool = recompute
        self.estimator: Estimator = estimator
        self.train: bool = train
        self.param_limit = param_limit

    def clear(self):
        self._cache = {}

    def solve(self, blocks: List[IRBlock],
              devices: Tuple[int],
              inflights: int,
              memory_limit: int,
              min_dp: int = 1, max_dp: int = 32,
              min_tp: int = 1, max_tp: int = 32,) -> Optional[StageSpec]:
        
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

        # grid search
        for dp in range(min_dp, min(len(devices), max_dp) + 1):
            # constraints: only search for gpu# of power of 2
            if not is_of_power2(dp): continue
            # get tp size
            if len(devices) % dp != 0: continue
            tp = len(devices) // dp

            if not (min_tp <= tp <= max_tp): continue
            # constraints: only search for gpu# of power of 2
            if not is_of_power2(tp): continue

            stage = self._solve(blocks, dp, tp, inflights, memory_limit)
            if stage is None:  # no solution
                continue
            if min_latency is None or stage.latency < min_latency:
                best_stage_spec = stage
                min_latency = stage.latency
        
        return best_stage_spec

    def _solve(self,
               blocks: List[IRBlock],
               dp_size: int,
               tp_size: int,
               inflights: int,
               memory_limit: int) -> Optional[StageSpec]:
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
            Optional[StageSpec]: best spec
        """
        key = (tuple(blocks), dp_size, tp_size)
        if key in self._cache:
            return self._cache[key]

        tp_mem_efficiency = 1.0 + 0.10 * (tp_size-1)
        tp_com_efficienty = 1.0 + 0.10 * (tp_size-1)

        total_act_memory = 0
        total_latency = 0
        for block in blocks:
            latency, act_memory = self.estimator(block.nodes)
            # activation memory
            act_memory = act_memory / (tp_size * dp_size) * tp_mem_efficiency
            # recompute granularity: per layer
            if self.train:
                total_act_memory = max(total_act_memory, act_memory) if self.recompute \
                    else total_act_memory + act_memory * inflights
            else:
                total_act_memory = max(total_act_memory, act_memory)
            # latency
            if self.recompute:
                latency = latency / 3 * 4  # suppose forward:backward=1:2
            latency = latency / (tp_size * dp_size) * tp_com_efficienty
            total_latency += latency


        optimizer_factor = 4 if self.train else 1

        # parameter size
        param_size = 0
        for block in blocks:
            for node in block.nodes:
                for tensor in node.inputs():
                    if isinstance(tensor, IRTensor) and tensor.is_attr():
                        # too large weight will bring memory fragment
                        factor = 1 # if tensor.byte_size() // t <= 1.5 * 1024 * 1024 * 1024 else 1.5
                        param_size += tensor.byte_size() * factor
        # consider gradient and adam optimizer (totally 3x param size)
        param_size = param_size * optimizer_factor / tp_size
        total_memory = param_size + total_act_memory

        if total_memory > memory_limit:
            total_latency = None

        if blocks[0].bid == 0:
            if self.param_limit is not None:
                if param_size >= self.param_limit * 1024 * 1024 * 1024:
                    total_latency = None

        if total_latency is None:
            spec = None
        else:
            spec = StageSpec(
                tuple(blocks),
                dp_size,
                tp_size,
                total_latency,
                total_memory
            )
        self._cache[key] = spec
        print(f'search {sum(len(blk.nodes) for blk in blocks)} ops | '
              f'tp={tp_size}, dp={dp_size} | '
              f'latency={None if total_latency is None else round(total_latency,2)} ms, '
              f'memory={round(total_memory/1024/1024/1024, 2)} GB')
        return spec
