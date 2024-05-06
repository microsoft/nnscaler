# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import List, Dict, Set, Optional
import logging

from cube.graph.graph import IRGraph
from cube.graph.segment import IRSegment
from cube.graph.schedule.schedplan import SchedulePlan

_logger = logging.getLogger(__name__)


class OrderSolver:

    def __init__(self):
        
        # segment -> segment.device
        self._device_cache: Dict[IRSegment, Set[int]] = {}

    def get_device(self, segment: IRSegment) -> Set[int]:
        if segment not in self._device_cache:
            self._device_cache[segment] = set(segment.device)
        return self._device_cache[segment]
    
    def clear(self):
        self._device_cache = {}

    def solve(self, graph: IRGraph, nmicros: int,
              sched_file: Optional[str] = None) -> SchedulePlan:
        """Search for a schedule plan for the given operator placement.

        The search will leverage 1f1b on part of segments and insert the rest
        segments into the schedule plan.

        Args:
            graph (IRGraph): the graph that has been grouped into segments
            nmicros (int): number of micro-batches
            sched_file (str or None): if provided, will use searched schedule file

        Returns:
            SchedulePlan: searched schedule plan.
        """
        self.clear()
        fstages = [s for s in graph.select(ntype=IRSegment, flatten=False) if s.isfw()]
        _logger.info(f'get {len(fstages)} forward stages')
        
        if sched_file is None:
            sched = self.sched_1f1b(graph, nmicros)
        else:
            sched = self.sched_tessel(graph, nmicros, sched_file)

        _logger.info(f'ordering plan:\n{sched.str(20)}')
        return sched

    def sched_1f1b(self, graph, nmicros: int) -> SchedulePlan:
        """1F1B schedule"""
        fstages = [s for s in graph.select(ntype=IRSegment, flatten=False) if s.isfw()]
        num_stages = len(fstages)

        _logger.info(f"initing 1f1b schedule template: "
                     f"{len(fstages)} stages, {nmicros} micro-batches")
        
        sched = SchedulePlan(graph, nmicros)

        wait_steps = [sid for sid in range(num_stages)]
        bw_ofst = [num_stages - 1 - sid for sid in range(num_stages)]
        total_steps = nmicros * 2 + (num_stages - 1) * 2

        for step in range(total_steps):
            for sid in range(num_stages):
                ofst = wait_steps[sid]
                if step < ofst: continue
                fw_idx = (step - ofst) // 2
                # forward or backward segment
                stage = fstages[sid] if (step - ofst) % 2 == 0 else fstages[sid].mirror
                mb_idx = fw_idx if (step - ofst) % 2 == 0 else fw_idx - bw_ofst[sid]
                # append for execution
                if mb_idx < 0 or mb_idx >= nmicros: continue
                sched.add_segment(stage, mb_idx, step)
        sched.finish()
        return sched

    def sched_tessel(self, graph: IRGraph, nmicros: int, load_sched_file: Optional[str] = None) -> SchedulePlan:
        """Tessel schedule
        
        Args:
            graph (IRGraph): the graph that has been grouped into segments
            nmicros (int): number of micro-batches
            load_sched_file (str or None): load schedule from file. Defaults to None.
        """
        from tessel.schedule.schedplan import SchedPlan as TSched
        from tessel.schedule.schedplan import Block as TBlock

        # create blocks
        segments = [s for s in graph.select(ntype=IRSegment, flatten=False)]
        blocks: List[TBlock] = []
        for stage in segments:
            if stage.isfw():
                blocks.append(TBlock(0, span=1, memory=1, btype="forward"))
            else:
                blocks.append(TBlock(0, span=1, memory=-1, btype="backward"))
        for idx, blk in enumerate(blocks):
            blk.gid = idx

        blk2seg: Dict[TBlock, IRSegment] = {}
        for block, segment in zip(blocks, segments):
            blk2seg[block.gid] = segment
        
        # setup dependencies
        # TODO

        if load_sched_file is not None:
            tsched = TSched.load(load_sched_file)
        else:
            raise NotImplementedError("Tessel search is not integrated yet")
        
        all_blks = tsched.all_blocks()
        tsched_nmicros = max(blk.mid for blk in all_blks) + 1
        blocks_per_microbatch = len(all_blks) // tsched_nmicros
        if blocks_per_microbatch != len(segments):
            raise RuntimeError(
                f"schedule plan is not compatible with the graph. "
                f"The plan has {blocks_per_microbatch} blocks, "
                f"but the graph has {len(segments)} segments")
        
        # unroll schedule to the runtime number of micro-batches
        tsched = tsched.unroll(nmicros)
        sched = SchedulePlan(graph, nmicros)
        for step in range(tsched.nsteps):
            tblocks = tsched.blocks(step)
            for tblock in tblocks:
                segment = blk2seg[tblock.gid]
                sched.add_segment(segment, tblock.mid, step, tblock.span)
        sched.finish()
        return sched
