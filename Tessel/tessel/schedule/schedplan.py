# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations
from typing import Dict, Set, Tuple, List, Optional
import json
import numpy as np
import logging

import more_itertools


StartEnd = Tuple[int, int]
Devices = Tuple[int, ...]


_logger = logging.getLogger(__name__)


class Block:

    def __init__(self, mid: int, span: int, memory: float, btype: str, _gid=None):
        assert span > 0
        # micro-batch index
        self.mid: int = mid
        self.span = span
        self.memory = memory
        assert btype in ('forward', 'backward')
        self.btype = btype
        self.before: Set[Block] = set()
        self.after: Set[Block] = set()
        # sub-graph index
        self.gid: Optional[int] = _gid

    @staticmethod
    def make_dependency(prev, next):
        prev.after.add(next)
        next.before.add(prev)

    def __repr__(self):
        return f'f{self.mid}' if self.btype == 'forward' else f'b{self.mid}'


class SchedPlan:

    def __init__(self, ndevs: int) -> None:
        
        self._ndevs = ndevs
        self._nsteps = 0
        self._blocks: Set[Block] = set()
        self._block_devices: Dict[Block, Tuple[int]] = dict()
        self._block_steps: Dict[Block, int] = dict()
        self._step_blocks: Dict[int, List[Block]] = {0:[]}
        self._plans: List[List[Optional[Block]]] = [[] for _ in range(ndevs)]
        # repetend start step and end step
        self.repetend: Optional[StartEnd] = None

    @property
    def nsteps(self) -> int:
        return self._nsteps

    @property
    def ndevs(self) -> int:
        return self._ndevs

    @property
    def plans(self) -> List[List[Optional[Block]]]:
        return self._plans

    def all_blocks(self) -> Set[Block]:
        return self._blocks
    
    def chain_blocks(self) -> List[Block]:
        """Sort all blocks by step from early to later

        Returns:
            List[Block]: sorted blocks
        """
        blocks = []
        for step in range(self.nsteps):
            step_blocks = self.blocks(step)
            step_blocks = sorted(step_blocks, key=lambda blk: self.device(blk)[0])
            blocks += step_blocks
        assert len(blocks) == len(self._blocks)
        return blocks

    def add_block(self, block: Block, device: List[int], step: int):
        """Add a block into schedule plan. 
        
        If the block is already inserted inside the scheduling plan,
        the block must have same step and device.

        Args:
            block (Block): block to add
            device (List[int]): list of device id
            step (int): the starting step
        
        Returns:
            None
        """
        if block in self._blocks:
            assert self.step(block) == step and tuple(self.device(block)) == tuple(device), (
                f"Repeated adding a block but has different device and starting step setup:\n"
                f"Try to add   : {block}-{device} on step {step}\n"
                f"Already exist: {block}-{self.device(block)} on step {step}"
            )
            return
        maxstep = step + block.span
        if maxstep > self._nsteps:
            for devplan in self._plans:
                devplan += [None] * (maxstep - self._nsteps)
            for t in range(self._nsteps, maxstep):
                self._step_blocks.setdefault(t, [])
            self._nsteps = maxstep
        self._blocks.add(block)
        self._block_devices[block] = tuple(device)
        self._block_steps[block] = step
        self._step_blocks.setdefault(step, []).append(block)
        for devid in device:
            for t in range(step, step + block.span):
                assert self._plans[devid][t] is None, f"Conflict block {block}-{device} add on device {devid} at step {step}"
                self._plans[devid][t] = block

    def add_block_seq(self, blocks: List[Optional[Block]], devices: List[Optional[Devices]]):
        """Add a sequence of blocks into schedule plan

        The None in blocks indicates an empty step, which will not place block

        Data dependency will be added from prior block to the next block.

        Args:
            blocks (List[Block or None]): list of blocks to add. None indicates an empty step
            devices (List[Devices or None]): list of devices to add. None indicates an empty step
        """
        assert len(blocks) == len(devices)
        step = 0
        for block, devs in zip(blocks, devices):
            if block is not None:
                self.add_block(block, devs, step)
            step += (block.span if block is not None else 1)
        blocks = [blk for blk in blocks if blk is not None]
        for blk1, blk2 in more_itertools.windowed(blocks, 2):
            Block.make_dependency(blk1, blk2)

    def blocks(self, step: int) -> Tuple[Block]:
        """Get blocks started at the given step

        Note:
            the returned blocks don't contain blocks that are
            started at the previous steps but not finished yet.

        Args:
            step (int): the starting step

        Returns:
            Tuple[Block]: list of blocks
        """
        return tuple(self._step_blocks[step])
    
    def step(self, block: Block) -> int:
        """Get the starting step of the block

        Args:
            block (Block): the block

        Returns:
            int: the starting step
        """
        return self._block_steps[block]
    
    def device(self, block: Block) -> Tuple[int]:
        """Get the device of the block
        
        Args:
            block (Block): the block

        Returns:
            Tuple[int]: list of device id
        """
        return self._block_devices[block]

    def device_blocks(self, devid: int) -> Tuple[Block]:
        """Get the blocks on the given device
        
        Args:
            devid (int): device id

        Returns:
            Tuple[Block]: tuple of blocks that happend on the device
        """
        blocks = []
        for blk in self.all_blocks():
            if devid in self.device(blk):
                blocks.append(blk)
        return tuple(blocks)
    
    def extract(self, from_step: int, to_step: int) -> SchedPlan:
        """Extract a sub-schedule plan from steps of [from_step, to_step)
        
        Args:
            from_step (int): start step
            to_step (int): end step

        Returns:
            SchedPlan: a new schedule plan
        """
        sched = SchedPlan(self.ndevs)
        for step in range(from_step, to_step):
            for block in self.blocks(step):
                sched.add_block(block, self.device(block), step-from_step)
        return sched
    
    def unroll(self, nmicros: int) -> SchedPlan:
        """Unroll repetend to `nmicros` microbatches

        Note:
            the new blocks in unrolled schedule are not set
            with any dependency.
        
        Args:
            nmicros (int): the total number of microbatches
        
        Returns:
            SchedPlan: a new unrolled schedule plan
        """
        assert self.repetend is not None
        rstart, rend = self.repetend
        # already existed number of microbatches
        mids = set(blk.mid for blk in self._blocks)
        assert len(mids) == max(mids) + 1, f"Microbatch index should be consecutive"
        nmids = len(mids)
        assert nmicros >= nmids, \
            f"Unroll to nmicros ({nmicros}) smaller than num-microbatches that are already in schedule ({nmids})"

        all_blocks: Dict[Tuple[int, int], Block] = {}
        for blk in self._blocks:
            assert blk.gid is not None
            all_blocks[(blk.gid, blk.mid)] = blk
    
        def get_block(gid: int, mid: int) -> Block:
            ref = all_blocks[(gid, 0)]
            return all_blocks.setdefault(
                (gid, mid), Block(mid, ref.span ,ref.memory, ref.btype, gid))

        in_repetend = lambda blk: blk in self._blocks and rstart <= self.step(blk) and self.step(blk) < rend

        # get repetend offset
        dev_span = []
        for devid in range(self.ndevs):
            blocks = [blk for blk in self._blocks if devid in self.device(blk)]
            blocks = [blk for blk in blocks if in_repetend(blk)]
            maxstep = max(self.step(blk) + blk.span for blk in blocks)
            minstep = min(self.step(blk) for blk in blocks)
            dev_span.append(maxstep - minstep)
        rspan = max(dev_span)

        rofst = 0
        for devid in range(self.ndevs):
            blocks = [blk for blk in self._blocks if devid in self.device(blk)]
            blocks = [blk for blk in blocks if in_repetend(blk)]
            for block in blocks:
                # after blocks
                ablocks = list(block.after)
                keys = [(blk.gid, blk.mid + 1) for blk in ablocks]
                ablocks = [get_block(*key) for key in keys]
                ablocks = [blk for blk in ablocks if in_repetend(blk)]
                astarts = [self.step(blk) for blk in ablocks]
                if len(astarts) != 0:
                    min_ofst = min(astarts) - self.step(block) - rspan
                    rofst = max(rofst, min_ofst)
                # before blocks
                bblocks = list(block.before)
                keys = [(blk.gid, blk.mid + 1) for blk in bblocks]
                bblocks = [get_block(*key) for key in keys]
                bblocks = [blk for blk in bblocks if in_repetend(blk)]
                bends = [self.step(blk) + blk.span for blk in bblocks]
                if len(bends) != 0:
                    min_ofst = max(bends) - self.step(block) - rspan
                    rofst = max(rofst, min_ofst)

        unrolled_plan = SchedPlan(self.ndevs)

        # warmup
        for step in range(rstart):
            blocks = self.blocks(step)
            for block in blocks:
                unrolled_plan.add_block(block, self.device(block), step)

        # steady
        rspan = max(dev_span)
        for mid_ofst in range(nmicros-nmids+1):
            for step in range(rstart, rend):
                for blk in self.blocks(step):
                    rblk = get_block(blk.gid, blk.mid + mid_ofst)
                    unrolled_plan.add_block(
                        rblk, self.device(blk),
                        step + (rspan + rofst) * mid_ofst
                    )

        # cooldown
        mid_ofst = nmicros - nmids
        for step in range(rend, self.nsteps):
            for blk in self.blocks(step):
                unrolled_plan.add_block(
                    get_block(blk.gid, blk.mid + mid_ofst), 
                    self.device(blk),
                    step + (rspan + rofst) * mid_ofst
                )
        
        unrolled_plan.repetend = (rstart, rend + (rspan + rofst) * mid_ofst)
        return unrolled_plan

    def copy(self, mid: Optional[int] = None) -> SchedPlan:
        """Copy the plan with the blocks assigned by the given microbatch index
        
        Args:
            mid (Optional[int], optional): microbatch index. Defaults to None (keep as same).
        
        Returns:
            SchedPlan: a new schedule plan
        """
        blks: Dict[Block, Block] = {}
        def new(block: Block):
            new_mid = block.mid if mid is None else mid
            return blks.setdefault(
                block, Block(new_mid, block.span, block.memory, block.btype, block.gid))

        sched = SchedPlan(self.ndevs)
        for block in self._blocks:
            blk = new(block)
            sched.add_block(blk, self.device(block), self.step(block))
            # set dependency
            blk.before = set(new(bblock) for bblock in block.before)
            blk.after = set(new(ablock) for ablock in block.after)
        return sched

    def peak_memory(self, device: int) -> float:
        peak_mem = 0
        curr_mem = 0
        added = set()
        for step in range(self.nsteps):
            if step >= len(self.plans[device]):
                break
            blk = self.plans[device][step]
            if blk is None or blk in added:
                continue
            curr_mem += blk.memory
            peak_mem = max(peak_mem, curr_mem)
            added.add(blk)
        return peak_mem

    def stride(self):
        """Remove empty steps inside the schedule"""
        step = 0
        while step < self.nsteps:
            if all(self.plans[devid][step] is None for devid in range(self.ndevs)):
                for blk in self._blocks:
                    if self.step(blk) > step:
                        self._block_steps[blk] -= 1
                for t in range(step + 1, self.nsteps):
                    if t in self._step_blocks:
                        self._step_blocks[t-1] = self._step_blocks[t]
                        del self._step_blocks[t]
                for devid in range(self.ndevs):
                    self._plans[devid].pop(step)
                self._nsteps -= 1
            else:
                step += 1

    def shift(self, block: Block, offset: int):
        """Shift block by offset steps
        
        Args:
            block (Block): the block to shift
            offset (int): the offset steps

        Raises:
            KeyError: if the block is not in the schedule plan
            RuntimeError: if the block cannot shift by the given offset

        Returns:
            None
        """
        if offset == 0: return
        if block not in self._blocks:
            raise KeyError(f"Block {block} not in schedule plan")
        curr_step = self.step(block)
        if curr_step + offset < 0:
            raise RuntimeError(f"Block {block} cannot shift by {offset} steps (happen before time 0)")
        start = curr_step + offset
        end = start + block.span
        # check time slot empty
        for step in range(start, end):
            for device in self.device(block):
                if step < self.nsteps and self.plans[device][step] not in (None, block):
                    raise RuntimeError(f"Block {block} cannot shift by {offset} steps (conflict with other blocks)")
        # expand steps if necessary
        if end >= self.nsteps:
            for devplan in self._plans:
                devplan += [None] * (end - self.nsteps)
            for t in range(self.nsteps, end):
                self._step_blocks.setdefault(t, [])
            self._nsteps = end
        # remove block from old steps
        self._step_blocks[curr_step].remove(block)
        for step in range(curr_step, curr_step + block.span):
            for devid in self.device(block):
                self._plans[devid][step] = None
        # update block to new steps
        self._block_steps[block] = start
        self._step_blocks.setdefault(start, []).append(block)
        for step in range(start, end):
            for devid in self.device(block):
                self._plans[devid][step] = block

    def tighten(self):
        """Tighten the schedule plan by making blocks happen as early as possible.
        
        The process doesn't change the execution order of blocks for each device.
        """
        for step in range(self.nsteps):
            blocks = self.blocks(step)
            for blk in blocks:
                devices = self.device(blk)
                # get available minimal start step without changing schedule order
                min_start = step
                while min_start > 0:
                    # empty step
                    if all(self.plans[devid][min_start-1] is None for devid in devices):
                        min_start -= 1
                    else:
                        break
                if min_start == step:
                    continue
                # consider data dependency
                pres = [pre for pre in blk.before if pre in self._blocks]
                end = max(self.step(pre) + pre.span for pre in pres) if len(pres) > 0 else 0
                min_start = max(end, min_start)
                if min_start == step:
                    continue
                # shift the block ahead if there are available steps
                for st in range(min_start, step):
                    available = True
                    for t in range(st, st + blk.span):
                        if any(self.plans[devid][t] not in (None, blk) for devid in devices):
                            available = False
                            break
                    if available:
                        self.shift(blk, st - step)
                        break
        # remove tail empty steps
        while self.nsteps > 0:
            if all(self.plans[devid][self.nsteps-1] is None for devid in range(self.ndevs)):
                for devid in range(self.ndevs):
                    self._plans[devid].pop()
                self._nsteps -= 1
            else:
                break

    def validate(self, complete: bool = True) -> bool:
        """Check whether the schedule plan is valid (no data dependency conflict)

        Args:
            complete (bool, optional): if True, requires all blocks
                in data dependency (block.after and block.before) should 
                appear in the schedule. Defaults to True.
        Returns:
            bool: True if valid
        """
        for step in range(self.nsteps):
            blks = self.blocks(step)
            for blk in blks:
                for bblk in blk.before:
                    if complete:
                        if bblk not in self._blocks:
                            _logger.error(f"Validation check fail: missing depdendent block {bblk}")
                            return False
                    if bblk in self._blocks:
                        if self.step(bblk) + bblk.span > step:
                            error_msg = (
                                f"Validation check fail: data dependency conflict: "
                                f"{bblk}(step={self.step(bblk)},device={self.device(bblk)})"
                                f" -> "
                                f"{blk}(step={self.step(blk)},device={self.device(blk)})\n"
                                f"{self}"
                            )
                            _logger.error(error_msg)
                            return False
        return True

    def __repr__(self) -> str:
        dscp = ''
        for devid in range(self.ndevs):
            step = 0
            while step < self.nsteps:
                if self.repetend is not None and step in self.repetend:
                    dscp += ' |'
                have_block = False
                for blk in self.blocks(step):
                    if devid in self.device(blk):
                        dscp += ' ' + '-'.join([repr(blk)] * blk.span)
                        have_block = True
                        step += blk.span
                        break
                if not have_block:
                    dscp += ' --'
                    step += 1
            dscp += '\n'
        return dscp

    def getstate(self) -> np.ndarray:
        """
        return state format: 2-D array of (M, N+2) shape,
        where M is number of microbatches, N is number of sub-graphs.
        
        (i, j) in (M, N) denotes the start time of block gid j of microbatch i 
        (*, N+1) and (*, N+2) denotes the start and end of the repetend, respectively.
        """
        nmicros = max(blk.mid for blk in self._blocks) + 1
        nstages = max(blk.gid for blk in self._blocks) + 1
        state = -np.ones((nmicros, nstages+2), dtype=int)
        for blk in self._blocks:
            step = self.step(blk)
            state[blk.mid, blk.gid] = step
        state[:,-2:] = self.repetend
        assert np.all(state >= 0)
        return state

    def loadstate(self, blocks: List[Block], devices: List[List[int]], state: np.ndarray):
        """Load the state from the state array"""
        getblock, getdevice = {}, {}
        for blk, devs in zip(blocks, devices):
            getblock[(blk.mid, blk.gid)] = blk
            getdevice[blk] = devs
        self.repetend = tuple(state[0,-2:])
        for mid in range(state.shape[0]):
            for gid in range(state.shape[1]-2):
                step = state[mid, gid]
                block = getblock[(mid, gid)]
                self.add_block(block, getdevice[block], step)

    def save(self, filename: str):
        """
        save the schedule plan to a json file
        """
        plans = {
            'ndevs': self.ndevs,
            'blocks': [],
            'repetend': self.repetend
        }
        for block in self._blocks:
            plans['blocks'].append({
                'mid': block.mid,
                'span': block.span,
                'memory': block.memory,
                'btype': block.btype,
                'gid': block.gid,
                'step': self.step(block),
                'device': self.device(block)
            })
        with open(filename, 'w') as f:
            json.dump(plans, f)

    @staticmethod
    def load(filename: str) -> SchedPlan:
        """Load a schedule plan from a json file
        
        Args:
            filename (str): the file name

        Returns:
            SchedPlan: a new schedule plan
        """
        with open(filename, 'r') as f:
            plan = json.load(f)
        ndevs = plan['ndevs']
        schedplan = SchedPlan(ndevs)
        for block in plan['blocks']:
            # block attr
            mid = block['mid']
            span = block['span']
            memory = block['memory']
            btype = block['btype']
            gid = block.get('gid', None)
            # schedule plan position
            start = block['step']
            device: List[int] = block['device']
            schedplan.add_block(
                Block(mid, span, memory, btype, gid),
                device, start
            )
        schedplan.repetend = tuple(plan['repetend'])
        return schedplan

    @staticmethod
    def concat(plans: List[SchedPlan]) -> SchedPlan:
        """Concat a list of schedule plans into one
        
        Args:
            plans (List[SchedPlan]): list of schedule plans

        Returns:
            SchedPlan: a new schedule plan
        """
        cplan = SchedPlan(plans[0].ndevs)
        step_ofst = 0
        for plan in plans:
            for block in plan.all_blocks():
                cplan.add_block(block, plan.device(block), plan.step(block) + step_ofst)
            step_ofst += plan.nsteps
        return cplan
