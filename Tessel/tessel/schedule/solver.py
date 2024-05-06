# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
A solver based solution for scheduling plan

python solver.py --premise vshape --nmicros 4 --ndevs 4 --memory 4
"""

from typing import List, Optional, Set, Dict, Iterable, Tuple
import sys
import more_itertools

from tessel.schedule.schedplan import SchedPlan, Block

import z3


def _z3_max(variables: Iterable[z3.ArithRef]) -> z3.ArithRef:
    res = None
    for var in variables:
        if res is None:
            res = var
            continue
        res = z3.If(var > res, var, res)
    assert res is not None, f"Require variables have at least one value: {variables}"
    return res


def _z3_min(variables: Iterable[z3.ArithRef]) -> z3.ArithRef:
    res = None
    for var in variables:
        if res is None:
            res = var
            continue
        res = z3.If(var < res, var, res)
    assert res is not None, f"Require variables have at least one value: {variables}"
    return res


class SolverBase:

    def __init__(self, ndevs: int):

        self._blocks: Set[Block] = set()
        self._ndevs = ndevs
        self._block_devices: Dict[Block, List[int]] = {}
        self._block_starts: Dict[Block, z3.ArithRef] = {} # the start step
        self._block_stops: Dict[Block, z3.ArithRef] = {}   # the end step
        self._mem: List[z3.ArithRef] = [None] * ndevs
        self._solution: Optional[z3.z3.ModelRef] = None
        self._solved = False
        # we experience extremely slow performance on z3.Optimize
        # self._solver = z3.Optimize()
        self._solver = z3.Solver()

    @property
    def nblocks(self) -> int:
        return len(self._blocks)

    @property
    def ndevs(self) -> int:
        return self._ndevs
    
    @property
    def solved(self) -> bool:
        return True
    
    def device(self, block: Block) -> List[int]:
        assert block in self._blocks
        return self._block_devices[block]

    def start(self, block: Block) -> z3.ArithRef:
        assert block in self._blocks, f"{block} not in the solver scope"
        return self._block_starts[block]

    def stop(self, block: Block) -> z3.ArithRef:
        assert block in self._blocks, f"{block} not in the solver scope"
        return self._block_stops[block]
    
    def add_block(self, block: Block, devs: List[int], step: int):
        assert block not in self._block_devices, f"Block {block} already exists"
        self._solved, self._solution = False, None
        self._block_devices[block] = devs
        start = z3.Int('blk' + str(len(self._block_devices)))
        end = start + block.span
        self._block_starts[block] = start
        self._block_stops[block] = end
        self._solver.add(start >= step)
        # intra-device: no overlapping constraints
        visited = set()
        for devid in devs:
            blocks = [blk for blk in self._blocks if devid in self.device(blk)]
            blocks = [blk for blk in blocks if blk not in visited]
            for blk in blocks:
                bstart = self.start(blk)
                bend = self.stop(blk)
                self._solver.add(z3.Or(bend <= start, end <= bstart))
            visited.update(blocks)
        self._blocks.add(block)

    def add_micro_plan(self, micro: SchedPlan):
        self._solved, self._solution = False, None
        for block in micro.all_blocks():
            step = micro.step(block)
            devs = micro.device(block)
            self.add_block(block, devs, step)
        for block in micro.all_blocks():
            for after_block in block.after:
                self.add_dependency_constraints([block, after_block])

    def add_dependency_constraints(self, blocks: List[Block]):
        self._solved, self._solution = False, None
        for idx in range(len(blocks) - 1):
            pre, post = blocks[idx], blocks[idx+1]
            pre_stop, post_start = self.stop(pre), self.start(post)
            self._solver.add(pre_stop <= post_start)
    
    def add_memory_constraints(self, dev_mem_limits: List[int]):
        """Add memory constraints for each device

        This can only be called after all blocks added
        """
        assert self.ndevs == len(dev_mem_limits)
        peak_mem_per_dev = []
        maxspan = sum(blk.span for blk in self._blocks)
        # add block stop time constraints
        for block in self._blocks:
            self._solver.add(self.stop(block) <= maxspan)
        # add memory constraints
        for devid in range(self.ndevs):
            peak = 0
            curr = 0
            blocks = [blk for blk in self._blocks if devid in self.device(blk)]
            for step in range(0, maxspan):
                mem = 0
                for block in blocks:
                    mem = z3.If(self.start(block) == step, block.memory, mem)
                curr = mem + curr
                peak = z3.If(curr > peak, curr, peak)
            self._solver.add(peak <= dev_mem_limits[devid])
            peak_mem_per_dev.append(peak)
        self._mem = peak_mem_per_dev

    def mono_mid_constraints(self):
        """Same subgraph should be executed in order of growing micro-batch index"""
        gid_blocks: Dict[int, List[Block]] = {}
        for block in self._blocks:
            assert block.gid is not None
            gid_blocks.setdefault(block.gid, []).append(block)
        for blocks in gid_blocks.values():
            blocks = sorted(blocks, key=lambda blk: blk.mid)
            if len(blocks) <= 1: continue
            for blk1, blk2 in more_itertools.windowed(blocks, 2):
                self._solver.add(self.start(blk1) < self.start(blk2))

    def solve(self, var, upper_var: int, lower_var: int, accept: Optional[int] = None, silence = True) -> Optional[int]:
        """Find lowest value for var given boundary of [lower_var, upper_var]
        
        Args:
            var (z3.ArithRef): the variable to be solved
            upper_var (int): the upper boundary of the variable
            lower_var (int): the lower boundary of the variable
            accept (Optional[int], optional): if provided, once find the result equal or smaller
                than the accept value, the solver will stop the search and return the found one.
                Defaults to None.
            silence (bool, optional): whether to print the progress. Defaults to True.

        Returns:
            Optional[int]: the found optimal value
        """
        self._solution = None
        upper, lower = upper_var, lower_var
        accept = lower_var if accept is None else accept
        accept = min(accept, upper - 1)

        # =============== use z3.Optimize() as self._solver ==============
        # self._solver.add(z3.And(lower <= var, var < upper))
        # self._solver.minimize(var)
        # opt = None
        # if self._solver.check() == z3.sat:
        #     self._solution = self._solver.model()
        #     opt = self._solution.eval(var).as_long()
        # ================================================================

        opt = upper
        while lower < upper and opt > accept:
            try_var = (lower + upper) // 2
            self._solver.push()
            self._solver.add(var <= try_var)
            if not silence:
                sys.stdout.write('checking solution of {:<4}.........'.format(try_var))
                sys.stdout.flush()
            if self._solver.check() == z3.sat:
                if not silence: print(f'Yes', flush=True)
                sys.stdout.flush()
                self._solution = self._solver.model()
                upper = opt = self._solution.eval(var).as_long()
            else:
                if not silence: print(f'Fail', flush=True)
                sys.stdout.flush()
                lower = try_var + 1
            self._solver.pop()
        if self._solution is not None:
            if not silence: print(f'find optimal solution of {opt}', flush=True)
            sys.stdout.flush()
        self._solved = True
        return opt if self._solution is not None else None

    def satisfy(self, var, upper: int) -> bool:
        """Check whether the solution exists"""
        self._solution = None
        self._solver.push()
        self._solver.add(var <= upper)
        ret = (self._solver.check() == z3.sat)
        self._solver.pop()
        return ret
    
    def solutions(self, var: z3.ArithRef, val: int) -> Iterable[SchedPlan]:
        """
        iterate all possible solutions given the searched solutions

        Yields:
            SchedPlan: the solution
        """
        assert self._solved, "Expected first call of solve"
        # step = self._solution.eval(var).as_long()
        self._solver.push()
        self._solver.add(var == val)
        while self._solver.check() == z3.sat:
            solution: Dict[Block, int] = dict()
            model = self._solver.model()
            for blk, t in self._block_starts.items():
                solution[blk] = model.eval(t).as_long()
            # solution to schedule plan
            schedplan = SchedPlan(self.ndevs)
            for blk, t in self._block_starts.items():
                step = model.eval(t).as_long()
                schedplan.add_block(blk, self.device(blk), step)
            yield schedplan
            block = [d() != model[d] for d in model]
            self._solver.add(z3.Or(block))
        self._solver.pop()


class StepOptimalSolver(SolverBase):

    def __init__(self, ndevs: int) -> None:
        super().__init__(ndevs)
        self._nsteps: z3.ArithRef = 0
        self._opt = None

    def init_nsteps(self):
        self._nsteps = 0
        for block in self._blocks:
            end = self.stop(block)
            self._nsteps = z3.If(end > self._nsteps, end, self._nsteps)

    def solve(self, memory: List[int], upper_time: Optional[int] = None, accept: Optional[int] = None, silence = True) -> Optional[int]:
        """Find step optimal plans given the time constraints of [0, upper_time-1]

        Args:
            memory (List[int]): the memory constraint of each device
            upper_time (int or None): the upper boundary of the time. Defaults to None.
            accept (int or None): if provided, once find the result equal or smaller
                than the accept value, the solver will stop the search and return the found one.
                Defaults to None.
            silence (bool, optional): whether to print the progress. Defaults to True.

        Returns:
            Optional[int]: the found optimal value
        """
        self.mono_mid_constraints()
        self._solution = None
        if not silence: print('memory constraints:', memory)
        self.add_memory_constraints(memory)

        self.init_nsteps()
        upper = upper_time if upper_time is not None \
            else sum(blk.span for blk in self._blocks) + 1
        self._opt = super().solve(self._nsteps, upper, 0, accept, silence)
        return self._opt

    def satisfy(self, memory: List[int]) -> bool:
        """Check whether the schedule plan exists"""
        self._solution = None
        self._solver.push()
        self.mono_mid_constraints()
        self.add_memory_constraints(memory)
        self.init_nsteps()
        upper = sum(blk.span for blk in self._blocks)
        ret = super().satisfy(self._nsteps, upper)
        self._solver.pop()
        return ret

    def solutions(self):
        return super().solutions(self._nsteps, self._opt) if self._opt is not None else []


class BubbleOptimalSolver(SolverBase):

    def __init__(self, ndevs: int):
        super().__init__(ndevs)
        self._nbubbles = None
        self._opt = None

    def init_bubble(self):
        """
        Bubble number is defined as summation of empty steps
        between consecutive blocks of a device.

        [----]      [----]   
        [----]  ->   [----]  Note: this will cause no bubble if not consider dependency
        [----]     [----]    

        Calculation:
            get max span (from start computation to the end of computation) among all devices
            inner_bubble = max_span - real_span
            outer_bubble = minimal offset in waiting dependency
            bubble = inner_bubble + outer_bubble
        """
        self._nbubbles = 0

        gid_mid_blocks: Dict[Tuple[int, int], Block] = {}
        for block in self._blocks:
            if block.gid is None: continue
            gid_mid_blocks[(block.gid, block.mid)] = block

        dev_span = []
        dev_step = []
        for devid in range(self.ndevs):
            blocks = [blk for blk in self._blocks if devid in self.device(blk)]
            # [min start-step, max end-step)
            minstep = _z3_min([self.start(block) for block in blocks])
            maxstep = _z3_max([self.stop(block) for block in blocks])
            span = maxstep - minstep
            dev_span.append(span)
            dev_step.append(sum(blk.span for blk in blocks))
        rspan = _z3_max(dev_span)

        rofst = 0
        for devid in range(self.ndevs):
            blocks = [blk for blk in self._blocks if devid in self.device(blk)]
            for block in blocks:
                # after blocks
                ablocks = list(block.after)
                keys = [(blk.gid, blk.mid + 1) for blk in ablocks]
                ablocks = [gid_mid_blocks[key] for key in keys if key in gid_mid_blocks]
                astarts = [self.start(blk) for blk in ablocks]
                if len(astarts) != 0:
                    min_ofst = _z3_min(astarts) - self.start(block) - rspan
                    rofst = _z3_max([rofst, min_ofst])
                # before blocks
                bblocks =  list(block.before)
                keys = [(blk.gid, blk.mid + 1) for blk in bblocks]
                bblocks = [gid_mid_blocks[key] for key in keys if key in gid_mid_blocks]
                bends = [self.start(blk) + blk.span for blk in bblocks]
                if len(bends) != 0:
                    min_ofst = _z3_max(bends) - self.start(block) - rspan
                    rofst = _z3_max([rofst, min_ofst])
        
        dev_bubbles = []
        for devid in range(self.ndevs):
            nbubbles = rspan - dev_step[devid] + rofst
            dev_bubbles.append(nbubbles)
        
        self._nbubbles = _z3_max(dev_bubbles)


        # dev_bubbles = [0] * self.ndevs
        # for devid in range(self.ndevs):
        #     # device bubble = internal empty slots + tail must wait slots
        #     blocks = [blk for blk in self._blocks if devid in self.device(blk)]
        #     # [min start-step, max end-step)
        #     minstep = _z3_min([self.start(block) for block in blocks])
        #     maxstep = _z3_max([self.start(block) + block.span for block in blocks])
        #     span = maxstep - minstep
        #     # get internal empty slots
        #     total_steps = sum(blk.span for blk in blocks)
        #     dev_bubbles[devid] = span - total_steps
        #     # get tail must wait slots: check dependency on the next repetend
        #     ofst = 0
        #     for block in blocks:
        #         # after blocks
        #         ablocks = [blk for blk in block.after]
        #         keys = [(blk.gid, blk.mid + 1) for blk in ablocks]
        #         ablocks = [gid_mid_blocks[key] for key in keys if key in gid_mid_blocks]
        #         astarts = [self.start(blk) for blk in ablocks]
        #         if len(astarts) != 0:
        #             min_start = _z3_min(astarts) - maxstep
        #             ofst = _z3_max([ofst, min_start])
        # 
        #         # before blocks
        #         bblocks = [blk for blk in block.before]
        #         keys = [(blk.gid, blk.mid + 1) for blk in bblocks]
        #         bblocks = [gid_mid_blocks[key] for key in keys if key in gid_mid_blocks]
        #         bends = [self.start(blk) + blk.span for blk in bblocks]
        #         if len(bends) != 0:
        #             min_start = _z3_max(bends) - maxstep
        #             ofst = _z3_max([ofst, min_start])
        # 
        #     dev_bubbles[devid] += ofst
        # 
        # for bubble in dev_bubbles:
        #     self._nbubbles = z3.If(z3.And(bubble >= self._nbubbles), bubble, self._nbubbles)

    def stride(self):
        """
        Constraints: every step should have at least one block in execution
        """
        maxstep = 0
        for block in self._blocks:
            end = self.start(block) + block.span
            maxstep = z3.If(end > maxstep, end, maxstep)
        for step in range(sum(block.span for block in self._blocks)):
            have_block = False
            for block in self._blocks:
                start, stop = self.start(block), self.stop(block)
                have_block = z3.Or(have_block, z3.And(start <= step, step < stop))
            self._solver.add(z3.If(step < maxstep, have_block, True))

    def solve(self, memory: List[int], upper_nbubbles: Optional[int] = None, accept: Optional[int] = None, silence=True) -> Optional[int]:
        """Find the lowest-bubble plan given the bubble range constraints [0, upper_nbubbles-1]

        Args:
            memory (List[int]): the memory constraint of each device
            upper_nbubbles (int or None): the upper boundary of the bubble. Defaults to None.
            accept (int or None): if provided, once find the result equal or smaller
                than the accept value, the solver will stop the search and return the found one.
                Defaults to None.
            silence (bool, optional): whether to print the progress. Defaults to True.
        
        Returns:
            Optional[int]: the found optimal value
        """
        self.mono_mid_constraints()
        # init memory
        if not silence: print('memory constraints:', memory)
        self.add_memory_constraints(memory)
        
        # init bubble
        self.init_bubble()
        upper_nbubbles = sum(block.span for block in self._blocks) + 1 \
            if upper_nbubbles is None else upper_nbubbles

        # stride the plan
        self.stride()

        self._opt = super().solve(self._nbubbles, upper_nbubbles, 0, accept, silence)
        return self._opt

    def solutions(self):
        return super().solutions(self._nbubbles, self._opt) if self._opt is not None else []