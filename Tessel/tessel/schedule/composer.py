# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import List, Tuple, Optional
import math
import sys
# import multiprocessing as mp

from .schedplan import SchedPlan, Block
from .repetend import MicroPicker
from .solver import StepOptimalSolver, BubbleOptimalSolver, SolverBase

from ..timer import CpuTimer


class Composer:

    @staticmethod
    def compose(micro: SchedPlan, memory: int) -> SchedPlan:
        # assign gid to blocks
        for gid, blk in enumerate(micro.chain_blocks()):
            blk.gid = gid
        ndevs = micro.ndevs
        peak_mem = [0] * ndevs
        for blk in micro.all_blocks():
            if blk.memory > 0:
                for devid in micro.device(blk):
                    peak_mem[devid] += blk.memory
        max_inflight_nmb = int(math.ceil(memory / max(peak_mem)))
        # search for the 
        best_sched, best_nbubbles = None, None
        for nr in range(1, max_inflight_nmb + 1):
            schedule, nbubbles = Composer.compose_n(micro, memory, nr)
            if schedule is None: continue
            if best_nbubbles is None or nbubbles < best_nbubbles:
                best_sched = schedule
                best_nbubbles = nbubbles
            if nbubbles == 0:
                break
        return best_sched

    @staticmethod
    def compose_n(micro: SchedPlan, memory: int, nmicros: int) -> Optional[SchedPlan]:
        """Search the best schedule for using the number of microbatches"""
        # assign gid to blocks
        for gid, blk in enumerate(micro.chain_blocks()):
            blk.gid = gid
        ndevs = micro.ndevs
        memory = [memory] * ndevs
        micros = [micro.copy(mid) for mid in range(nmicros)]

        schedule_parts = [None, None, None]  # warmup / repetend / cooldown
        nbubbles = micros[0].nsteps + 1
        for warmup_blks, repetend_blks, cooldown_blks, devices in MicroPicker.pick(micros):
            warmup_devs = [devices[blk] for blk in warmup_blks]
            repetend_devs = [devices[blk] for blk in repetend_blks]
            cooldown_devs = [devices[blk] for blk in cooldown_blks]
            warmup_post_mem = Composer.memory(warmup_blks, warmup_devs, ndevs)
            repetend_post_mem = Composer.memory(warmup_blks + repetend_blks, warmup_devs + repetend_devs, ndevs)
            
            # step 1: construct a repetend
            repetend_pre_mem = [memory[devid] - warmup_post_mem[devid] for devid in range(ndevs)]
            CpuTimer().start('repetend')
            repetend, case_nbubbles = Composer.construct(
                repetend_blks, repetend_devs, ndevs, repetend_pre_mem, nbubbles, optimizer=BubbleOptimalSolver)
            # repetend, case_nbubbles = Composer.construct(
            #     repetend_blks, repetend_devs, ndevs, mem, nbubbles, optimizer=StepOptimalSolver)
            CpuTimer().stop('repetend')
            if repetend is None: continue
            
            # step 2: validate warmup
            CpuTimer().start('warmup')
            if case_nbubbles == 0:
                warmup, _ = Composer.construct(warmup_blks, warmup_devs, ndevs, memory,
                                               optimizer=StepOptimalSolver)
                exist = (warmup is not None)
            else:
                exist = Composer.satisfy(warmup_blks, warmup_devs, ndevs, memory) 
            CpuTimer().stop('warmup')
            if not exist: continue
            
            # step 3: validate cooldown
            cooldown_pre_mem = [memory[devid] - repetend_post_mem[devid] for devid in range(ndevs)]
            CpuTimer().start('cooldown')
            if case_nbubbles == 0:
                cooldown, _ = Composer.construct(cooldown_blks, cooldown_devs, ndevs, cooldown_pre_mem,
                                         optimizer=StepOptimalSolver)
                exist = (cooldown is not None)
            else:
                exist = Composer.satisfy(cooldown_blks, cooldown_devs, ndevs, cooldown_pre_mem)
            CpuTimer().stop('cooldown')
            if not exist: continue
            
            print(f'find a better solution solution: bubbles per repetend: {case_nbubbles}')

            assert case_nbubbles < nbubbles
            nbubbles = case_nbubbles
            schedule_parts = [
                warmup if nbubbles == 0 else (warmup_blks, warmup_devs, memory),
                repetend,
                cooldown if nbubbles == 0 else (cooldown_blks, cooldown_devs, cooldown_pre_mem),
            ]

            if case_nbubbles == 0:
                print('> early stop as find 0-bubble plans')
                break
            print(f'> setting repetend maximal bubbles: {nbubbles}')

        repetend = schedule_parts[1]
        if repetend is None:  # no solution
            return None

        # search for warmup parts
        if nbubbles == 0:
            warmup, repetend, cooldown = schedule_parts
        else:
            print(f'> constructing warmup and cooldown parts...')
            CpuTimer().start('warmup')
            warmup_blks, warmup_devs, warmup_mem = schedule_parts[0]
            warmup, _ = Composer.construct(warmup_blks, warmup_devs, ndevs, warmup_mem,
                                           optimizer=StepOptimalSolver)
            CpuTimer().stop('warmup')
            assert warmup is not None

            # search for cooldown parts
            CpuTimer().start('cooldown')
            cooldown_blks, cooldown_devs, cooldown_mem = schedule_parts[2]
            cooldown, _ = Composer.construct(cooldown_blks, cooldown_devs, ndevs, cooldown_mem,
                                             optimizer=StepOptimalSolver)
            CpuTimer().stop('cooldown')
            assert cooldown is not None

        schedule = SchedPlan.concat([warmup, repetend, cooldown])
        schedule.repetend = (warmup.nsteps, warmup.nsteps + repetend.nsteps)

        return schedule

    @staticmethod
    def compose_fast(micro: SchedPlan, memory: int,
                     wc_ratio: Tuple[int, int] = (128, 0.05),
                     flip: bool = True) -> Optional[SchedPlan]:
        """Search the schedule by directly constructing the repetend
        
        Args:
            micro (SchedPlan): the micro-batch plan
            memory (int): memory limit
            wc_ratio (Tuple[int, int]): 
                total number of microbatches,
                accepted warmup / cooldown bubble ratio for the schedule

        Returns:
            SchedPlan or None: the searched schedule plan
        """
        # assign gid to blocks
        for gid, blk in enumerate(micro.chain_blocks()):
            blk.gid = gid
        ndevs = micro.ndevs
        memory = [memory] * ndevs

        # step 1: condense the micro-batch plan to squeeze out bubbles
        # by not considering data dependency
        compact = SchedPlan(micro.ndevs)

        def free_steps(micro: SchedPlan, devid: int, step_range: Tuple[int, int]) -> bool:
            for t in range(*step_range):
                order = micro.plans[devid]
                if t < len(order) and order[t] is not None:
                    return False
            return True
        
        added = set()
        for devid in range(micro.ndevs):
            blocks = micro.device_blocks(devid)
            # primary criteron: schedule blocks with multiple devices to two sides,
            # secondary criteron: blocks in time step 
            def criteron(blk):
                ndevs = len(micro.device(blk))
                if ndevs > 1:
                    ndevs = 0-ndevs if blk.btype == 'forward' else ndevs
                else:
                    ndevs = 0
                return (ndevs, micro.step(blk))
            blocks = sorted(blocks, key=criteron)
            # blocks = sorted(blocks, key=lambda blk: len(micro.device(blk)))
            # ================ optimization ==================
            if flip and devid % 2 == 1:
                blocks = reversed(blocks)
            # ================================================
            max_step = 0
            for blk in blocks:
                if blk in added: continue
                # find a step to locate
                step = 0
                while not free_steps(compact, devid, (step, step + blk.span)):
                    step += 1
                    if step > 10000:
                        raise RuntimeError("Internal error: loop doesn't terminate")
                # locate the block
                compact.add_block(blk, micro.device(blk), step)
                max_step = max(max_step, step + blk.span)
                added.add(blk)
        
        print(f'> plan compaction:\n{compact}')

        # step 2: assign micro-batch indices
        rchain_blocks = list(reversed(micro.chain_blocks()))
        blk2mid = {}
        for idx, blk in enumerate(rchain_blocks):
            mid = 0
            # print(f'{blk}{compact.device(blk)}: {blk.after}')
            for succ_blk in rchain_blocks[:idx]:
                if succ_blk in blk.after:
                    # print(f'> {blk}{compact.device(blk)} -> {succ_blk}{compact.device(succ_blk)}')
                    if compact.step(succ_blk) >= compact.step(blk) + blk.span:
                        mid = max(mid, blk2mid[succ_blk])
                    else:
                        mid = max(mid, blk2mid[succ_blk] + 1)
            blk2mid[blk] = mid

        # step 3: construct repetend
        nmicros = max(blk2mid.values()) + 1
        micros: List[SchedPlan] = [micro.copy(mid) for mid in range(nmicros)]

        repetend = SchedPlan(micro.ndevs)
        for idx, blk in enumerate(micro.chain_blocks()):
            step = compact.step(blk)
            mid = blk2mid[blk]
            micro_blk = micros[mid].chain_blocks()[idx]
            assert micros[mid].device(micro_blk) == micro.device(blk), \
                f"Internal error: device mismatch: {micros[mid].device(micro_blk)} != {micro.device(blk)}"
            devices = micros[mid].device(micro_blk)
            repetend.add_block(micro_blk, devices, step)
        
        if not repetend.validate(complete=False):
            raise ValueError(f"Internal error: invalid repetend:\n{repetend}")
        print(f'> composed repetend ({nmicros} micros):\n{repetend}')
        sys.stdout.flush()

        # step 4: construct warmup and cooldown
        
        chain_blocks: List[List[Block]] = []
        blk2devices = {}
        for m in micros:
            blks = m.chain_blocks()
            chain_blocks.append(blks)
            blk2devices.update({blk: m.device(blk) for blk in blks})

        warmup_blks, cooldown_blks = [], []
        for idx, blk in enumerate(micro.chain_blocks()):
            mid = blk2mid[blk]
            # warmup blocks
            for warmup_mid in range(mid):
                blk = chain_blocks[warmup_mid][idx]
                warmup_blks.append(blk)
            # cooldown blocks
            for cooldown_mid in range(mid + 1, nmicros):
                blk = chain_blocks[cooldown_mid][idx]
                cooldown_blks.append(blk)
        repetend_blks = repetend.chain_blocks()
        
        warmup_devs = [blk2devices[blk] for blk in warmup_blks]
        cooldown_devs = [blk2devices[blk] for blk in cooldown_blks]
        repetend_devs = [blk2devices[blk] for blk in repetend_blks]

        # check memory limit
        warmup_post_mem = Composer.memory(warmup_blks, warmup_devs, ndevs)
        repetend_peak_mem = [repetend.peak_memory(devid) for devid in range(ndevs)]
        post_repetend_peak_mem = [warmup_post_mem[devid] + repetend_peak_mem[devid] for devid in range(ndevs)]
        if any(post_repetend_peak_mem[devid] > memory[devid] for devid in range(ndevs)):
            raise RuntimeError(
                f"Out of memory capacity: memory at least requires {post_repetend_peak_mem} = "
                f"{warmup_post_mem} (post-warmup) + {repetend_peak_mem} (peak-repetend), "
                f"but the memory limit is {memory}")
    
        repetend_post_mem = Composer.memory(
            warmup_blks + repetend_blks, warmup_devs + repetend_devs, ndevs)

        nsteps = repetend.nsteps
        total_micros, accept_ratio = wc_ratio
        max_bubble = nsteps * total_micros * accept_ratio

        # with mp.Pool() as pool:
        #     result1 = pool.apply_async(
        #         Composer.construct, (warmup_blks, warmup_devs, ndevs, warmup_peak_mem,
        #                              None, warmup_accept, StepOptimalSolver))
        #     result2 = pool.apply_async(
        #         Composer.construct, (cooldown_blks, cooldown_devs, micro.ndevs, cooldown_pre_mem,
        #                              None, cooldown_accept, StepOptimalSolver))
        # 
        #     warmup, _ = result1.get()
        #     cooldown, _ = result2.get()

        CpuTimer().start('warmup')
        warmup_peak_mem = [memory[devid] - repetend.peak_memory(devid) for devid in range(ndevs)]
        warmup_comp = sum(blk.span * len(blk2devices[blk]) for blk in warmup_blks) // ndevs
        warmup_accept = int(warmup_comp + max_bubble)
        print(f'> warmup accept time steps: {warmup_accept}', flush=True)
        warmup, _ = Composer.construct(warmup_blks, warmup_devs, ndevs, warmup_peak_mem,
                                       accept=warmup_accept, optimizer=StepOptimalSolver)
        print(f'> warmup before tighten nsteps: {warmup.nsteps}\n', flush=True)
        warmup.tighten()
        CpuTimer().stop('warmup')
        if warmup is None:
            raise RuntimeError('Fail to find warmup schedule, check the memory limits')
        
        CpuTimer().start('cooldown')
        cooldown_pre_mem = [memory[devid] - repetend_post_mem[devid] for devid in range(ndevs)]
        cooldown_comp = sum(blk.span * len(blk2devices[blk]) for blk in cooldown_blks) // ndevs
        cooldown_accept = int(cooldown_comp + max_bubble)
        print(f'> cooldown accept time steps: {cooldown_accept}', flush=True)
        cooldown, _ = Composer.construct(cooldown_blks, cooldown_devs, micro.ndevs, cooldown_pre_mem,
                                         accept=cooldown_accept, optimizer=StepOptimalSolver)
        print(f'> cooldown before tighten nsteps: {cooldown.nsteps}\n', flush=True)
        cooldown.tighten()
        CpuTimer().stop('cooldown')
        if cooldown is None:
            raise RuntimeError('Fail to find cooldown schedule, check the memory limits')

        print(f'> finish search')
        schedule = SchedPlan.concat([warmup, repetend, cooldown])
        schedule.repetend = (warmup.nsteps, warmup.nsteps + repetend.nsteps)
        # check validation
        assert schedule.validate(), f"Invalid schedule:\n{schedule}"
        print(f'> validate schedule: OK')
        return schedule

    @staticmethod
    def construct(blocks: List[Block], devices: List[Tuple[int]],
                  ndevs: int, memory: Tuple[int],
                  upper: Optional[int]=None, 
                  accept: Optional[int]=None,
                  optimizer: Optional[SolverBase] = StepOptimalSolver) -> Tuple[Optional[SchedPlan], Optional[int]]:
        """Construct a schedule given the blocks, devices and memory constraints.

        Args:
            blocks (List[Block]): blocks to be scheduled
            devices (List[Tuple[int]]): devices of each block
            ndevs (int): number of devices
            memory (Tuple[int]): memory constraints
            upper (int or None): upper bound of the number of steps. Defaults to None.
            accept (int or None): accept the solution if the solver target is equal or less than accept.
                Defaults to None, finding the optimal solution.
            optimizer (SolverBase or None): optimizer to be used. Defaults to StepOptimalSolver.
        
        Returns:
            SchedPlan or None: the schedule plan
            int or None: the optimized solver target
        """
        
        solver = optimizer(ndevs)
        # step 1 add block inside solver
        for block, devs in zip(blocks, devices):
            solver.add_block(block, devs, 0)
        # step 2 setup dependency
        for blk1 in blocks:
            for blk2 in blocks:
                if blk2 in blk1.after:
                    # print(f'> add dependency: {blk1}-dev{devices[idx1]} -> {blk2}-dev{devices[idx2]}')
                    solver.add_dependency_constraints([blk1, blk2])
        # step 3 construct
        lowest = solver.solve(memory, upper, accept, silence=True)
        if lowest is None:
            print(f"{optimizer.__name__}: Fail to find a solution given boundary constraints ( solution > {upper} (upper) )\n")
            return None, None
        for schedplan in solver.solutions():
            # assert schedplan.nsteps == nsteps
            return schedplan, lowest

    @staticmethod
    def satisfy(blocks: List[Block], devices: List[Tuple[int]],
                ndevs: int, memory: Tuple[int]) -> bool:
        """Check the existence of a schedule"""
        solver = StepOptimalSolver(ndevs)
        # step 1 add block inside solver
        for block, devs in zip(blocks, devices):
            solver.add_block(block, devs, 0)
        # step 2 setup dependency
        for blk1 in blocks:
            for blk2 in blocks:
                if blk2 in blk1.after:
                    # print(f'> add dependency: {blk1}-dev{devices[idx1]} -> {blk2}-dev{devices[idx2]}')
                    solver.add_dependency_constraints([blk1, blk2])
        return solver.satisfy(memory)

    @staticmethod
    def memory(blocks: List[Block], devices: List[Tuple[int]], ndevs: int) -> Tuple[int]:
        """
        Calculate memory after executing all blocks
        """
        memory: Tuple[int] = [0] * ndevs
        for block, device in zip(blocks, devices):
            for devid in device:
                memory[devid] += block.memory
        return memory
