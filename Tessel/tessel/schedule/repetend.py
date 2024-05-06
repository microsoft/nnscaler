# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import List, Tuple, Dict
import itertools
import math
from tessel.schedule.schedplan import SchedPlan, Block


Devices = Tuple[int]


class MicroPicker:

    @staticmethod
    def select(nblocks: int, nmicros: int):
        for split in itertools.combinations(list(range(1, nblocks)), nmicros-1):
            split = sorted(list(split))
            mids = []
            last_index = 0
            for mid, index in enumerate(split):
                mids += [nmicros-1-mid] * (index-last_index)
                last_index = index
            mids += [0] * (nblocks-last_index)
            assert len(mids) == nblocks
            yield mids
    
    @staticmethod
    def pick(micros: List[SchedPlan]) -> Tuple[List[Block], List[Block], List[Block], Dict[Block, Devices]]:
        """Yield warmup, repetend, cooldown blocks and block2device mapping

        Args:
            micros (List[SchedPlan]): list of placement of micro-batches
        
        Yields:
            warmup blocks, repetend blocks, cooldown blocks, block2device mapping
        """
        nmicros = len(micros)
        # collect device mapping
        block2device = {}
        for micro in micros:
            for block in micro.all_blocks():
                block2device[block] = micro.device(block)

        ref = micros[0]
        blocks = ref.chain_blocks()
    
        # nspace = math.comb(len(blocks)-1, nmicros-1)
        space = tuple(MicroPicker.select(len(blocks), nmicros))
        nspace = len(space)
        # TODO: support multi-branch
        for pidx, mids in enumerate(space):
            print(f'[{pidx}/{nspace}] assigning mids: {mids}')
            warmup, repetend, cooldown = [], [], []
            # collect repetend blocks
            for idx, (mid, block) in enumerate(zip(mids, blocks)):
                blk = micros[mid].chain_blocks()[idx]
                repetend.append(blk)
            # collect warmup and cooldown blocks
            for mid, micro in enumerate(micros):
                for block in micro.all_blocks():
                    if block in repetend: continue
                    idx = micro.chain_blocks().index(block)
                    if idx < mids.index(mid):
                        warmup.append(block)
                    else:
                        cooldown.append(block)
            print(f'warmup: {warmup}\nrepetend: {repetend}\ncooldown: {cooldown}')
            yield warmup, repetend, cooldown, block2device
