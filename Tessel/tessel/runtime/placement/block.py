# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import List, Optional, Tuple
import more_itertools
import logging

from cube.ir.cten import IRCell
from cube.ir.operator import IRFwOperation
from cube.graph.function.anchor import IRGraphAnchor
from cube.graph.segment import IRSegment


_logger = logging.getLogger(__name__)


class IRBlock:
    """
    IRBlock represents a sub-graph that contains multiple consecutive nodes.

    The IRBlock is the searching granularity in stages, where all the inside
    operators will be partitioned and placed on a same device group.
    
    Each block can be applied by constraints of tensor/data parallelism size
    and its device region for execution.
    """

    def __init__(self, nodes: Tuple[IRCell], bid: int = None):
        self.nodes = tuple(nodes)
        self.bid : int = bid

        # search constraints
        self.min_tp: int = 1
        self.max_tp: int = 32
        self.min_dp: int = 1
        self.max_dp: int = 32
        # the device region to execute this block
        self.devices: Optional[Tuple[int]] = None

    @property
    def nnodes(self) -> int:
        return len(self.nodes)

    def node(self, idx: int) -> IRCell:
        """Get the node at index"""
        return self.nodes[idx]

    def fwops(self) -> List[IRFwOperation]:
        """Get all forward operators"""
        fwops = []
        for node in self.nodes:
            if isinstance(node, IRSegment):
                fwops += node.select(ntype=IRFwOperation)
            else:
                assert isinstance(node, IRFwOperation)
                fwops.append(node)
        return fwops

    def constrain_tp_size(self, min_size: int = 1, max_size: Optional[int] = None):
        """Add constraint of minimal and maximal tensor parallelism size.

        Warnings:
            This will orverride the previous setting.

        Args:
            min_size (int): Minimal tensor parallelism size. Defualt 1, which means not
                apply tensor parallelism
            max_size (int): Maximal tensor parallelism size. Defualt None, which means
                apply any size of tensor parallelism
        
        Returns:
            None
        """
        self.min_tp = min_size
        self.max_tp = max_size

    def constrain_tp_size(self, min_size: int = 1, max_size: Optional[int] = None):
        """Add constraint of minimal and maximal data parallelism size.

        Warnings:
            This will orverride the previous setting.

        Args:
            min_size (int): Minimal data parallelism size. Defualt 1, which means not
                apply data parallelism
            max_size (int): Maximal data parallelism size. Defualt None, which means
                apply any size of data parallelism
        
        Returns:
            None
        """
        self.min_dp = min_size
        self.max_dp = max_size

    def constrain_devices(self, devices: Optional[Tuple[int]] = None):
        """Add constraint that this layer_op can only be executed on the device set.

        Note:
            this will also apply constraint on tensor/data parallelism size.
        """
        self.devices = tuple(devices) if devices is not None else devices
        if devices is not None:
            max_size = len(devices)
            # update tensor parallelism constraint
            if self.max_tp is None:
                self.max_tp = max_size
            else:
                self.max_tp = min(max_size, self.max_tp)
            if self.min_tp > self.max_tp:
                raise RuntimeError(f"Adding constraint: executing on {max_size} devices, "
                                   f"but got minimal tensor parallelism size of {self.min_tp}")
            # update data parallelism constraint
            if self.max_dp is None:
                self.max_dp = max_size
            else:
                self.max_dp = min(max_size, self.max_dp)
            if self.min_dp > self.max_dp:
                raise RuntimeError(f"Adding constraint: executing on {max_size} devices, "
                                   f"but got minimal data parallelism size of {self.min_dp}")

    def __repr__(self):
        nids = f'[{self.nodes[0].cid}-{self.nodes[-1].cid}]' if len(self.nodes) > 0 else '[]'
        return f'Block(lid={self.bid}, nnodes={self.nnodes}, nids={nids})'


def blocking(fnodes: List[IRFwOperation], max_num: int) -> List[IRBlock]:
    """Group operators into IRBlocks.

    Note the segments will be considered as a single IRBlock. For other operators
    that are not in IRSegment, they will be grouped using IRGraphAnchor.
    
    .. todo:: Support auto-blocking without IRGraphAnchor 

    Args:
        graph (IRGraph): the graph
        max_num (int): number of maximal layers.
    
    Returns:
        blocks (List[IRBlock]): list of IRBlock
    """
    blocks: List[IRBlock] = []
    segment_blocks: List[IRBlock] = []
    for group in more_itertools.split_before(fnodes, lambda n: isinstance(n, (IRGraphAnchor, IRSegment))):
        while len(group) > 0 and isinstance(group[0], IRSegment):
            if isinstance(group[0], IRSegment):
                node = group.pop(0)
                blk = IRBlock((node,))
                blocks.append(blk)
                segment_blocks.append(blk)
        if len(group) == 0: continue
        blocks.append(IRBlock(group))
    assert sum(blk.nnodes for blk in blocks) == len(fnodes)
    # shrink the blocks to match the max_num
    nblocks = len(blocks)
    merged = 0
    while merged < nblocks - max_num:
        have_merged = False
        for block in list(blocks[::2]):
            idx = blocks.index(block)
            if idx + 1 >= len(blocks):
                continue
            lhs, rhs = blocks[idx], blocks[idx+1]
            if lhs.nnodes == 1 and isinstance(lhs.node(0), IRSegment):
                continue
            if rhs.nnodes == 1 and isinstance(rhs.node(0), IRSegment):
                continue
            blk = IRBlock(lhs.nodes + rhs.nodes)
            blocks[idx] = blk
            blocks.pop(idx+1)
            merged += 1
            have_merged = True
            if merged >= nblocks - max_num: break
        if not have_merged:
            _logger.warning(f'fail to merge blocks due to user constraints. The search time may increase.')
            break
    if nblocks > max_num:
        _logger.info(f'shrink search space by considering {len(blocks)} blocks (originally {nblocks})')

    if len(blocks) == 1:
        _logger.warning(f'Detected only one block, this may due to lack of IRGraphAnchor. '
                        f'In this case, only SPMD parallelism is applied.')

    # setup block ids
    for idx, blk in enumerate(blocks):
        blk.bid = idx

    # setup constraints
    for segblk in segment_blocks:
        devices = segblk.node(0).device
        if len(devices) > 0:
            segblk.constrain_devices(tuple(devices))
    
    return blocks