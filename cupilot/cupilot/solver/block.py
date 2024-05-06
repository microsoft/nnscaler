# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from typing import List, Optional, Tuple, Callable
import more_itertools as mitr
import logging

from cube.ir.cten import IRCell
from cube.ir.operator import IRFwOperation
from cube.graph.function.anchor import IRGraphAnchor
from cube.graph.graph import IRGraph
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

    def __init__(self, graph: IRGraph, nodes: Tuple[IRFwOperation], bid: int = None):
        self.graph: IRGraph = graph
        self.nodes = tuple(nodes)
        if not len(self.nodes) > 0:
            raise ValueError(f'IRBlock must contain at least one node')
        self.bid : int = bid

        self._standalone: bool = False
        self._segment: Optional[IRSegment] = None
        
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
    
    @property
    def standalone(self) -> bool:
        """Check whether this block will exclude from staged-spmd search. Default False."""
        return self._standalone
    
    def mark_standalone(self, fn: Callable):
        """Mark this block to exclude from staged-spmd search
        
        Args:
            fn (Callable): the function to be applied on the block
        """
        self._standalone = True
        segment = self.graph.group(self.nodes)
        fn(segment)
        for node in segment.nodes():
            if node.name == 'multiref' or isinstance(node, IRGraphAnchor):
                continue
            if len(node.device) == 0:
                raise RuntimeError(
                    f"Expected fn to completely partition and assign node."
                    f"But got un-assigned node {node.name}[{node.cid}]"
                )
        self._segment = segment

    @property
    def segment(self) -> Optional[IRSegment]:
        """Get segment for the standalone block"""
        return self._segment

    def node(self, idx: int) -> IRCell:
        """Get the node at index"""
        return self.nodes[idx]

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
        if max_size is not None:
            self.max_tp = max_size

    def constrain_dp_size(self, min_size: int = 1, max_size: Optional[int] = None):
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
        if max_size is not None:
            self.max_dp = max_size

    def constrain_devices(self, devices: Optional[Tuple[int]] = None):
        """Add constraint that this IRBlock can only be executed on the given device set.

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

    def __repr__(self):
        nids = f'[{self.nodes[0].cid}-{self.nodes[-1].cid}]' if len(self.nodes) > 0 else '[]'
        return f'Block(lid={self.bid}, nnodes={self.nnodes}, nids={nids})'

    @staticmethod
    def merge(blocks: List[IRBlock]) -> IRBlock:
        """Merge multiple blocks into one block
        
        All the constraints will be removed.

        Args:
            blocks (List[IRBlock]): list of IRBlock

        Returns:
            IRBlock: the merged IRBlock
        """
        blocks = list(blocks)
        if len(blocks) == 0:
            raise TypeError("Cannot merge zero blocks into one block")
        graph = blocks[0].graph
        nodes = []
        for blk in blocks:
            nodes += list(blk.nodes)
        return IRBlock(graph, tuple(nodes))

    @staticmethod
    def blocking(graph: IRGraph) -> List[IRBlock]:
        """Group operators into IRBlocks

        .. todo:: Support auto-blocking without IRGraphAnchor 

        Args:
            graph (IRGraph): the graph

        Returns:
            List[IRBlock]: list of IRBlock
        """
        fnodes = graph.select(ntype=IRFwOperation)
        blocks = []
        for group in mitr.split_before(fnodes, lambda n: isinstance(n, IRGraphAnchor)):
            if len(group) == 0: continue
            blocks.append(IRBlock(graph, tuple(group)))
        if len(blocks) == 1:
            _logger.warning(f'Detected only 1 block, this may due to lack of IRGraphAnchor. '
                            f'In this case, only SPMD parallelism is applied.')
        return blocks

    @staticmethod
    def shrink_blocks(blocks: List[IRBlock], max_block_num: int) -> List[IRBlock]:
        """Merge IRBlocks into coarser IRBlocks

        1) The consecutive blocks with same device constraints will be merged into
        one block.

        2) Then all other consecutive blocks without device constraints will be merged
        into one until the number of blocks is less than max_block_num.

        3) The standalone blocks will not be merged.
        
        Args:
            blocks (List[IRBlock]): list of IRBlock
            max_block_num (int): maximal number of IRBlocks

        Returns:
            List[IRBlock]: list of IRBlock
        """
        if len(blocks) == 0:
            return []
        graph = blocks[0].graph

        # shrink the blocks with same device constraints
        blocks = list(mitr.split_when(blocks, \
            lambda prev, curr: \
                (prev.standalone or curr.standalone) or \
                (prev.devices is None or prev.devices != curr.devices)))
        for idx in range(len(blocks)):
            if len(blocks[idx]) > 1:
                nodes = []
                for blk in blocks[idx]:
                    nodes += list(blk.nodes)
                blocks[idx] = [IRBlock(graph, nodes)]
        blocks = list(mitr.flatten(blocks))
        assert len(blocks) > 0

        # shrink the blocks to match the max_num
        nblocks = len(blocks)
        merged = 0
        while merged < nblocks - max_block_num:
            have_merged = False
            for block in list(blocks[::2]):
                idx = blocks.index(block)
                if idx + 1 >= len(blocks):
                    continue
                lhs, rhs = blocks[idx], blocks[idx+1]
                if lhs.standalone or rhs.standalone:
                    continue
                if (lhs.devices is not None) and \
                   (rhs.devices is not None) and \
                   lhs.devices != rhs.devices:
                    continue
                devices = lhs.devices if lhs.devices is not None else rhs.devices

                blk = IRBlock(graph, lhs.nodes + rhs.nodes)
                blk.min_tp = max(lhs.min_tp, rhs.min_tp)
                blk.max_tp = min(lhs.max_tp, rhs.max_tp)
                blk.min_dp = max(lhs.min_dp, rhs.min_dp)
                blk.max_dp = min(lhs.max_dp, rhs.max_dp)
                blk.devices = devices

                blocks[idx] = blk
                blocks.pop(idx+1)
                merged += 1
                have_merged = True
                if merged >= nblocks - max_block_num: break
            if not have_merged:
                _logger.warning(f'fail to merge blocks due to user constraints. The search time may increase.')
                break
        if nblocks > max_block_num:
            _logger.info(f'shrink search space by considering {len(blocks)} blocks (originally {nblocks})')

        if len([blk for blk in blocks if not blk.standalone]) == 1:
            _logger.warning(f'Detected only one non-standalone block, this may due to lack of IRGraphAnchor. '
                            f'In this case, only SPMD parallelism is applied.')

        # setup block ids
        for idx, blk in enumerate(blocks):
            blk.bid = idx

        return blocks
