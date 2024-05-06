# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Optional, List, Tuple
import logging

from cube.graph.graph import IRGraph
from cube.ir.operator import IRFwOperation

SubGraph = Tuple[IRFwOperation]

_logger = logging.getLogger(__name__)


def tensor_parallelism(graph: IRGraph, node: IRFwOperation, 
                       idx: int, dim: int, devs: List[int]):
    """Tensor parallelism"""
    sub_nodes = [node] if len(devs) == 1 \
        else graph.partition(node, node.algorithms('dim'), idx=idx, dim=dim, num=len(devs))
    for devid, sub_node in zip(devs, sub_nodes):
        graph.assign(sub_node, devid)
    return sub_nodes


def nested_tensor_parallelism(graph: IRGraph, node: IRFwOperation,
                              idxs: Tuple[Optional[int]], dims: Tuple[Optional[int]], nums: Tuple[int],
                              devs: List[int]) -> List[IRFwOperation]:
    """Nested tensor parallelism"""
    sub_nodes = [node]
    for (idx, dim, num) in zip(idxs, dims, nums):
        for _ in range(len(sub_nodes)):
            node = sub_nodes.pop(0)
            if idx is None:
                # replicate
                pnodes = [node] if num == 1 else graph.replicate(node, times=num)
            else:
                # partition
                pnodes = [node] if num == 1 else graph.partition(node,
                    node.algorithms('dim'), idx=idx, dim=dim, num=num)
            sub_nodes += pnodes
    assert len(sub_nodes) == len(devs)
    for sub_node, device in zip(sub_nodes, devs):
        graph.assign(sub_node, device)
    return sub_nodes


def replicate(graph: IRGraph, node: IRFwOperation, devs: List[int]):
    """Replicate parallelism"""
    sub_nodes = [node] if len(devs) == 1 else graph.replicate(node, len(devs))
    for devid, sub_node in zip(devs, sub_nodes):
        graph.assign(sub_node, devid)
    return sub_nodes