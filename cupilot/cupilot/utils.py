# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Tuple, Optional, List, Dict
import more_itertools as mitr

from cube.graph import IRGraph
from cube.graph.segment import IRSegment
from cube.graph.function import IRGraphAnchor
from cube.ir.operator import IRFwOperation
from cube.graph.function.dimops import TransformRule

from .plan.plan import ParallelSpec
from .solver.block import IRBlock

import logging

_logger = logging.getLogger(__name__)


def auto_multiref(graph: IRGraph, plan: ParallelSpec):
    """automated multiref for tensors that are partitioned differently
    
    Warning:
        this may not work if user already partitions some operators
        that doesn't align the parallel spec plan.
    """
    # get parallel strategy
    node2config = dict()
    for stage in plan.stages:
        for cid, split in stage.tp_spec.items():
            node2config[cid] = split

    segments = graph.select(ntype=IRSegment, flatten=False)
    for ftensor in graph.full_tensors():
        if ftensor.is_grad(): continue
        if len(graph.consumers(ftensor)) <= 1: continue
        consumers, ctensors = graph.consumers(ftensor), graph.ctensors(ftensor)
        splits = set()
        for consumer, ctensor in zip(consumers, ctensors):
            if consumer.cid not in node2config:
                splits.add(None)
            else:
                spec = node2config[consumer.cid]
                if spec is None:
                    splits.add(None)
                else:
                    idx, dim = spec
                    rule: TransformRule = consumer.algorithms('dim').infer(idx, dim, 1)
                    split = rule.inputs()[consumer.inputs().index(ctensor)]
                    splits.add(split)
        if len(splits) > 1 and ftensor.requires_grad:
            _logger.info(f"apply multiref for tensors {ftensor.tid}({ftensor.shape}) of nodes: "
                         f"{', '.join(c.name for c in consumers)}")
            graph.multiref(ftensor)

    for segment in segments:
        auto_multiref(segment, plan)


def get_anchors(graph: IRGraph, nodes: Tuple[IRFwOperation]) -> Tuple[Optional[IRGraphAnchor]]:
    """Get the last IRGraphAnchor node that executes before each node
    
    Args:
        graph (IRGraph): the graph
        nodes (List[IRFwOperation]): forward nodes

    Returns:
        Tuple[IRGraphAnchor or None]: the closest IRGraphAnchor before each node
    """
    anchors = [None] * len(nodes)
    last_anchor = None
    for node in graph.select(ntype=IRFwOperation, flatten=False):
        if node in nodes:
            idx = nodes.index(node)
            anchors[idx] = last_anchor
        if isinstance(node, IRGraphAnchor):
            last_anchor = node
    return anchors


def stage_blocks(graph: IRGraph, blocks: List[IRBlock], spec: ParallelSpec) -> List[List[IRSegment]]:
    """Group forward operators into segments following parallel spec"""

    blocks = [blk for blk in blocks if not blk.standalone]
    fnodes = mitr.flatten(blk.nodes for blk in blocks)
    nstages = len(spec.stages)

    devs2sidx: Dict[int, int] = {}
    curr_ndevs = 0
    for sidx, stage in enumerate(spec.stages):
        stage_ndevs = stage.tp_size * stage.dp_size
        stage_devs = tuple(idx + curr_ndevs for idx in range(stage_ndevs))
        devs2sidx[stage_devs] = sidx
        curr_ndevs += stage_ndevs

    node2sidx: Dict[int, int] = {}
    for sidx, stage in enumerate(spec.stages):
        for cid in stage.tp_spec:
            node2sidx[cid] = sidx

    block2sidx: Dict[IRBlock, int] = {}
    for block in blocks:
        if block.standalone:
            continue
        for node in block.nodes:
            sidx = node2sidx.get(node.cid, None)
            if sidx is not None:
                block2sidx[block] = sidx
                break
        else:
            raise KeyError(f'cannot find any node in block {block} that is in parallel spec')
    
    for block in blocks:
        sidx = block2sidx[block]
        for node in block.nodes:
            node2sidx[node.cid] = sidx
    
    sidx2nodes: Dict[int, List] = {}
    group = mitr.bucket(fnodes, lambda n: node2sidx[n.cid])
    for sidx in range(nstages):
        sidx2nodes[sidx] = list(group[sidx])
    
    stages: List[IRSegment] = []
    for sidx in range(nstages):
        snodes = sidx2nodes[sidx]
        if len(snodes) == 0:
            continue
        segment = graph.group(snodes)
        stages.append(segment)

    remain_fwops = graph.select(ntype=IRFwOperation, flatten=False)
    assert len(remain_fwops) == 0, \
        f'expected no IRFwops outside, but found {remain_fwops}'
    assert len(stages) == nstages, \
        f'expected {nstages} stages, but found {len(stages)}'
    return stages
