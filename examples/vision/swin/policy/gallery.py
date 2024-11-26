#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from typing import List

import more_itertools as mitr
import itertools

from nnscaler import ComputeConfig
from nnscaler.graph import IRGraph
from nnscaler.graph.function.anchor import IRGraphAnchor
from nnscaler.graph.schedule.predefined import PredefinedSched
from nnscaler.graph.segment import IRSegment
from nnscaler.ir.operator import IRBpOperation, IRDataOperation, IRFwOperation

from examples.utils import tensor_parallelism, replica, group_to_layers

import logging
_logger = logging.getLogger(__name__)


def coshard(graph: IRGraph, node: IRFwOperation, devs: List[int], colocate: int,
             idx: int, dim: int):
    algo = node.algorithms('dim')
    sub_nodes = graph.partition(node, algo, idx=idx, dim=dim, num=colocate*len(devs))
    assert sub_nodes is not None
    graph.recompute(sub_nodes)
    for devid in devs:
        for coid in range(colocate):
            sub_node = sub_nodes[devid * colocate + coid]
            graph.assign(sub_node, devid)
    return sub_nodes


def pas_megatron(graph: IRGraph, cfg: ComputeConfig):
    """Megatron-way tensor parallelism"""
    devs = list(range(cfg.plan_ngpus))

    # skip mutliref because the partition of tensors in transformer are the same for all tensors

    # attention
    for attn in graph.select(name='window_attn'):
        tensor_parallelism(graph, attn, idx=1, dim=0, devs=devs)
    # feedforward
    for ffn in graph.select(name='feedforward'):
        tensor_parallelism(graph, ffn, idx=1, dim=0, devs=devs)

    # replicate other nodes
    for node in graph.select(ntype=(IRFwOperation, IRDataOperation)):
        if len(node.device) == 0:
            replica(graph, node, devs)
    return graph


def pas_mesh_shard(graph: IRGraph, cfg: ComputeConfig):
    """
    Coshard policy example

    It will partition a tensor `colocate*plan_ngpus` subtensors,
        and each device will have `colocate` subtensors.

    This can save GPU memory when work with recompute
    """
    devs = list(range(cfg.plan_ngpus))

    # skip mutliref because the partition of tensors in transformer are the same for all tensors

    # attention
    for attn in graph.select(name='window_attn'):
        # _tp(graph, attn, tp_devs, idx=1, dim=0)
        coshard(graph, attn, devs, colocate=2, idx=1, dim=0)
    # feedforward
    for ffn in graph.select(name='feedforward'):
        # _tp(graph, ffn, tp_devs, idx=1, dim=0)
        coshard(graph, ffn, devs, colocate=4, idx=1, dim=0)

    # replicate other nodes
    for node in graph.select(ntype=(IRFwOperation, IRDataOperation)):
        if len(node.device) == 0:
            replica(graph, node, devs)
    return graph


def pas_1f1b(graph: IRGraph, cfg: ComputeConfig):
    """1F1B schedule"""
    num_stages = cfg.pas_config['pipeline_nstages']
    nmicros = cfg.pas_config['pipeline_nmicros']
    scheduler = cfg.pas_config.get('pipeline_scheduler', '1f1b')
    if num_stages != cfg.plan_ngpus:
        raise ValueError('1F1B schedule requires num_stages == plan_ngpus')

    # group to transformer layers
    transformers = group_to_layers(graph.select(ntype=IRFwOperation))
    stages = mitr.divide(num_stages, transformers)
    stages = [list(itertools.chain(*s)) for s in stages]
    graph.staging([t[0] for t in stages])

    # staging
    stages: List[IRSegment] = graph.select(ntype=IRSegment, flatten=False)
    stages = [s for s in stages if s.isfw()]

    for idx, stage in enumerate(stages):
        for fnode in stage.nodes():
            graph.assign(fnode, idx)

    # replicate dataloader
    for node in graph.select(ntype=IRDataOperation):
        replica(graph, node, list(range(cfg.plan_ngpus)))
    # apply 1f1b schedule
    cfg.apply_pipeline_scheduler(graph, num_stages, nmicros, scheduler)
    return graph
