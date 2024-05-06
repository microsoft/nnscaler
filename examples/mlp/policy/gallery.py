# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import List
from cube.graph import IRGraph
from cube.graph.segment import IRSegment
from cube.ir.operator import IRDataOperation, IRFwOperation
from cube.graph.schedule.predefined import PredefinedSched

from examples.utils import tensor_parallelism, replica, create_mesh


def PASSingle(graph: IRGraph, resource, **kwargs):
    """Single device"""
    assert resource.ngpus == 1, "only apply for single gpu case"
    for node in graph.nodes():
        if isinstance(node, (IRDataOperation, IRFwOperation)):
            graph.assign(node, 0)
    return graph


def PASData(graph: IRGraph, resource, **kwargs):
    """Data Parallellism"""
    devs = list(range(resource.ngpus))
    for node in graph.select(ntype=IRFwOperation):
        tensor_parallelism(graph, node, idx=0, dim=0, devs=devs)
    for node in graph.select(ntype=IRDataOperation):
        replica(graph, node, devs=devs)
    return graph


def PASCol(graph: IRGraph, resource, **kwargs):
    """Linear Column Parallel"""
    devs = list(range(resource.ngpus))
    for node in graph.select(name='linear'):
        tensor_parallelism(graph, node, idx=1, dim=0, devs=devs)
    for node in graph.select(ntype=(IRFwOperation, IRDataOperation)):
        if len(node.device) == 0:
            replica(graph, node, devs=devs)
    return graph


def PASRow(graph: IRGraph, resource, **kwargs):
    """Linear Row Parallel"""
    devs = list(range(resource.ngpus))
    for node in graph.select(name='linear'):
        tensor_parallelism(graph, node, idx=0, dim=1, devs=devs)
    for node in graph.select(ntype=(IRFwOperation, IRDataOperation)):
        if len(node.device) == 0:
            replica(graph, node, devs=devs)
    return graph


def PASMegatronTP(graph: IRGraph, resource, **kwargs):
    """Linear Hybrid Parallelism (Megatron)"""
    devs = list(range(resource.ngpus))
    for idx, node in enumerate(graph.select(name='linear')):
        tensor_parallelism(graph, node, idx=1, dim=idx%2, devs=devs)
    for node in graph.select(ntype=(IRFwOperation, IRDataOperation)):
        if len(node.device) == 0:
            replica(graph, node, devs=devs)
    return graph


def PASMegatron(graph: IRGraph, resource, nmicros: int, tp_size: int,  **kwargs):

    num_stages = resource.ngpus // tp_size
    _, tp_mesh = create_mesh(resource.ngpus, (num_stages, tp_size))

    # group to sub-graphs
    linears = graph.select(name='linear')
    stage_start_nodes = linears[::len(linears) // num_stages][:num_stages]
    graph.staging(stage_start_nodes)

    segments = graph.select(ntype=IRSegment, flatten=False)
    fsegs = [seg for seg in segments if seg.isfw()]

    for sid, segment in enumerate(fsegs):
        # get tensor parallel group
        tp_group = tp_mesh[sid]
        for idx, node in enumerate(segment.nodes()):
            if node.name == 'linear':
                tensor_parallelism(graph, node, idx=1, dim=idx%2, devs=tp_group)
            else:
                replica(node, devs=tp_group)
    
    for dl in graph.select(ntype=IRDataOperation):
        replica(dl, devs=list(range(resource.ngpus)))

    PredefinedSched.sched_1f1b(graph, nmicros, num_stages)
    return graph

