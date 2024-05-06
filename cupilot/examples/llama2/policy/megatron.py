from typing import List

from cube.graph.graph import IRGraph
from cube.graph.segment import IRSegment
from cube.graph.function.anchor import IRGraphAnchor
from cube.ir.operator import IRFwOperation
from cube.ir.operator import IRDataOperation
from cube.graph.schedule.predefined import PredefinedSched

from cupilot.parallel.spmd import nested_tensor_parallelism, replicate

import more_itertools as mitr


def megatron_policy(graph: IRGraph, resource,
                    nmicros: int,
                    tp_size: int,
                    pp_size: int,
                    recompute: bool) -> IRGraph:
    """Megatron policy"""
    print(f'> megatron policy config: tp={tp_size}, pp={pp_size}, recompute={recompute}')
    dp_size = resource.ngpus // (pp_size * tp_size)
    assert dp_size * pp_size * tp_size == resource.ngpus

    fnodes = graph.select(ntype=IRFwOperation)
    layers = list(mitr.split_before(fnodes, lambda n: isinstance(n, IRGraphAnchor)))
    if recompute:
        for layer in layers:
            graph.recompute(tuple(layer))
    assert len(layers) > 0
    head, transformers, tail = layers[0], layers[1:-1], layers[-1]
    transformers[0] = head + transformers[0]
    transformers[-1] = transformers[-1] + tail
    layers = transformers
    print(f'> group forward nodes into {len(layers)} layers')

    # pipeline stage assignment
    stages = mitr.divide(pp_size, layers)
    stages = [tuple(s) for s in stages]
    print(f'> pipeline layer number: {[len(s) for s in stages]}')
    stages = [tuple(mitr.flatten(layers)) for layers in stages]
    graph.staging([s[0] for s in stages])
    stages: List[IRSegment] = \
        [seg for seg in graph.select(ntype=IRSegment, flatten=False) if seg.isfw()]
    assert len(stages) == pp_size

    # replicate dataloader
    devices = tuple(range(resource.ngpus))
    for dl in graph.select(ntype=IRDataOperation):
        replicate(graph, dl, devices)
    
    # tensor parallelism + data parallelism
    for stage in stages:
        sdevs = devices[:dp_size * tp_size]
        for node in stage.nodes():
            if node.name == 'attention':
                idxs, dims = (0, 2), (0, 0)
            elif node.name == 'mlp':
                idxs, dims = (0, 1), (0, 0)
            else:
                idxs, dims = (0, None), (0, None)
            nested_tensor_parallelism(graph, node, idxs, dims,
                                      (dp_size, tp_size), sdevs)
        devices = devices[dp_size * tp_size:]
    assert len(devices) == 0
    PredefinedSched.sched_1f1b(graph, nmicros, len(stages))
    return graph
