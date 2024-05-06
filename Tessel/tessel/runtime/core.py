# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Piper policy

https://openreview.net/attachment?id=-U9I0f2S7W&name=supplementary_material

The implementation is a little bit adapted to fit with cube's view
"""
from typing import List, Callable

from cube.graph.function.anchor import IRGraphAnchor
from cube.graph import IRGraph
from cube.graph.segment import IRSegment

from .estimator import Estimator
from .placement.stage import StageSolver, ParallelSpec
from .placement.spmd import SpmdSolver
from .placement.block import IRBlock
from .config import TesselConfig


def staged_spmd(blocks: List[IRBlock],
                num_devices: int,
                mem_limit: int,
                config: TesselConfig):
    """Search for best placement

    Args:
        graph (IRGraph)
        resource (EnvResource)
        mbs (int): micro-batch size
        nmicros (int): number of microbatches
        recompute (bool): whether perform recompute stage-wisely
        tp_func (Callable): expert function of applying tensor parallelism,
            which takes graph, and node and devices as input

    Returns:
        ParallelSpec
    """
    estimator = Estimator(config.db_cache)
    spmd_solver = SpmdSolver(estimator, config.recompute, True, config.param_limit_gb)
    print(f'> search [initialize]: profiling model...')

    fnodes = []
    for block in blocks:
        fnodes += list(block.nodes)
    latency, memory  = estimator(fnodes)
    print(f'> search [estimation]: single device latency: {latency} ms, memory: {memory/1024/1024/1024} GB')
    # save profiled database
    print(f'> search [dump]: saving profiled database...')
    estimator.save()

    stage_solver = StageSolver(spmd_solver,
                               config.max_dp_size,
                               config.max_tp_size,
                               config.max_pp_size,
                               config.min_pp_size)
    parallel_spec: ParallelSpec = stage_solver.solve(blocks, num_devices, mem_limit)
    assert parallel_spec is not None, f"no solution"
    return parallel_spec


def instantiate(graph: IRGraph,
                num_devices: int,
                fsegments: List[IRSegment],
                spec: ParallelSpec,
                tp_func: Callable):
    """Instantiate the graph following the parallel spec"""
    
    assert len(fsegments) == len(spec.stages)
    devices = list(range(num_devices))
    for sid, segment in enumerate(fsegments):
        stage = spec.stages[sid]
        dp, tp = stage.dp_size, stage.tp_size
        stage_devices, devices = devices[:dp*tp], devices[dp*tp:]
        # apply data parallelism
        if dp > 1:
            raise NotImplementedError("Only support data parallelism size to be 1")
        # apply tensor parallelism
        print(f'> applying tp={tp}, dp={dp} for stage {sid}')
        for node in segment.nodes():
            if node.name == 'multiref' or isinstance(node, IRGraphAnchor):
                continue
            sub_nodes = tp_func(graph, node, stage_devices)
            for sub_node in sub_nodes:
                assert len(sub_node.device) > 0, f"tp_func should assign devices to the node"

    assert len(devices) == 0, f'not all devices are used (remaining {len(devices)})'
    return graph
