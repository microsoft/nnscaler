# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Callable, List, Union, Dict, Tuple
import more_itertools as mitr

from cube.graph.schedule.predefined import PredefinedSched

from cube.ir.operator import IRFwOperation, IRDataOperation
from cube.graph.graph import IRGraph, IRSegment
from cube.graph.function import IRGraphAnchor
from cube.graph.schedule.schedplan import SchedulePlan as CSched
from cube.graph.function.dimops import IRDimops
from cube.ir.cten import IRCell

from tessel.schedule.schedplan import SchedPlan as TSched
from tessel.schedule.schedplan import Block as TBlock
from .config import TesselConfig


def _recompute(graph: IRGraph):
    """Apply recompute for each layer"""
    fnodes = graph.select(ntype=IRFwOperation)
    layers = mitr.split_before(fnodes, lambda n: isinstance(n, IRGraphAnchor))
    for layer in layers:
        graph.recompute(list(layer))


def replica(graph: IRGraph, node: IRCell, devs: List[int]) -> List[IRDimops]:
    """Replicate a node"""
    sub_nodes = [node] if len(devs) == 1 else graph.replicate(node, len(devs))
    for devid, sub_node in zip(devs, sub_nodes):
        graph.assign(sub_node, devid)
    return sub_nodes


def tensor_parallelism(graph: IRGraph, node: IRDimops, idx: int, dim: int, devices: Tuple[int]) -> List[IRDimops]:
    """Tensor parallelism on a node"""
    sub_nodes = [node] if len(devices) == 1 \
        else graph.partition(node, node.algorithms('dim'), idx=idx, dim=dim, num=len(devices))
    for devid, sub_node in zip(devices, sub_nodes):
        graph.assign(sub_node, devid)
    return sub_nodes


def PASFullTP(graph: IRGraph,
              resource,
              mbs: int,
              nmicros: int,
              tp_func: Callable,
              config: TesselConfig) -> IRGraph:
    
    config = TesselConfig() if config is None else config

    if config.recompute:
        _recompute(graph)
    
    devices = tuple(range(resource.ngpus))
    for fnode in graph.select(ntype=IRFwOperation):
        if fnode.name == 'multiref' or isinstance(fnode, IRGraphAnchor):
            continue
        tp_func(graph, fnode, devices)
    
    # replicate dataloader to all devices
    for dl in graph.select(ntype=IRDataOperation):
        if len(dl.device) == 0:
            replica(graph, dl, list(range(resource.ngpus)))
    return graph


def PAS1F1B(graph: IRGraph,
            resource,
            mbs: int,
            nmicros: int,
            premise: Callable,
            config: TesselConfig,
            sched = '1f1b') -> IRGraph:

    config = TesselConfig() if config is None else config
    config.max_dp_size = mbs if config.max_dp_size is None \
        else min(mbs, config.max_dp_size)
    
    mem_limit = resource.gpus[0].memory - 2 * 1024 * 1024 * 1024  # reserve 2GB
    print(f'> memory limit: {mem_limit} bytes')

    if config.recompute:
        _recompute(graph)

    premise(graph, resource.ngpus, mbs, mem_limit, config)
    assert not any(isinstance(node, IRFwOperation) for node in graph.nodes()), \
        "Premise should call graph.blocking() or graph.staging()"

    segments = graph.select(ntype=IRSegment, flatten=False)
    fsegments: List[IRSegment] = [seg for seg in segments if seg.isfw()]

    # replicate dataloader to all devices
    for dl in graph.select(ntype=IRDataOperation):
        if len(dl.device) == 0:
            replica(graph, dl, list(range(resource.ngpus)))
    
    assert all(len(seg.device) < resource.ngpus for seg in fsegments)

    if not graph.train:
        PredefinedSched.sched_infer_pipe(graph, nmicros, len(fsegments))
    else:
        if sched == '1f1b':
            PredefinedSched.sched_1f1b(graph, nmicros, len(fsegments))
        elif sched == 'gpipe':
            PredefinedSched.sched_gpipe(graph, nmicros, len(fsegments))
        else:
            raise RuntimeError
    return graph


def PAS1F1BPlus(graph: IRGraph,
                resource,
                mbs: int,
                nmicros: int,
                premise: Callable,
                config: TesselConfig,) -> IRGraph:
    
    config = TesselConfig() if config is None else config
    config.max_dp_size = mbs if config.max_dp_size is None \
        else min(mbs, config.max_dp_size)

    if config.recompute:
        _recompute(graph)
    
    mem_limit = resource.gpus[0].memory - 2 * 1024 * 1024 * 1024  # reserve 2GB
    print(f'> memory limit: {mem_limit} bytes')

    premise(graph, resource.ngpus, mbs, mem_limit, config)
    assert not any(isinstance(node, IRFwOperation) for node in graph.nodes()), \
        "Premise should call graph.blocking() or graph.staging()"
    
    # replicate dataloader to all devices
    for dl in graph.select(ntype=IRDataOperation):
        if len(dl.device) == 0:
            replica(graph, dl, list(range(resource.ngpus)))

    segments = graph.select(ntype=IRSegment, flatten=False)
    fsegments: List[IRSegment] = [seg for seg in segments if seg.isfw()]
    nstages = len([seg for seg in fsegments if len(seg.device) < resource.ngpus])
    PredefinedSched.sched_1f1b_plus(graph, nmicros, nstages)
    return graph


def PASChimera(graph: IRGraph,
               resource,
               mbs: int,
               nmicros: int,
               premise: Callable,
               config: TesselConfig,) -> IRGraph:
    """Chimera Direct policy"""
    config = TesselConfig() if config is None else config
    config.max_dp_size = mbs if config.max_dp_size is None \
        else min(mbs, config.max_dp_size)

    if config.recompute:
        _recompute(graph)

    mem_limit = resource.gpus[0].memory - 2 * 1024 * 1024 * 1024  # reserve 2GB
    print(f'> memory limit: {mem_limit} bytes')

    premise(graph, resource.ngpus, mbs, mem_limit, config)
    assert not any(isinstance(node, IRFwOperation) for node in graph.nodes()), \
        "Premise should call graph.blocking() or graph.staging()"

    segments = graph.select(ntype=IRSegment, flatten=False)
    fsegments: List[IRSegment] = [seg for seg in segments if seg.isfw()]
    assert len(fsegments) == 4

    # replicate dataloader to all devices
    for dl in graph.select(ntype=IRDataOperation):
        if len(dl.device) == 0:
            replica(graph, dl, list(range(resource.ngpus)))

    PredefinedSched.sched_chimera_direct(graph, nmicros, len(fsegments))
    return graph


def _create_tblocks(graph: IRGraph):
    segments = graph.select(ntype=IRSegment, flatten=False)
    assert len(segments) > 0
    blocks = []
    for segment in segments:
        if segment.isfw():
            blocks.append(TBlock(0, span=1, memory=1, btype="forward"))
        else:
            blocks.append(TBlock(0, span=1, memory=-1, btype="backward"))
    for idx, blk in enumerate(blocks):
        blk.gid = idx
    # setup dependencies
    for idx1, blk1 in enumerate(blocks):
        for idx2, blk2 in enumerate(blocks):
            if idx1 == idx2: continue
            if graph.depends(segments[idx1], segments[idx2]):
                TBlock.make_dependency(blk1, blk2)
            if segments[idx1].isfw() and segments[idx1].mirror is segments[idx2]:
                TBlock.make_dependency(blk1, blk2)
    return blocks


def _schedule(graph: IRGraph, tsched: Union[str, TSched], nmicros: int, blk2seg: Dict[TBlock, IRSegment]) -> CSched:
    """
    Translate a searched schedplan of Tessel into Cube SchedulePlan runtime

    Args:
        graph (IRGraph): staged IRGraph
        schedplan (Union[TSched, str]): Tessel SchedPlan instance or file (saved in json format)
    
    Returns:
        CSched: cube schedule plan
    """
    tsched: TSched = tsched if isinstance(tsched, TSched) else TSched.load(tsched)
    # unroll the plan
    tsched = tsched.unroll(nmicros)
    csched = CSched(graph, nmicros)
    for step in range(tsched.nsteps):
        tblocks = tsched.blocks(step)
        for tblock in tblocks:
            csched.add_segment(blk2seg[tblock.gid], tblock.mid, step, tblock.span)
    csched.finish()
    return csched


def PASTessel(graph: IRGraph,
              resource,
              mbs: int,
              nmicros: int,
              premise: Callable,
              config: TesselConfig,
              load_sched: str) -> IRGraph:
    """policy entry for tessel.

    Args:
        graph (IRGraph)
        resource (EnvResource)
        mbs (int): micro-batch size
        nmicros (int): number of micro-batches
        premise (Callable): function to determine the graph.
            It takes inputs of (graph, num_devices, mem_limit, config)

    Returns:
        IRGraph
    """
    config = TesselConfig() if config is None else config
    config.max_dp_size = mbs if config.max_dp_size is None \
        else min(mbs, config.max_dp_size)

    if config.recompute:
        _recompute(graph)
    
    mem_limit = resource.gpus[0].memory - 2 * 1024 * 1024 * 1024  # reserve 2GB
    print(f'> memory limit: {mem_limit} bytes')

    premise(graph, resource.ngpus, mbs, mem_limit, config)
    assert not any(isinstance(node, IRFwOperation) for node in graph.nodes()), \
        "Premise should call graph.blocking() or graph.staging()"

    # replicate dataloader to all devices
    for dl in graph.select(ntype=IRDataOperation):
        if len(dl.device) == 0:
            replica(graph, dl, list(range(resource.ngpus)))

    # assign block sub-graph index
    blocks = _create_tblocks(graph)
    segments = graph.select(ntype=IRSegment, flatten=False)
    block2seg: Dict[TBlock, IRSegment] = {}
    for block, segment in zip(blocks, segments):
        block2seg[block.gid] = segment

    print(f'> loading schedule plan from {load_sched}')
    tsched = TSched.load(load_sched)

    print(f'> get composed schedule:\n{tsched}')
    csched = _schedule(graph, tsched, nmicros, block2seg)
    return graph
