# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""GPT policy gallery for MPMD Parallelism"""
from cube.graph import IRGraph
from cube.graph.segment import IRSegment
from cube.ir.operator import IRDataOperation, IRFwOperation
from cube.graph.schedule.predefined import PredefinedSched

from examples.utils import create_mesh, tensor_parallelism, replica, group_to_layers


def PASRoundRobin(graph: IRGraph, resource, **kwargs):
    """
    roundrobin scheduling
    """
    fnodes = [node for node in graph.nodes() if isinstance(node, IRFwOperation)]
    
    # group to transformer layers
    transformers = group_to_layers(fnodes)
    
    for lid, transformer in enumerate(transformers):
        stage_id = lid % resource.ngpus
        print(f'assigning {lid} transformer to stage {stage_id}')
        for node in transformer:
            graph.assign(node, stage_id)

    for node in graph.nodes():
        if len(node.device) == 0:
            graph.assign(node, 0)

    return graph


def PAS1F1B(graph: IRGraph, resource, nmicros: int = 16, **kwargs):
    """1F1B schedule"""
    num_stages = resource.ngpus
    num_microbatch = nmicros

    # group to transformer layers
    fnodes = [node for node in graph.nodes() if isinstance(node, IRFwOperation)]
    transformers = group_to_layers(fnodes)
    assert len(transformers) >= num_stages

    # staging
    fstages = [[] for _ in range(num_stages)]
    nlayer_per_stage = (len(transformers) // resource.ngpus)
    for lid, fnodes in enumerate(transformers):
        stage_id = min(lid // nlayer_per_stage, num_stages - 1)
        fstages[stage_id] += fnodes  
    graph.staging(tuple(stages[0] for stages in fstages))

    # stage to device
    fsegments = [seg for seg in graph.nodes() if isinstance(seg, IRSegment) and seg.isfw()]
    assert len(fsegments) == num_stages
    for devid, segment in enumerate(fsegments):
        graph.assign(segment, devid)
    
    for node in graph.nodes():
        if isinstance(node, IRDataOperation):
            graph.assign(node, 0)

    if graph.train:
        PredefinedSched.sched_1f1b(graph, num_microbatch, num_stages)
    else:
        PredefinedSched.sched_infer_pipe(graph, num_microbatch, num_stages)
    return graph


def PASMegatron(graph: IRGraph, resource,
                tp_size: int = 2, dp_size: int = 1,
                nmicros: int = 16, **kwargs ):
    """Megatron policy for hybrid data-tensor-pipeline parallelism"""
    pp_size = resource.ngpus // (dp_size * tp_size)
    num_microbatch = nmicros

    # device mesh
    dp_groups, pp_groups, tp_groups = \
        create_mesh(resource.ngpus, (dp_size, pp_size, tp_size))
    print(f'dp groups: {dp_groups}')
    print(f'pp groups: {pp_groups}')
    print(f'tp groups: {tp_groups}')

    def get_device(dp_idx: int, pp_idx: int, tp_idx: int, ) -> int:
        return tp_groups[dp_idx * pp_size + pp_idx][tp_idx]

    # group to transformer layers
    transformers = group_to_layers(graph.select(ntype=IRFwOperation))

    # group to stage: set each stage operators
    fstages = [[] for _ in range(pp_size)]
    nlayer_per_stage = (len(transformers) // pp_size)
    for lid, fnodes in enumerate(transformers):
        stage_id = min(lid // nlayer_per_stage, pp_size - 1)
        fstages[stage_id] += fnodes  
    graph.staging(tuple(stages[0] for stages in fstages))

    dataloader = graph.select(ntype=IRDataOperation)[0]
    bs = dataloader.output(0).shape[0]

    # partition dataloader
    dls = replica(graph, dataloader, [0]*dp_size) # graph.partition(dataloader, dataloader.algorithms('data'), num=dp_size)
    for dp_idx, dl in enumerate(dls):
        # only stage 0 needs dataloader
        devices = [get_device(dp_idx, 0, tp_idx) for tp_idx in range(tp_size)]
        replica(graph, dl, devices)
    
    fstages = [stage for stage in graph.select(ntype=IRSegment, flatten=False) if stage.isfw()]
    assert len(fstages) > 0
    for pp_idx, fstage in enumerate(fstages):
        for fnode in fstage.nodes():
            if len(fnode.inputs()) == 0: continue # anchor
            if fnode.name == 'self_attention' or fnode.name == 'feedforward':
                fnodes = tensor_parallelism(graph, fnode, idx=1, dim=0, devs=[0]*tp_size)
            elif fnode.name == 'embedding':
                fnodes = tensor_parallelism(graph, fnode, idx=1, dim=0, devs=[0]*tp_size)
            elif fnode.name == 'linear': # the last embeding linear
                fnodes = tensor_parallelism(graph, fnode, idx=1, dim=0, devs=[0]*tp_size)
            elif fnode.name == 'sum':
                fnodes = tensor_parallelism(graph, fnode, idx=0, dim=2, devs=[0]*tp_size)
            else:
                fnodes = replica(graph, fnode, [0]*tp_size)
            # data parallel
            for tp_idx, fnode in enumerate(fnodes):
                dp_devices = [get_device(dp_idx, pp_idx, tp_idx) for dp_idx in range(dp_size)]
                batch_dim = fnode.input(0).shape.index(bs)
                tensor_parallelism(graph, fnode, idx=0, dim=batch_dim, devs=dp_devices)
    PredefinedSched.sched_1f1b(graph, num_microbatch, pp_size)
    return graph
