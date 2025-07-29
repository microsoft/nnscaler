from .spmd_solver import calc_optimal_spmd_plan, analysis_pretty_printer
from .pipeline_solver import calc_optimal_pp_plan
from .autodist_config import AutoDistConfig
from .model_graph import ModelGraph, estimate_mem_lower_bound
from .descs import *
from .util import replica, partition_node

from nnscaler.graph import IRGraph
from nnscaler.graph.segment import IRSegment
from nnscaler.ir.operator import IRDataOperation, IRFwOperation
from nnscaler.graph.function import IRDimops
from nnscaler.graph.function.anchor import IRGraphAnchor
from nnscaler.graph.function.pyfunc import IRPyFunc
from nnscaler.graph.schedule.predefined import PredefinedSched

import json
import os
import logging
import time
from typing import Dict, List
from pathlib import Path
from collections import defaultdict

_logger = logging.getLogger(__name__)

__all__ = [
    'parallelize_graph',
]


def check_env(autodist_config: AutoDistConfig):
    arch_dir = Path(autodist_config.profile_dir)
    if not arch_dir.exists():
        _logger.info(f'create folder: {arch_dir}')
        arch_dir.mkdir(parents=True, exist_ok=True)

def pre_estimate_mem(graph: ModelGraph):
    '''
    Estimate a rough lower bound of memory consumption per device. Exit if the model is too large
    for allocated resources.
    '''

    def to_mb(size):
        return size // 1024 // 1024

    def to_gb(size):
        return to_mb(size) // 1024

    # calculate sizes of activations, buffers and parameters, exit if the model is
    # too large for allocated resources
    param_mem, buffer_mem, activation_mem = graph.query_mem(0, graph.op_num - 1)
    _logger.info(
        f'param mem {to_mb(param_mem)} MB, buff mem {to_mb(buffer_mem)} MB, activation mem {to_mb(activation_mem)} MB'
    )
    plan_ngpus = graph.autodist_config.mesh_desc.ngpus
    if graph.autodist_config.zero_stage == 1:
        zero_group_size = graph.autodist_config.world_size // graph.autodist_config.zero_ngroups
    elif graph.autodist_config.zero_stage == 0:
        zero_group_size = plan_ngpus
    else:
        raise RuntimeError(
            f'invalid zero stage {graph.autodist_config.zero_stage}')
    min_single_dev_mem = estimate_mem_lower_bound(
        param_mem=param_mem,
        buffer_mem=buffer_mem,
        activation_mem=activation_mem,
        plan_ngpus=plan_ngpus,
        zero_group_size=zero_group_size,
        cfg=graph.autodist_config,
    )
    min_single_dev_mem += graph.min_recompute_mem
    _logger.info(
        f'estimated minimum memory per device {to_mb(min_single_dev_mem)} MB')
    mem_constraint = graph.autodist_config.memory_constraint
    if min_single_dev_mem > mem_constraint * 1024 * 1024 * 1024:
        raise RuntimeError(
            f'est min mem: {to_gb(min_single_dev_mem)} GB vs mem constraint: {mem_constraint} GB, '
            + 'model is too large for current resources, try to ' +
            'reduce batch size, add more devices or increase zero group size')


def calc_parallel_plan(graph: IRGraph,
                       autodist_config: AutoDistConfig) -> PipelineSearchOutput:
    _logger.info(autodist_config)
    check_env(autodist_config)

    autodist_graph = ModelGraph(ir_graph=graph, autodist_config=autodist_config)
    pre_estimate_mem(autodist_graph)

    recompute_groups = autodist_graph.recompute_groups
    recompute_groups = [
        [node.cid for node in group] for group in recompute_groups
    ]

    if autodist_config.pipeline:
        pp_out = calc_optimal_pp_plan(autodist_graph, autodist_config)
    else:
        pp_out = calc_optimal_spmd_plan(autodist_graph, autodist_config)
    pp_out.desc.recompute_groups = recompute_groups
    pp_out.stage_mems = [mem for mem in pp_out.stage_mems]
    return pp_out


def parallelize_graph(graph: IRGraph,
                      autodist_config: AutoDistConfig) -> IRGraph:
    segments: List[IRSegment] = graph.select(ntype=IRSegment)
    if segments:
        raise RuntimeError('assume there is no segment in the graph')

    if autodist_config.load_plan_path:
        _logger.info(f'load plan from {autodist_config.load_plan_path}')
        with open(autodist_config.load_plan_path, 'r') as f:
            search_out_json = json.load(f)
        search_out = PipelineSearchOutput.from_json(search_out_json)
    else:
        compile_start_time = time.time()
        search_out = calc_parallel_plan(graph, autodist_config)
        compile_cost_time = time.time() - compile_start_time

        if autodist_config.save_plan_path:
            _logger.info(f'save plan to {autodist_config.save_plan_path}')
            with open(autodist_config.save_plan_path, 'w') as f:
                json.dump(search_out.to_json(), f, indent=2)

    _logger.info(f'use plan with e2e time/s {1000 * search_out.e2e_time:.2f}ms')
    pp_desc = search_out.desc

    cid2node: Dict[int, IRFwOperation] = dict()
    for node in graph.nodes():
        if isinstance(node, IRFwOperation):
            cid2node[node.cid] = node

    # set recompute groups
    for group in pp_desc.recompute_groups:
        nodes = [cid2node[cid] for cid in group]
        graph.recompute(nodes)

    # graph staging
    if len(pp_desc.spmd_descs) > 1:
        # add multiref for shared parameters across stages
        shared_param2stage_info = defaultdict(dict)
        for ftensor in graph.attributes():
            if not ftensor.is_param():
                continue
            for ctensor, consumer in zip(graph.ctensors(ftensor),
                                         graph.consumers(ftensor)):
                if ctensor.grad is None:
                    continue
                for stage_idx, stage_desc in enumerate(pp_desc.spmd_descs):
                    if consumer.cid in stage_desc.partition_descs:
                        if len(stage_desc.partition_descs[
                                consumer.cid].desc) != 1:
                            raise RuntimeError(
                                f'node {consumer} has more than one partition dim'
                            )
                        (p_idx, p_dim), p_num = stage_desc.partition_descs[
                            consumer.cid].desc[0]
                        if p_idx != -1 and consumer.inputs()[p_dim] == ftensor:
                            raise RuntimeError(
                                f'node {consumer} has partitioned input {ftensor}'
                            )
                        is_replicated = p_idx == -1
                        if stage_idx not in shared_param2stage_info[ftensor]:
                            shared_param2stage_info[ftensor][stage_idx] = []
                        shared_param2stage_info[ftensor][stage_idx].append(
                            is_replicated)

        for ftensor, stage_info in shared_param2stage_info.items():
            if len(stage_info) == 1:
                continue
            # special case: all stages have only one gpu
            stage_idxs = list(stage_info.keys())
            stage_sizes = [
                pp_desc.spmd_descs[i].mesh_desc.ngpus for i in stage_idxs
            ]
            if all([s == 1 for s in stage_sizes]):
                continue
            # check whether all partitioned
            # In AutoDist, shared parameters are not allowed to be partitioned.
            # As a result, the related operator is replicated or in data parallel.
            has_replicated = False
            for stage_idx, replicate_info in stage_info.items():
                if any(replicate_info):
                    has_replicated = True
                    break
            if has_replicated:
                _logger.info(f'add multiref for shared param {ftensor}')
                graph.multiref(ftensor)

        stages = []
        for spmd_desc in pp_desc.spmd_descs:
            stage = []
            for cid in spmd_desc.partition_descs:
                if cid not in cid2node:
                    raise RuntimeError(f'node {cid} not found in {cid2node}, make sure the plan is correct')
                stage.append(cid2node[cid])
            stages.append(stage)
        graph.staging([s[0] for s in stages])
        stages = graph.select(ntype=IRSegment, flatten=False)
        stages = [s for s in stages if s.isfw()]
    else:
        stages = [graph]

    # TODO: check pipeline_nstages when ready.
    # if autodist_config.pipeline and len(stages) != autodist_config.pipeline_nstages:
    #     raise RuntimeError("pipeline_nstages doesn't match the number of stages (based on your pipeline_pivots config) in the plan")

    # add multiref to a tensor when
    # 1. it is not a grad tensor
    # 2. it has more than one consumers
    # 3. consumers are different operators or in different partitions
    for stage, spmd_desc in zip(stages, pp_desc.spmd_descs):
        for ftensor in stage.full_tensors():
            if ftensor.is_grad():
                continue
            if len(stage.consumers(ftensor)) <= 1:
                continue
            consumers = stage.consumers(ftensor)
            splits = set()
            for consumer in consumers:
                if consumer.cid in spmd_desc.partition_descs:
                    node_desc = spmd_desc.partition_descs[consumer.cid].desc
                    if len(node_desc) != 1:
                        raise RuntimeError(
                            f'node {consumer} has more than one partition desc')
                    (p_idx, p_dim), p_num = node_desc[0]
                else:
                    _logger.warning(
                        f'node {consumer} is not in any partition desc')
                    p_idx, p_dim, p_num = -1, -1, spmd_desc.mesh_desc.ngpus
                repr_str = f'{consumer.signature}-{p_idx}-{p_dim}-{p_num}'
                splits.add(repr_str)
            if len(splits) > 1:
                _logger.debug(f'add multiref {consumers}')
                stage.multiref(ftensor)

    # partition and assign nodes to devices
    # TODO(yizhu1): network topo aware device map
    offset = 0
    for idx, (spmd_desc, stage) in enumerate(zip(pp_desc.spmd_descs, stages)):
        cur_ngpus = spmd_desc.mesh_desc.ngpus
        dev = [offset + i for i in range(cur_ngpus)]
        stage_info_str = f'stage {idx} on devices {dev} with mem {search_out.stage_mems[idx]:.2f} GB'
        _logger.info(f'\nautodist plan analysis for {stage_info_str}:\n\n{analysis_pretty_printer(spmd_desc.analysis)}')
        offset += cur_ngpus
        for node in stage.nodes():
            if isinstance(node, IRFwOperation):
                if isinstance(
                        node,
                    (IRGraphAnchor, IRPyFunc)) or node.name == 'multiref':
                    continue
                if node.cid in spmd_desc.partition_descs:
                    p_desc = spmd_desc.partition_descs[node.cid]
                    partition_node(node, graph, dev, p_desc)
                    if isinstance(node, IRDimops):
                        _logger.debug(
                            f'apply {node} with {node.anno} at {node.comment}, plan: {p_desc}'
                        )
                    else:
                        _logger.debug(
                            f'replicate non-IRDimops {node.signature} with {node.comment}'
                        )
                else:
                    replica(graph, node, dev)
                    _logger.debug(
                        f'NOT included in plan, replicate {node.signature} with {node.comment}'
                    )

    for dl in graph.select(ntype=IRDataOperation):
        replica(graph, dl, devs=list(range(autodist_config.mesh_desc.ngpus)))

    # apply 1f1b schedule
    if len(stages) > 1:
        PredefinedSched.sched_1f1b(
            graph,
            autodist_config.update_freq,
            len(stages),
        )

    return graph
