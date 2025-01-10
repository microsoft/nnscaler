#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from .spmd_solver import calc_optimal_spmd_plan, analysis_pretty_printer
from .pipeline_solver import calc_optimal_pp_plan
from .autodist_config import AutoDistConfig
from .model_graph import ModelGraph, estimate_mem_lower_bound
from .descs import *
from .util import replica, partition_node

from nnscaler.graph import IRGraph
from nnscaler.graph.segment import IRSegment
from nnscaler.ir.operator import IRDataOperation, IRFwOperation
from nnscaler.ir.tensor import IRSubTensor
from nnscaler.ir import IRCell
from nnscaler.graph.function import IRDimops
from nnscaler.graph.function.anchor import IRGraphAnchor
from nnscaler.graph.function.pyfunc import IRPyFunc
from nnscaler.graph.schedule.predefined import PredefinedSched

import json
import logging
import copy
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
        search_out = calc_parallel_plan(graph, autodist_config)

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

    def subtensor_desc(t):
        return (t.indmap, t.grad is not None)
    tensor_split_info = defaultdict(dict)
    for ftensor in graph.full_tensors():
        if ftensor.is_grad():
            continue
        consumers = graph.consumers(ftensor)
        if not consumers:
            continue
        for consumer in consumers:
            find_desc = False
            for stage_idx, stage_desc in enumerate(pp_desc.spmd_descs):
                if consumer.cid not in stage_desc.partition_descs:
                    continue
                find_desc = True
                node_desc = stage_desc.partition_descs[consumer.cid].desc
                if len(node_desc) != 1:
                    raise RuntimeError(f'node {consumer} is partitioned along multiple dims')

                (p_idx, p_dim), p_num = node_desc[0]
                if p_idx == -1:
                    partitioned_node = consumer
                else:
                    partitioned_nodes = consumer.algorithm('dim').instantiate(idx=p_idx, dim=p_dim, num=p_num)
                    if partitioned_nodes is None:
                        raise RuntimeError(f'node {consumer} cannot be partitioned by {p_idx}-{p_dim}-{p_num}')
                    partitioned_node = partitioned_nodes[0]

                if stage_idx not in tensor_split_info[ftensor]:
                    tensor_split_info[ftensor][stage_idx] = set()
                for input in partitioned_node.inputs():
                    if isinstance(input, IRSubTensor) and input.parent == ftensor:
                        if p_idx == -1 and stage_desc.mesh_desc.ngpus > 1:
                            tensor_split_info[ftensor][stage_idx].add(('REPLICATED', subtensor_desc(input)))
                        else:
                            # special case: if the stage has only one gpu, we treat it as partitioned
                            tensor_split_info[ftensor][stage_idx].add(('PARTITIONED', subtensor_desc(input)))
                break
            assert find_desc, f'node {consumer} not found in any stage'

    # graph staging
    if len(pp_desc.spmd_descs) > 1:
        # add multiref for shared parameters across stages
        # note that we have constrained that shared parameters cannot
        # be partitioned in SPMDSolver.
        for ftensor, stage_info in tensor_split_info.items():
            if not ftensor.is_param():
                continue
            splits = set()
            find_replicated = False
            for stage_splits in stage_info.values():
                splits.update(stage_splits)
                if any(s[0] == 'REPLICATED' for s in stage_splits):
                    find_replicated = True
            splits = list(splits)
            if len(splits) > 1 or find_replicated:
                _logger.info(f'add multiref for shared param {ftensor}')
                graph.multiref(ftensor, comment='shared param')

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

    # add multiref to an activation tensor when the states of the tensor and its grad are different
    # among consumers and current segment's outputs
    for idx, (stage, spmd_desc) in enumerate(zip(stages, pp_desc.spmd_descs)):
        for ftensor in stage.full_tensors():
            if ftensor.is_grad() or ftensor.is_param():
                continue
            if idx not in tensor_split_info[ftensor]:
                continue
            splits = copy.deepcopy(tensor_split_info[ftensor][idx])
            for output in IRCell.get_objects_from_complex(stage.outputs()):
                if isinstance(output, IRSubTensor) and output.parent == ftensor:
                    splits.add(('REPLICATED', subtensor_desc(output)))
            if len(splits) > 1:
                _logger.debug(f'add multiref for {ftensor} in stage {stage}')
                stage.multiref(ftensor, comment='activation')

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
