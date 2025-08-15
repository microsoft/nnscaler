#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""
Policy Writing Guidelines

Users can write the policy following the steps:

1. Apply multiref
    If all consumers of a full tensor consume the same subtensor (the partitions are exactly the same), we can skip this step.
2. Apply recompute (if needed)
3. Graph staging (pipeline only)
4. Graph partition & assign
5. Apply schedule (pipeline only)

Note the steps 1, 2, 3 must be finished before any graph partition.

IRDataOperation is recommended to be replicated to all devices.
"""

import logging
from typing import List, Optional, TYPE_CHECKING
import random

import torch
import more_itertools as mitr

from nnscaler.autodist.apis import parallelize_graph
from nnscaler.autodist.autodist_config import AutoDistConfig
from nnscaler.graph import IRGraph
from nnscaler.graph.function.anchor import IRGraphAnchor
from nnscaler.graph.function.dimops import IRDimops
from nnscaler.graph.segment import IRSegment
from nnscaler.ir.operator import IRDataOperation, IRFwOperation
from nnscaler.ir import IRCell, IRSubTensor, IRFullTensor


if TYPE_CHECKING:
    from nnscaler.parallel import ComputeConfig


_logger = logging.getLogger(__name__)


def _tp(graph: IRGraph, node: IRDimops, devs: List[int], idx: int, dim: int):
    if len(devs) > 1:
        sub_nodes = graph.partition(
            node, node.algorithm('dim'), idx=idx, dim=dim, num=len(devs))
    else:
        sub_nodes = [node]
    for devid, sub_node in zip(devs, sub_nodes):
        graph.assign(sub_node, devid)
    return sub_nodes


def _replica(graph: IRGraph, node, devs: List[int]):
    sub_nodes = graph.replicate(node, times=len(devs))
    for devid, sub_node in zip(devs, sub_nodes):
        graph.assign(sub_node, devid)
    return sub_nodes


def is_tensor_in_output(t: IRFullTensor, graph: IRSegment) -> bool:
    for output in IRCell.get_objects_from_complex(graph.outputs()):
        if isinstance(output, IRSubTensor) and output.parent == t:
            return True
    return False


def auto_multiref(graph: IRGraph):
    for ftensor in graph.full_tensors():
        if ftensor.is_grad(): continue
        in_output = int(is_tensor_in_output(ftensor, graph))
        if len(graph.consumers(ftensor)) + in_output > 1:
            graph.multiref(ftensor, comment='auto_multiref')


def pas_dp(graph: IRGraph, cfg: 'ComputeConfig'):
    """
    pure data parallelism policy
    """
    ngpus = cfg.plan_ngpus
    if ngpus != 1:
        raise ValueError("Data parallelism only supports 1 plan GPU")

    # no partition is done, so we can skip multiref safely
    for node in graph.select(ntype=(IRFwOperation, IRDataOperation)):
        _replica(graph, node, [0])
    return graph


def pas_tp(graph: IRGraph, cfg: 'ComputeConfig'):
    """
    random tensor parallelism inside a scale unit, and dp across scale units
    """
    ngpus = cfg.plan_ngpus
    # get the current random state
    state = random.getstate()

    seed = cfg.pas_config.get('seed', 1)  # by default we fix the seed for test reproducibility
    random.seed(seed)
    devs = list(range(ngpus))

    auto_multiref(graph)

    for node in graph.select(ntype=(IRFwOperation, IRDataOperation)):
        if node.name == 'multiref' or isinstance(node, IRGraphAnchor):
            continue
        if isinstance(node, IRDimops):
            configs = node.transform_space()
            if len(configs) == 0:
                _replica(graph, node, devs)
            else:
                configs = sorted(configs, reverse=True,
                                key=lambda config: node.input(config[0]).shape[config[1]])
                random.shuffle(configs)
                for (idx, dim) in configs:
                    if node.input(idx).shape[dim] % len(devs) != 0: continue
                    if node.algorithm('dim').satisfy(idx=idx, dim=dim, num=len(devs)):
                        _tp(graph, node, devs, idx, dim)
                        break
                else:
                    _replica(graph, node, devs)
        else:
            _replica(graph, node, devs)

    # restore the random state
    random.setstate(state)
    return graph


def pas_pp(graph: IRGraph, cfg: 'ComputeConfig'):
    """
    pipeline parallelism inside a scale unit, and dp across scale units
    """
    nstages = cfg.pas_config.get('pipeline_nstages', 'auto')
    if nstages != 'auto' and nstages != cfg.plan_ngpus:
        raise ValueError("pas_pp requires pipeline_nstages == plan_ngpus")
    return pas_hybrid(graph, cfg)


def pas_data(graph: IRGraph, env_resource: 'ComputeConfig'):
    """
    tensor partition on batch dimension inside a scale unit, and dp across scale units
    """
    ngpus = env_resource.plan_ngpus
    auto_multiref(graph)

    batch_dim = 0
    for dl in graph.select(ntype=IRDataOperation):
        _replica(graph, dl, list(range(ngpus)))

    for node in graph.nodes():
        if isinstance(node, IRFwOperation):
            try:
                algo = node.algorithm('dim')
                idx = 0
                sub_nodes = graph.partition(
                    node, algo, idx=idx, dim=batch_dim, num=ngpus)
            except Exception:
                sub_nodes = graph.replicate(node, ngpus)

            for idx, node in enumerate(sub_nodes):
                graph.assign(node, idx)
    return graph


def pas_hybrid(graph: IRGraph, cfg: 'ComputeConfig'):
    """
    pipeline and tensor parallelism inside a scale unit, and dp across scale units
    """
    if not cfg.use_end2end:
        raise ValueError("Hybrid policy only supports end2end module")

    ngpus: int = cfg.plan_ngpus
    nstages = cfg.pas_config.get('pipeline_nstages', 'auto')
    if nstages == 'auto':
        nstages = cfg.plan_ngpus
    nmicros = cfg.pas_config['pipeline_nmicros']
    scheduler = cfg.pas_config.get('pipeline_scheduler', '1f1b')
    pp_size = cfg.pas_config.get('pp_size', nstages)

    if nstages % pp_size != 0:
        raise ValueError(f'invalid pp_size {pp_size} for nstages {nstages}')
    if ngpus % pp_size != 0:
        raise ValueError(f'invalid pp_size {pp_size} for ngpus {ngpus}')
    tp_size = ngpus // pp_size


    auto_multiref(graph)
    fnodes = graph.select(ntype=IRFwOperation)
    stages = mitr.divide(nstages, fnodes)
    stages = [list(s) for s in stages]
    for idx, stage in enumerate(stages):
        _logger.info(f'> stage {idx}: {stage[0]}')
    graph.staging([s[0] for s in stages])

    stages: List[IRSegment] = graph.select(ntype=IRSegment, flatten=False)
    stages = [s for s in stages if s.isfw()]
    assert len(stages) == nstages, "Internal Error"

    # stage-wise tensor parallelism
    curr_devices = list(range(ngpus))
    for idx, stage in enumerate(stages):
        idx = idx % pp_size
        devs = curr_devices[idx * tp_size: (idx + 1)* tp_size]
        for node in stage.nodes():
            try:
                _tp(graph, node, devs, idx=0, dim=0)
            except Exception as e:
                _replica(graph, node, devs)

    # replicate dataloader
    for dl in graph.select(ntype=IRDataOperation):
        _replica(graph, dl, devs=list(range(ngpus)))

    cfg.apply_pipeline_scheduler(graph, nstages, nmicros, scheduler)
    return graph


def pas_autodist(graph: IRGraph, cfg: 'ComputeConfig') -> IRGraph:
    pas_cfg = cfg.pas_config

    update_freq = pas_cfg.get('update_freq', 1)
    if isinstance(update_freq, (tuple, list)):
        update_freq = update_freq[0]

    # optional parameters

    # Note we don't directly pass pipeline_nstages to autodist.
    # when `pipeline_nstages == 'auto'`, we will check if there are options incompatible with pipeline.
    # if we find incompabible options (here use_async_reducer and pipeline_pivots),
    # we will disable pipeline effectively by setting it to 1.
    pipeline_nstages = pas_cfg.get('pipeline_nstages', 'auto')

    if pipeline_nstages == 'auto':
        if not pas_cfg.get('pipeline_pivots'):
            pipeline_nstages = 1
        if not cfg.use_end2end or cfg.use_async_reducer:
            pipeline_nstages = 1
    elif pipeline_nstages > 1:
        # the user manually enabled pipeline, should not disable, so raise
        if not pas_cfg.get('pipeline_pivots'):
            raise ValueError("pipeline_pivots must be set to enable pipeline")
        if not cfg.use_end2end:
            raise ValueError("explore_pipeline cannot be enabled if use_end2end is False")
        if cfg.use_async_reducer:
            raise ValueError("explore_pipeline cannot be enabled if use_async_reducer is True")
    else:
        if pas_cfg.get('pipeline_pivots'):
            raise ValueError("pipeline_pivots must not be set because pipeline is disabled by pipeline_nstages<=1")

    pipeline_scheduler = pas_cfg.get('pipeline_scheduler', '1f1b')
    if pipeline_scheduler != '1f1b':
        raise ValueError(f"Only 1f1b scheduler is supported in autodist.")

    mesh_col = pas_cfg.get('max_partition_degree', cfg.plan_ngpus)
    if cfg.plan_ngpus % mesh_col != 0:
        raise ValueError(f"plan_ngpus {cfg.plan_ngpus} should be divisible by max_partition_degree {mesh_col}")
    mesh_row = cfg.plan_ngpus // mesh_col
    if pipeline_nstages == 1 and mesh_row != 1:
        raise ValueError("mesh_row should be 1 if pipeline is not enabled")
    memory_constraint = pas_cfg.get('mem_constraint', -1)
    task_name = pas_cfg.get('task_name', '_')
    use_memory_efficient_fp16 = pas_cfg.get('use_memory_efficient_fp16', False)
    use_memory_efficient_bf16 = pas_cfg.get('use_memory_efficient_bf16', False)
    use_fp16 = pas_cfg.get('use_fp16', use_memory_efficient_fp16)
    use_bf16 = pas_cfg.get('use_bf16', use_memory_efficient_bf16)
    re_profile = pas_cfg.get('re_profile', False)
    verbose = pas_cfg.get('verbose', False)
    load_plan_path = pas_cfg.get('load_plan_path', None)
    save_plan_path = pas_cfg.get('save_plan_path', None)
    partition_constraints_path = pas_cfg.get('partition_constraints_path', '')
    recompute_modules = pas_cfg.get('recompute_modules', '')
    pipeline_pivots = pas_cfg.get('pipeline_pivots', '')
    max_pipeline_bubble_ratio = pas_cfg.get('max_pipeline_bubble_ratio', 0.2)
    max_pipeline_unbalance_ratio = pas_cfg.get('max_pipeline_unbalance_ratio', 0.5)
    use_apex_fused_adam_v2 = pas_cfg.get('use_apex_fused_adam_v2', False)
    parallel_profile = pas_cfg.get('parallel_profile', True)
    transient_mem_coef = pas_cfg.get('transient_mem_coef', 2)

    task_name = f'{task_name}_{cfg.plan_ngpus}gpus_{update_freq}update_freq'
    if memory_constraint == -1:
        # consider memory fragmentation and other buffers, use 80% of the memory
        memory_constraint = int(0.8 * torch.cuda.mem_get_info()[1] / 1024 /
                                1024 / 1024)
    if cfg.use_zero:
        zero_stage = 1
        zero_ngroups = cfg.zero_ngroups
    else:
        zero_stage = 0
        zero_ngroups = 1
    if use_fp16 or use_bf16:
        support_inkernel_cast = use_apex_fused_adam_v2
        if use_memory_efficient_fp16 or use_memory_efficient_bf16:
            # Check fairseq/optim/fused_adam.py
            # If memory efficient:
            # Considered in opt_resident_mem: fp32 moment1, fp32 moment2.
            # Considered in opt_transient_mem: fp32 weight, fp32 gradient,
            # because fp16 weight and gradient are casted to fp32.
            # Here weight_mem is in fp16, so multiply by (2+2).
            opt_resident_coef = 4
            opt_transient_coef = 0 if support_inkernel_cast else 4
        else:
            # If not memory efficient:
            # Considered in opt_resident_mem: fp32 moment1, fp32 moment2, fp32 weight.
            # Considered in opt_transient_mem: fp32 gradient,
            # because fp16 gradient are casted to fp32.
            # Here weight_mem is in fp16, so multiply by (2+2+2).
            opt_resident_coef = 6
            # inkernel cast between fp32 weight and fp16 grad has not support
            opt_transient_coef = 2 if support_inkernel_cast else 2
    else:
        # Considered in opt_resident_mem: fp32 moment1, fp32 moment2
        # Considered in opt_transient_mem: 0
        # Here weight_mem is in fp32, so multiply by (1+1).
        opt_resident_coef = 2
        opt_transient_coef = 0

    autodist_cfg = AutoDistConfig(
        mesh_row=mesh_row,
        mesh_col=mesh_col,
        update_freq=update_freq,
        task_name=task_name,
        is_train=not cfg.inference_only,
        ignore_small_tensor_threshold=524288,  # 0.5 MB is a good threshold to reduce search time and make the result correct, will refine later
        memory_granularity=524288,             # 0.5 MB is a good threshold to reduce search time and make the result correct, will refine later
        consider_mem=True,
        partition_constraints_path=partition_constraints_path,
        memory_constraint=memory_constraint,
        opt_resident_coef=opt_resident_coef,
        opt_transient_coef=opt_transient_coef,
        verbose=verbose,
        re_profile=re_profile,
        world_size=cfg.runtime_ngpus,
        recompute_modules=recompute_modules,
        zero_stage=zero_stage,
        zero_ngroups=zero_ngroups,
        load_plan_path=load_plan_path,
        save_plan_path=save_plan_path,
        pipeline_pivots=pipeline_pivots,
        pipeline_nstages=pipeline_nstages,
        max_pipeline_bubble_ratio=max_pipeline_bubble_ratio,
        max_pipeline_unbalance_ratio=max_pipeline_unbalance_ratio,
        parallel_profile=parallel_profile,
        transient_mem_coef=transient_mem_coef,
    )

    return parallelize_graph(graph, autodist_cfg)
