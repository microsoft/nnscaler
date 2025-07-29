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


if TYPE_CHECKING:
    from nnscaler.parallel import ComputeConfig


_logger = logging.getLogger(__name__)


def _tp(graph: IRGraph, node: IRDimops, devs: List[int], idx: int, dim: int):
    if len(devs) > 1:
        sub_nodes = graph.partition(
            node, node.algorithms('dim'), idx=idx, dim=dim, num=len(devs))
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

    # by default we fix the seed for test reproducibility
    seed = cfg.pas_config.get('seed', 1)
    random.seed(seed)
    devs = list(range(ngpus))

    for ftensor in graph.full_tensors():
        if ftensor.is_grad():
            continue
        if len(graph.consumers(ftensor)) > 1:
            graph.multiref(ftensor)

    for node in graph.select(ntype=(IRFwOperation, IRDataOperation)):
        if node.name == 'multiref' or isinstance(node, IRGraphAnchor):
            continue
        if isinstance(node, IRDimops):
            configs = node.transform_space()
            if len(configs) == 0:
                _replica(graph, node, devs)
            if node.name == "view":
                _replica(graph, node, devs)
            else:
                configs = sorted(configs, reverse=True,
                                 key=lambda config: node.input(config[0]).shape[config[1]])
                random.shuffle(configs)
                for (idx, dim) in configs:
                    if node.input(idx).shape[dim] % len(devs) != 0:
                        continue
                    if node.algorithms('dim').satisfy(idx=idx, dim=dim, num=len(devs)):
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
    if cfg.pipeline_nstages != cfg.plan_ngpus:
        raise ValueError("pipeline_nstages should be equal to plan_ngpus")
    return pas_hybrid(graph, cfg)


def pas_data(graph: IRGraph, env_resource: 'ComputeConfig'):
    """
    tensor partition on batch dimension inside a scale unit, and dp across scale units
    """
    ngpus = env_resource.plan_ngpus
    # auto multi-ref
    for ftensor in graph.full_tensors():
        if len(graph.consumers(ftensor)) > 1:
            graph.multiref(ftensor, [[n] for n in graph.consumers(ftensor)])

    batch_dim = 0
    for dl in graph.select(ntype=IRDataOperation):
        _replica(graph, dl, list(range(ngpus)))

    for node in graph.nodes():
        if isinstance(node, IRFwOperation):
            try:
                algo = node.algorithms('dim')
                idx = 0
                sub_nodes = graph.partition(
                    node, algo, idx=idx, dim=batch_dim, num=ngpus)
            except Exception:
                sub_nodes = graph.replicate(node, ngpus)

            for idx, node in enumerate(sub_nodes):
                graph.assign(node, idx)
    return graph


from nnscaler.ir import IRCell, IRSubTensor, IRFullTensor
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
            graph.multiref(ftensor)
        
            
def pas_moe(graph: IRGraph, cfg: 'ComputeConfig'):
    auto_multiref(graph)
    return pas_hybrid(graph, cfg)
    
    
_record = -1
def pas_hybrid(graph: IRGraph, cfg: 'ComputeConfig'):
    """
    pipeline and tensor parallelism inside a scale unit, and dp across scale units
    """
    if not cfg.use_pipeline:
        raise ValueError("pipeline should be enabled")

    ngpus: int = cfg.plan_ngpus
    nstages = cfg.pipeline_nstages
    tp_size: int = cfg.plan_ngpus // nstages
    if ngpus % tp_size != 0:
        raise ValueError(f'invalid tp_size {tp_size} for ngpus {ngpus}')
    pp_size = ngpus // tp_size

    fnodes = graph.select(ntype=IRFwOperation)
    stages = mitr.divide(pp_size, fnodes)
    stages = [list(s) for s in stages]
    for idx, stage in enumerate(stages):
        _logger.info(f'> stage {idx}: {stage[0]}')
    graph.staging([s[0] for s in stages])

    stages: List[IRSegment] = graph.select(ntype=IRSegment, flatten=False)
    stages = [s for s in stages if s.isfw()]
    assert len(stages) == pp_size, "Internal Error"

    # stage-wise tensor parallelism
    curr_devices = list(range(ngpus))
    for stage in stages:
        for node in stage.nodes():
            devs = curr_devices[:tp_size]
            try:
                # _tp(graph, node, devs, idx=0, dim=0)
                if node.comment.endswith(', in forward,  return self.w2(F.silu(self.w1(x)) * self.w3(x))'):
                    if node.name == "linear":
                        global _record
                        _record += 1
                        if _record % 3 in [0, 1]:
                            _tp(graph, node, devs, idx=1, dim=0)
                        elif _record % 3 in [2]:
                            _tp(graph, node, devs, idx=1, dim=1)
                    elif node.name in ["mul", "silu"]:
                        _tp(graph, node, devs, idx=0, dim=2)
                elif node.name == "nnscaler_moe_gmm":
                    _tp(graph, node, devs, idx=3, dim=0)
                else:
                    raise
                    # _tp(graph, node, devs, idx=0, dim=0)
            except Exception as e:
                _replica(graph, node, devs)
        curr_devices = curr_devices[tp_size:]
    assert len(
        curr_devices) == 0, f"remaining devices: {curr_devices} not used"

    # replicate dataloader
    for dl in graph.select(ntype=IRDataOperation):
        _replica(graph, dl, devs=list(range(ngpus)))

    cfg.apply_pipeline_scheduler(graph)
    return graph

def pas_flex(graph: IRGraph, cfg: 'ComputeConfig'):
    """
    pipeline and tensor parallelism inside a scale unit, and dp across scale units
    """
    if not cfg.use_pipeline:
        raise ValueError("pipeline should be enabled")

    ngpus: int = cfg.plan_ngpus
    nstages = cfg.pipeline_nstages
    tp_size: int = cfg.plan_ngpus // nstages
    if ngpus % tp_size != 0:
        raise ValueError(f'invalid tp_size {tp_size} for ngpus {ngpus}')
    pp_size = ngpus // tp_size

    fnodes = graph.select(ntype=IRFwOperation)
    stages = mitr.divide(pp_size, fnodes)
    stages = [list(s) for s in stages]
    for idx, stage in enumerate(stages):
        _logger.info(f'> stage {idx}: {stage[0]}')
    graph.staging([s[0] for s in stages])

    stages: List[IRSegment] = graph.select(ntype=IRSegment, flatten=False)
    stages = [s for s in stages if s.isfw()]
    assert len(stages) == pp_size, "Internal Error"

    # stage-wise tensor parallelism
    curr_devices = list(range(ngpus))
    for stage in stages:
        for node in stage.nodes():
            devs = curr_devices[:tp_size]
            try:
                _tp(graph, node, devs, idx=0, dim=0)
            except Exception as e:
                _replica(graph, node, devs)
        curr_devices = curr_devices[tp_size:]
    assert len(
        curr_devices) == 0, f"remaining devices: {curr_devices} not used"

    # replicate dataloader
    for dl in graph.select(ntype=IRDataOperation):
        _replica(graph, dl, devs=list(range(ngpus)))

    cfg.apply_pipeline_scheduler(graph)
    return graph


def pas_autodist(graph: IRGraph, cfg: 'ComputeConfig') -> IRGraph:
    pas_cfg = cfg.pas_config

    # required parameters
    update_freq = pas_cfg.get('update_freq', 1)
    if isinstance(update_freq, (tuple, list)):
        update_freq = update_freq[0]
    if cfg.use_pipeline and update_freq != cfg.pipeline_nmicros:
        raise ValueError("pipeline_nmicros should be equal to update_freq")

    # optional parameters
    mesh_col = pas_cfg.get('mesh_col', cfg.plan_ngpus)
    if mesh_col != cfg.plan_ngpus:
        raise ValueError("mesh_col should be equal to plan_ngpus")
    mem_constraint = pas_cfg.get('mem_constraint', -1)
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
    use_apex_fused_adam_v2 = pas_cfg.get('use_apex_fused_adam_v2', False)

    mesh_row = 1
    ngpus = mesh_row * mesh_col
    task_name = f'{task_name}_{ngpus}gpus_{update_freq}update_freq'
    if mem_constraint == -1:
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
        # 0.5 MB is a good threshold to reduce search time and make the result correct, will refine later
        ignore_small_tensor_threshold=524288,
        # 0.5 MB is a good threshold to reduce search time and make the result correct, will refine later
        memory_granularity=524288,
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
        pipeline=cfg.use_pipeline,
        pipeline_pivots=pipeline_pivots,
        pipeline_nstages=cfg.pipeline_nstages,
    )

    return parallelize_graph(graph, autodist_cfg)
