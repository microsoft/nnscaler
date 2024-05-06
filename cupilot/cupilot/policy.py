# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Policy entry
"""
from typing import Callable, Optional, List, Tuple
import logging
import warnings
import torch

from cube.graph.function.anchor import IRGraphAnchor
from cube.graph.function.dimops import IRDimops
from cube.graph.graph import IRGraph
from cube.graph.segment import IRSegment
from cube.ir.operator import IRFwOperation, IRDataOperation
from cube.runtime.device import DeviceGroup
from cube.flags import CompileFlag

from .plan import ParallelSpec
from .estimator.profiler import Estimator
from .estimator.cost_model import CostModel
from .config import CupilotConfig
from .solver import SpmdSolver
from .solver import StageSolver
from .solver import OrderSolver
from .solver.block import IRBlock
from .constraints import Constraints
from .parallel.spmd import nested_tensor_parallelism, replicate
from .utils import auto_multiref, stage_blocks

SubGraph = Tuple[IRFwOperation]

_logger = logging.getLogger(__name__)



def staged_spmd(graph: IRGraph, resource,
                blocks: List[IRBlock],
                constraints: Constraints,
                config: CupilotConfig) -> ParallelSpec:
    """Parallelization plan search core for spatial placement.

    Note the searching is not stable due to un-derterministic ILP
    solver. To make consistent execution, this function should be called on one rank.
    The search results need to be broadcasted to other ranks.
    
    Args:
        graph (IRGraph): execution cube graph
        resource: execution resource
        blocks (List[IRBlock]): sub-graphs
        constraints (Constraints): user constraints
        config (CupilotConfig): policy search configuration.

    Returns: 
        spec (ParallelSpec): parallelization plan
    """
    mem_limit = resource.gpus[0].memory - 2 * 1024 * 1024 * 1024  # 2GB for nccl memory
    mem_limit *= config.memory_fraction
    _logger.info(f'device memory capacity: {mem_limit}')

    # step 1. build profiler
    estimator = Estimator(cache=config.db_cache)

    # step 2. build cost model for communiation
    _logger.info(f'building cost model...')
    CostModel.ndevs_per_node = DeviceGroup().local_world_size
    cost_model = CostModel(graph, estimator, constraints)
    latency, memory = estimator(graph.select(ntype=IRFwOperation))
    _logger.info(f'estimate single device latency per microbatch: {round(latency, 2)} ms, '
                 f'memory: {round(memory/1024/1024/1024, 2)} GB')
    _logger.info(f'saving profiled database to {config.db_cache}...')
    estimator.save()

    standalone_blocks = [blk for blk in blocks if blk.standalone]
    solver_blocks = [blk for blk in blocks if not blk.standalone]
    _logger.info(f'find {len(standalone_blocks)} standalone blocks and {len(solver_blocks)} solver blocks')

    # step 3. search spmd for standalone blocks
    for blk in standalone_blocks:
        if blk.standalone:
           pass  # TODO

    # step 4. search staged_spmd for remaining blocks

    # - initialize memory and computation cost for constrained nodes
    init_mem_cost: List[float] = [0.0] * resource.ngpus
    init_comp_cost: List[float] = [0.0] * resource.ngpus
    for node, devices in constraints.op_place.items():
        latency, memory = estimator((node,))
        for devid in devices:
            # FIXME: consider recompute and inflight micro-batches
            # init_mem_cost[devid] += memory / len(devices)
            init_comp_cost[devid] += latency / len(devices)
    for devid in range(resource.ngpus):
        mem_gb = round(init_mem_cost[devid] / (1024**3), 2)
        lat_s = round(init_comp_cost[devid] / 1000, 2)
        _logger.info(f'device [{devid}]: init memory cost: {mem_gb} GB, computation cost: {lat_s} s')

    # - create spmd / stage solver
    spmd_solver = SpmdSolver(cost_model, config.recompute, memory_saving=config.memory_saving)
    if config.dev0_mem_limit_gb is not None:
        spmd_solver.add_device_mem_limit(0, config.dev0_mem_limit_gb)

    stage_solver = StageSolver(
        spmd_solver,
        max_d=config.max_dp_size,
        max_t=config.max_tp_size,
        max_p=config.max_pp_size
    )
    # - search with stage solver
    spec = stage_solver.solve(
        solver_blocks,
        resource.ngpus,
        memory_limit_bytes=mem_limit,
        init_mem_cost=tuple(init_mem_cost),
        init_comp_cost=tuple(init_comp_cost)
    )

    # TODO: add standalone blocks in spec
    return spec


def policy(graph: IRGraph, resource,
           nmicros: int,
           mbs: int,
           constrain_fn: Optional[Callable] = None,
           load_spec_file: Optional[str] = None,
           save_spec_file: Optional[str] = None,
           config: Optional[CupilotConfig] = None) -> IRGraph:
    """
    CuPilot policy entry

    Args:
        graph (IRGraph): execution cube graph
        resource: execution resource
        nmicros (int): number of micro-batch size
        mbs (int): micro-batch size
        constrain_fn (Callable[[IRGraph, Resource, List[IRBlock], Constraints], List[IRBlock]]):
            expert constraint function for the search. 
            The function takes graph, resource, List[IRBlock] and Constraints as inputs,
            and should return List[IRBlocks]. Inside constrain_fn, expert can modify
            the granularity of blocks and add constraints to the graph nodes.
        load_spec_file (str | None): load parallel spec
        save_spec_file (str | None): save parallel spec
        config (CupilotConfig | None): policy search configuration.

    Returns:
        graph (IRGraph): transformed graph
    """
    config = CupilotConfig() if config is None else config
    config.max_dp_size = mbs if config.max_dp_size is None \
        else min(mbs, config.max_dp_size)
    
    # TODO: support estimator with zero
    if config.zero_size > 0:
        CompileFlag.use_zero = True
        CompileFlag.zero_ngroups = config.zero_size

    blocks: List[IRBlock] = IRBlock.blocking(graph)

    # setup recompute
    if config.recompute:
        if len(blocks) == 1:
            warnings.warn(
                'The policy relies on IRGraphAnchor as recompute granularity. '
                'Since only one IRBlock is found, recompute will not be applied.')
        else:
            for blk in blocks:
                graph.recompute(blk.nodes)

    # expertee specifies constraints
    constraints = Constraints()
    if constrain_fn is not None:
        blocks = constrain_fn(graph, resource, blocks, constraints)
        if not all(isinstance(blk, IRBlock) for blk in blocks):
            raise ValueError('Expected constrain_fn returns a list of IRBlock')
        
        # setup block device constraints
        # note: if one of the node inside a block has been constrained to a device group,
        # all the nodes inside a block can only be mapped to the same device group
        for node, (algo, nums) in constraints.op_trans.items():
            if nums is None: continue
            min_tp, max_tp = nums
            # find the block that contains the node
            for blk in blocks:
                if node in blk.nodes:
                    if node in constraints.op_place:
                        devices = sorted(set(constraints.op_place[node]))
                        # this will also include tp size constraints
                        blk.constrain_devices(devices)
                    else:
                        blk.constrain_tp_size(min_tp, max_tp)
                    break
            else:
                raise KeyError(f"Cannot find the constrained node: {node.name}[{node.cid}] in blocks")


    # shrink search space by merging blocks
    blocks = IRBlock.shrink_blocks(blocks, config.max_block_num)
    # blocks = blocks[:4] + IRBlock.shrink_blocks(blocks[4:], config.max_block_num-4)

    # staged_spmd search -- only apply on rank 0 to ensure deterministic
    if DeviceGroup().rank == 0:
        if isinstance(load_spec_file, str):
            _logger.info(f'loading spec from {load_spec_file}...')
            spec = ParallelSpec.load(load_spec_file, graph)
        else:
            spec = staged_spmd(graph,
                               resource, 
                               blocks,
                               constraints,
                               config)
        _logger.info(f'parallel spec results:\n{spec.str(nmicros)}')

        if isinstance(save_spec_file, str):
            _logger.info(f'saving spec to {save_spec_file}...')
            spec.save(save_spec_file)

        state: str = spec.getstate()
        state = torch.tensor([ord(c) for c in state], dtype=torch.int, device=torch.cuda.current_device())
        # notify each node
        for rank in range(DeviceGroup().local_world_size, DeviceGroup().world_size, DeviceGroup().local_world_size):
            _logger.info(f'notify rank {rank} has finished searching...')
            torch.distributed.send(torch.tensor([state.size(0)], device=torch.cuda.current_device()), dst=rank)
            torch.distributed.send(state, dst=rank)
    
    else:
        _logger.info('waiting for rank 0 to finish searching...')
        length = torch.tensor([0], device=torch.cuda.current_device())
        torch.distributed.recv(length, src=0)
        state = torch.empty(length.item(), dtype=torch.int, device=torch.cuda.current_device())
        torch.distributed.recv(state, src=0)
        state = ''.join([chr(c) for c in state.tolist()])
        spec = ParallelSpec.loadstate(state)
        _logger.info(f'parallel spec results:\n{spec.str(nmicros)}')
    
    _logger.info(f'instantiate plan...')

    # group nodes into stages
    fstages: List[IRSegment] = stage_blocks(graph, blocks, spec)
    _logger.info(f'staged-spmd: grouped into {len(fstages)} stages')

    # apply multiref to the nodes where a same tensor is partitioned differently on the nodes.
    auto_multiref(graph, spec)

    # replicate data loader
    devices = list(range(resource.ngpus))
    for dl in graph.select(ntype=IRDataOperation):
        replicate(graph, dl, devices)

    # apply constraints
    for node, devices in constraints.op_place.items():
        algo = constraints.op_trans[node][0]
        num = len(devices)
        if algo is None:
            sub_nodes = graph.replicate(node, times=num)
        else:
            idx, dim = algo
            algo = node.algorithms('dim')
            sub_nodes = graph.partition(node, algo, idx=idx, dim=dim, num=num)
        for snode, device in zip(sub_nodes, devices):
            graph.assign(snode, device)
        # FIXME: the node should also be transformed at batch dimension
        # if the dp_size > 1

    # spmd in stages
    devices = list(range(resource.ngpus))
    for sidx, stage in enumerate(spec.stages):
        tp, dp = stage.tp_size, stage.dp_size
        tp_spec = stage.tp_spec
        stage_devices, devices = devices[:tp*dp], devices[tp*dp:]
        _logger.info(f'applying tp={tp}, dp={dp} for stage {sidx}...')
        for node in fstages[sidx].nodes():
            if isinstance(node, IRGraphAnchor) or node.name == 'multiref':
                continue
            if len(node.device) > 0: continue  # applied by constraints
            if node.cid not in tp_spec:
                _logger.warning(f'node {node.name}({node.cid}) not in tp spec, use replicate')
                replicate(graph, node, stage_devices)
                continue
            tp_strategy = tp_spec[node.cid] if node.cid in tp_spec else None
            
            idxs, dims, nums = [], [], []
            # append data parallelism config
            # FIXME: this may lead to partition error if the node
            # can not be partitioned at idx=0,dim=0.
            idxs.append(0 if isinstance(node, IRDimops) else None)
            dims.append(0 if isinstance(node, IRDimops) else None)
            nums.append(dp)
            # append tensor parallelism config
            idxs.append(None if tp_strategy is None else tp_strategy[0])
            dims.append(None if tp_strategy is None else tp_strategy[1])
            nums.append(tp)
            # apply nested tensor parallelism
            nested_tensor_parallelism(
                graph, node, tuple(idxs), tuple(dims), tuple(nums), stage_devices)

    # at this point, the transformation and placement are done,
    # the graph is only composed by IRSegment.

    # search for schedule
    sched = OrderSolver().solve(graph, nmicros, config.order_plan)
    # _logger.info(f'searched schedule:\n{sched.str(show_max_steps=20)}')
    return graph
