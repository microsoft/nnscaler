#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from typing import List, Tuple, Union, Callable, Dict
import json
import os
from os import listdir
from pathlib import Path
import logging
import multiprocessing
import torch

from nnscaler.graph import IRGraph
from nnscaler.ir.cten import IRTensor
from nnscaler.profiler.database import ProfileDataBase, ProfiledMetrics
from nnscaler.algorithm.ops.dimops import gen_partitions
from nnscaler.ir.operator import IRFwOperation
from nnscaler.graph.function.pyfunc import IRPyFunc
from nnscaler.graph.function.anchor import IRGraphAnchor
from nnscaler.graph.function.dimops import DimopSplit, IRDimops
import nnscaler.resources

from .autodist_config import AutoDistConfig

_logger = logging.getLogger(__name__)

_DEFAULT_COMM_DATA_PATH = nnscaler.resources.files() / 'profile/mi200/comm'


def _piecewise_estimator(xs: List[float], ys: List[float], x: float) -> float:
    """
    Piecewise linear estimator.

    Args:
        xs: x coordinates of the points.
        ys: y coordinates of the points.
        x: x coordinate of the query point.

    Returns:
        y coordinate of the query point.
    """
    if x <= xs[0]:
        return ys[0]
    # Communication profile results vary across a large data range, e,g. x1 < x2 but y1 > y2.
    # To make sure the returned time is always positive, using linear approximation when the
    # message size is very large (>512MB).
    if x >= xs[-1]:
        assert xs[-1] > 0 and ys[
            -1] > 0, f'Unexpected val x={x}, xs={xs}, ys={ys}'
        if xs[-1] < 512:
            _logger.warning(
                f'Estimation may be inaccurate for x={x} MB, xs={xs[-1]} MB, ys={ys[-1]} s'
            )
        return x / xs[-1] * ys[-1]
    for i in range(len(xs) - 1):
        if xs[i] <= x < xs[i + 1]:
            return ys[i] + (x - xs[i]) * (ys[i + 1] - ys[i]) / (xs[i + 1] -
                                                                xs[i])
    raise RuntimeError(f'x={x}, xs={xs}, ys={ys}, should not reach here')


def _filter_nodes(graph: IRGraph, db: ProfileDataBase) -> List[List[IRFwOperation]]:
    visited_nodes = set()
    node_to_profile = list()
    for node in graph.select(ntype=IRFwOperation):
        if isinstance(node, (IRGraphAnchor, IRPyFunc)):
            continue
        hash_code = node.signature + ' : ' + db._serialize(node)
        if hash_code in visited_nodes:
            continue
        node_to_profile.append(node)
        visited_nodes.add(hash_code)
    return node_to_profile


def _group_nodes(node_to_profile: List[IRFwOperation], group_num: int) -> List[List[IRFwOperation]]:
    node_groups = [[] for _ in range(group_num)]
    for i, node in enumerate(node_to_profile):
        node_groups[i % group_num].append(node)
    return node_groups


def _profile_nodes(nodes: List[IRFwOperation], db: ProfileDataBase, partition_degree: int, re_profile: bool):
    ret = list()
    for node in nodes:
        if isinstance(node, IRDimops):
            partition_nodes = gen_partitions(node,
                                             partition_degree,
                                             base=partition_degree,
                                             depth=1)
        else:
            partition_nodes = [node]
        for partition_node in partition_nodes:
            profiled_metrics: ProfiledMetrics = db.profile(partition_node, override=re_profile)
            ret.append((partition_node.signature, db._serialize(partition_node), profiled_metrics))
    return ret


def _profile_graph(dilled_info: str, dev_id: int, partition_degree: int, re_profile: bool, comp_profile_path: str, result: multiprocessing.Queue):
    import dill
    torch.cuda.set_device(dev_id)

    id_state, dilled_graph = dill.loads(dilled_info)
    graph = IRGraph.from_dill(id_state, dilled_graph)
    db = ProfileDataBase()
    db.load_ops(comp_profile_path)
    node_to_profile = _filter_nodes(graph, db)
    nodes = _group_nodes(node_to_profile, group_num=torch.cuda.device_count())[dev_id]
    ret = _profile_nodes(nodes, db, partition_degree, re_profile)
    _logger.info(f'device {dev_id} finished profiling {len(nodes)} nodes')
    result.put(ret)


def _load_comm_data(profile_dir: Path, plan_ngpus: int) -> Dict[str, Dict[str, List[Tuple[float, float]]]]:
    '''
    Load communication profile data from the profile directory. If the data is not found, use the default data
    at _DEFAULT_COMM_DATA_PATH. Note that in autodist's current design, we only consider the communication
    cost across 2^n devices, where n is an positive integer. For example, if plan_ngpus is 8, we will try to
    load intra_2.json, intra_4.json, and intra_8.json from the profile directory. If any of the files is not
    found, we will use the default data as well.
    '''
    def loader(path: Path):
        if not os.path.exists(path):
            return False, None
        info = {}
        dev = 2
        while dev <= plan_ngpus:
            fname = f'intra_{dev}.json'
            if not (path / fname).exists():
                return False, None
            with open(path / fname, 'r') as f:
                info[fname] = json.load(f)
            dev *= 2
        return True, info

    comm_path = profile_dir / 'comm'
    success, comm_info = loader(comm_path)
    if not success:
        _logger.warning(f'Communication profile data not found, using default data at {_DEFAULT_COMM_DATA_PATH}')
        success, comm_info = loader(Path(_DEFAULT_COMM_DATA_PATH))
    if not success:
        raise RuntimeError(f'Communication profile data is not compatible with plan_ngpus {plan_ngpus}')
    return comm_info


class CostDatabase:

    def __init__(self, graph: IRGraph, profile_dir: str, plan_ngpus: int, memory_granularity: int, ignore_small_tensor_threshold: int):
        self.graph = graph

        self.profile_dir = Path(profile_dir)
        self.db = ProfileDataBase()
        self.comp_profile_path = self.profile_dir / 'comp'
        if not self.comp_profile_path.exists():
            self.comp_profile_path.mkdir(parents=True)
        self.db.load_ops(self.comp_profile_path)

        self.comm_info = _load_comm_data(self.profile_dir, plan_ngpus)

        self.memory_granularity = memory_granularity
        self.ignore_small_tensor_threshold = ignore_small_tensor_threshold

    def profile_comp(self, partition_degree: int, parallel_profile: bool, re_profile: bool):
        def insert_profile_info(info: List[Tuple[str, str, ProfiledMetrics]]):
            for sign, serialized, profiled_metrics in info:
                _logger.debug(f'profiled {sign} in {serialized} with {profiled_metrics}')
                if not self.db.exist_serialized(sign, serialized):
                    self.db.insert(sign, serialized, profiled_metrics)

        if parallel_profile:
            _logger.info('Profiling in parallel')
            # use spawn to make sure the profiling process is independent from each other
            # and the main process, this is also required by torch
            mp_context = multiprocessing.get_context('spawn')

            results = mp_context.Queue()
            processes = []
            for i in range(torch.cuda.device_count()):
                p = mp_context.Process(target=_profile_graph,
                                       args=(self.graph.dumps(), i, partition_degree, re_profile, self.comp_profile_path, results))
                processes.append(p)
                p.start()

            # put queue.get() before join to avoid deadlock
            for p in processes:
                ret = results.get()
                insert_profile_info(ret)
            results.close()

            for p in processes:
                p.join()
        else:
            _logger.info('Profiling in serial')
            node_to_profile = _filter_nodes(self.graph, self.db)
            ret = _profile_nodes(node_to_profile, self.db, partition_degree, re_profile)
            insert_profile_info(ret)

        self.db.dump_ops(self.comp_profile_path, override=True)

    def exist(self, node: IRFwOperation) -> bool:
        return self.db.exist(node)

    def query_profiled_metrics(
        self, obj: Union[IRFwOperation, 'CubeOperator', 'OpPartition']
    ) -> ProfiledMetrics:
        node = obj if isinstance(obj, IRFwOperation) else obj.ir_cell
        if not self.exist(node):
            raise RuntimeError(f'cannot find {node} in the profile database')
        return self.db.query(node)

    def round(self, mem):
        if mem % self.memory_granularity == 0:
            return mem
        else:
            return (mem + self.memory_granularity) // self.memory_granularity * self.memory_granularity

    def filter_then_sum(self, tensor_sizes: Tuple[int], mask=[]):
        # assert len(tensor_sizes) == len(
        #     mask), f'len(tensor_sizes) is not equal to len(masks)'
        masked_sizes = [i * j for i, j in zip(tensor_sizes, mask)]
        return sum(masked_sizes)

    def get_mems(self, op_partition):
        memory_types = ['train', 'infer', 'input', 'param', 'buffer']
        memory_results = {}
        for memory_type in memory_types:
            if isinstance(op_partition.operator.ir_cell, IRDimops):
                mem = self.query_single_mem(op_partition, memory_type)
            else:
                mem = 0
            memory_results[memory_type] = mem
        return memory_results

    def get_mem_and_buffer(
        self,
        op_partition,
        is_train: bool,
        stage_num: int,
        world_size: int,
        plan_ngpus: int,
        zero_stage: int,
        zero_ngroups: int,
        opt_resident_coef: float,
        opt_transient_coef: float
    ) -> Tuple[int, int, int, int, int]:
        """
        Get the memory consumption and buffer memory consumption of a partition option.

        Args:
            op_partition: the partition option to be calculated
            is_train: whether the partition is for training
            stage_num: the number of stages
            world_size: the total number of devices
            plan_ngpus: the number of GPUs planned
            zero_stage: the zero optimization stage
            zero_ngroups: the number of zero optimization groups
            opt_resident_coef: the coefficient for optimizer resident memory
            opt_transient_coef: the coefficient for optimizer transient memory

        Returns:
            node_mem: the memory consumption of the partition option
            node_buffer: the buffer memory consumption of the partition option
            activation_mem: the activation memory consumption of the partition option
            opt_transient_mem: the optimizer transient memory consumption of the partition option
            input_mem: the input memory consumption of the partition option
        """
        memory_results = self.get_mems(op_partition)
        activation_mem = memory_results['train']
        if zero_stage not in [0, 1]:
            raise RuntimeError(f'invalid zero stage {zero_stage}')

        # estimate optimizer memory consumption for training.
        # no gradient no memory consumption,
        # weight_mem should be 0 when require_grad is false.
        opt_resident_mem, opt_transient_mem = 0, 0
        if is_train and memory_results['param'] > 0:
            if zero_stage == 0:
                weight_mem = memory_results['param']
            else:
                # if zero-1 is used, we assume the full weight is distributed equally
                # among all devices
                weight_mem = self.query_single_mem(op_partition, 'full_weight')
            opt_resident_mem = opt_resident_coef * weight_mem
            opt_transient_mem = opt_transient_coef * weight_mem
            if zero_stage == 1:
                if op_partition.is_replicated():
                    assert world_size % plan_ngpus == 0, f'world_size {world_size} is not divisible by ngpus {plan_ngpus}'
                    scale_factor = world_size // plan_ngpus
                    divisor = scale_factor // zero_ngroups
                else:
                    assert world_size % zero_ngroups == 0
                    divisor = world_size // zero_ngroups
                opt_resident_mem = opt_resident_mem // divisor
                opt_transient_mem = opt_transient_mem // divisor

        # optimizer state + saved activation tensors for backward + param
        # + gradients + buffer tensors (has deduplicated with the saved tensors)
        node_mem = opt_resident_mem + memory_results['train'] + 2 * memory_results['param'] + memory_results['buffer']
        node_mem = node_mem + (stage_num - 1) * activation_mem if is_train else node_mem
        node_buffer = max(memory_results.values()) if is_train else memory_results['infer']

        if node_mem != 0:
            def to_mb(x):
                return x / 1024 / 1024

            _logger.debug(
                f'{op_partition.operator.ir_cell.cid}, {op_partition.ir_cell}, '
                + f'node mem: {to_mb(node_mem)} MB, '
                + f'activation mem: {to_mb(activation_mem)} MB, '
                + f'optimizer transient mem: {to_mb(opt_transient_mem)} MB'
            )

        return node_mem, node_buffer, activation_mem, opt_transient_mem, memory_results['input']

    def query_single_mem(self, obj, memory_type, round=True) -> int:
        """
        Query memory size of a single operator or partition.
        OpPartition represents one partition of an operator.
        CubeOperator represents the full operator before partitioning.

        'input' is the total bytes of the input tensors excluding parameter and buffer tensors.
        'param' is the total bytes of the parameter tensors.
        'buffer' is the total bytes of the buffer tensors.
        'infer' is the peak bytes during op inference.
        'train' is the total bytes of the saved activation tensors for backward.
        'full_weight' is the total bytes of the weight of the full operator.

        Args:
            obj: OpPartition or CubeOperator
            memory_type: 'input', 'param', 'infer', 'train', 'full_weight'
            round: whether to round the memory size up to the nearest multiple of memory_granularity

        Returns:
            memory size in bytes
        """
        from .op_partition import OpPartition
        from .cube_operator import CubeOperator
        if isinstance(obj, OpPartition):
            masks = self.gen_masks(obj.operator)
        else:
            assert isinstance(obj, CubeOperator)
            masks = self.gen_masks(obj)
        if memory_type == 'full_weight' and isinstance(obj, OpPartition):
            profiled_metrics = self.query_profiled_metrics(obj.operator)
        else:
            profiled_metrics = self.query_profiled_metrics(obj)

        if memory_type == 'input':
            mask = masks['input']
            ret = self.filter_then_sum(profiled_metrics.in_mem_info, mask)
        elif memory_type == 'param':
            mask = masks['param']
            ret = self.filter_then_sum(profiled_metrics.param_mem_info, mask)
        elif memory_type == 'buffer':
            mask = masks['buffer']
            ret = self.filter_then_sum(profiled_metrics.buffer_mem_info, mask)
        elif memory_type == 'infer':
            ret = profiled_metrics.infer_memory
        elif memory_type == 'train':
            mask = masks['train']
            ret = self.filter_then_sum(profiled_metrics.train_mem_info, mask)
        elif memory_type == 'full_weight':
            mask = masks['param']
            ret = self.filter_then_sum(profiled_metrics.param_mem_info, mask)
        else:
            raise ValueError(
                f'Invalid memory_type {memory_type} provided. Choose from: ' +
                "'input', 'param', 'buffer', 'infer', 'train', 'full_weight'.")
        if round:
            return self.round(ret)
        else:
            return ret

    def query_comp_time(self,
                        op_or_partition: Union['CubeOperator', 'OpPartition'],
                        recompute: bool = False,
                        is_train: bool = True):
        profiled_metrics = self.query_profiled_metrics(op_or_partition)
        if not is_train:
            return profiled_metrics.fw_span / 1000
        if recompute:
            return (profiled_metrics.fw_span + profiled_metrics.bw_span +
                    profiled_metrics.fw_span) / 1000
        else:
            return (profiled_metrics.fw_span + profiled_metrics.bw_span) / 1000

    def primitive_to_cost(self, dev_num: int, byte_size: int, primitive: str):
        if byte_size == 0:
            return 0
        size_mb = byte_size / 1024 / 1024
        device_setting = f'intra_{dev_num}.json'
        sizes_in_mb, times_in_s = self.comm_info[device_setting][primitive]
        est_time = _piecewise_estimator(sizes_in_mb, times_in_s, size_mb)
        assert est_time >= 0, f'{primitive} {dev_num} comm size: {size_mb} MB, est time: {est_time} s'
        return est_time

    def calc_weight_update_time(self, cur_partition) -> float:
        """
        Calculate communication cost for weight update. Currently cost is evaluated
        by allreduce.

        Args:
            cur_partition: one partition option of the operator

        Returns:
            communication cost in seconds
        """
        # partition_dims and partition_nums represent a concrete partition option of a node
        # if the element in partition_dims is -1, it means the node is replicated.
        # currently, len of partition_dims is 1, we only support partitioning one dimension
        partition_dims = cur_partition.partition_dims
        partition_nums = cur_partition.partition_nums
        # TODO: remove this assertion, support partitioning multiple dimensions
        assert len(
            partition_dims
        ) == 1, f'expect len(partition_dims) == 1, got {len(partition_dims)}'
        full_weight_mem = self.query_single_mem(cur_partition,
                                                'full_weight',
                                                round=False)
        partitioned_weight_mem = self.query_single_mem(cur_partition,
                                                       'param',
                                                       round=False)

        if partitioned_weight_mem == 0:
            return 0
        if full_weight_mem % partitioned_weight_mem == 0:
            mem_weight_spatial_num = full_weight_mem // partitioned_weight_mem
        else:
            # when setting memory granularity > 1, possible that the two numbers are not divisible
            mem_weight_spatial_num = (full_weight_mem + partitioned_weight_mem
                                     ) // partitioned_weight_mem

        replica_num = 1
        for i, partition_dim_name in enumerate(partition_dims):
            if partition_dim_name == -1:
                replica_num *= partition_nums[i]
        all_num = 1
        for num in cur_partition.partition_nums:
            all_num *= num
        weight_update_num = all_num // (mem_weight_spatial_num * replica_num)
        if weight_update_num == 1:
            return 0
        comm_time = self.primitive_to_cost(dev_num=weight_update_num,
                                           primitive='all reduce',
                                           byte_size=partitioned_weight_mem)

        return comm_time

    def estimate_comm_cost(self, src_p, dst_p, is_forward) -> float:
        """
        Estimate communication cost between src partition and dst partition.
        Currently the communication is only for activation tensors.

        Args:
            src_p: the partition of source operator
            dst_p: the partition of destination operator
            is_forward: whether the communication is for only forward pass
                or only backward pass

        Returns:
            communication cost in seconds
        """
        assert len(src_p.partition_nums) == 1 and len(dst_p.partition_nums) == 1

        def comm_cost(tensor: IRTensor, num_devices: int, src_split: DimopSplit,
                      dst_split: DimopSplit, dst_replica: bool,
                      is_forward: bool):
            """
            Calculate communication cost for a single tensor.
            Note for data parallel, we don't consider allreduce cost as it
            will only be performed at the last of iteration.

            Args:
                tensor: the tensor to be communicated
                num_devices: number of devices
                src_split: the split info of the tensor in the source operator
                dst_split: the split info of the tensor in the destination operator
                dst_replica: whether the destination operator is replicated
                is_forward: whether the communication is for only forward pass or
                    only backward pass

            Returns:
                communication cost in seconds
            """
            assert not dst_split.isV()
            assert not tensor.is_attr()
            byte_size = tensor.byte_size()

            def helper(primitive: str):
                return self.primitive_to_cost(num_devices, byte_size, primitive)

            # R: replicated, V: value split, D: dim split
            if src_split.isR():
                if dst_split.isR():
                    if dst_replica:
                        return 0.0
                    else:
                        # identity-allreduce
                        if is_forward:
                            return 0.0
                        else:
                            return helper('all reduce')
                elif dst_split.isD():
                    # split-allgather
                    if is_forward:
                        return 0.0
                    else:
                        return helper('all gather')
            if src_split.isV():
                if dst_split.isR():
                    # allreduce-identity
                    if dst_replica:
                        if is_forward:
                            return helper('all reduce')
                        else:
                            return 0.0
                    else:
                        # allreduce-allreduce
                        return helper('all reduce')
                elif dst_split.isD():
                    if is_forward:
                        return helper('reduce scatter')
                    else:
                        return helper('all gather')
            if src_split.isD():
                # allgahter-reducescatter or allgather-split
                if dst_split.isR():
                    if is_forward:
                        return helper('all gather')
                    else:
                        if dst_replica:
                            return 0.0
                        else:
                            return helper('reduce scatter')
                # all2all-all2all or identity-identity
                if dst_split.isD():
                    return 0.0 if src_split.dims == dst_split.dims else helper(
                        'all to all')
            raise NotImplementedError(
                f'Unknown split type: {src_split} -> {dst_split}')

        src_p_dim, src_p_num = src_p.partition_dims[0], src_p.partition_nums[0]
        dst_p_dim, dst_p_num = dst_p.partition_dims[0], dst_p.partition_nums[0]
        assert src_p_num == dst_p_num
        src_idx, src_dim = src_p.operator.dim_id2pos(src_p_dim)
        dst_idx, dst_dim = dst_p.operator.dim_id2pos(dst_p_dim)
        rule_src, rule_dst = None, None
        if src_idx != -1:
            rule_src = src_p.operator.ir_cell.algorithm('dim').infer(
                src_idx, src_dim, src_p_num)
        if dst_idx != -1:
            rule_dst = dst_p.operator.ir_cell.algorithm('dim').infer(
                dst_idx, dst_dim, dst_p_num)
        cost = 0.0
        for i, src_t in enumerate(src_p.operator.ir_cell.outputs()):
            for j, dst_t in enumerate(dst_p.operator.ir_cell.inputs()):
                if src_t == dst_t:
                    if not is_forward and not src_t.requires_grad:
                        # if the activation does not require grad,
                        # then no backward communication.
                        cost += 0.0
                    else:
                        cost += comm_cost(
                            src_t, src_p_num,
                            rule_src.outputs()[i]
                            if rule_src is not None else DimopSplit(r=True),
                            rule_dst.inputs()[j] if rule_dst is not None else
                            DimopSplit(r=True), dst_idx == -1, is_forward)
                    break
        return cost

    def gen_masks(self, op):
        masks = {}
        profiled_metrics = self.query_profiled_metrics(op)
        inputs = profiled_metrics.in_mem_info
        param = profiled_metrics.param_mem_info
        buffer = profiled_metrics.buffer_mem_info
        train_m = profiled_metrics.train_mem_info

        def helper(mems):
            return [
                0 if mem < self.ignore_small_tensor_threshold else 1
                for mem in mems
            ]

        in_mask = helper(inputs)
        for idx in op.omit_recompute_in_idx:
            in_mask[idx] = 0
        param_mask = helper(param)
        for idx in op.omit_param_idx:
            param_mask[idx] = 0
        train_m_mask = helper(train_m)
        for idx in op.omit_train_idx:
            train_m_mask[idx] = 0
        buffer_mask = helper(buffer)
        for idx in op.omit_buffer_idx:
            buffer_mask[idx] = 0
        # no need to deduplicate inputs, because input tensors are transient.
        # the saved input tensors for backward have been considered in train_m.

        masks = {
            'input': in_mask,
            'param': param_mask,
            'train': train_m_mask,
            'buffer': buffer_mask,
        }
        return masks
