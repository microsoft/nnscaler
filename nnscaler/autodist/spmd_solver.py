from .model_graph import ModelGraph, collect_depth2scope_nodes
from .cube_operator import CubeOperator
from .descs import *
from .cost_database import CostDatabase
from .autodist_config import AutoDistConfig
from .op_partition import OpPartition, generate_partitions
from nnscaler.graph.function.dimops import IRDimops
from nnscaler.ir import IRTensor

import os
import copy
import time
import json
import yaml
import numpy
import logging
from dataclasses import dataclass, asdict
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple, List, Set, Any

__all__ = [
    'SPMDSolver',
    'calc_optimal_spmd_plan',
    'analysis_pretty_printer',
]

_logger = logging.getLogger(__name__)

_PLAN_ANALYSIS_LIST_TIME_TOP_NUM = 10
_PLAN_ANALYSIS_LIST_PARTITIONS_TOP_NUM = 2
_PLAN_ANALYSIS_MODULE_MAX_DEPTH = 4
_PLAN_ANALYSIS_MODULE_TOP_NUM = 3


@dataclass
class PartitionCostDesc:
    # the computation time of a partition
    comp_time: float
    # the communication cost when updating the weights
    # currently, it is estimated by allreduce time
    weight_update_time: float
    # the minimum memory required for the partition, including
    # 1. activation memory
    # 2. weight memory (weight needs gradient)
    # 3. gradient memory, assume the same type as weight memory
    # 4. buffer memory (weight does not need gradient)
    # 5. optimizer resident memory: like 1st and 2nd moment in Adam
    mem: int
    # input memory size, used in recompute
    in_mem: int
    # additional transient mem cost: currently use the maximum tensor size
    transient_mem: int
    # sum of the activation tensor size
    activation_mem: int
    # transient memory size in the optimizer, e,g. some optimizers cast
    # the weight and gradient before stepping
    opt_transient_mem: int
    # comm_time[i][j] is the communication time between current partition
    # and i-th producer's j-th partition
    comm_time: List[List[float]]

    def __repr__(self):
        contents = dict()
        for k, v in asdict(self).items():
            if 'mem' in k:
                k_in_mb = k + ' (MB)'
                contents[k_in_mb] = v // 1024 // 1024
            else:
                contents[k] = v
        return str(contents)

@dataclass
class ModuleMemCostDesc:
    total_cost: int
    resident_mem: int
    activation_mem: int
    opt_transient_mem: int
    recompute_mem: int
    transient_mem: int

    def __repr__(self):
        contents = dict()
        for k, v in asdict(self).items():
            k_in_mb = k + ' (MB)'
            contents[k_in_mb] = v // 1024 // 1024
        return str(contents)


class SPMDSolver:

    def __init__(
        self,
        graph: ModelGraph,
        autodist_config: AutoDistConfig,
        mesh_desc: MeshDesc,
        stage_num: int = 1,
        micro_batch_num: int = 1,
    ):
        self.mesh_desc = mesh_desc
        if mesh_desc.row != 1:
            raise RuntimeError(f'mesh row should be 1, but got {mesh_desc.row}')
        self.device_num = mesh_desc.col
        self.autodist_config = autodist_config
        self.micro_batch_num = micro_batch_num
        self.mem_bound = autodist_config.memory_constraint
        self.verbose = autodist_config.verbose
        self.is_train = autodist_config.is_train
        self.graph = graph
        self.pcs: Dict[str, Dict[str, PartitionConstraint]] = dict()
        self.non_used_pcs: Set[PartitionConstraint] = set()
        if autodist_config.pc_path:
            self._load_partition_constraints(autodist_config.pc_path)
        else:
            _logger.info('no partition constraint is loaded')

        self.cost_database = graph.cost_database
        self.cost_database.profile_comp(self.device_num)
        self.stage_num = stage_num

        # assume the dataflow graph is
        #        a
        #       / \
        #      b   c
        #      |  / \
        #      d  e  f
        #      | /   |
        #      g     h
        # the ops are stored in a topological order [a, b, d, c, e, f, g, h]
        # in spmd solver, dynamic programming is used to find the optimal partition plan
        # dp[p(u), M] is the optimal plan for the subgraph ending with u in partition state p,
        # with memory bound M. if v is the predecessor of u in the topological order, then
        # dp[p(u), M] = min(dp[q(v), M - mem(p(u))] + comm_cost(p(u), q(v))) + comp_cost(p(u)) + comm_cost(p(u))
        # where q(v) is the partition state of v, mem(p(u)) is the memory cost of p(u),
        # comm_cost(p(u), q(v)) is the communication cost between p(u) and q(v), and
        # comp_cost(p(u)) is the computation cost of p(u), comm_cost(p(u)) is the communication
        # cost of p(u) (like the allreduce cost in model update).
        # However, u and v may not be connected in the dataflow graph, like [d, c], [e, f], [f, g]
        # and [g, h] in the example above. To calculate the communication cost between p(u) and q(v),
        # we need to store additional information in the partition state. For example, we need to maintain
        # the partition state of node d in the partition state of node c, so that we can calculate the
        # communication cost when reaching node g.
        # to achieve this, we calcuate the 'cut ops' for each node, which is the set of nodes that
        # need to be maintained in the partition state of the current node. The cut ops for the example
        # above are:
        # a: [a]
        # b: [a, b]
        # d: [a, d]
        # c: [d, c]
        # e: [d, c, e]
        # f: [d, e, f]
        # g: [f, g]
        # h: [h]
        self.initialize()

    def initialize(self):
        self.build_cut_ops()
        self.init_op_partitions()
        self.build_following_relationships()
        self.calc_partition_info()

    def _load_partition_constraints(self, pc_path: str):
        pc_path = Path(pc_path)
        if pc_path.exists():
            try:
                with open(pc_path, 'r') as f:
                    pc_yamls = yaml.safe_load(f)
                    for pc_yaml in pc_yamls:
                        pc = PartitionConstraint.from_yaml(pc_yaml)
                        self.non_used_pcs.add(pc)
                        cur_name_pcs = self.pcs.setdefault(pc.name, {})
                        if pc.parent_module in cur_name_pcs:
                            _logger.warning(
                                f'find duplicate partition constraint in {pc.parent_module}, omit {pc}'
                            )
                        else:
                            cur_name_pcs[pc.parent_module] = pc
            except Exception:
                _logger.exception(
                    f'fail to load partition constraints from {pc_path}')
                self.pcs = dict()
        else:
            _logger.warning(f'pc path {pc_path} does not exist')

    def get_operator(self, idx: int) -> CubeOperator:
        return self.graph.operator_list[idx]

    def get_op_partition_count(self, idx: int) -> int:
        return len(self._op_partitions[idx])

    def init_op_partitions(self):
        '''
        Autodist adopts a heuristic to force operators to be replicated if the following conditions
        are satisfied:
        1. the operator does not have a batch dimension, i.e., the operator is not data dependent
        2. the sum of inputs and outputs size is smaller than `force_replica_threshold`, which implies
        that the operator is small enough to be replicated
        '''
        force_replica_threshold = 0
        for operator in self.graph.operator_list:
            # In modern deep learning models
            # 1. norm operators are in the backbone of the model (layernorm in transformer, batchnorm in CNN, etc.)
            # 2. the output size of a norm operator is related to the batch size (do not replicate norm operators)
            # As a result, if the sum of inputs and outputs size of an operator is smaller than
            # the minimum output size of a norm operator, replicate it is safe.
            if 'norm' in operator.op_name.lower():
                norm_size = operator.out_tensors[0].nelement()
                if force_replica_threshold == 0:
                    force_replica_threshold = norm_size
                else:
                    force_replica_threshold = min(force_replica_threshold,
                                                  norm_size)
        _logger.info(f'force_replica_threshold is {force_replica_threshold}')

        def should_force_replica(operator: CubeOperator) -> bool:
            if operator.has_batch_dim:
                return False
            cnt = 0
            for item in operator.ir_cell.inputs():
                if isinstance(item, IRTensor):
                    cnt += item.nelement()
            for item in operator.ir_cell.outputs():
                if isinstance(item, IRTensor):
                    cnt += item.nelement()
            return cnt < force_replica_threshold

        # Do not allow to partition shared parameters currently
        param2consumers = defaultdict(list)
        for operator in self.graph.operator_list:
            for tensor in operator.ir_cell.inputs():
                if isinstance(tensor, IRTensor) and tensor.is_param():
                    param2consumers[tensor].append(operator.ir_cell)
        shared_param_constraints = defaultdict(set)
        for param, consumers in param2consumers.items():
            if len(consumers) == 1:
                continue
            _logger.info(f'find shared parameter {param} in {consumers}')
            for consumer in consumers:
                if not isinstance(consumer, IRDimops):
                    # always replicate non-dimops
                    continue
                idx = consumer.inputs().index(param)
                shape_anno = consumer.anno.input(idx)
                for dim_anno in shape_anno.dims:
                    shared_param_constraints[consumer].add(dim_anno.name)

        def is_valid_partition(operator: CubeOperator, p_ids: List[Any],
                               p_nums: List[int]) -> bool:
            '''
            use a function to filter invalid partitions. Note: the partition
            representation will be refined in the future.
            Args:
                operator (CubeOperator): the operator to be partitioned
                p_ids  (List[Any]): the partition identifiers, -1 means replicated
                p_nums (List[int]): the partition numbers
            Returns:
                bool: True if the partition is valid, False otherwise

            Examples:
                >>> # matmul with annotation m k+, k+ n -> m n
                >>> # partition on 8 devices, possible inputs
                >>> # partition along dim 'm'
                >>> is_valid_partition(matmul, ['m',], [8,])
                >>> # replicated across all devices
                >>> is_valid_partition(matmul, [-1,], [8,])
                >>> # partition along dim 'm', 'n' and 'k' (currently not supported)
                >>> # each device has a partial value with shape:
                >>> m // 2 k // 2, k // 2 n // 2 -> m // 2, n // 2
                >>> is_valid_partition(matmul, ['m', 'n', 'k'], [2, 2, 2])
                >>> # partition along dim 'm' and -1 (currently not supported)
                >>> # each device has a complete value with shape:
                >>> m // 2 k, k n -> m // 2, n
                >>> is_valid_partition(matmul, ['m', -1], [2, 4])
            '''
            if len(p_ids) != len(p_nums):
                raise RuntimeError(
                    f'invalid partition {p_ids} {p_nums} for {operator.op_name}'
                )

            if self.mesh_desc.col == 1:
                return True

            # in order to reduce search space and simplify the communication pattern,
            # we constrain operators to be partitioned along only one dimension
            for u, v in zip(p_ids, p_nums):
                if v != self.mesh_desc.col:
                    return False

            if len(p_ids) != 1:
                raise RuntimeError(
                    f'exactly one dimension should be partitioned, but got {p_ids} {p_nums}'
                )

            # force replica for non-dimops
            if not isinstance(operator.ir_cell, IRDimops):
                return p_ids[0] == -1

            p_idx, p_dim = operator.dim_id2pos(p_ids[0])

            if operator.ir_cell in shared_param_constraints and isinstance(
                    operator.ir_cell, IRDimops):
                if (p_ids[0] != -1) and (
                        p_ids[0] in shared_param_constraints[operator.ir_cell]):
                    return False

            if operator.op_name in self.pcs:
                if not isinstance(operator.ir_cell, IRDimops):
                    raise RuntimeError(
                        f'operator {operator.op_name} is not a dimops, check the partition constraint'
                    )
                nested_module_type = '.'.join(
                    [module_type.__name__ for _, module_type in operator.ir_cell.module_stack.items()])
                candidate_pcs: List[Tuple[int, PartitionConstraint]] = []
                for pc in self.pcs[operator.op_name].values():
                    name_pos = nested_module_type.rfind(pc.parent_module)
                    if name_pos == -1:
                        continue
                    # use the length of the parent module name to find the closest partition constraint
                    candidate_pcs.append([len(pc.parent_module), pc])
                candidate_pcs.sort(key=lambda x: -x[0])
                if candidate_pcs:
                    selected_pc = candidate_pcs[0][1]
                else:
                    selected_pc = None
                if selected_pc is not None:
                    _logger.debug(
                        f'find partition constraint {selected_pc} for {operator.ir_cell} {nested_module_type}'
                    )
                    self.non_used_pcs.discard(selected_pc)
                    for u, v in zip(p_ids, p_nums):
                        if u == -1:
                            if not selected_pc.replica_allowed:
                                return False
                        else:
                            allowed_pids = [
                                operator.pos2dim_id(pos)
                                for pos in selected_pc.allowed_partition_dims
                            ]
                            if u not in allowed_pids:
                                return False

            if p_ids[0] != -1:
                if not operator.ir_cell.algorithms('dim').satisfy(
                        p_idx, p_dim, p_nums[0]):
                    return False
            return True

        def build_op_partitions(operator: CubeOperator) -> List[OpPartition]:
            # force replica for non-dimops
            if not isinstance(operator.ir_cell, IRDimops):
                candidates = [((-1,), (self.device_num,))]
            else:
                if should_force_replica(operator):
                    _logger.debug(f'force replica {operator.ir_cell}')
                    candidates = [((-1,), (self.device_num,))]
                elif self.device_num == 1:
                    candidates = [((-1,), (1,))]
                else:
                    # python set is not stable
                    p_dims = [-1] + sorted(operator.parallelable_dims)

                    candidates = generate_partitions(p_dims, self.device_num)

            op_partitions = []
            for dim_ids, p_nums in candidates:
                if is_valid_partition(operator, dim_ids, p_nums):
                    op_partitions.append(
                        OpPartition(
                            partition_dims=dim_ids,
                            partition_nums=p_nums,
                            operator=operator,
                        ))

            return op_partitions

        # generate partitions for each operator
        self._op_partitions: List[List[OpPartition]] = list()
        replicated_ops = defaultdict(list)
        for i, operator in enumerate(self.graph.operator_list):
            self._op_partitions.append(build_op_partitions(operator))
            if not self._op_partitions[-1]:
                raise RuntimeError(
                    f'node {operator} has no valid partition, check profiler, partition constraint and filter'
                )
            if len(self._op_partitions[-1]) == 1 and \
               isinstance(operator.ir_cell, IRDimops):
                if operator.ir_cell == self._op_partitions[-1][0].ir_cell:
                    replicated_ops[operator.ir_cell.signature].append(
                        operator.ir_cell)
        if replicated_ops:
            for signature, ops in replicated_ops.items():
                _logger.debug(f'find {len(ops)} replicated {signature}')
                for op in ops:
                    _logger.debug(f'\t{op}\n\t{op.comment}\n\n')
        if self.non_used_pcs:
            _logger.warning(
                f'find unused partition constraints {self.non_used_pcs}')
        _logger.info('finish building op partitions')

    # use a union-find set to find the oldest operator in a following chain
    def get_father_id(self, i):
        if self.father_ids[i] == i:
            return i
        self.father_ids[i] = self.get_father_id(self.father_ids[i])
        return self.father_ids[i]

    def build_following_relationships(self):
        # self.producers[i]: the indices of the operators that produce tensors for the i-th operator
        self.producers: List[List[int]] = list()
        for i, operator in enumerate(self.graph.operator_list):
            self.producers.append([
                self.graph.get_op_idx(producer)
                for producer in operator.producers
            ])

        # important: build following relationships
        #        a
        #       / \
        #      b   c
        #      |   |
        #      d   e
        #      |   |
        #      f   g
        #       \ /
        #        h
        # a: layer norm
        # b, c: reshape
        # d, e: transpose
        # f, g: view
        # h: matmul
        # assume operators are stored in a topological order [a, b, d, f, c, e, g, h]
        # in order to reduce the search space and keep the partition plan optimal,
        # we group some operators into 4 following chains
        # 1. a
        # 2. b -> d -> f
        # 3. c -> e -> g
        # 4. h
        # in a chain, there are no communication adapters between operators.
        # follow_ids[i] is the index of the operator that i follows, if follow_ids[i] = i, i is the oldest operator in the chain
        # father_ids[i] is the index of the oldest operator in the following chain that i belongs to
        # in the example above,
        # follow_ids = [0, 1, 1, 2, 4, 4, 5, 7]
        # father_ids = [0, 1, 1, 1, 4, 4, 4, 7]
        self.follow_ids = list(range(self.graph.op_num))
        self.father_ids = list(range(self.graph.op_num))

        for i, op in enumerate(self.graph.operator_list):
            # - op consumes tensors from only one producer
            # - op has only one input tensor
            # - the producer has only one input tensor
            if len(self.producers[i]) == 1:
                if len(op.in_tensors) == 1:
                    j = self.producers[i][0]
                    # constrain the following chain starts from a unary operator
                    if len(self.graph.operator_list[j].in_tensors) == 1:
                        self.follow_ids[i] = j
                        self.father_ids[i] = self.get_father_id(j)

        _logger.info('finish building following relationships')

        # after follow, only keep the newest one in cut ops
        for i in range(self.graph.op_num):
            fathers = set()
            pre_cut_ops = copy.copy(self.cut_ops[i])
            self.cut_ops[i] = []
            for j in range(len(pre_cut_ops)):
                u = pre_cut_ops[-1 - j]
                if self.get_father_id(u) in fathers:
                    continue
                else:
                    fathers.add(self.get_father_id(u))
                    self.cut_ops[i].append(u)
            self.cut_ops[i].sort()

        def find_idx_map(src_op, tgt_op):
            ret = []
            for i, src_t in enumerate(src_op.ir_cell.outputs()):
                if not isinstance(src_t, IRTensor):
                    continue
                for j, tgt_t in enumerate(tgt_op.ir_cell.inputs()):
                    if not isinstance(tgt_t, IRTensor):
                        continue
                    if src_t == tgt_t:
                        ret.append((i, j))
            return ret

        # After building following relationships for each operator, we can build the following chains
        # for each operator's each partition. The communication cost between partitions in a
        # following chain is 0, e,g. no communication adapter will be generated.
        # p_fathers[i][j]:
        # assume i-th operator is in the following chain indexed by fi = get_father_id(i),
        # i-th operator's j-th partition follows fi-th operator's p_fathers[i][j]-th partition
        # For example, there is a following chain composed of 3 operators:
        # x1 = layer_norm(x0), annotation: a, b, c^ -> a, b, c^
        # x2 = permute(x1, [0, 2, 1]), annotation: a, b, c -> a, c, b
        # x3 = gelu(x2), annotation: a, b, c -> a, b, c
        # assume x0's shape is [2, 1024, 4096] and the device number is 2
        # then the partitions for each operators are:
        # layer_norm: [(-1,), (2,)], [('a',), (2,)], [('b',), (2,)]
        # permute: [(-1,), (2,)], [('a',), (2,)], [('b',), (2,)], [('c',), (2,)]
        # gelu: [(-1,), (2,)], [('a',), (2,)], [('b',), (2,)], [('c',), (2,)]
        # the p_fathers for each operator are:
        # layer_norm: [0, 1, 2]
        # permute: [0, 1, 2, -1]
        # gelu: [0, 1, -1, 2]
        def calc_father4op_partition():
            p_fathers = []
            father_id2preserved_pids = {}
            for i in range(self.graph.op_num):
                fi = self.get_father_id(i)
                if fi == i:
                    p_fathers.append(list(range(
                        self.get_op_partition_count(i))))
                    father_id2preserved_pids[i] = set(p_fathers[-1])
                else:
                    cur_p_fathers = [-1] * self.get_op_partition_count(i)
                    for producer in self.producers[i]:
                        if self.get_father_id(producer) != fi:
                            continue
                        # assume there is only one tensor from producer to consumer
                        idx_map = find_idx_map(self.get_operator(producer),
                                               self.get_operator(i))
                        if len(idx_map) != 1:
                            raise RuntimeError(
                                f'find multiple or no idx_map {idx_map}')
                        u, v = idx_map[0]
                        for j, tgt_p in enumerate(self._op_partitions[i]):
                            have_changed = False
                            p_father = -1
                            for k, src_p in enumerate(
                                    self._op_partitions[producer]):
                                # use shape to check follow relationship between partitions
                                # TODO: is this correct? what if the shape is the same but the partition is different?
                                if src_p.ir_cell.outputs()[u].shape == tgt_p.ir_cell.inputs()[v].shape and \
                                not src_p.is_partial_val:
                                    p_producer = p_fathers[producer][k]
                                    if p_producer == -1:
                                        p_father = -1
                                    else:
                                        if not have_changed:
                                            p_father = p_producer
                                    have_changed = True
                            # if p_father = -1, this partition will be filtered out
                            if cur_p_fathers[j] != -1:
                                assert p_father == cur_p_fathers[
                                    j], f'{i} {self.get_operator(i).ir_cell} {fi} {self.get_operator(fi).ir_cell}'
                            cur_p_fathers[j] = p_father
                    p_fathers.append(cur_p_fathers)
                    # -1 will be filtered out in the intersection operation below
                    father_id2preserved_pids[fi] = father_id2preserved_pids[
                        fi].intersection(set(p_fathers[-1]))
            return p_fathers, father_id2preserved_pids

        p_fathers, father_id2preserved_pids = calc_father4op_partition()

        # filter useless partitions in following chains
        for i in range(self.graph.op_num):
            filtered_partitions = []
            fi = self.get_father_id(i)
            for p_father, partition in zip(p_fathers[i],
                                           self._op_partitions[i]):
                if p_father in father_id2preserved_pids[fi]:
                    filtered_partitions.append(partition)
            self._op_partitions[i] = filtered_partitions
            if not filtered_partitions:
                raise RuntimeError(
                    f'fail to find valid partition for {self.get_operator(i).ir_cell}'
                )

        self.p_fathers, _ = calc_father4op_partition()

        # reorder partition
        for i in range(self.graph.op_num):
            p_num = self.get_op_partition_count(i)
            if p_num == 1:
                continue
            if self.get_father_id(i) == i:
                continue
            cur_p_fathers = self.p_fathers[i]
            partitions = [None] * p_num
            for j, p_father in enumerate(self.p_fathers[i]):
                if p_father == -1:
                    raise RuntimeError(f'find -1 in p_fathers for operator {i}')
                partitions[p_father] = self._op_partitions[i][j]
            self._op_partitions[i] = partitions
            self.p_fathers[i] = list(range(p_num))

        _logger.info('finish filtering useless partitions')

    def calc_partition_cost(self, op_idx: int, partition_idx: int):
        """
        Calculate the latency, memory, and communication features of a partition option.

        Args:
            op_idx: the index of the current op
            partition_idx: the index of the current partition option

        Returns:
            a PartitionCostDesc object containing the calculated features
        """
        micro_batch_num = self.micro_batch_num
        is_train = self.autodist_config.is_train
        tgt_p = self._op_partitions[op_idx][partition_idx]
        if is_train:
            # only calculate the communication cost for the weight that requires gradient
            weights_require_grad = []
            for in_tensor in tgt_p.operator.ir_cell.inputs():
                if isinstance(in_tensor, IRTensor) and in_tensor.is_param():
                    weights_require_grad.append(in_tensor.requires_grad)
            # currently not support the case that there are two weights, one requires grad, the other not
            # TODO: support this case when we encounter it.
            assert all(weights_require_grad) or not any(
                weights_require_grad
            ), f'expect all weights require grad or not, got {weights_require_grad}'
            if isinstance(tgt_p, IRDimops) and any(weights_require_grad):
                weight_comm_time = self.cost_database.calc_weight_update_time(
                    cur_partition=tgt_p)
            else:
                weight_comm_time = 0
        else:
            weight_comm_time = 0

        if not self.autodist_config.consider_mem:
            node_mem, node_buffer, act_mem, opt_transient_mem, in_mem = 0, 0, 0, 0, 0
        else:
            node_mem, node_buffer, act_mem, opt_transient_mem, in_mem = self.cost_database.get_mem_and_buffer(
                tgt_p, self.is_train, self.stage_num)

        # communication cost induced by partitioning activation tensors of the given op partition
        comm_vecs = []
        for producer in self.producers[op_idx]:
            comm_vec = [0.0] * self.get_op_partition_count(producer)
            for k, src_p in enumerate(self._op_partitions[producer]):
                fw_comm_time = self.cost_database.estimate_comm_cost(
                    src_p, tgt_p, True)
                if is_train and src_p.operator.ir_cell.mirror is not None:
                    bw_comm_time = self.cost_database.estimate_comm_cost(
                        src_p, tgt_p, False)
                else:
                    bw_comm_time = 0
                intra_time = micro_batch_num * (fw_comm_time + bw_comm_time)
                # double check the follow chain
                if self.get_father_id(op_idx) == self.get_father_id(
                        producer) and intra_time == 0:
                    if src_p.operator.ir_cell.mirror is not None:
                        if self.p_fathers[op_idx][
                                partition_idx] != self.p_fathers[producer][k]:
                            _logger.warning(
                                f'Unexpected comm cost, set to inf: {src_p.ir_cell} to {tgt_p.ir_cell}'
                            )
                            intra_time = float('inf')
                comm_vec[k] = intra_time

            comm_vecs.append(comm_vec)

        if isinstance(tgt_p.ir_cell, IRDimops):
            comp_time = self.cost_database.query_comp_time(
                op_or_partition=tgt_p,
                recompute=tgt_p.operator.recompute,
                is_train=is_train)
        else:
            comp_time = 0.0

        return PartitionCostDesc(
            comp_time=micro_batch_num * comp_time,
            weight_update_time=weight_comm_time,
            mem=node_mem,
            in_mem=in_mem,
            transient_mem=node_buffer,
            activation_mem=act_mem,
            opt_transient_mem=opt_transient_mem,
            comm_time=comm_vecs,
        )

    def calc_partition_info(self):
        self.partition_info: List[List[PartitionCostDesc]] = list()
        for i in range(self.graph.op_num):
            cur_info = []
            _logger.debug(f'calc partition info for {self.get_operator(i)}')
            for j in range(self.get_op_partition_count(i)):
                cost_desc = self.calc_partition_cost(i, j)
                if cost_desc.comp_time == float('inf'):
                    _logger.warning(
                        f'profile error {self.get_operator(i).ir_cell}, reset compute time to 0.0'
                    )
                    cost_desc.comp_time = 0.0
                cur_info.append(cost_desc)
                _logger.debug(f'{self._op_partitions[i][j]} {cost_desc}')
            self.partition_info.append(cur_info)
        _logger.info('finish spmd solver initializetion')

    def estimate_min_mem(self, start: int, end: int) -> int:
        '''
        different from the estimation in ModelGraph, this function
        in smaller granularity, i.e., the memory cost of a partition.
        it helps to reduce the search cost in pipeline parallelism.

        Args:
            start (int): the left index of the interval
            end   (int): the right index of the interval

        Returns:
            int: the estimated minimum memory cost of the interval in bytes
        '''
        node_mem, act_mem, opt_mem, tmp_mems = 0, 0, 0, []
        for i in range(start, end + 1):
            cur_node_mem, cur_act_mem, cur_opt_mem, tmp_mem = [], [], [], []
            for j, tgt_p in enumerate(self._op_partitions[i]):
                p_cost_desc = self.partition_info[i][j]
                cur_node_mem.append(p_cost_desc.mem)
                cur_act_mem.append(p_cost_desc.activation_mem)
                cur_opt_mem.append(p_cost_desc.opt_transient_mem)
                tmp_mem.append(p_cost_desc.transient_mem)
            node_mem += min(cur_node_mem)
            act_mem += min(cur_act_mem)
            opt_mem += min(cur_opt_mem)
            tmp_mems.append(min(tmp_mem))
        min_mem = node_mem - act_mem + max(act_mem, opt_mem)
        if not tmp_mems:
            raise RuntimeError('fail to estimate min mem')
        tmp_mems.sort()
        tmp_mems.reverse()
        if len(tmp_mems) == 1 or not self.autodist_config.is_train:
            return min_mem + tmp_mems[0]
        else:
            return min_mem + tmp_mems[0] + tmp_mems[1]

    def gen_min_mem_plan_greedy(self, start: int,
                                end: int) -> List[Tuple[int, int]]:
        '''
        generate the minimum memory plan for the interval [start, end] in a greedy way.
        for each operator, we choose the partition with the minimum memory cost.
        NOTE: do not guarantee the plan satisfies the memory constraint.

        Args:
            start (int): the left index of the interval
            end   (int): the right index of the interval

        Returns:
            List[Tuple[int, int]]: the minimum memory plan
        '''
        plan = []
        for i in range(start, end + 1):
            cur_mem = []
            for desc in self.partition_info[i]:
                cur_mem.append(desc.mem)
            plan.append((i, cur_mem.index(min(cur_mem))))
        return plan

    def calc_mem_cost(self, plan: List[Tuple[int, int]]) -> ModuleMemCostDesc:
        '''
        calculate the memory cost of the plan

        Args:
            plan (List[Tuple[int, int]]): the plan to be evaluated

        Returns:
            ModuleMemCostDesc: the memory cost of the plan in bytes
        '''

        mem, act_mem, opt_transient_mem, transient_mem = 0, 0, 0, []
        for op_idx, p_idx in plan:
            desc = self.partition_info[op_idx][p_idx]
            if self.graph.operator_list[op_idx].recompute:
                if self.graph.operator_list[op_idx].recompute_start_op:
                    mem += desc.in_mem
                mem += desc.mem - desc.activation_mem
            else:
                mem += desc.mem
                act_mem += desc.activation_mem
            opt_transient_mem += desc.opt_transient_mem
            transient_mem.append(desc.transient_mem)
        cost = mem - act_mem + max(act_mem, opt_transient_mem)

        start, end = plan[0][0], plan[-1][0]
        recompute_mem_cost = 0
        for group in self.graph.recompute_group_idxs:
            cur_start, cur_end = group[0], group[-1]
            # do not consider the recompute cost when it is out of the current stage
            if cur_start > end or cur_end < start:
                continue
            if cur_start >= start and cur_end <= end:
                cur_recompute_mem_cost = 0
                for i in range(cur_start, cur_end + 1):
                    p_cost_desc = self.partition_info[i][plan[i - start][1]]
                    cur_recompute_mem_cost += p_cost_desc.activation_mem
                recompute_mem_cost = max(recompute_mem_cost,
                                         cur_recompute_mem_cost)
        cost += recompute_mem_cost

        # A heuristic that helps to estimate the memory cost accurately.
        # It is hard to fully reuse large memory blocks in the cached allocator.
        # In training and inference, we use the top 2 largest inference transient
        # memory cost. In training, we double the cost as a result of the backward pass.
        if transient_mem:
            transient_mem.sort()
            transient_mem.reverse()
            if len(transient_mem) == 1:
                transient_mem_cost = transient_mem[0]
            else:
                transient_mem_cost = transient_mem[0] + transient_mem[1]
            if self.autodist_config.is_train:
                transient_mem_cost *= 2
        cost += transient_mem_cost
        return ModuleMemCostDesc(cost, mem, act_mem, opt_transient_mem, recompute_mem_cost, transient_mem_cost)

    def calc_inner_time_cost(self, plan: List[Tuple[int, int]]) -> float:
        '''
        calculate the inner time cost of the plan: computation time + weight update time

        Args:
            plan (List[Tuple[int, int]]): the plan to be evaluated

        Returns:
            float: the inner time cost of the plan
        '''
        cost = 0.0
        for op_idx, p_idx in plan:
            desc = self.partition_info[op_idx][p_idx]
            cost += desc.comp_time + desc.weight_update_time
        return cost

    def calc_intra_time_cost(self, plan: List[Tuple[int, int]]) -> float:
        '''
        calculate the intra time cost of the plan: communication time between operators

        Args:
            plan (List[Tuple[int, int]]): the plan to be evaluated

        Returns:
            float: the intra time cost of the plan
        '''
        cost = 0.0
        op_idx2p_idx: Dict[int, int] = dict(plan)
        for op_idx, p_idx in plan:
            desc = self.partition_info[op_idx][p_idx]
            for k, comm_vec in enumerate(desc.comm_time):
                producer = self.producers[op_idx][k]
                if not producer in op_idx2p_idx:
                    continue
                cost += comm_vec[op_idx2p_idx[producer]]
        return cost

    def build_cut_ops(self):
        cid2idx = {}
        for i, op in enumerate(self.graph.operator_list):
            cid2idx[op.ir_cell.cid] = i
        out_degs = [len(op.consumers) for op in self.graph.operator_list]
        unclosed_idx = set()
        self.cut_ops: List[List[int]] = list()
        for i, op in enumerate(self.graph.operator_list):
            for pred in op.producers:
                pred_idx = cid2idx[pred.ir_cell.cid]
                assert pred_idx in unclosed_idx
                out_degs[pred_idx] -= 1
                if out_degs[pred_idx] == 0:
                    unclosed_idx.remove(pred_idx)
            ret = list(unclosed_idx) + [i]
            ret.sort()
            self.cut_ops.append(ret)
            if len(op.consumers) > 0:
                unclosed_idx.add(i)

    def _solve_by_ilp(self, start: int, end: int) -> SPMDSearchOutput:
        import pulp
        import multiprocessing
        from pulp import LpVariable, LpProblem, LpMinimize, LpStatus, lpSum, lpDot
        tic = time.time()

        # 1. define the variables
        # s[i][j] = 1 if the i-th operator selects the j-th partition
        s = []
        # e[i][j][k] = 1, the i-th edge's source selects the j-th partition and the destination selects the k-th partition
        e = []

        num_nodes = 0
        for i in range(start, end + 1):
            fi = self.get_father_id(i)
            p_num = self.get_op_partition_count(i)
            if fi == i or fi < start:
                if p_num == 1:
                    s.append([1])
                else:
                    num_nodes += 1
                    s.append(
                        LpVariable.matrix(f's[{i}]', (range(p_num),),
                                          cat='Binary'))
            else:
                s.append(s[fi - start])

        num_edges = 0
        for dst in range(start, end + 1):
            for src in self.producers[dst]:
                j = dst - start
                i = src - start
                # in pipeline parallelism, the producer may be in the previous stage
                # omit the communication cost in this case
                if i < 0:
                    continue
                if len(s[i]) == 1:
                    e.append(s[j])
                elif len(s[j]) == 1:
                    e.append(s[i])
                else:
                    num_edges += 1
                    e.append(
                        LpVariable.matrix(f'e[{i},{j}]',
                                          (range(len(s[i]) * len(s[j])),),
                                          cat='Binary'))

        # NOTE: comment temporarily, refine it later
        # 2. set initial value for warm start
        # plan = self.gen_min_mem_plan_greedy(start, end)
        # for op_idx, p_idx in plan:
        #     s_idx = op_idx - start
        #     if len(s[s_idx]) == 1:
        #         continue
        #     for i in range(len(s[s_idx])):
        #         s[s_idx][i].setInitialValue(i == p_idx)

        # 3. define the objective function
        prob = LpProblem('SPMD', LpMinimize)
        # inner cost
        obj = 0
        for i in range(start, end + 1):
            cost = []
            for desc in self.partition_info[i]:
                cost.append(desc.comp_time + desc.weight_update_time)
            obj += lpDot(s[i - start], cost)

        # intra communication cost
        offset = 0
        for dst in range(start, end + 1):
            dst_p_num = self.get_op_partition_count(dst)
            j = dst - start
            for idx, src in enumerate(self.producers[dst]):
                if src < start:
                    continue
                src_p_num = self.get_op_partition_count(src)
                i = src - start
                cost = [0 for _ in range(src_p_num * dst_p_num)]
                for k, desc in enumerate(self.partition_info[dst]):
                    for l in range(src_p_num):
                        cost[l * dst_p_num + k] = desc.comm_time[idx][l]
                obj += lpDot(e[offset], cost)
                offset += 1
        assert offset == len(e)

        prob += obj
        # 4. define the constraints

        # 4.1. each node can only choose one partition
        for i in range(start, end + 1):
            fi = self.get_father_id(i)
            if fi == i or fi < start:
                prob += lpSum(s[i - start]) == 1

        # 4.2. satisfy memory constraint
        mem = 0
        act_mem = 0
        opt_transient_mem = 0
        max_act_opt_transient = LpVariable('max_act_opt_transient', lowBound=0)
        max_transient = LpVariable('max_transient', lowBound=0)
        for i in range(start, end + 1):
            cur_mem = []
            cur_in_mem = []
            cur_act_mem = []
            cur_param_mem = []
            cur_opt_transient_mem = []
            cur_transient_mem = []
            for desc in self.partition_info[i]:
                cur_mem.append(desc.mem)
                cur_in_mem.append(desc.in_mem)
                cur_act_mem.append(desc.activation_mem)
                cur_param_mem.append(desc.mem - desc.activation_mem)
                cur_opt_transient_mem.append(desc.opt_transient_mem)
                cur_transient_mem.append(desc.transient_mem)
            if not self.graph.operator_list[i].recompute:
                mem += lpDot(s[i - start], cur_mem)
                act_mem += lpDot(s[i - start], cur_act_mem)
            else:
                if self.graph.operator_list[i].recompute_start_op:
                    mem += lpDot(s[i - start], cur_in_mem)
                mem += lpDot(s[i - start], cur_param_mem)
            opt_transient_mem += lpDot(s[i - start], cur_opt_transient_mem)
            prob += lpDot(s[i - start], cur_transient_mem) <= max_transient
        recompute_mem = LpVariable('recompute_mem', lowBound=0)
        for group in self.graph.recompute_group_idxs:
            cur_start, cur_end = group[0], group[-1]
            if cur_start > end or cur_end < start:
                continue
            if cur_start >= start and cur_end <= end:
                cur_group_mem = 0
                for i in range(cur_start, cur_end + 1):
                    cur_act_mem = []
                    for desc in self.partition_info[i]:
                        cur_act_mem.append(desc.activation_mem)
                    cur_group_mem += lpDot(s[i - start], cur_act_mem)
                prob += cur_group_mem <= recompute_mem
            else:
                _logger.warning(
                    f'interval {start} {end} and recompute group {cur_start} {cur_end} overlap'
                )
        prob += act_mem <= max_act_opt_transient
        prob += opt_transient_mem <= max_act_opt_transient
        if self.autodist_config.is_train:
            transient_coef = 4
        else:
            transient_coef = 2
        prob += mem - act_mem + max_act_opt_transient + transient_coef * max_transient + recompute_mem <= self.mem_bound

        # 4.3. constraint over e
        offset = 0
        for dst in range(start, end + 1):
            for src in self.producers[dst]:
                if src < start:
                    continue
                dst_p_num = self.get_op_partition_count(dst)
                src_p_num = self.get_op_partition_count(src)
                if dst_p_num == 1 or src_p_num == 1:
                    offset += 1
                    continue
                prob += lpSum(e[offset]) == 1
                j = dst - start
                i = src - start
                for row in range(src_p_num):
                    prob += lpSum([
                        e[offset][row * dst_p_num + col]
                        for col in range(dst_p_num)
                    ]) <= s[i][row]
                for col in range(dst_p_num):
                    prob += lpSum([
                        e[offset][row * dst_p_num + col]
                        for row in range(src_p_num)
                    ]) <= s[j][col]
                offset += 1
        assert offset == len(e)
        assert 'PULP_CBC_CMD' in pulp.listSolvers(onlyAvailable=True), (
            "Please install ILP solvers by 'sudo apt install coinor-cbc'")

        solver = pulp.PULP_CBC_CMD(mip=True,
                                   msg=self.verbose,
                                   timeLimit=600,
                                   threads=multiprocessing.cpu_count())

        prob.solve(solver)
        status = prob.status
        objective = pulp.value(prob.objective)
        # corner case: no variables
        if num_nodes == 0:
            assert num_edges == 0
            objective = obj.constant
        else:
            objective = float(objective) if objective is not None else -1.0
        _logger.debug(f'\n {prob}')
        _logger.debug(
            f'status: {status}, objective: {objective}, time: {time.time() - tic}'
        )
        if prob.status in [pulp.LpStatusInfeasible] or objective < 0:
            return None

        def get_non_zero_index(binary_vector):
            """Get the index of non-zero item in a vector."""
            ct = 0
            ret = None
            for i, elem in enumerate(binary_vector):
                if pulp.value(elem):
                    ret = i
                    ct += 1
            assert ct == 1
            return ret

        s_val = [-1] * (end - start + 1)
        for i in range(start, end + 1):
            s_val[i - start] = get_non_zero_index(s[i - start])
        e_val = [-1] * len(e)
        offset = 0
        for dst in range(start, end + 1):
            for src in self.producers[dst]:
                if src < start:
                    continue
                j = dst - start
                i = src - start
                e_val[offset] = get_non_zero_index(e[offset])
                i_spec_index = e_val[offset] // len(s[j])
                j_spec_index = e_val[offset] % len(s[j])
                assert s_val[i] == i_spec_index
                assert s_val[j] == j_spec_index
                offset += 1
        plans = []
        all_time_cost = objective
        inner_time_cost = 0
        for i in range(start, end + 1):
            plans.append((i, s_val[i - start]))
            p_cost_desc = self.partition_info[i][s_val[i - start]]
            inner_time_cost += p_cost_desc.comp_time + p_cost_desc.weight_update_time
        mem_cost = self.calc_mem_cost(plans).total_cost
        return SPMDSearchOutput(self.partition_path2desc(plans),
                                mem_cost / 1024 / 1024 / 1024, all_time_cost,
                                inner_time_cost)

    def do_ilp(self, intervals: List[Tuple[int, int]],
               topk: int) -> List[List[SPMDSearchOutput]]:
        if topk != 1:
            raise RuntimeError('topk != 1 is not supported')
        ret = []
        for start, end in intervals:
            solver_out = self._solve_by_ilp(start, end)
            if solver_out is not None:
                ret.append([solver_out])
            else:
                ret.append([])
            _logger.debug(f'finish solving interval {start} {end}')
        return ret

    def do_dp(self, intervals: List[Tuple[int, int]],
              topk: int) -> List[List[SPMDSearchOutput]]:
        import cppimport.import_hook
        import nnscaler.autodist.dp_solver as dp_solver

        mode = 0 if self.is_train else 1
        mem_div = 64
        mem_bound = int(self.mem_bound) // mem_div
        solver = dp_solver.DPSolver(self.autodist_config.verbose, mode, mem_bound, mem_div, topk)
        for start, end in intervals:
            solver.add_interval(start, end)
        for idx in range(self.graph.op_num):
            solver.add_node(idx, self.father_ids[idx], self.cut_ops[idx],
            self.producers[idx], self.get_op_partition_count(idx))
            for i, partition in enumerate(self._op_partitions[idx]):
                p_cost_desc = self.partition_info[idx][i]
                solver.add_partition(idx, i, p_cost_desc.comp_time + p_cost_desc.weight_update_time,
                p_cost_desc.mem // mem_div, p_cost_desc.transient_mem // mem_div,
                p_cost_desc.activation_mem // mem_div, p_cost_desc.opt_transient_mem // mem_div,
                self.p_fathers[idx][i], p_cost_desc.comm_time)
        solver.solve()
        ret = []
        for start, end in intervals:
            cpp_results = solver.get_results(start, end)
            descs = []
            for result in cpp_results:
                desc = self.partition_path2desc(result.path)
                descs.append(SPMDSearchOutput(desc, result.memory * mem_div / 1024 / 1024 / 1024, result.all_time, result.inner_time))
            ret.append(descs)
        return ret

    def analyze_plan(self, plan: List[Tuple[int, int]]) -> Dict[str, Any]:
        """
        Analyze the given plan and return the analysis results.
        The analysis includes:
        - Computation Related
            - the total computation time
            - the top-10 operators that consume the most computation time
            - detailed partition plans for the top-2 operators that consume the most computation time
        - Communication Related
            - the total communication time
            - the top-10 operators that consume the most communication time
        - Memory Related
            - the top-3 modules that consume the most memory in depth 1-4
        - Detailed Partition Plans for each IRDimops

        Args:
            plan (List[Tuple[int, int]]): the plan to be analyzed

        Returns:
            Dict[str, Any]: the analysis results
        """
        ret = dict()
        start, end = plan[0][0], plan[-1][0]

        # top 10 operators grouped by signature that:
        # - consume the most computation time
        # - consume the most communication time
        sig2comp_time = dict()
        sig2comm_time = dict()
        op_idx2comp_time = dict()
        op_idx2comm_time = dict()
        dimops_split_info = list()
        sig2split_info = dict()
        comp_time_sum, comm_time_sum = 0, 0
        for op_idx, p_idx in plan:
            desc = self.partition_info[op_idx][p_idx]
            node = self.graph.operator_list[op_idx].ir_cell
            sig = node.signature
            if sig not in sig2comp_time:
                sig2comp_time[sig] = 0
            if sig not in sig2split_info:
                sig2split_info[sig] = []
            sig2comp_time[sig] += desc.comp_time
            op_idx2comp_time[op_idx] = desc.comp_time
            comm_cost = 0
            for k, comm_vec in enumerate(desc.comm_time):
                producer = self.producers[op_idx][k]
                # do not consider the communication cost between the node in the interval
                # to its producer outside the interval currently
                if start <= producer <= end:
                    producer_p_idx = plan[producer - start][1]
                    comm_cost += comm_vec[producer_p_idx]
            op_idx2comm_time[op_idx] = comm_cost
            if isinstance(node, IRDimops):
                partition_repr = (repr(node), repr(node.anno), node.comment, repr(self._op_partitions[op_idx][p_idx]))
                split_info = (partition_repr, desc.comp_time, comm_cost)
                dimops_split_info.append(split_info)
                sig2split_info[sig].append(split_info)
            if comm_cost == 0:
                continue
            if sig not in sig2comm_time:
                sig2comm_time[sig] = 0
            sig2comm_time[sig] += comm_cost
            comp_time_sum += desc.comp_time
            comm_time_sum += comm_cost

        sig2comp_time = sorted(sig2comp_time.items(), key=lambda x: x[1], reverse=True)
        comp_sig_num = min(_PLAN_ANALYSIS_LIST_TIME_TOP_NUM, len(sig2comp_time))
        sig2comp_time = sig2comp_time[:comp_sig_num]
        top_comp_time_sum = sum([x[1] for x in sig2comp_time])
        ret['comp_time_sum'] = comp_time_sum
        ret[f'top_comp_time'] = sig2comp_time
        ret[f'top_comp_time_sum'] = top_comp_time_sum

        # in addition list partition plans for top comp time operators
        top_op_split_info = {}
        for sig, _ in sig2comp_time[:min(_PLAN_ANALYSIS_LIST_PARTITIONS_TOP_NUM, len(sig2comp_time))]:
            top_op_split_info[sig] = sig2split_info[sig]
        ret['top_op_split_info'] = top_op_split_info

        sig2comm_time = sorted(sig2comm_time.items(), key=lambda x: x[1], reverse=True)
        comm_sig_num = min(_PLAN_ANALYSIS_LIST_TIME_TOP_NUM, len(sig2comm_time))
        sig2comm_time = sig2comm_time[:comm_sig_num]
        top_comm_time_sum = sum([x[1] for x in sig2comm_time])
        ret['comm_time_sum'] = comm_time_sum
        ret[f'top_comm_time'] = sig2comm_time
        ret[f'top_comm_time_sum'] = top_comm_time_sum

        # similar to analysis in the raw graph, we list the top-3 modules that:
        # - consume the most computation time
        # - consume the most communication time
        # - consume the most memory
        # to reduce the complexity, we only consider the modules:
        # - 1 <= depth <= _PLAN_ANALYSIS_MODULE_MAX_DEPTH
        # - in the interval [start, end]
        # - composed of more than one operator
        ret['module_analysis'] = {}
        op_idx2plan_offset = {op_idx: i for i, (op_idx, _) in enumerate(plan)}
        depth2scope_nodes = collect_depth2scope_nodes(self.graph.scope_tree_root)
        for depth, scope_nodes in depth2scope_nodes.items():
            if depth == 0 or depth > _PLAN_ANALYSIS_MODULE_MAX_DEPTH:
                continue
            content = {'comp_time': [], 'comm_time': [], 'mem': []}
            info = list()
            for scope_node in scope_nodes:
                # currently do not consider the module that is not in the interval
                if scope_node.start < start or scope_node.end > end:
                    continue
                # skip modules composed of only one operator, since they are covered
                # at the operator level analysis
                if scope_node.start == scope_node.end:
                    continue
                comp_time, comm_time = 0, 0
                for op_idx in range(scope_node.start, scope_node.end + 1):
                    comp_time += op_idx2comp_time[op_idx]
                    comm_time += op_idx2comm_time[op_idx]
                sub_plan_start = op_idx2plan_offset[scope_node.start]
                sub_plan_end = op_idx2plan_offset[scope_node.end]
                sub_plan = plan[sub_plan_start:sub_plan_end + 1]
                mem_cost = self.calc_mem_cost(sub_plan)
                info.append((scope_node.get_full_name(), comp_time, comm_time, mem_cost))
            # sort by comp_time
            info.sort(key=lambda x: x[1], reverse=True)
            for i in range(min(_PLAN_ANALYSIS_MODULE_TOP_NUM, len(info))):
                name, comp_time, _, _ = info[i]
                content['comp_time'].append((name, comp_time))
            # sort by comm_time
            info.sort(key=lambda x: x[2], reverse=True)
            for i in range(min(_PLAN_ANALYSIS_MODULE_TOP_NUM, len(info))):
                name, _, comm_time, _ = info[i]
                content['comm_time'].append((name, comm_time))
            # sort by mem
            info.sort(key=lambda x: x[3].total_cost, reverse=True)
            for i in range(min(_PLAN_ANALYSIS_MODULE_TOP_NUM, len(info))):
                name, _, _, mem = info[i]
                content['mem'].append((name, repr(mem)))
            ret['module_analysis'][depth] = content

        # TODO: generate a visualization of the plan like torch.profiler
        ret['dimops_split_info'] = dimops_split_info

        return ret

    def solve(self, intervals: List[Tuple[int, int]],
              topk: int) -> List[SPMDSearchOutput]:
        '''
        generate the optimal partition plan for operators in the interval [start, end] by
        integer linear programming (ILP) or dynamic programming (DP). Communication cost
        between the node in the interval to its producer outside the interval is not considered.

        Args:
            intervals (List[Tuple[int, int]]): the intervals to be solved
            topk (int): the number of top-k plans for each interval

        Returns:
            List[List[SPMDSearchOutput]]: the top-k partition plans for each interval
        '''
        if self.autodist_config.solver == 'ilp':
            return self.do_ilp(intervals, topk)
        elif self.autodist_config.solver == 'dp':
            return self.do_dp(intervals, topk)
        else:
            raise RuntimeError(
                f'unsupported solver {self.autodist_config.solver}')

    def partition_path2desc(
            self, plans: List[Tuple[int, int]]) -> Dict[int, NodePartitionDesc]:
        '''
        convert the partition representation: (op_idx, partition_idx) to (op_cid, partition_desc)

        Args:
            plans (List[Tuple[int, int]]): the partition plan to be converted

        Returns:
            Dict[int, NodePartitionDesc]: the converted partition plan
        '''
        partitions = [self._op_partitions[u][v] for u, v in plans]

        partition_descs = {}
        for p in partitions:
            op = p.operator
            p_info = tuple([
                (op.dim_id2pos(dim), num)
                for dim, num in zip(p.partition_dims, p.partition_nums)
            ])
            partition_descs[op.ir_cell.cid] = NodePartitionDesc(desc=p_info)

        return TensorParallelDesc(partition_descs=partition_descs,
                                  mesh_desc=self.mesh_desc,
                                  recompute_groups=[],
                                  analysis=self.analyze_plan(plans))


def analysis_pretty_printer(analysis: Dict[str, Any]) -> str:
    ret = ''
    ret += f'Total computation time: {1000.0 * analysis["comp_time_sum"]:.2f} ms\n'
    ret += f'Top {_PLAN_ANALYSIS_LIST_TIME_TOP_NUM} of operators that consume the most computation time:\n'
    for sig, time in analysis[f'top_comp_time']:
        ret += f'    {sig}: {1000.0 * time:.2f} ms\n'
    ret += f'Top {_PLAN_ANALYSIS_LIST_TIME_TOP_NUM} of operators computation time sum: {1000.0 * analysis["top_comp_time_sum"]:.2f} ms\n'
    ret += '\n'
    ret += f'Top {_PLAN_ANALYSIS_LIST_PARTITIONS_TOP_NUM} operators split info:\n'
    for sig, split_info in analysis[f'top_op_split_info'].items():
        ret += f'    {sig}:\n'
        for partition_repr, comp_time, comm_time in split_info:
            node_repr, anno, comment, partition_info = partition_repr
            ret += f'        {node_repr}\n'
            ret += f'        {comment}\n'
            ret += f'        {anno}, {partition_info}, comp_time: {1000.0 * comp_time:.2f} ms, comm_time: {1000.0 * comm_time:.2f} ms\n\n'
        ret += '\n'
    ret += f'Total communication time: {1000.0 * analysis["comm_time_sum"]:.2f} ms\n'
    ret += f'Top {_PLAN_ANALYSIS_LIST_TIME_TOP_NUM} operators that consume the most communication time:\n'
    for sig, time in analysis[f'top_comm_time']:
        ret += f'    {sig}: {1000.0 * time:.2f} ms\n'
    ret += f'Top {_PLAN_ANALYSIS_LIST_TIME_TOP_NUM} of operators communication time sum: {1000.0 * analysis[f"top_comm_time_sum"]:.2f} ms\n'
    ret += '\n'
    ret += 'Module analysis:\n'
    for depth, content in analysis['module_analysis'].items():
        ret += f'Depth {depth}:\n'
        ret += f'    Top {_PLAN_ANALYSIS_MODULE_TOP_NUM} modules that consume the most computation time:\n'
        for name, time in content['comp_time']:
            ret += f'        {name}: {1000.0 * time:.2f} ms\n'
        ret += f'    Top {_PLAN_ANALYSIS_MODULE_TOP_NUM} modules that consume the most communication time:\n'
        for name, time in content['comm_time']:
            ret += f'        {name}: {1000.0 * time:.2f} ms\n'
        ret += f'    Top {_PLAN_ANALYSIS_MODULE_TOP_NUM} modules that consume the most memory:\n'
        for name, mem_desc in content['mem']:
            ret += f'        {name}: {mem_desc}\n'
    return ret


def calc_optimal_spmd_plan(
        model_graph: ModelGraph,
        autodist_config: AutoDistConfig) -> PipelineSearchOutput:
    '''
        calculate the optimal sigle-program-multiple-data plan for the input graph,
        the returned plan is wrapped in a PipelineSearchOutput object

        Args:
            model_graph (ModelGraph): the wrapped input IRGraph
            autodist_config (AutoDistConfig): the configuration for AutoDist

        Returns:
            PipelineSearchOutput: the optimal plan
    '''
    spmd_solver = SPMDSolver(
        graph=model_graph,
        mesh_desc=autodist_config.mesh_desc,
        autodist_config=autodist_config,
        stage_num=1,
        micro_batch_num=autodist_config.update_freq,
    )

    spmd_outs = spmd_solver.solve([(0, model_graph.op_num - 1)], 1)[0]
    if not spmd_outs:
        raise RuntimeError(
            'fail to find a valid partition plan, ' \
            'try to increase device number or reduce batch size'
        )
    spmd_out = spmd_outs[0]
    pp_desc = PipelineParallelDesc(
        spmd_descs=[spmd_out.desc],
        recompute_groups=spmd_out.desc.recompute_groups,
        mesh_desc=spmd_out.desc.mesh_desc,
    )
    pp_out = PipelineSearchOutput(
        desc=pp_desc,
        e2e_time=spmd_out.all_time,
        stage_mems=[spmd_out.memory],
        stage_all_times=[spmd_out.all_time],
        stage_comp_times=[spmd_out.comp_time],
    )
    return pp_out
