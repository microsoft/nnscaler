#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from __future__ import annotations

from nnscaler.graph.graph import IRGraph
from nnscaler.graph.function.anchor import IRGraphAnchor
from nnscaler.graph.function.dimops import IRDimops
from nnscaler.ir.operator import IRFwOperation
from nnscaler.ir.cten import IRObject, IRTensor
from .cube_operator import CubeOperator
from .autodist_config import AutoDistConfig
from .cost_database import CostDatabase

from dataclasses import dataclass
from collections import deque
import logging
import copy
from typing import List, Tuple, Dict, Any, Callable

_logger = logging.getLogger(__name__)


# expect ops with output tensors are all IRDimops
def should_include(node: IRFwOperation):
    return any(isinstance(t, IRTensor) for t in node.outputs())


def calc_flops(node: IRFwOperation):
    if 'torch.nn.functional.linear' in node.signature:
        assert len(node.inputs()) >= 2
        assert len(node.outputs()) == 1
        ret = 2 * node.inputs()[0].nelement()
        if len(node.inputs()[1].shape) == 2:
            ret = ret * node.inputs()[1].shape[0]
        return ret
    elif 'torch.bmm' in node.signature:
        # this function do not support broadcast
        assert len(node.inputs()) == 2
        assert len(node.outputs()) == 1
        b, m, k = node.inputs()[0].shape
        _, _, n = node.inputs()[1].shape
        return 2 * b * m * n * k
    elif 'torch.matmul' in node.signature:
        assert len(node.inputs()) == 2
        assert len(node.outputs()) == 1
        lhs, rhs = node.inputs()
        out = node.outputs()[0]
        if len(lhs.shape) == 1 and len(rhs.shape) == 1:
            # vector-vector
            ret = lhs.nelement()
        elif len(lhs.shape) == 2 and len(rhs.shape) == 2:
            # matrix-vector
            m, k = lhs.shape
            _, n = rhs.shape
            ret = m * n * k
        elif len(lhs.shape) == 1 and len(rhs.shape) == 2:
            # vector-matrix
            k, n = rhs.shape
            ret = k * n
        elif len(lhs.shape) == 2 and len(rhs.shape) == 1:
            # matrix-vector
            m, k = lhs.shape
            ret = m * k
        elif len(lhs.shape) > 2 or len(rhs.shape) > 2:
            ret = out.nelement()
            if len(lhs.shape) > 2:
                ret = ret * lhs.shape[-1]
            elif len(rhs.shape) > 2:
                ret = ret * rhs.shape[-2]
        else:
            raise RuntimeError(
                f'unsupported matmul {lhs.shape}, {rhs.shape}, {out.shape}')
        return 2 * ret
    return 0


def estimate_mem_lower_bound(
    param_mem: int,
    buffer_mem: int,
    activation_mem: int,
    plan_ngpus: int,
    zero_group_size: int,
    cfg: AutoDistConfig,
) -> float:
    '''
    Given memory consumption of parameters, buffers and activations, and the number of
    pipeline stages (counting from the last stage, including itself), calculate the
    minimum possible memory consumption of each device.
    Assume the activation memory is shared with transient optimizer memory, since activations
    have been deallocated before optimizer's step.
    The minimum memory consumption is achieved when:
    1. activations, parameters, buffers and gradients are distributed evenly across plan_ngpus
    2. the optimizer memory is distributed evenly across zero_group_size (when zero stage 1 is enabled) or plan_ngpus
    '''
    opt_resident_mem = cfg.opt_resident_coef * param_mem
    opt_transient_mem = cfg.opt_transient_coef * param_mem

    # avg memory cost of activation, param (grad), buffer
    activation_mem = activation_mem / plan_ngpus
    param_mem = param_mem / plan_ngpus
    buffer_mem = buffer_mem / plan_ngpus

    # avg opt mem
    if cfg.zero_stage == 1:
        opt_resident_mem = opt_resident_mem / zero_group_size
        opt_transient_mem = opt_transient_mem / zero_group_size
    elif cfg.zero_stage == 0:
        opt_resident_mem = opt_resident_mem / plan_ngpus
        opt_transient_mem = opt_transient_mem / plan_ngpus
    else:
        raise RuntimeError(f'invalid zero stage {cfg.zero_stage}')

    min_single_dev_mem = max(opt_transient_mem, activation_mem) + 2 * param_mem + buffer_mem + opt_resident_mem
    return min_single_dev_mem


def aggregate_common_mem(sub_nodes: List[IRFwOperation],
                         check_connected: bool) -> Tuple[int, int, int]:
    """
    Aggregate the memory size of input tensors, parameter tensors and buffer tensors
    in the subgraph.
    Use IRObject as edges to find the connected components, and check the connectivity
    of the subgraph if check_connected is True.

    Args:
        sub_nodes: a list of IRFwOperation from the whole graph
        check_connected: whether to check the connectivity of the subgraph

    Returns:
        in_mem: the memory size of input tensors
        param_mem: the memory size of parameter tensors
        buffer_mem: the memory size of buffer tensors
    """

    def _unfold_complex(data):
        if isinstance(data, (list, tuple)):
            ret = []
            for d in data:
                ret += _unfold_complex(d)
            return ret
        elif isinstance(data, dict):
            ret = []
            for _, d in data.items():
                ret += _unfold_complex(d)
            return ret
        elif isinstance(data, slice):
            return _unfold_complex([data.start, data.stop, data.step])
        elif isinstance(data, IRObject):
            return [data]
        else:
            return []

    object2producer: Dict[IRObject, IRFwOperation] = dict()
    for node in sub_nodes:
        for complex_output in node.outputs():
            for t in _unfold_complex(complex_output):
                assert isinstance(t, IRObject)
                if t in object2producer:
                    raise RuntimeError(f'tensor {t} has multiple producers')
                object2producer[t] = node

    # use union set to check whether the subgraph is connected
    node2father: Dict[IRDimops, IRDimops] = dict()
    for node in sub_nodes:
        node2father[node] = node

    def get_father(node):
        father = node2father[node]
        if node == father:
            return father
        else:
            father = get_father(father)
            node2father[node] = father
            return father

    def merge(lhs, rhs):
        lhs_father = get_father(lhs)
        rhs_father = get_father(rhs)
        node2father[rhs_father] = lhs_father

    edges: Dict[IRFwOperation, List[IRFwOperation]] = dict()
    in2consumer: Dict[IRObject, List[IRFwOperation]] = dict()
    for node in sub_nodes:
        # deal with both inputs and kwargs to track connecting edges
        complex_inputs = list(node.inputs()) + list(node.kwargs.values())
        for complex_input in complex_inputs:
            for t in _unfold_complex(complex_input):
                assert isinstance(t, IRObject)
                if t not in object2producer:
                    if t not in in2consumer:
                        in2consumer[t] = []
                    in2consumer[t].append(node)
                    continue
                src = object2producer[t]
                if src not in edges:
                    edges[src] = []
                edges[src].append(node)
                merge(src, node)

    for _, consumers in in2consumer.items():
        for i in range(len(consumers) - 1):
            merge(consumers[i], consumers[i + 1])

    components = set()
    for node, father in node2father.items():
        components.add(get_father(node))

    if check_connected and len(components) > 1:
        for i, father in enumerate(components):
            _logger.info(f'{i}-th component')
            for node, _ in node2father.items():
                if get_father(node) == father:
                    _logger.info(node)
        raise RuntimeError('more than one connect component')

    in_mem, param_mem, buffer_mem = 0, 0, 0
    for t, _ in in2consumer.items():
        if not isinstance(t, IRTensor):
            continue
        if t.is_param():
            param_mem += t.byte_size()
        elif t.is_buffer():
            buffer_mem += t.byte_size()
        else:
            in_mem += t.byte_size()
    return in_mem, param_mem, buffer_mem


def aggregate_train_mem(sub_nodes: List[IRFwOperation], db) -> int:
    visited_tensors: Set[IRTensor] = set()
    train_mem = 0
    for node in sub_nodes:
        metrics = db.query(node)
        if metrics is None:
            # if the node is not in the database, skip it currently
            continue
        train_mem2in_idx = metrics.train_mem2in_idx
        train_mem_info = metrics.train_mem_info
        for mem, in_idx in zip(train_mem_info, train_mem2in_idx):
            if in_idx == -1:
                train_mem += mem
            else:
                t = node.inputs()[in_idx]
                # `t` also can be any other unhashable var, if we set ? in annotation
                if isinstance(t, IRTensor) and t not in visited_tensors:
                    train_mem += mem
                    visited_tensors.add(t)
    return train_mem


class ScopeNode:

    def __init__(self,
                 name: str,
                 module_type: Any,
                 parent=None,
                 node: IRFwOperation = None,
                 depth: int = 0,
                 leaf_size: int = 0,
                 flops: int = 0,
                 fw_span: float = 0,
                 start: int = 0,
                 end: int = 0):
        self.name = name
        self.module_type = module_type
        self.children = []
        self.parent = parent
        self.node = node
        self.depth = depth
        self.leaf_size = leaf_size
        self.flops = flops
        self.fw_span = fw_span
        self.in_mem = 0
        self.train_mem = 0
        self.param_mem = 0
        self.buffer_mem = 0
        self.start = start
        self.end = end

    def get_full_name(self):
        if self.module_type is None:
            return self.name
        return f'{self.name}, {self.module_type.__name__}'

    def insert(self, node: IRFwOperation, module_info: List[Tuple[str, Any]],
               flops: int, fw_span: float, idx: int):
        self.leaf_size += 1
        self.flops += flops
        self.fw_span += fw_span
        if len(module_info) == 0:
            child = ScopeNode(node.signature,
                              None,
                              parent=self,
                              node=node,
                              depth=self.depth + 1,
                              leaf_size=1,
                              flops=flops,
                              fw_span=fw_span,
                              start=idx,
                              end=idx)
            self.children.append(child)
            return child
        module_path, module_type = module_info[0]
        for i, child in enumerate(self.children):
            if child.name == module_path:
                if i == len(self.children) - 1:
                    return child.insert(node,
                                        module_info[1:],
                                        flops,
                                        fw_span,
                                        idx=idx)
                else:
                    _logger.warning(
                        f'{node} with {module_info} used multiple times')
        child = ScopeNode(module_path,
                          module_type,
                          parent=self,
                          depth=self.depth + 1)
        ret = child.insert(node, module_info[1:], flops, fw_span, idx=idx)
        self.children.append(child)
        return ret

    @property
    def is_leaf(self):
        return self.node is not None

    @property
    def is_root(self):
        return self.parent is None

    def select(self, func):
        if func(self):
            return [self]
        ret = []
        for child in self.children:
            ret += child.select(func)
        return ret

    # time complexity: O(depth * #nodes)
    def pull_up(self, db):
        # leaf node
        if self.node is not None:
            if not isinstance(self.node, IRFwOperation):
                raise RuntimeError(f'expect IRFwOperation, got {self.node}')
            if isinstance(self.node, IRDimops):
                profiled_metrics = db.query(self.node)
                if profiled_metrics is not None:
                    self.in_mem = sum(profiled_metrics.in_mem_info)
                    self.train_mem = sum(profiled_metrics.train_mem_info)
                    self.param_mem = sum(profiled_metrics.param_mem_info)
                    self.buffer_mem = sum(profiled_metrics.buffer_mem_info)
                else:
                    raise RuntimeError(f'cannot find {self.node} in db')
            else:
                if should_include(self.node):
                    _logger.warning(
                        f'detect a non-IRDimops {self.node.signature} ' + \
                        f'at {self.node.comment} that produces tensors')
            return [self.node]
        sub_nodes = []
        for child in self.children:
            sub_nodes += child.pull_up(db)
        # a sub-module can have more than one connected component, like RoPE
        # we check the connectivity only when self is the root node
        self.in_mem, self.param_mem, self.buffer_mem = aggregate_common_mem(
            sub_nodes, self.parent is None)
        self.train_mem = aggregate_train_mem(sub_nodes, db)
        self.start = self.children[0].start
        self.end = self.children[-1].end
        return sub_nodes

    def query(self, start: int, end: int, cache: Dict[Tuple[int, int], Any],
              leaf_handler: Callable[int, Any], merger: Callable[List[Any],
                                                                 Any]):
        '''
        Boost the query by segment tree and cache
        Args:
            start: the left index of nodes
            end: the right index of nodes
            cache: the cache for query
            leaf_handler: the handler for leaf nodes
            merger: the merger for sub-intervals

        Returns:
            the result of the query
        '''
        if not (self.start <= start and end <= self.end):
            raise RuntimeError(
                f'[{start}, {end}] not in [{self.start}, {self.end}]')
        if (start, end) in cache:
            return cache[(start, end)]
        if start == end:
            ret = leaf_handler(start)
        else:
            # break the interval into sub-intervals
            def get_intersection(x1, y1, x2, y2):
                return max(x1, x2), min(y1, y2)

            sub_rets = []
            for child in self.children:
                x, y = get_intersection(start, end, child.start, child.end)
                if x > y:
                    continue
                sub_rets.append(child.query(x, y, cache, leaf_handler, merger))
            ret = merger(sub_rets)
        cache[(start, end)] = ret
        return ret

    def __repr__(self):
        if self.node is not None:
            return ''
        desc = '  ' * self.depth
        info = [
            self.name,
            str(self.module_type),
            f'depth: {self.depth}',
            f'size: {self.leaf_size}',
            'FLOPs: {0:.3g}B'.format(self.flops / 1e9),
            'fw_span: {0:.3g}ms'.format(self.fw_span),
            # TODO: the node may be a IRPytfunc whose fw_span = 0 currently
            'FLOPS: {0:.3g}T'.format(0. if self.fw_span == 0. else self.flops /
                                     self.fw_span / 1e9),
            'in_mem: {0:.3g}MB'.format(self.in_mem / 1024 / 1024),
            'train_mem: {0:.3g}MB'.format(self.train_mem / 1024 / 1024),
            'param_mem: {0:.3g}MB'.format(self.param_mem / 1024 / 1024),
            'buffer_mem: {0:.3g}MB'.format(self.buffer_mem / 1024 / 1024)
        ]
        desc = desc + ', '.join(info) + '\n'
        for child in self.children:
            desc += child.__repr__()
        return desc


def collect_depth2scope_nodes(root: ScopeNode) -> Dict[int, List[ScopeNode]]:
    depth2scope_nodes: Dict[int, List[ScopeNode]] = dict()

    def dfs(node: ScopeNode):
        if node.depth not in depth2scope_nodes:
            depth2scope_nodes[node.depth] = []
        depth2scope_nodes[node.depth].append(node)
        for child in node.children:
            dfs(child)

    dfs(root)
    return depth2scope_nodes


def analyze_base_graph(root: ScopeNode) -> None:
    '''
    Analyze the input graph's structure and statistics based on profiling results.
    NOTE: if the input graph contains operators that consumes or generates extremely
    large tensors, the profiling result may be incorrect. User should check the
    partition plan's analysis later.
    '''
    depth2scope_nodes = collect_depth2scope_nodes(root)

    # Similar to deepspeed profiler, we list top3 modules in terms of
    # params, buffers, activation mem and fw_span
    show_num = 3
    def get_val(node: ScopeNode, key: str):
        # pretty print the memory size in MB and span in ms for ScopeNode
        val = getattr(node, key)
        if 'mem' in key:
            return f'{val / 1024 / 1024:.2f} MB'
        elif 'span' in key:
            return f'{val:.2f} ms'
        else:
            raise RuntimeError(f'invalid key {key}')

    def build_info(nodes: List[ScopeNode], key: str):
        info = list()
        sorted_nodes = sorted(nodes, key=lambda x: getattr(x, key), reverse=True)
        for node in sorted_nodes[:min(show_num, len(sorted_nodes))]:
            info.append((node.get_full_name(), get_val(node, key)))
        return info

    visual_contents = dict()
    for depth, scope_nodes in depth2scope_nodes.items():
        # ignore the root node, since it doesn't have module info
        if depth == 0:
            continue
        visual_contents[depth] = dict()
        for key in ['param_mem', 'fw_span', 'train_mem', 'buffer_mem']:
            visual_contents[depth][key] = build_info(scope_nodes, key)

    ret = '-' * 25 + 'nnScaler Graph Profiling Result' + '-' * 25 + '\n\n'
    for depth, contents in visual_contents.items():
        ret += f'depth {depth}\n'
        for key, info in contents.items():
            ret += f'    {key} - {info}\n'
    return ret


# a class to store statistics of a continuous sub-sequence
# in the initial graph's topology sequence
@dataclass
class IntervalInfo:
    start: int
    end: int
    fw_span: float
    param_mem: int
    buffer_mem: int
    activation_mem: int

    def equivalent(self, other):
        if self.end - self.start != other.end - other.start:
            return False
        if self.fw_span != other.fw_span:
            return False
        if self.param_mem != other.param_mem:
            return False
        if self.buffer_mem != other.buffer_mem:
            return False
        if self.activation_mem != other.activation_mem:
            return False
        # TODO(yizhu1): check whether the operators are the same
        return True


class ModelGraph:

    def __init__(self, ir_graph: IRGraph, autodist_config: AutoDistConfig):
        self.ir_graph = ir_graph
        self.autodist_config = autodist_config
        self.cost_database = CostDatabase(self.ir_graph, self.autodist_config)
        self.cost_database.profile_comp(partition_degree=1)

        self.scope_tree_root = self.reconstruct_scope_tree()
        self.scope_leaf_nodes = self.scope_tree_root.select(lambda x: x.is_leaf)

        self.min_recompute_mem, self.recompute_groups = self.init_recompute_nodes()

        self.operator_list: List[CubeOperator] = []
        self._ir_cell2idx: Dict[IRFwOperation, int] = dict()
        self.init_operators()

        self._query_fw_span_cache: Dict[Tuple[int, int], float] = dict()
        self._query_mem_cache = dict()

    @property
    def op_num(self):
        return len(self.operator_list)

    def get_op_idx(self, op: CubeOperator):
        return self._ir_cell2idx[op.ir_cell]

    def reconstruct_scope_tree(self):
        fw_cube_nodes = self.ir_graph.select(ntype=IRFwOperation)
        root = ScopeNode('root', None)
        db = self.cost_database.db

        for i, node in enumerate(fw_cube_nodes):
            # filter out the anchor nodes, since they don't have module stack
            if isinstance(node, IRGraphAnchor):
                continue
            if isinstance(node, IRDimops):
                if not self.cost_database.exist(node):
                    fw_span = 0
                else:
                    fw_span = self.cost_database.query_profiled_metrics(
                        node).fw_span
            else:
                fw_span = 0
            module_info = []
            for module_path, module_type in node.module_stack.items():
                module_info.append((module_path.split('.')[-1], module_type))
            root.insert(node, module_info, calc_flops(node), fw_span, idx=i)

        root.pull_up(db)
        _logger.debug('\n' + root.__repr__())
        _logger.info('\n' + analyze_base_graph(root))

        return root

    def get_pipeline_pivots(self) -> List[int]:
        '''
        To reduce the search space, we only consider limited number of pivot
        operators which break the model into several pipeline stages.
        Currently, user's guidance (autodist_config.pipeline_pivots) is required.

        Returns:
            the indices of pivot operators in the operator list
        '''
        # TODO(yizhu1): check recompute_modules are between pivots
        if not self.autodist_config.pipeline:
            raise RuntimeError('pipeline is not enabled')
        pp_pivot_modules = self.autodist_config.pipeline_pivots.split(',')
        pp_pivot_modules = [module for module in pp_pivot_modules if module]
        if not pp_pivot_modules:
            raise RuntimeError('pipeline_pivots is empty')

        def filter_func(scope_node):
            if scope_node.is_leaf:
                return False
            for module in pp_pivot_modules:
                if scope_node.is_root:
                    continue
                if not isinstance(scope_node.module_type, type):
                    raise RuntimeError(
                        f'expect type, got {scope_node.module_type}')
                if module == scope_node.module_type.__name__:
                    return True
            return False

        pivot_modules = self.scope_tree_root.select(filter_func)
        node2idx: Dict[IRFwOperation, int] = dict()
        for i, op in enumerate(self.operator_list):
            node2idx[op.ir_cell] = i
        pivot_idxs = []
        for module in pivot_modules:
            leaf_nodes = module.select(lambda x: x.is_leaf)
            pivot_idxs.append(node2idx[leaf_nodes[0].node])
        if not pivot_idxs:
            raise RuntimeError(f'cannot find any pivot in {pp_pivot_modules}')
        return pivot_idxs

    def calc_interval_info(self, start: int, end: int) -> IntervalInfo:
        '''
        calculate the interval info of nodes in [start, end]
        '''
        fw_span = self.query_fw_span(start, end)
        param_mem, buffer_mem, activation_mem = self.query_mem(start, end)
        return IntervalInfo(start, end, fw_span, param_mem, buffer_mem,
                            activation_mem)

    def group_pipeline_intervals(self) -> List[List[IntervalInfo]]:
        '''
        Group the pipeline intervals with the same interval info. It is used to
        reduce the search time of a stage's (interval) spmd plan: only one
        interval in a group needs to be searched.

        Returns:
            a list of groups, each group contains a list of intervals
        '''
        idxs = [0] + self.get_pipeline_pivots() + [self.op_num]
        len2intervals: Dict[int, List[List[IntervalInfo]]] = dict()
        for i in range(len(idxs) - 1):
            start = idxs[i]
            for j in range(i + 1, len(idxs)):
                end = idxs[j] - 1
                length = end - start + 1
                cur_interval = self.calc_interval_info(start, end)
                if length not in len2intervals:
                    len2intervals[length] = [[cur_interval]]
                else:
                    found_equal = False
                    for group in len2intervals[length]:
                        if group[0].equivalent(cur_interval):
                            group.append(cur_interval)
                            found_equal = True
                            break
                    if not found_equal:
                        len2intervals[length].append([cur_interval])
        ret = []
        for _, groups in len2intervals.items():
            ret += groups
        return ret

    def query_fw_span(self, start: int, end: int) -> float:
        '''
        Time complexity: O(log(#nodes))
        Args:
            start: the left index of the operator list
            end: the right index of the operator list

        Returns:
            the forward span of operators in [start, end]
        '''

        def leaf_handler(idx):
            return self.scope_leaf_nodes[idx].fw_span

        def merger(sub_rets):
            return sum(sub_rets)

        return self.scope_tree_root.query(
            start,
            end,
            self._query_fw_span_cache,
            leaf_handler,
            merger,
        )

    def init_recompute_nodes(self):
        recompute_modules = self.autodist_config.recompute_modules.split(',')
        recompute_modules = [
            module for module in recompute_modules if len(module) > 0
        ]
        if len(recompute_modules) == 0:
            return 0, []

        def fetch_module(scope_node: ScopeNode, prefix: List[str]):
            if scope_node.node is not None:
                return []
            if scope_node.is_root:
                next_prefix = copy.deepcopy(prefix)
            else:
                next_prefix = prefix + [scope_node.module_type.__name__]
            cur_name = '.'.join(next_prefix)
            for module in recompute_modules:
                if module in cur_name:
                    return [scope_node]
            ret = []
            for child in scope_node.children:
                ret += fetch_module(child, next_prefix)
            return ret

        modules = fetch_module(self.scope_tree_root, [])
        train_mem = 0
        for module in modules:
            train_mem = max(train_mem, module.train_mem)
        # calculate the lower bound of memory consumption for recompute
        # assume the activation memory is evenly distributed across devices
        min_recompute_mem = train_mem / self.autodist_config.ngpus
        _logger.info(f'estimated recompute mem {min_recompute_mem / 1024 / 1024} MB')

        def fetch_nodes(scope_node):
            if scope_node.node is not None:
                return [scope_node.node]
            ret = []
            for child in scope_node.children:
                ret += fetch_nodes(child)
            return ret

        recompute_groups = []
        for module in modules:
            recompute_groups.append(fetch_nodes(module))
        return min_recompute_mem, recompute_groups

    def label_ops(self, operator_list: List[CubeOperator]):
        # NOTE: complicated input composed of tensors are not considered, like list of tensors
        # label the tensors that are shared by multiple operators, examples:
        # 1. the embedding matrix is shared by embedding lookup and the last linear layer
        # 2. the activation tensor is shared by query, key and value projections in transformer
        # label the operators that have been set to recompute
        counted_tensors: Set[IRTensor] = set()
        counted_in_tensors: Set[IRTensor] = set()
        recompute_nodes: Set[IRFwOperation] = set()
        for group in self.recompute_groups:
            recompute_nodes.update(group)
        for operator in operator_list:
            if not isinstance(operator.ir_cell, IRDimops):
                continue
            # deduplicate activation tensors
            # train_mem2in_idx only includes activation tensors without param/buffer tensors
            train_mem2in_idx = self.cost_database.query_profiled_metrics(
                operator).train_mem2in_idx
            for i, idx in enumerate(train_mem2in_idx):
                if idx == -1:
                    continue
                tensor = operator.ir_cell.inputs()[idx]
                assert isinstance(tensor, IRTensor), f'expect tensor, but get {type(tensor)}'
                if tensor.tid in counted_tensors:
                    operator.omit_train_idx.append(i)
                else:
                    counted_tensors.add(tensor.tid)

            # deduplicate parameter and buffer tensors
            # assume the traverse order of input tensors is the same as
            # the order in profiling
            in_idx, b_idx, w_idx = -1, -1, -1
            for in_tensor in operator.in_tensors:
                if in_tensor.is_param():
                    assert not in_tensor.is_buffer()
                    w_idx += 1
                    if in_tensor.tid in counted_tensors:
                        operator.omit_param_idx.append(w_idx)
                    else:
                        counted_tensors.add(in_tensor.tid)
                elif in_tensor.is_buffer():
                    assert not in_tensor.is_param()
                    b_idx += 1
                    if in_tensor.tid in counted_tensors:
                        operator.omit_buffer_idx.append(b_idx)
                    else:
                        counted_tensors.add(in_tensor.tid)
                else:
                    in_idx += 1
                    # avoid an input tensor is counted multiple times
                    # when it is shared by multiple operators on the
                    # border of recompute groups. For example, if tensor
                    # x is consumed by two operators a and b who are on the
                    # border of a recompute group, x should not be counted twice.
                    if in_tensor.tid in counted_in_tensors:
                        operator.omit_recompute_in_idx.append(in_idx)
                    else:
                        counted_in_tensors.add(in_tensor.tid)
            if operator.ir_cell in recompute_nodes:
                operator.recompute = True

        # label border operators for recompute groups
        for group in self.recompute_groups:
            output_tensors: Set[IRTensor] = set()
            for node in group:
                for t in node.outputs():
                    if isinstance(t, IRTensor):
                        output_tensors.add(t)
            for node in group:
                is_border = False
                for t in node.inputs():
                    if isinstance(t, IRTensor) and not t.is_attr():
                        if not t in output_tensors:
                            is_border = True
                            break
                if is_border:
                    op = operator_list[self._ir_cell2idx[node]]
                    op.recompute_start_op = True
                    train_mem2in_idx = self.cost_database.query_profiled_metrics(
                        op).train_mem2in_idx
                    for idx, tensor in enumerate(op.in_tensors):
                        if tensor.is_attr():
                            continue
                        if tensor in output_tensors:
                            # avoid count multiple times when the input is another
                            # border operator's output
                            op.omit_recompute_in_idx.append(idx)
                        else:
                            # avoid count multiple times when the input has been
                            # saved by the recompute interface
                            if idx in train_mem2in_idx:
                                i = train_mem2in_idx.index(idx)
                                if i not in op.omit_train_idx:
                                    op.omit_train_idx.append(i)

    def query_mem(self, start: int, end: int) -> Tuple[int, int, int]:
        '''
        calculate memory consumption of operators in [start, end]
        Time complexity: O(log(#nodes))

        Args:
            start: the left index of the operator list
            end: the right index of the operator list

        Returns:
            (param_mem, buffer_mem, activation_mem)
        '''
        db_inst = self.cost_database

        def leaf_handler(idx):
            op = self.operator_list[idx]
            if not isinstance(op.ir_cell, IRDimops):
                return 0, 0, 0
            param_mem = db_inst.query_single_mem(op, 'param', round=False)
            buffer_mem = db_inst.query_single_mem(op, 'buffer', round=False)
            # set the activation memory to 0 if the operator is set to recompute.
            # the memory is considered in `min_recompute_mem` instead
            activation_mem = 0 if op.recompute else db_inst.query_single_mem(
                op, 'train', round=False)
            return param_mem, buffer_mem, activation_mem

        def merger(sub_rets):
            param_mem, buffer_mem, activation_mem = 0, 0, 0
            for ret in sub_rets:
                param_mem += ret[0]
                buffer_mem += ret[1]
                activation_mem += ret[2]
            return param_mem, buffer_mem, activation_mem

        return self.scope_tree_root.query(start, end, self._query_mem_cache,
                                          leaf_handler, merger)

    def init_operators(self):
        cube_nodes = self.ir_graph.select(ntype=IRFwOperation)
        cube_nodes = [
            node for node in cube_nodes if not isinstance(node, IRGraphAnchor)
        ]
        operator_list = []

        tid2consumers = {}
        for i, ir_cell in enumerate(cube_nodes):
            operator_list.append(CubeOperator(ir_cell=ir_cell))
            for t in ir_cell.inputs():
                if isinstance(t, IRTensor):
                    if t.tid not in tid2consumers:
                        tid2consumers[t.tid] = []
                    tid2consumers[t.tid].append(operator_list[-1])

        # init producer and consumer relations
        for src_op_idx in range(len(operator_list) - 1):
            src_op = operator_list[src_op_idx]
            for t in src_op.ir_cell.outputs():
                if not isinstance(t, IRTensor):
                    continue
                # graph outputs (like loss) have no consumer
                if t.tid not in tid2consumers:
                    continue
                for dst_op in tid2consumers[t.tid]:
                    src_op.add_consumer(dst_op)
                    dst_op.add_producer(src_op)

        # Infer batch dims
        # Assume operators with parameters consume and generate tensors
        # with batch dim. A search is followed to propagate the possible
        # batch dim to the whole graph.
        seed_ops = []
        visited = set()
        for op in operator_list:
            if any([t.is_param() for t in op.in_tensors]):
                _logger.debug(f'add seed op {op.ir_cell}')
                seed_ops.append(op)
                visited.add(op.ir_cell.cid)
        dq = deque(seed_ops)
        while len(dq) > 0:
            op = dq.popleft()
            op.has_batch_dim = True
            for consumer in op.consumers:
                if consumer.ir_cell.cid not in visited:
                    visited.add(consumer.ir_cell.cid)
                    dq.append(consumer)
        for op in operator_list:
            if not op.has_batch_dim:
                _logger.debug(f'{op.ir_cell} don\'t have batch dim')

        if len(operator_list) != len(self.scope_leaf_nodes):
            raise RuntimeError(
                f'expect {len(self.scope_leaf_nodes)} operators, got {len(operator_list)}'
            )
        for i, op in enumerate(operator_list):
            self._ir_cell2idx[op.ir_cell] = i
        self.label_ops(operator_list)
        self.operator_list = operator_list

        self._recompute_group_idxs: List[List[int]] = list()
        for recompute_group in self.recompute_groups:
            interval = []
            for node in recompute_group:
                interval.append(self._ir_cell2idx[node])
            start, end = interval[0], interval[-1]
            if end - start + 1 != len(interval):
                raise RuntimeError('recompute nodes are not continuous')
            self._recompute_group_idxs.append(interval)
            self.operator_list[end].recompute_last_op = True

    @property
    def recompute_group_idxs(self) -> List[List[int]]:
        return self._recompute_group_idxs
