# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Cost model for intra-op plan search
"""
from typing import List, Callable, Tuple, Dict
import numpy as np

from cube.graph import IRGraph
from cube.ir.cten import IRTensor
from cube.ir.operator import IRFwOperation
from cube.graph.function.anchor import IRGraphAnchor
from cube.graph.function.dimops import IRDimops, TransformRule, DimopSplit

from .profiler import Estimator
from ..constraints import Constraints

import logging
_logger = logging.getLogger(__name__)


DistSpec = Dict[int, Tuple[Tuple[int, int]]]


class CommCost:
    """
    Get communication cost in milliseconds
    """

    ndevs_per_node: int = 8
    intra_node_gbps = 600  # actually nvlink have more than this, but it cannot fully saturate the bandwidth
    inter_node_gbps = 100  # consider 100 gbps ib network

    @staticmethod
    def set_bandwidth(intra_node: int, inter_node: int):
        """Set bandwidth in Gbps"""
        CommCost.intra_node_gbps = intra_node
        CommCost.inter_node_gbps = inter_node

    @staticmethod
    def get_bandwidth(ranks: List[int]) -> float:
        """Get bandwidth in btypes per seconds

        TODO: support with real runtime information
        """
        
        if len(ranks) < CommCost.ndevs_per_node:
            return CommCost.intra_node_gbps / 8 * 1e9
        else:
            return CommCost.inter_node_gbps / 8  * 1e9

    @staticmethod
    def allreduce_cost(tensor: IRTensor, num_devices: int) -> float:
        bandwidth = CommCost.get_bandwidth(list(range(num_devices))) 
        return 2 * (num_devices - 1) * tensor.byte_size() / num_devices / bandwidth * 1000

    @staticmethod
    def alltoall_cost(tensor: IRTensor, num_devices: int) -> float:
        bandwidth = CommCost.get_bandwidth(list(range(num_devices)))
        # bandwidth in all-to-all is really worse and better not to use
        bandwidth = bandwidth / 2
        return tensor.byte_size() / num_devices / num_devices * (num_devices - 1) / bandwidth * 1000

    @staticmethod
    def allgather_cost(tensor: IRTensor, num_devices: int) -> float:
        # bandwidth in allgather can only be half due to torch implementation issues
        # return 1e6
        bandwidth = CommCost.get_bandwidth(list(range(num_devices))) / 2.98
        return tensor.byte_size() / num_devices * (num_devices - 1) / bandwidth * 1000

    @staticmethod
    def reducescatter_cost(tensor: IRTensor, num_devices: int) -> float:
        # bandwidth in reduce-scatter can only be half due to torch implementation issues
        # return 1e6
        bandwidth = CommCost.get_bandwidth(list(range(num_devices))) / 2.38
        return tensor.byte_size() / num_devices * (num_devices - 1) / bandwidth * 1000


class CostModel:
    """CostModel for SPMD solver search

    The cost model considers the transformation space of each operaotr, and
    builds edges between data-dependent operators and their communication
    cost under each transformation pairs.

    Note nodes that have been assigned to devices will not be considered in
    the cost model.
    """

    def __init__(self, graph: IRGraph, estimator: Callable, constraints: Constraints, max_tp: int = 32):

        self.graph: IRGraph = graph
        self.estimator: Estimator = estimator
        self.constraints = constraints

        # [node.cid] = (None, (idx1, dim1), ...)
        self.partition_algos: Dict[int, Tuple[int, int]] = {}
        # [node.cid][ndevs] = np.array(...)
        self.comp_cost = {}
        self.mem_cost = {}

        # producer.cid -> [consumer.cid, ...]
        self.edges: Dict[int, List[int]] = {}
        ftensor_producer: Dict[IRTensor, int] = {}

        # note we only need to build models for un-assigned operators as
        # assigned opeartors are already constrained by users.
        # TODO: However, we still need to consider communication cost
        # between constrained nodes and un-constrainted nodes
        all_fnodes = graph.select(ntype=IRFwOperation)
        fnodes = self.get_search_nodes(all_fnodes)
        
        # setup producer
        for producer in fnodes:
            for t in producer.outputs():
                if isinstance(t, IRTensor):
                    ftensor_producer[t.parent] = producer.cid
        # connect consumer
        for consumer in fnodes:
            for t in consumer.inputs():
                if isinstance(t, IRTensor) and not t.is_attr():
                    if t.parent in ftensor_producer:
                        pcid = ftensor_producer[t.parent]
                        self.edges.setdefault(pcid, []).append(consumer.cid)

        # setup transformation space and their cost
        for fnode in fnodes:
            train = fnode.mirror is not None
            # build transformation space
            self.partition_algos[fnode.cid] = self.get_transform_space(fnode)
            # build computation and communication cost
            self.comp_cost[fnode.cid] = {}
            self.mem_cost[fnode.cid] = {}
            num = 1
            while num <= max_tp:
                node_comp_cost = []
                node_mem_cost = []
                for config in self.partition_algos[fnode.cid]:
                    if num > 1 and config is not None:
                        split = (config[0], config[1], num)
                    else:
                        split = None
        
                    try:
                        infer_span, infer_mem, train_span, train_mem = \
                            self.estimator.perf(fnode, split, train)
                    except Exception as e:
                        _logger.error(f'fail to profile node: {fnode.name}[{fnode.cid}], split: {split}, saving profiled data...')
                        self.estimator.save()
                        raise e

                    if train:
                        node_comp_cost.append(train_span)
                        node_mem_cost.append(train_mem)
                    else:
                        node_comp_cost.append(infer_span)
                        node_mem_cost.append(infer_mem)
                self.comp_cost[fnode.cid][num] = np.array(node_comp_cost, dtype=float)
                self.mem_cost[fnode.cid][num] = np.array(node_mem_cost, dtype=int)
                num *= 2

    def get_transform_space(self, node: IRFwOperation) -> List[Tuple[int, int]]:
        """
        Get the transform space of a node
        
        None indicates replicate
        """
        if node in self.constraints.op_trans:
            if node in self.constraints.op_place:
                raise KeyError(
                    f"Unexpected node: {node.name} to consider transformation space, "
                    f"since it is determined by additional placement constraints"
                )
            algo, _ = self.constraints.op_trans[node]
            # algo None indicates only to replicate
            algo = algo if algo is None else tuple(algo)
            return [algo,]
        if isinstance(node, IRDimops):
            return [None] + node.transform_space()
            # params = [t for t in node.inputs() if isinstance(t, IRTensor) and t.is_attr()]
            # # must be partitioned for computation-intensive ops
            # if len(params) > 0 and node.name not in light_op_names:
            #     return list(node.transform_space())
            # # can be partitioned or replicated for computation-light ops
            # else:
            #     return [None] + node.transform_space()
        return [None]

    def get_comp_cost(self, fnode: IRFwOperation, num_devices: int) -> np.ndarray:
        """
        Get computation cost related to different partition strategies
        """
        # return np.zeros(len(self.partition_algos[fnode.cid]), dtype=float)
        return self.comp_cost[fnode.cid][num_devices]

    def get_comm_cost(self, fnode: IRFwOperation, num_devices) -> np.ndarray:
        """
        Get communication cost for a node given a strategy

        This only calucates the cases for partitioning on value dimension

        @return cost: np.ndarray: 1-D array of the cost on allreduce
        """
        cost = []
        for strategy in self.partition_algos[fnode.cid]:
            if strategy is None:
                cost.append(0.)
                continue
            s_cost = 0
            idx, dim = strategy
            rule: TransformRule = fnode.algorithms('dim').infer(idx, dim, num_devices)
            for idx, output in enumerate(rule.outputs()):
                if output.isV():
                    s_cost += CommCost.allreduce_cost(fnode.output(idx), num_devices)
            cost.append(s_cost)
        return np.array(cost, dtype=float)
    
    def get_pair_reshard_cost(self, fnode_src: IRFwOperation, fnode_dst: IRFwOperation, 
                              num_devices: int) -> np.ndarray:
        """
        Get cost of resharding between two nodes
        @return cost: np.ndarray: 1-D tensor of (nsrc * ndst,) shape,
            nsrc is the number of partitioned ways of the source node
            ndst is the number of partitioned ways of the destination node
        """
        nsrc = len(self.partition_algos[fnode_src.cid])
        ndst = len(self.partition_algos[fnode_dst.cid])
        cost = np.zeros((nsrc, ndst), dtype=float)

        def comm_cost(tensor: IRTensor, num_devices: int,
                      src_split: DimopSplit, dst_split: DimopSplit, dst_replica: bool):
            # note for data parallel, we don't consider allreduce cost as it
            # will only be performed at the last of iteration.
            if tensor.is_attr(): return 0.0
            if src_split.isV() or src_split.isR():
                # identity-allreduce or identity-identity
                if dst_split.isR():
                    return 0.0 if dst_replica else CommCost.allreduce_cost(tensor, num_devices)
                # split-allgather
                if dst_split.isD():
                    return CommCost.allgather_cost(tensor, num_devices)
            if src_split.isD():
                # allgahter-reducescatter or allgather-split
                if dst_split.isR():
                    return CommCost.allgather_cost(tensor, num_devices) if dst_replica else \
                           CommCost.allgather_cost(tensor, num_devices) + CommCost.reducescatter_cost(tensor, num_devices)
                # all2all-all2all or identity-identity
                if dst_split.isD():
                    return 0.0 if src_split == dst_split else 2 * CommCost.alltoall_cost(tensor, num_devices)
            raise NotImplementedError(f"Unknown split type: {src_split} -> {dst_split}")

        # FIXME: need consider cases that an operator has multiple **same** inputs
        tensors: Dict[IRTensor, Tuple[int, int]] = {}
        for idx, output in enumerate(fnode_src.outputs()):
            tensors[output.parent] = [idx]
        for idx, input in enumerate(fnode_dst.inputs()):
            if not isinstance(input, IRTensor): continue
            tensors.setdefault(input.parent, []).append(idx)
        tensors = {t: tuple(v) for t, v in tensors.items() if len(v) == 2}

        for i, strategy_src in enumerate(self.partition_algos[fnode_src.cid]):

            rule_src = None
            if strategy_src is not None:
                idx, dim = strategy_src
                rule_src = fnode_src.algorithms('dim').infer(idx, dim, num_devices)
            
            for j, strategy_dst in enumerate(self.partition_algos[fnode_dst.cid]):
                rule_dst = None
                if strategy_dst is not None:
                    idx, dim = strategy_dst
                    rule_dst = fnode_dst.algorithms('dim').infer(idx, dim, num_devices)

                for tensor, (idx_src, idx_dst) in tensors.items():
                    cost[i, j] += comm_cost(
                        tensor, num_devices, 
                        rule_src.outputs()[idx_src] if rule_src is not None else DimopSplit(r=True),
                        rule_dst.inputs()[idx_dst] if rule_dst is not None else DimopSplit(r=True),
                        strategy_dst is None
                    )
        return cost

    def get_edges(self, nodes: List[IRFwOperation]) -> Dict[IRFwOperation, Tuple[IRFwOperation]]:
        """
        Get edges of a subgraph
        """
        edges: Dict[IRFwOperation, List[IRFwOperation]] = {}
        cid2nodes: Dict[int, IRFwOperation] = {n.cid : n for n in nodes}
        for node in nodes:
            if node.cid in self.edges:
                edges[node] = [cid2nodes[cid] for cid in self.edges[node.cid] if cid in cid2nodes]
        return edges

    def get_search_nodes(self, nodes: List[IRFwOperation]) -> List[IRFwOperation]:
        """Get nodes that are needed to search for SPMD solver

        Nodes will be excluded:
            - nodes that are in constraints
            - nodes are assigned by devices (from stand-along blocks)
            - IRGraphAnchor (won't apear in final execution)
            - identity, multiref (system-side management)

        .. todo::
            consider communication cost between constrained nodes and
            un-constrainted nodes

        Args:
            nodes (List[IRFwOperation]): un-filtered forward operations

        Returns:
            List[IRFwOperation]: filtered forward operations
        """
        fnodes: List[IRFwOperation] = []
        for fnode in nodes:
            # nodes are already determined of transformation and placement
            if fnode in self.constraints.op_trans:
                if fnode in self.constraints.op_place:
                    continue
            # nodes from standalone blocks
            if len(fnode.device) > 0: continue
            if not isinstance(fnode, IRFwOperation): continue
            if isinstance(fnode, IRGraphAnchor) or fnode.name in ('multiref', 'identity'):
                continue
            fnodes.append(fnode)
        return fnodes
