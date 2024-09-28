#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from typing import List, Union, Callable, Optional, Tuple
import logging

import torch

import nnscaler
from nnscaler.graph import IRGraph
from nnscaler.graph.segment import IRSegment
from nnscaler.graph.function.dimops import IRDimops
from nnscaler.graph.gener.rvd.intra import IntraAutoPlacer
from nnscaler.ir.operator import IRDataOperation, IRFwOperation
from nnscaler.ir.cten import IRCell
from nnscaler.ir.tensor import IRFullTensor
from nnscaler.graph.function.anchor import IRGraphAnchor
import nnscaler.runtime
from nnscaler.utils import print_each_rank

import numpy as np


_logger = logging.getLogger(__name__)


def create_mesh(ngpus: int, group_num: Tuple[int]) -> Tuple[Tuple[Tuple[int]]]:
    """
    Create hybrid (nested) groups given the each group number.

    The product of group_num should be same with total devices.

    e.g., 6 device to 2 x 3 mesh will results [dim][group_id] = tuple[int]:
        (
            ( (0,1,2), (3,4,5) ),
            ( (0,3), (2,5), (3,6) ),
        )
    """
    group_num = np.array(group_num)
    cnt = np.prod(group_num)
    assert cnt == ngpus, 'total device not match'
    grid = np.arange(cnt).reshape(tuple(group_num))
    dims = list(range(len(group_num)))
    outputs = []
    for dim, num in enumerate(group_num):
        remain = ngpus // num
        order = tuple(dims[:dim] + dims[dim+1:] + [dim])
        grid_dim = np.transpose(grid, order).reshape((remain,num))
        grid_dim = grid_dim.tolist()
        outputs.append(tuple(tuple(ranks) for ranks in grid_dim))
    assert len(outputs) == len(group_num)
    return tuple(outputs)


def group_to_layers(fnodes) -> List[List[IRCell]]:
    # group to layers
    transformers: List[List[IRFwOperation]] = []
    anchors = [node for node in fnodes if isinstance(node, IRGraphAnchor)]
    indices = [fnodes.index(anchor) for anchor in anchors]
    for lid, idx in enumerate(indices):
        fnodes[idx+1].comment = f'===> start of layer {lid}'
        start = idx if lid != 0 else 0
        end = indices[lid+1] if lid + 1 < len(anchors) else len(fnodes)
        transformers.append(fnodes[start:end])
    for lid in range(len(transformers) - 1):
        if transformers[lid][-1].name == 'multiref':
            node = transformers[lid].pop()
            transformers[lid+1].insert(0, node)
    return transformers


def _tp_autoplace(segment: IRSegment, ftensor: IRFullTensor,
                  producers: List[IRFwOperation], devs: List[int],
                  sub_nodes: List[IRFwOperation]) -> List[int]:
    """decide the devices of the partitioned `sub-nodes` to achieve optimal communication

    Args:
        segment (IRSegment): segment of the ftensor
        ftensor (IRFullTensor): the tensor to be partitioned
        producers (List[IRFwOperation]): producers of the ftensor
        devs (List[int]): devices to be placed
        sub_nodes (List[IRFwOperation]): partitioned nodes

    Returns:
        List[int]: devices of the partitioned `sub-nodes`
    """
    if ftensor.is_param() or len(producers) != len(sub_nodes):
        _logger.warning(f"skip auto placer due to condition not matched: "
                        f"nproducers: {len(producers)}, nconsumers: {len(sub_nodes)}, "
                        f"producer name: {producers[0].name if len(producers) > 0 else None}")
        devs = sorted(list(devs))
    else:
        devs = IntraAutoPlacer.auto_place(segment, ftensor, producers, sub_nodes)
    return devs

# tensor parallelism
def tensor_parallelism(graph: IRGraph, node: IRDimops,
                       idx: int, dim: int, devs: List[int],
                       autoplace: bool = False) -> List[IRDimops]:
    """Apply tensor parallelism of a node to devs"""
    if len(devs) == 1:
        graph.assign(node, devs[0])
        return [node]
    # transformation
    algo = node.algorithms('dim')
    sub_nodes = graph.partition(node, algo, idx=idx, dim=dim, num=len(devs))
    assert sub_nodes is not None

    if autoplace:
        segment = graph.segment(node)
        devs = _tp_autoplace(segment, node.input(idx).parent,
                             segment.producers(node.input(idx).parent),
                             devs, sub_nodes)
    # assign
    for devid, sub_node in zip(devs, sub_nodes):
        graph.assign(sub_node, devid)
    return sub_nodes


# replica
def replica(graph: IRGraph, node: Union[IRFwOperation, IRDataOperation],
            devs: List[int]) -> List[Union[IRFwOperation, IRDataOperation]]:
    """Replicate a forward node or dataloader to devs"""
    if len(devs) == 1:
        graph.assign(node, devs[0])
        return [node]
    sub_nodes = graph.replicate(node, times=len(devs))
    for devid, sub_node in zip(devs, sub_nodes):
        graph.assign(sub_node, devid)
    return sub_nodes


def get_policy(modules: List, name: str) -> Callable:
    """Get policy from modules

    Note every rank should enter this function simutaneously.

    Args:
        modules (List): list of modules
        name (str): name of policy

    Returns:
        Callable: policy
    """
    for module in modules:
        if name in module.__dict__:
            print_each_rank(f'using policy from {module.__name__}.{name}')
            return module.__dict__[name]
    policies = []
    for module in modules:
        policies += list(policy for policy in module.__dict__.keys() if policy.startswith('PAS'))
    raise ValueError(f"policy {name} not found. Candidates: {policies}")


def init_random():
    np.random.seed(1)
    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1)
