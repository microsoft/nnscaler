# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""GPT policy gallery for MPMD Parallelism"""

from typing import List

from cube.graph import IRGraph
from cube.graph.function.pyfunc import IRPyFunc
from cube.ir.operator import IRBpOperation, IRDataOperation, IRFwOperation

from examples.utils import tensor_parallelism, replica


# coshard
def coshard(graph: IRGraph, node: IRFwOperation, devs: List[int], colocate: int,
             idx: int, dim: int):
    algo = node.algorithms('dim')
    sub_nodes = graph.partition(node, algo, idx=idx, dim=dim, num=colocate*len(devs))
    assert sub_nodes is not None
    graph.recompute(sub_nodes)
    for devid in devs:
        for coid in range(colocate):
            sub_node = sub_nodes[devid * colocate + coid]
            graph.assign(sub_node, devid)
    return sub_nodes


def PASSingle(graph: IRGraph, resource, **kwargs):
    """Single-device execution"""
    assert resource.ngpus == 1
    for node in graph.nodes():
        if not isinstance(node, IRBpOperation):
            graph.assign(node, 0)
    return graph


def PASDP(graph: IRGraph, resource, **kwargs):
    """Data parallelism"""
    devs = list(range(resource.ngpus))
    dataloader = graph.select(ntype=IRDataOperation)[0]
    bs = dataloader.output(0).shape[0]
    # replicate dataloader
    replica(graph, dataloader, devs)
    # partition forward operators
    for node in graph.select(ntype=IRFwOperation):
        if isinstance(node, IRPyFunc):
            graph.assign(node, 0)
            continue
        if len(node.inputs()) == 0: continue
        batch_dim = node.input(0).shape.index(bs)
        tensor_parallelism(graph, node, idx=0, dim=batch_dim, devs=devs)
    return graph


def PASMegatronTP(graph: IRGraph, resource, **kwargs):
    """Megatron-way tensor parallelism"""
    devs = list(range(resource.ngpus))
    # attention
    for attn in graph.select(name='self_attention'):
        tensor_parallelism(graph, attn, idx=1, dim=0, devs=devs)
    # feedforward
    for ffn in graph.select(name='feedforward'):
        tensor_parallelism(graph, ffn, idx=1, dim=0, devs=devs)
    # partition embed
    for embed in graph.select(name='embedding'):
        tensor_parallelism(graph, embed, idx=1, dim=0, devs=devs)
    # partition last linear
    linears = graph.select(name='linear')
    tensor_parallelism(graph, linears[-1], idx=1, dim=0, devs=devs)
    # partition loss
    sums = graph.select(name='sum')
    tensor_parallelism(graph, sums[0], idx=0, dim=2, devs=devs)
    # replica other nodes
    for node in graph.select(ntype=(IRFwOperation, IRDataOperation)):
        if len(node.device) == 0:
            replica(graph, node, devs)
    return graph


def PASMeshShard(graph: IRGraph, resource, **kwargs):
    """Coshard policy for long sequence"""
    devs = list(range(resource.ngpus))
    # attention
    for attn in graph.select(name='self_attention'):
        # tensor_parallelism(graph, attn, idx=1, dim=0, devs)
        coshard(graph, attn, devs, colocate=2, idx=1, dim=0)
    # feedforward
    for ffn in graph.select(name='feedforward'):
        # tensor_parallelism(graph, ffn, idx=1, dim=0, devs)
        coshard(graph, ffn, devs, colocate=4, idx=1, dim=0)
    # replica other nodes
    for node in graph.select(ntype=(IRFwOperation, IRDataOperation)):
        if len(node.device) == 0:
            replica(graph, node, devs)
    return graph
