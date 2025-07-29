from .descs import NodePartitionDesc
from nnscaler.graph import IRGraph
from nnscaler.ir.operator import IRFwOperation

import struct
from pathlib import Path
from typing import List
from collections import deque


def int2byte(val):
    return struct.pack('i', val)


def int4byte(val):
    return struct.unpack('i', val)[0]


def double2byte(val):
    return struct.pack('d', val)


def double4byte(val):
    return struct.unpack('d', val)[0]


def get_default_profile_path():
    return Path.home() / '.cache' / 'nnscaler' / 'autodist' / '1.0' / get_node_arch()


def get_node_arch():
    import torch
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(torch.cuda.current_device()).replace(
            ' ', '_')
    else:
        return 'cpu'  # although we don't support cpu now, we still need to return something for testing


# tensor parallelism
def tp(graph: IRGraph, node: IRFwOperation, devs: List[int], idx: int,
       dim: int):
    algo = node.algorithms('dim')
    sub_nodes = graph.partition(node, algo, idx=idx, dim=dim, num=len(devs))
    assert sub_nodes is not None
    for devid, sub_node in zip(devs, sub_nodes):
        graph.assign(sub_node, devid)
    return sub_nodes


def replica(graph: IRGraph, node: IRFwOperation, devs: List[int]):
    sub_nodes = graph.replicate(node, times=len(devs))
    for devid, sub_node in zip(devs, sub_nodes):
        graph.assign(sub_node, devid)
    return sub_nodes


def partition_node(node: IRFwOperation, graph: IRGraph, devs: [int],
                   desc: NodePartitionDesc) -> None:
    min_dev_index = min(devs)
    tp_size = len(devs)
    info = desc.desc

    dq = deque()
    dq.append((node, (0, tp_size)))
    for (idx, dim), num in info:

        cur_nodes = []
        while dq:
            u, (low, high) = dq.popleft()
            assert (high - low) % num == 0
            inc = (high - low) // num
            sub_intervals = list(
                map(lambda x: (low + x * inc, low + (x + 1) * inc),
                    list(range(num))))
            if idx == -1 and dim == -1:
                sub_nodes = graph.replicate(u, times=num)
            else:
                assert idx >= 0 and dim >= 0
                algo = u.algorithms('dim')
                sub_nodes = graph.partition(u, algo, idx=idx, dim=dim, num=num)
            for i in range(num):
                cur_nodes.append((sub_nodes[i], sub_intervals[i]))

        for cur_node in cur_nodes:
            dq.append(cur_node)

    while dq:
        u, (low, high) = dq.popleft()
        assert high - low == 1
        graph.assign(u, low + min_dev_index)
