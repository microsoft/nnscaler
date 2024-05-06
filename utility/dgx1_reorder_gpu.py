# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Reorder GPU index by finding DGX-1 topology Find dgx topology

┌───────────┐
1 = 0 = 4 = 5
‖ x |   | x ‖
2 = 3 = 7 = 6
└───────────┘

"""
from typing import List
import subprocess
import numpy as np

_kConnType = {
    "NV1": 1,
    "NV2": 2,
    "NODE": 3,
    "X": -1,
}

_kConnTypeStr = {val: key for key, val in _kConnType.items()}



def get_topology():
    cmds = [
        'nvidia-smi',
        'topo',
        '-m',
    ]

    proc = subprocess.Popen(cmds, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    outputs = stdout.decode('utf-8').split('\n')

    outputs = [out for out in outputs if out.startswith('GPU')]
    ngpus = len(outputs)
    print(f'Detected GPU number: {ngpus}')

    topology = np.empty((ngpus, ngpus), dtype=int)
    for src, output in enumerate(outputs):
        connections = output.split('\t')[1:1+ngpus]
        for dst, link in enumerate(connections):
            link = link.replace(" ", "")
            assert link in _kConnType, f"Find link not in DGX-1 topology: {link}"
            topology[src, dst] = _kConnType[link]
    return topology


def topology_repr(topology: np.ndarray, reorder: List[int]):
    reorder = list(reorder)
    ngpus = topology.shape[0]
    reorder_topo = np.empty((ngpus, ngpus), dtype=object)
    for src in range(ngpus):
        for dst in range(ngpus):
            link = _kConnTypeStr[topology[src, dst]]
            reorder_topo[reorder.index(src), reorder.index(dst)] = link
    maxlen = max(len(key) for key in _kConnType)
    dscp = ''
    for gidx, line in enumerate(reorder_topo):
        dscp += f'GPU{gidx}: '+ ' '.join(link.ljust(maxlen) for link in line) + '\n'
    return dscp


def reorder(topology: np.ndarray) -> np.ndarray:
    """
    Reorder GPU according to DGX-1 topology

     ┌───────────┐
     1 = 0 = 4 = 5
     ‖ x |   | x ‖
     2 = 3 = 7 = 6
     └───────────┘
    """
    ngpus = topology.shape[0]
    # find NV2 ring
    ring = [0]
    while len(ring) < ngpus:
        nv2s = np.where(topology[ring[-1]] == _kConnType['NV2'])[0]
        find_next = False
        for gid in nv2s:
            if gid not in ring:
                ring.append(gid)
                find_next = True
                break
        assert find_next
    ring = np.array(ring, dtype=int)
    print(f'Get ring: {ring}')
    # find fc
    for idx, src in enumerate(ring):
        is_fc = True
        pairs = [
            (src, ring[(idx + 3) % len(ring)]),
            (src, ring[(idx + 2) % len(ring)]),
            (ring[(idx+1) % len(ring)], ring[(idx+3) % len(ring)])
        ]
        for src, dst in pairs:
            if topology[src, dst] != _kConnType['NV1']:
                is_fc = False
                break
        if is_fc:
            break
    assert is_fc, f"Cannot find FC group."
    ring = np.roll(ring, 0-idx)
    return ring

        
if __name__ == '__main__':
    topology = get_topology()
    print('original topology:')
    print(topology_repr(topology, list(range(topology.shape[0]))))
    reorder = reorder(topology)
    print('reorder topology:')
    print(topology_repr(topology, reorder))
    print(
        f"Command need to be added into environment:\n"
        f"export CUDA_VISIBLE_DEVICES={','.join(str(gid) for gid in reorder)}"
    )
