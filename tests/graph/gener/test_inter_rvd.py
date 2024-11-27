#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from typing import List, Tuple
import nnscaler
from nnscaler.ir.tensor import IRFullTensor
from nnscaler.graph.gener.rvd.layout import RVDLayout, RVDInspector
from nnscaler.graph.gener.rvd.inter import InterPathFinder
import numpy as np

from .test_intra_rvd import enable_reduce_scatter_adapter  # noqa


def factors(k: int, num: int) -> List[Tuple[int]]:
    """
    get all possible sequence k1 * k2 * .. k_{num} = k
    """
    if num == 1: return [(k,)]
    res = []
    for i in range(1, k):
        if k % i != 0: continue
        for sub_res in factors(k // i, num - 1):
            res.append((i,) + sub_res)
    return res


def test_one_f_case():
    fshape = [128, 256, 512]

    src_r, src_v, src_d = 1,4,(1,1,2)
    dst_r, dst_v, dst_d = 2,1,(2,1,2)
    src_rvd = (src_r, src_v) + src_d
    dst_rvd = (dst_r, dst_v) + dst_d

    pndevs = np.prod(src_rvd)
    cndevs = np.prod(dst_rvd)

    ftensor = IRFullTensor(shape=fshape, name='tensor', requires_grad=False)

    pdevs = list(range(pndevs))
    fp_rvd = RVDLayout.grid(ftensor, r=src_r, v=src_v, dims=src_d, devices=pdevs)

    cdevs = list(range(pndevs, pndevs + cndevs))
    fc_rvd = RVDLayout.grid(ftensor, r=dst_r, v=dst_v, dims=dst_d, devices=cdevs)

    rvds = InterPathFinder.get_optimal_path(ftensor, src_rvd, dst_rvd)
    assert rvds == (('p', 1, 4, 1, 1, 2), ('p', 1, 1, 4, 1, 2), ('c', 1, 1, 4, 1, 2), ('c', 2, 1, 2, 1, 2))

    fprims = InterPathFinder.path(fp_rvd, fc_rvd)
    assert len(fprims) == 14
    # producer part, v->d, so reduce_scatter
    assert fprims[0].signature == 'nnscaler.runtime.adapter.reduce_scatter'
    assert fprims[0].device == [0, 2, 4, 6]
    assert fprims[1].signature == 'nnscaler.runtime.adapter.reduce_scatter'
    assert fprims[1].device == [1, 3, 5, 7]
    # inter part move
    src_devs = set()
    dst_devs = set()
    for i in range(8):
        assert fprims[2 + i].signature == 'nnscaler.runtime.adapter.move'
        src_devs.add(fprims[2 + i].kwargs['src'])
        dst_devs.add(fprims[2 + i].kwargs['dst'])

    assert src_devs == set([0, 1, 2, 3, 4, 5, 6, 7])
    assert dst_devs == set([8, 9, 10, 11, 12, 13, 14, 15])

    # consumer part, d->v, so all_gather
    assert fprims[10].signature == 'nnscaler.runtime.adapter.all_gather'
    assert fprims[10].device == [8, 12]
    assert fprims[11].signature == 'nnscaler.runtime.adapter.all_gather'
    assert fprims[11].device == [9, 13]
    assert fprims[12].signature == 'nnscaler.runtime.adapter.all_gather'
    assert fprims[12].device == [10, 14]
    assert fprims[13].signature == 'nnscaler.runtime.adapter.all_gather'
    assert fprims[13].device == [11, 15]


def test_all_f_cases_fix_placement():
    fshape = [128, 256, 512]
    ftensor = IRFullTensor(shape=fshape, name='tensor', requires_grad=False)

    pndevs = 4
    cndevs = 8

    ndims = len(fshape) + 2
    for src_rvd in factors(pndevs, ndims):
        for dst_rvd in factors(cndevs, ndims):
            if src_rvd == dst_rvd or src_rvd[1] < dst_rvd[1]: continue
            print(f'test generating | source rvd: {src_rvd}, destination rvd: {dst_rvd}')
            pdevs = list(range(pndevs))
            fp_rvd = RVDLayout.grid(ftensor, r=src_rvd[0], v=src_rvd[1], dims=src_rvd[2:], devices=pdevs)

            cdevs = list(range(pndevs, pndevs + cndevs))
            fc_rvd = RVDLayout.grid(ftensor, r=dst_rvd[0], v=dst_rvd[1], dims=dst_rvd[2:],devices=cdevs)

            _ = InterPathFinder.path(fp_rvd, fc_rvd)
            rvds = InterPathFinder.get_optimal_path(ftensor, src_rvd, dst_rvd)
            print(f"==> path: {'->'.join(str(rvd) for rvd in rvds)}")

    # should not raise any exception
    assert True
