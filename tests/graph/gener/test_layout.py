#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from typing import List
import numpy as np

from nnscaler.graph.gener.rvd.layout import RVDLayout
from nnscaler.ir.tensor import IRFullTensor, IRSubTensor


def test_rvd_layout():
    fshape = [128, 256, 512]
    ftensor = IRFullTensor(shape=fshape, name='tensor', requires_grad=True)

    fsrc_r, fsrc_v, fsrc_d = 2,8,(1,1,2)

    ndevs = fsrc_r * fsrc_v * np.prod(np.array(fsrc_d))

    fpdevs = list(range(ndevs))
    fp_rvd = RVDLayout.grid(ftensor, r=fsrc_r, v=fsrc_v, dims=fsrc_d, devices=fpdevs)
    assert True
    assert fp_rvd.R == fsrc_r
    assert fp_rvd.V == fsrc_v
    assert fp_rvd.D == fsrc_d
    assert len(fp_rvd.subtensors) == ndevs
    assert fp_rvd.mat.shape == (fsrc_r, fsrc_v, *fsrc_d)
    # 0/1 are replicated. They should be the same.
    assert np.array_equal(fp_rvd.mat[0], fp_rvd.mat[1])
    mat = fp_rvd.mat[0]
    # check valmap
    for i in range(fp_rvd.V):
        for j in mat[i].flatten():
            j: IRSubTensor
            assert j.valmap == (i, fp_rvd.V)

    # check idxmap
    mat: List[IRSubTensor] = fp_rvd.mat[0][0].flatten().tolist()
    for i in range(0, len(mat)//2):
        assert mat[i].indmap == ((0, fshape[0]), (0, fshape[1]), (0, fshape[2]//2))
        assert mat[i + len(mat)//2].indmap == ((0, fshape[0]), (0, fshape[1]), (fshape[2]//2, fshape[2]))

    # check device
    assert all([s.device == (i,) for i, s in enumerate(fp_rvd.mat.flatten().tolist())])

