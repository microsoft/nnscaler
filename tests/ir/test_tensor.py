#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from nnscaler.ir.tensor import IRSubTensor, IRFullTensor

import pytest


def test_tensor_grad():
    ftensor = IRFullTensor((128, 512), requires_grad=True)
    subtensor = ftensor.tosub()

    assert isinstance(ftensor.grad, IRFullTensor)
    subtensor.grad = ftensor.grad.tosub()

    assert isinstance(subtensor.grad, IRSubTensor)

    ftensor.requires_grad = False
    assert ftensor.grad is None
    assert subtensor.grad is None
    assert subtensor.requires_grad is False


def test_continous():
    ftensor = IRFullTensor((128, 512), requires_grad=True)
    with pytest.raises(ValueError):
        IRSubTensor.is_dim_continous([], dim=0)

    indmap = []
    for dimlen in ftensor.shape:
        indmap.append((0, dimlen))
    indmap[0] = (0, 2)
    sub1 = ftensor.select(tuple(indmap), (0, 1))
    indmap[0] = (2, 4)
    sub2 = ftensor.select(tuple(indmap), (0, 1))
    indmap[0] = (4, 6)
    sub3 = ftensor.select(tuple(indmap), (0, 1))

    assert IRSubTensor.is_dim_continous([sub1, sub2, sub3], dim=0)
    assert not IRSubTensor.is_dim_continous([sub1, sub2, sub3], dim=1)
    assert not IRSubTensor.is_dim_continous([sub1, sub3], dim=0)
