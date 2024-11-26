#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""
pytest unit_tests/graph/function/test_dataloader.py
"""

import torch

from nnscaler.ir.cten import IRObject
from nnscaler.ir.tensor import IRFullTensor
from nnscaler.ir.operator import IRDataOperation


def test_data_operation():

    data_op = IRDataOperation(
        IRObject('dataloader'),
        [IRFullTensor(shape=[32, 256, 512]).tosub(),
         IRFullTensor(shape=[32, 128, 224]).tosub(),])
    
    # cannot be partitioned
    assert not hasattr(data_op, 'algorithms')
    # test input / output
    assert all(isinstance(out, IRObject) for out in data_op.outputs())
    assert all(isinstance(inp, IRObject) for inp in data_op.inputs())
    # can be replicated
    data_op_replica = data_op.replicate()
    assert data_op_replica.input(0) == data_op.input(0)
    assert data_op_replica.output(0) == data_op.output(0)
    assert data_op_replica.output(1) == data_op.output(1)
    assert data_op_replica.cid == data_op.cid
