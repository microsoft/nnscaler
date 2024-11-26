#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import pytest
from nnscaler.ir.tensor import IRFullTensor
from nnscaler.graph.gener.concurrent import ConcurrentGener, CompileFlag, \
    AllToAllPrim, ReduceScatterPrim, _logger
from ...utils import catch_log


def test_path_retry():
    ftensor = IRFullTensor((128, 512), requires_grad=True)
    indmap = []
    for dimlen in ftensor.shape:
        indmap.append((0, dimlen))
    indmap[0] = (0, 2)
    sub1 = ftensor.select(tuple(indmap), (0, 1))
    indmap[0] = (2, 4)
    sub2 = ftensor.select(tuple(indmap), (0, 1))
    indmap[0] = (4, 6)
    sub3 = ftensor.select(tuple(indmap), (0, 1))

    wrong_called = False
    right_called = False
    def path_with_reduce_scatter(*args, **kwargs):
        nonlocal wrong_called, right_called
        if not CompileFlag.disable_reduce_scatter_adapter:
            # the parameter is fake, just for testing
            wrong_called = True
            return [ReduceScatterPrim([sub1, sub2], [sub3], dim=0), AllToAllPrim([sub1, sub3], [sub2], idim=0, odim=1)]
        else:
            right_called = True
            return [AllToAllPrim([sub1, sub2], [sub3], idim=0, odim=1)]

    with catch_log(_logger, 'WARNING') as log_stream:
        assert ConcurrentGener._path(path_with_reduce_scatter, None, None, None)
        assert right_called and wrong_called
        assert 'Detected invalid AllToAllPrim' in log_stream.getvalue()

    called = 0
    def path_without_rc(*args, **kwargs):
        nonlocal called
        called += 1
        return [AllToAllPrim([sub1, sub3], [sub2], idim=0, odim=1)]

    with pytest.raises(RuntimeError, match='Invalid primitives detected.*'):
        with catch_log(_logger) as log_stream:
            ConcurrentGener._path(path_without_rc, None, None, None)

    assert called == 1
    assert 'Detected invalid AllToAllPrim' not in log_stream.getvalue()
