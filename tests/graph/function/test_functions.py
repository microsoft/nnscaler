#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

### Only test the anno creation in these tests

from functools import reduce
from operator import add
from nnscaler.graph.function.dimops import IRDimops, OpAnno
import nnscaler.graph.function.function as F
from nnscaler.ir.cten import IR, IRObject, IRTensor

import pytest
import torch
import numpy as np
import math

from nnscaler.ir.tensor import IRFullTensor


def o(value):
    return IRObject(value=value)


def assert_anno(op, expected):
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == expected


def test_handle_broadcast_multi():
    ins_anno, out_anno = F._handle_broadcast_multi([IRTensor([4]), IRTensor([3, 4]), IRTensor([2, 3, 4])])
    assert ins_anno[0] == ['c']
    assert ins_anno[1] == ['b', 'c']
    assert ins_anno[2] == ['a', 'b', 'c']
    assert out_anno == ['a', 'b', 'c']

    ins_anno, out_anno = F._handle_broadcast_multi([IRTensor([1]), IRTensor([2, 1, 4]), IRTensor([2, 3, 4])])
    assert ins_anno[0] == ['1']
    assert ins_anno[1] == ['a', '1', 'c']
    assert ins_anno[2] == ['a', 'b', 'c']
    assert out_anno == ['a', 'b', 'c']


def test_Full():
    op = F.Full([1, 2, 3], 1.)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == ' -> 1 2 3'

    op = F.Full([], 1.)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == ' -> 1'


def test_Expand():
    inp = IRTensor([10, 1])
    out = IRTensor([10, 2])
    op = F.Expand(inp, 10, 2)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b^ -> a 2'
    assert op.kwargs['size'] == [-1, 2]

    op.new([inp], [out], size=[10, 2])
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b^ -> a 2'

    with pytest.raises(ValueError):
        F.Expand(inp, (10, 2), 1)

    with pytest.raises(ValueError):
        F.Expand(inp, 1, (10, 2))

    with pytest.raises(ValueError):
        F.Expand(inp, 1)

    with pytest.raises(ValueError):
        F.Expand(inp, (5, 2))

    op = F.Expand(inp, -1, o(1))
    assert_anno(op,  'a b^ -> a 1')
    assert op.kwargs['size'][0] == -1
    assert op.kwargs['size'][1].value == 1

    op = F.Expand(inp, -1, 1)
    assert_anno(op,  'a b^ -> a 1')
    assert op.kwargs['size'] == [-1, 1]

    assert_anno(F.Expand(inp, -1, o(2)),  'a b^ -> a 2')
    assert_anno(F.Expand(inp, o((10, 2))),  'a^ b^ -> 10 2')

    op = F.Expand(inp, o(10), o(2))
    assert_anno(op,  'a^ b^ -> 10 2')
    assert op.kwargs['size'][0].value == 10
    assert op.kwargs['size'][1].value == 2

    op = F.Expand(inp, o(10), o(-1))
    assert_anno(op,  'a^ b^ -> 10 b^')
    assert op.kwargs['size'][0].value == 10
    assert op.kwargs['size'][1].value == -1

    op = F.Expand(inp, 10, 10, 2)
    assert_anno(op, 'b c^ -> 10 b 2')
    assert op.kwargs['size'] == [10, -1, 2]


def test_variadic_extraction():
    def o(value):
        return IRObject(value=value)
    assert F.extract_variadic([]) == ([], [])
    assert F.extract_variadic(1) == ([1], [False])
    assert F.extract_variadic([1]) == ([1], [False])
    assert F.extract_variadic((1,)) == ([1], [False])
    assert F.extract_variadic([1, 2]) == ([1, 2], [False, False])

    assert F.extract_variadic(o([])) == ([], [])
    assert F.extract_variadic(o(1)) == ([1], [True])
    assert F.extract_variadic(o([1])) == ([1], [True])
    assert F.extract_variadic(o((1,))) == ([1], [True])
    assert F.extract_variadic(o([1, 2])) == ([1, 2], [True, True])

    assert F.extract_variadic([1, o(2)]) == ([1, 2], [False, True])
    assert F.extract_variadic([1, o(2), 3, o(4)]) == ([1, 2, 3, 4], [False, True, False, True])

    with pytest.raises(ValueError, match='.*nested.*'):
        F.extract_variadic([1, o([2, 3])])
    with pytest.raises(ValueError, match='.*nested.*'):
        F.extract_variadic([1, [2, 3]])
    with pytest.raises(ValueError, match='Unsupported type.*'):
        F.extract_variadic(True)
    with pytest.raises(ValueError, match='Unsupported type.*'):
        F.extract_variadic([1, True])
    with pytest.raises(ValueError, match='Unsupported type.*'):
        F.extract_variadic(o([1, True]))


def test_Repeat():
    inp = IRTensor([3])
    op = F.Repeat(inp, (4, 2))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'b^ -> 4 (2 b^)'

    inp = IRTensor([3])
    op = F.Repeat(inp, (4, 1))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'b -> 4 b'

    inp = IRTensor([3])
    op = F.Repeat(inp, 4, 1)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'b -> 4 b'

    inp = IRTensor([3, 2])
    op = F.Repeat(inp, (4, 2))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a^ b^ -> (4 a^) (2 b^)'

    inp = IRTensor([3, 2])
    op = F.Repeat(inp, 4, 2, 1)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'b^ c -> 4 (2 b^) c'

    inp = IRTensor([3, 2])
    op = F.Repeat(inp, (4, 2), 1)  # the args(1) is ignored
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a^ b^ -> (4 a^) (2 b^)'

    inp = IRTensor([3, 2])
    op = F.Repeat(inp, (4, 1))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a^ b -> (4 a^) b'

    inp = IRTensor([3, 2])
    with pytest.raises(ValueError):
        op = F.Repeat(inp, (2))

    with pytest.raises(ValueError, match='.*nested.*'):
        op = F.Repeat(inp, 4, (4, 2))

    inp = IRTensor([3])
    op = F.Repeat(inp, o((4, 2)))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'b^ -> 4 (2 b^)'

    inp = IRTensor([3])
    op = F.Repeat(inp, (4, o(1)))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'b^ -> 4 b^'

    inp = IRTensor([3, 2])
    op = F.Repeat(inp, (4, o(2)))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a^ b^ -> (4 a^) (2 b^)'

    inp = IRTensor([3, 2])
    op = F.Repeat(inp, (o(4), 1))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a^ b -> (4 a^) b'

    inp = IRTensor([3, 2])
    op = F.Repeat(inp, 4, 2, o(1), 2)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'c^ d^ -> 4 2 c^ (2 d^)'

    inp = IRTensor([3, 2])
    with pytest.raises(ValueError):
        op = F.Repeat(inp, o(2))


def test_Topk():
    op = F.Topk(IRTensor([3, 4, 5]), 3)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b c^ -> a b 3, a b 3'
    op = F.Topk(IRTensor([3, 4, 5]), 3, dim = 1)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b^ c -> a 3 c, a 3 c'


def test_Nonzero():
    op = F.Nonzero(IRTensor([3, 4, 5]), as_tuple=True)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a^ b^ c^ -> ?, ?, ?'
    op = F.Nonzero(IRTensor([3, 4, 5]), as_tuple=False)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a^ b^ c^ -> ?'


def test_Where():
    op = F.Where(IRTensor([3, 4]), IRTensor([3, 4]), IRTensor([3, 4]))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b, a b, a b -> a b'
    op = F.Where(IRTensor([3, 4]), IRTensor([4]), IRTensor([3, 4]))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b, b, a b -> a b'
    op = F.Where(IRTensor([3, 4]), IRTensor([1]), IRTensor([3, 4]))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b, 1, a b -> a b'
    op = F.Where(IRTensor([3, 4]), 1, IRTensor([3, 4]))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b, ?, a b -> a b'
    op = F.Where(IRTensor([3, 4]), IRTensor([3, 4]), IRTensor([4]))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b, a b, b -> a b'
    op = F.Where(IRTensor([3, 4]), IRTensor([3, 4]), IRTensor([1]))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b, a b, 1 -> a b'
    op = F.Where(IRTensor([3, 4]), IRTensor([3, 4]), 1)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b, a b, ? -> a b'
    op = F.Where(IRTensor([3, 4]), 1, 2)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b, ?, ? -> a b'


def test_FullSlice():
    op = F.FullSlice(IRTensor([2, 3, 4]), 1, [1.2, -1], 2)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a^ b^ c^, ?, ?, ? -> 2'

    op = F.FullSlice(IRTensor([2, 3, 4]), 1, 2, 3)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a^ b^ c^, ?, ?, ? -> 1'
    op = F.FullSlice(IRTensor([2, 3, 4]), ..., 2)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b c^, ?, ?, ? -> a b'
    op = F.FullSlice(IRTensor([2, 3, 4]), 1, 2, slice(0, 3, 2))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a^ b^ c^, ?, ?, ? -> 2'
    op = F.FullSlice(IRTensor([2, 3, 4]), 1, 2, slice(1, 10, 1))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a^ b^ c^, ?, ?, ? -> 3'
    op = F.FullSlice(IRTensor([2, 3, 4]), 1, None, ..., None, slice(0, 2, None))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a^ b c^, ?, ?, ?, ?, ? -> 1 b 1 2'

    with pytest.raises(RuntimeError):
        op = F.FullSlice(IRTensor([2, 3, 4]), slice(1, IRTensor([2]), 3))
    op = F.FullSlice(IRTensor([3, 4]), None, 0, slice(0, 4, 2))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a^ b^, ?, ?, ? -> 1 2'
    op = F.FullSlice(IRTensor([3, 4]), [0,2], 0)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a^ b^, ?, ? -> 2'
    op = F.FullSlice(IRTensor([3, 4]), [[0,1], [1,2]], 0)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a^ b^, ?, ? -> 2 2'
    op = F.FullSlice(IRTensor([3, 4]), IRFullTensor([2]))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a^ b, c -> c b'
    op = F.FullSlice(IRTensor([3, 4]), IRFullTensor([2,2]))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a^ b, c d -> c d b'
    op = F.FullSlice(IRTensor([3, 4]), [True, False, True])
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a^ b^, ? -> ?'
    op = F.FullSlice(IRTensor([3, 4]), IRFullTensor([3], dtype=torch.bool), 0)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a^ b^, c^, ? -> ?'
    op = F.FullSlice(IRTensor([3, 4]), [True, False, True], [0,1])
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a^ b^, ?, ? -> ?'
    op = F.FullSlice(IRTensor([3, 4]), [True, False, True], IRFullTensor([2,2]))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a^ b^, ?, c^ d^ -> ?'
    op = F.FullSlice(IRTensor([3, 4]), IRFullTensor([3, 4], dtype=torch.bool))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a^ b^, c^ d^ -> ?'
    op = F.FullSlice(IRTensor([3, 4]), IRFullTensor([3]), IRFullTensor([3]))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a^ b^, c^, d^ -> ?'
    op = F.FullSlice(IRTensor([3, 4]), IRFullTensor([2,2]), IRFullTensor([2,2]))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a^ b^, c^ d^, e^ f^ -> ?'


def test_GetItem():
    # obj is IRTensor, index is IRTensor
    op = F.GetItem(IRTensor([4, 2]), IRTensor([3, 5], dtype=torch.int64))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a^ b, c d -> c d b'
    op = F.GetItem(IRTensor([4, 2]), IRTensor([3], dtype=torch.int64))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a^ b, c -> c b'
    op = F.GetItem(IRTensor([3, 4, 2]), IRTensor([3], dtype=torch.int64))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a^ b c, d -> d b c'
    op = F.GetItem(IRTensor([3, 4, 2]), IRTensor([3, 5], dtype=torch.int64))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a^ b c, d e -> d e b c'

    # obj is IRTensor, index is not IRTensor, will call FullSlice
    op = F.GetItem(IRTensor([3, 4, 2]), 0)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a^ b c, ? -> b c'
    op = F.GetItem(IRTensor([3, 4, 2]), [0, 1])
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a^ b^ c, ?, ? -> c'
    op = F.GetItem(IRTensor([3, 4, 2]), slice(0, 3, 2))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a^ b c, ? -> 2 b c'
    op = F.GetItem(IRTensor([3, 4, 2]), [slice(None), IRTensor([3, 5], dtype=torch.int64)])
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b^ c, ?, d e -> a d e c'
    op = F.GetItem(IRTensor([3, 4, 2]), [slice(None), IRTensor([4, 2], dtype=torch.bool)])
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a^ b^ c^, ?, d^ e^ -> ?'

    # obj is IRObject
    op = F.GetItem(IRObject(value=[3, 4, 5], is_constant=False), IRObject(value=0, is_constant=False), signature='operator.getitem')
    assert op.outputs()[0].value == 3 and op.outputs()[0].is_constant == True
    op = F.GetItem(IRObject(value=[3, 4, 5], is_constant=False), IRObject(value=slice(0, 2, 1), is_constant=False), signature='operator.getitem')
    assert op.outputs()[0].value == [3, 4] and op.outputs()[0].is_constant == True
    op = F.GetItem(IRObject(value=[3, 4, 5], is_constant=False), 0, signature='operator.getitem')
    assert op.outputs()[0].value == 3 and op.outputs()[0].is_constant == True
    op = F.GetItem(IRObject(value=[3, 4, 5], is_constant=False), slice(0, 2, 1), signature='operator.getitem')
    assert op.outputs()[0].value == [3, 4] and op.outputs()[0].is_constant == True

    # obj is not a IRObject, index is a IRObject
    op = F.GetItem([1, 2, 3], IRObject(value=0, is_constant=False), signature='operator.getitem')
    assert op.outputs()[0].value == 1 and op.outputs()[0].is_constant == False

    # direct call obj[index]
    op = F.GetItem([3, 4, 2], 1)
    assert op == 4
    op = F.GetItem([3, 4, 2], slice(0, 2, 1))
    assert op == [3, 4]


def test_Max():
    op = F.Max(IRTensor([2, 3, 4]))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a^ b^ c^ -> 1'
    op = F.Max(IRTensor([2, 3, 4]), 1, True)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b^ c -> a 1 c, a 1 c'
    op = F.Max(IRTensor([2, 3, 4]), 1, False)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b^ c -> a c, a c'
    op = F.Max(IRTensor([2, 3, 4]), 1)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b^ c -> a c, a c'
    op = F.Max(IRTensor([2, 3, 4]), IRTensor([4]))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b c, c -> a b c'
    op = F.Max(IRTensor([2, 3, 4]), IRObject(value=2, is_constant=False), True)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b c^ -> a b 1, a b 1'
    op = F.Max(IRTensor([2, 3, 4]), 2,IRObject(value=True, is_constant=False))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b c^ -> a b 1, a b 1'
    op = F.Max(IRTensor([2, 3, 4]), 2,IRObject(value=False, is_constant=False))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b c^ -> a b, a b'
    op = F.Max(IRTensor([2, 3, 4]), IRObject(value=2, is_constant=False),IRObject(value=True, is_constant=False))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b c^ -> a b 1, a b 1'
    op = F.Max(IRTensor([2, 3, 4]), IRObject(value=2, is_constant=False),IRObject(value=IRObject(value=True), is_constant=False))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b c^ -> a b 1, a b 1'
    op = F.Max(IRTensor([2, 3, 4]), IRObject(value=2, is_constant=False),IRObject(value=None, is_constant=False))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b c^ -> a b, a b'
    op = F.Max(IRTensor([2, 3, 4]), IRObject(value=2, is_constant=False))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b c^ -> a b, a b'


def test_Squeeze():
    op = F.Squeeze(IRTensor([2, 1, 4, 1]))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a^ b c^ d -> a^ c^'
    op = F.Squeeze(IRTensor([2, 1, 4, 1]), 1)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b c d -> a c d'
    op = F.Squeeze(IRTensor([2, 1, 4, 1]), 0)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a^ b c d -> a^ b c d'
    op = F.Squeeze(IRTensor([2, 1, 4, 1]), -1)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b c d -> a b c'
    op = F.Squeeze(IRTensor([2, 1, 4, 1]), -2)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b c^ d -> a b c^ d'

    op = F.Squeeze(IRTensor([2, 1, 4, 1]), (-1, -2))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b c^ d -> a b c^'
    op = F.Squeeze(IRTensor([2, 1, 4, 1]), (1, -1))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b c d -> a c'
    op = F.Squeeze(IRTensor([2, 1, 4, 1]), (1, -2))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b c^ d -> a c^ d'
    op = F.Squeeze(IRTensor([2, 1, 4, 1]), (0, -2))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a^ b c^ d -> a^ b c^ d'


def test_Unsqueeze():
    op = F.Unsqueeze(IRTensor([2, 4]), 0)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b -> 1 a b'
    op = F.Unsqueeze(IRTensor([2, 4]), 1)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b -> a 1 b'
    op = F.Unsqueeze(IRTensor([2, 4]), 2)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b -> a b 1'
    op = F.Unsqueeze(IRTensor([2, 4]), -3)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b -> 1 a b'
    op = F.Unsqueeze(IRTensor([2, 4]), -2)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b -> a 1 b'
    op = F.Unsqueeze(IRTensor([2, 4]), -1)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b -> a b 1'


def test_ScaledDotProductAttention():
    op = F.ScaledDotProductAttention(IRTensor([8, 128, 64]), IRTensor([8, 256, 64]), IRTensor([8, 256, 32]), None, 0.05)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a e d^, a b^ d^, a b^ c -> a e c'
    op = F.ScaledDotProductAttention(IRTensor([8, 128, 64]), IRTensor([8, 256, 64]), IRTensor([8, 256, 32]), None, 0.05, True)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a e^ d^, a b^ d^, a b^ c -> a e^ c'
    op = F.ScaledDotProductAttention(IRTensor([16, 8, 128, 64]), IRTensor([16, 8, 256, 64]), IRTensor([16, 8, 256, 32]), IRTensor([128, 256]), 0.05)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b f^ e^, a b c^ e^, a b c^ d, f^ c^ -> a b f^ d'
    op = F.ScaledDotProductAttention(IRTensor([16, 8, 128, 64]), IRTensor([16, 8, 256, 64]), IRTensor([16, 8, 256, 32]), IRTensor([1, 128, 256]), 0.05)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b f^ e^, a b c^ e^, a b c^ d, 1 f^ c^ -> a b f^ d'
    op = F.ScaledDotProductAttention(IRTensor([16, 8, 128, 64]), IRTensor([16, 8, 256, 64]), IRTensor([16, 8, 256, 32]), IRTensor([1, 8, 128, 256]), 0.05)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b f^ e^, a b c^ e^, a b c^ d, 1 b f^ c^ -> a b f^ d'
    op = F.ScaledDotProductAttention(IRTensor([16, 8, 128, 64]), IRTensor([16, 8, 256, 64]), IRTensor([16, 8, 256, 32]), IRTensor([1, 1, 256]), 0.05)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b f^ e^, a b c^ e^, a b c^ d, 1 1 c^ -> a b f^ d'
    op = F.ScaledDotProductAttention(IRTensor([16, 8, 128, 64]), IRTensor([16, 8, 256, 64]), IRTensor([16, 8, 256, 32]), IRTensor([1, 8, 128, 1]), 0.05)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b f^ e^, a b c^ e^, a b c^ d, 1 b f^ 1 -> a b f^ d'



def test_NewTensor():
    op = F.NewTensor(torch.tensor(1))
    assert op.signature == 'nnscaler.runtime.function.tensor'
    assert repr(op.anno) == ' -> 1^'
    assert op.kwargs['data'] == 1

    op = F.NewTensor(torch.tensor([1,2]))
    assert op.signature == 'nnscaler.runtime.function.tensor'
    assert repr(op.anno) == ' -> 2^'
    assert op.kwargs['data'] == [1,2]

    obj = IRObject(value=np.array([1,2]))
    op = F.NewTensor(obj)
    assert repr(op.anno) == ' -> 2^'
    assert op.kwargs['data'] == obj

    op = F.NewTensor(np.array([[1],[2],[3]]))
    assert repr(op.anno) == ' -> 3^ 1^'
    assert op.kwargs['data'] == [[1],[2],[3]]


def test_Setitem():
    set_val = IRObject(value=4, is_constant=False)
    op = F.SetItem(IRObject(value=[1, 2, 3]), 0, set_val)
    assert op.outputs()[0].value == [set_val, 2, 3]
    assert op.outputs()[0].is_constant

    op = F.SetItem(IRObject(value=[1, 2, 3], is_constant=False), 0, set_val)
    assert op.outputs()[0].value == [set_val, 2, 3]
    assert not op.outputs()[0].is_constant

    op = F.SetItem(IRTensor([3, 4, 5]), IRObject(value=slice(0, 5, 1)), IRObject(value=1.))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a^ b^ c^, ?, ? -> a^ b^ c^'

    op = F.SetItem(IRTensor([3, 4, 5]), IRTensor([3, 4, 5]), IRObject(value=1.))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a^ b^ c^, d^ e^ f^, ? -> a^ b^ c^'

    op = F.SetItem(IRTensor([3, 4, 5]), IRTensor([3]), IRObject(value=0), IRObject(value=0), IRObject(value=1.))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a^ b^ c^, d^, ?, ?, ? -> a^ b^ c^'


def test_Len():
    op = F.Len([1, 2, 3], signature='builtins.len')
    assert op.outputs()[0].value == 3

    op = F.Len(IRObject(value=[1, 2, 3], is_constant=False), signature='builtins.len')
    assert op.outputs()[0].value == 3 and not op.outputs()[0].is_constant


def test_Min():
    op = F.Min(IRTensor([2, 3, 4]))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a^ b^ c^ -> 1'
    op = F.Min(IRTensor([2, 3, 4]), 1, True)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b^ c -> a 1 c, a 1 c'
    op = F.Min(IRTensor([2, 3, 4]), 1, False)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b^ c -> a c, a c'
    op = F.Min(IRTensor([2, 3, 4]), 1)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b^ c -> a c, a c'
    op = F.Min(IRTensor([2, 3, 4]), IRTensor([4]))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b c, c -> a b c'
    op = F.Min(IRTensor([4]), IRTensor([2, 3, 4]))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'c, a b c -> a b c'
    op = F.Min(IRTensor([2, 3, 4]), IRObject(value=2, is_constant=False), True)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b c^ -> a b 1, a b 1'
    op = F.Min(IRTensor([2, 3, 4]), 2,IRObject(value=True, is_constant=False))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b c^ -> a b 1, a b 1'
    op = F.Min(IRTensor([2, 3, 4]), 2,IRObject(value=False, is_constant=False))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b c^ -> a b, a b'
    op = F.Min(IRTensor([2, 3, 4]), IRObject(value=2, is_constant=False),IRObject(value=True, is_constant=False))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b c^ -> a b 1, a b 1'
    op = F.Min(IRTensor([2, 3, 4]), IRObject(value=2, is_constant=False),IRObject(value=None, is_constant=False))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b c^ -> a b, a b'
    op = F.Min(IRTensor([2, 3, 4]), IRObject(value=2, is_constant=False))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b c^ -> a b, a b'
    op = F.Min(IRTensor([2, 3, 4]), IRObject(value=2, is_constant=False),IRObject(value=IRObject(value=True), is_constant=False))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b c^ -> a b 1, a b 1'


def test_FullLike():
    op = F.FullLike(IRTensor([2, 1, 4, 1]), 1.)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b c d -> a b c d'
    op_int = F.FullLike(IRTensor([3, 2]), 5)
    assert len(op_int._annos_candidates) == 1 and op_int._annos_candidates[0] == 'a b -> a b'
    op_true = F.FullLike(IRTensor([2, 2]), 1., requires_grad=True)
    assert len(op_true._annos_candidates) == 1 and op_true._annos_candidates[0] == 'a b -> a b'
    op_float = F.FullLike(IRTensor([1, 2],dtype=torch.int), 1, dtype=torch.float)
    assert len(op_float._annos_candidates) == 1 and op_float._annos_candidates[0] == 'a b -> a b'


def test_Log():
    result = F.Log(2)
    assert result == math.log(2)
    input_tensor = torch.rand(1, 2, 3)
    op = F.Log(input_tensor)
    assert torch.allclose(op, torch.log(input_tensor)) and op.shape == (1, 2, 3)
    op = F.Log(IRTensor([1,2,3]))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b c -> a b c'
    op = F.Log(IRObject(value=6, is_constant=False), signature='math.log')
    assert op.outputs()[0].value == math.log(6) and not op.outputs()[0].is_constant


def test_ZerosLike():
    op = F.ZerosLike(IRTensor([2, 1, 4, 1]))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b c d -> a b c d'
    op_true = F.ZerosLike(IRTensor([2, 2]), requires_grad=True)
    assert len(op_true._annos_candidates) == 1 and op_true._annos_candidates[0] == 'a b -> a b'
    op_float = F.ZerosLike(IRTensor([1, 2],dtype=torch.int), dtype=torch.float)
    assert len(op_float._annos_candidates) == 1 and op_float._annos_candidates[0] == 'a b -> a b'


def test_OnesLike():
    op = F.OnesLike(IRTensor([2, 1, 4, 1]))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b c d -> a b c d'
    op_true = F.OnesLike(IRTensor([2, 2]), requires_grad=True)
    assert len(op_true._annos_candidates) == 1 and op_true._annos_candidates[0] == 'a b -> a b'
    op_float = F.OnesLike(IRTensor([1, 2],dtype=torch.int), dtype=torch.float)
    assert len(op_float._annos_candidates) == 1 and op_float._annos_candidates[0] == 'a b -> a b'


def test_addmm():
    op = F.Addmm(IRTensor([2, 3]), mat1=IRTensor([2, 7]), mat2=IRTensor([7, 3]))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b, a k^, k^ b -> a b'
    op = F.Addmm(IRTensor([1, 3]), mat1=IRTensor([2, 7]), mat2=IRTensor([7, 3]))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == '1 b, a k^, k^ b -> a b'
    op = F.Addmm(IRTensor([2, 1]), mat1=IRTensor([2, 7]), mat2=IRTensor([7, 3]))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a 1, a k^, k^ b -> a b'
    op = F.Addmm(IRTensor([1, 1]), mat1=IRTensor([2, 7]), mat2=IRTensor([7, 3]))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == '1 1, a k^, k^ b -> a b'
    op = F.Addmm(IRTensor([3]), mat1=IRTensor([2, 3]), mat2=IRTensor([3, 3]))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'b, a k^, k^ b -> a b'
    op = F.Addmm(IRTensor([7]), mat1=IRTensor([2, 3]), mat2=IRTensor([3, 7]))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'b, a k^, k^ b -> a b'


def test_type():
    op = F.Type(IRTensor([2,3],dtype=None),torch.float32)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == '* -> *' and op.kwargs['dtype'] == torch.float32
    op = F.Type(IRTensor([3, 5], dtype=torch.int64),torch.float32)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == '* -> *'  and op.kwargs['dtype'] == torch.float32
    op = F.Type(IRTensor([3, 5], dtype=torch.int64),IRObject(value=torch.float32, is_constant=False), signature='torch.type')
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == '* -> *'  and op.kwargs['dtype'].value == torch.float32
    op = F.Type(IRTensor([3, 5], dtype=torch.int64),dtype=IRObject(value=None, is_constant=False), signature='torch.type')
    assert op.outputs()[0].value == "torch.int64"
    op = F.Type(IRTensor([3, 5], dtype=torch.int64),dtype=torch.int64)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == '* -> *' and op.kwargs['dtype'] == torch.int64
    op = F.Type(IRTensor([3, 5], dtype=torch.int64),"torch.int64")
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == '* -> *' and op.kwargs['dtype'] == 'torch.int64'
    op = F.Type(IRTensor([3, 5], dtype=torch.int64), signature='torch.type')
    assert op.outputs()[0].value == "torch.int64"


def test_to():
    with pytest.raises(ValueError, match='.*is not a valid argument.*'):
        op = F.To(IRTensor([2, 3], dtype=torch.float32), xx=None)

    with pytest.raises(ValueError, match='.*is not a valid argument.*'):
        op = F.To(IRTensor([2, 3], dtype=torch.float32), 0, xx=None)

    with pytest.raises(ValueError, match='.*is not a valid argument.*'):
        op = F.To(IRTensor([2, 3], dtype=torch.float32), torch.float32, xx=None)

    # 1st overload
    op = F.To(IRTensor([2, 3], dtype=torch.float32))  # No arguments
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == '* -> *' and not op.kwargs
    op = F.To(IRTensor([2, 3], dtype=torch.float32), None)  # only None
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == '* -> *' and not op.kwargs
    op = F.To(IRTensor([2, 3], dtype=torch.float32), torch.device('cuda:0'))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == '* -> *' and not op.kwargs
    op = F.To(IRTensor([2, 3], dtype=torch.float32), 'cuda:0')
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == '* -> *' and not op.kwargs
    op = F.To(IRTensor([2, 3], dtype=torch.float32), 0)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == '* -> *' and not op.kwargs
    op = F.To(IRTensor([2, 3], dtype=torch.float32), device=None)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == '* -> *' and not op.kwargs
    op = F.To(IRTensor([2, 3], dtype=torch.float32), device=torch.device('cuda:0'))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == '* -> *' and not op.kwargs
    op = F.To(IRTensor([2, 3], dtype=torch.float32), device=0)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == '* -> *' and not op.kwargs
    op = F.To(IRTensor([2, 3], dtype=torch.float32), device='cuda:0')
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == '* -> *' and not op.kwargs

    op = F.To(IRTensor([2, 3], dtype=torch.float32), 0, None)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == '* -> *' \
        and op.kwargs == {'dtype': None}

    op = F.To(IRTensor([2, 3], dtype=torch.float32), 0, None, True)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == '* -> *' \
        and op.kwargs == {'dtype': None, 'non_blocking': True}

    op = F.To(IRTensor([2, 3], dtype=torch.float32), 0, None, True, None)
    # Note type of copy is None, which is not correct.
    # because currently we don't do type checking
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == '* -> *' \
        and op.kwargs == {'dtype': None, 'non_blocking': True, 'copy': None}

    # 1st overload with options
    op = F.To(IRTensor([2, 3], dtype=torch.float32), 0, copy=True)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == '* -> *' \
        and op.kwargs == {'copy': True}

    op = F.To(IRTensor([2, 3], dtype=torch.float32), 0, None, copy=True)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == '* -> *' \
        and op.kwargs == {'dtype': None, 'copy': True}

    op = F.To(IRTensor([2, 3], dtype=torch.float32), 0, copy=True, non_blocking=True, memory_format=torch.contiguous_format)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == '* -> *' \
        and op.kwargs == {'copy': True, 'non_blocking': True, 'memory_format': torch.contiguous_format}

    # 1st overload with duplicate options
    with pytest.raises(ValueError):
        op = F.To(IRTensor([2, 3], dtype=torch.float32), 0, copy=True, device=None)

    # 2nd overload
    op = F.To(IRTensor([2, 3]), torch.float32)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == '* -> *' and op.kwargs == {'dtype': torch.float32}

    op = F.To(IRTensor([2, 3]), dtype=torch.float32)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == '* -> *' and op.kwargs == {'dtype': torch.float32}

    op = F.To(IRTensor([2, 3]), torch.float32, False)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == '* -> *' \
        and op.kwargs == {'dtype': torch.float32, 'non_blocking': False}

    op = F.To(IRTensor([2, 3]), torch.float32, False, copy=False, memory_format=torch.contiguous_format)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == '* -> *' \
        and op.kwargs == {'dtype': torch.float32, 'non_blocking': False,
                          'copy': False, 'memory_format': torch.contiguous_format}

    # duplicate options
    with pytest.raises(ValueError):
        op = F.To(IRTensor([2, 3]), torch.float32, False, non_blocking=True)

    # 3rd overload
    op = F.To(IRTensor([3, 5], dtype=torch.int64), IRTensor(dtype=torch.float32))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == '* -> *' and op.kwargs == {'dtype': torch.float32}
    op = F.To(IRTensor([3, 5], dtype=torch.int64), IRTensor(dtype=torch.float32), True)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == '* -> *' and op.kwargs == {'dtype': torch.float32, 'non_blocking': True}
    op = F.To(IRTensor([3, 5], dtype=torch.int64), IRTensor(dtype=torch.float32), True, None)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == '* -> *' and op.kwargs == {'dtype': torch.float32, 'non_blocking': True, 'copy': None}
    op = F.To(IRTensor([3, 5], dtype=torch.int64), IRTensor(dtype=torch.float32), True, copy=True)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == '* -> *' and op.kwargs == {'dtype': torch.float32, 'non_blocking': True, 'copy': True}

    # duplicate options
    with pytest.raises(ValueError):
        op = F.To(IRTensor([2, 3]), IRTensor(dtype=torch.float32), False, non_blocking=True)
    # too many positional arguments
    with pytest.raises(ValueError, match='.*too many positional arguments.*'):
        op = F.To(IRTensor([3, 5], dtype=torch.int64), IRTensor(dtype=torch.float32), True, None, True)


def test_outer():
    op = F.Outer(IRTensor([2]), vec2=IRTensor([2]))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'n, m -> n m'


def test_erf():
    op = F.Erf(IRTensor([2]))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a -> a'
    op = F.Erf(IRTensor([2,3]))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b -> a b'
    op = F.Erf(IRTensor([2,3,4]))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b c -> a b c'
    op = F.Erf(IRTensor([2,3,4,5]))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b c d -> a b c d'


def test_mul_or_multiply():
    op = F.Mul(IRTensor([1,2]),100)
    assert len(op._annos_candidates) == 2 and op._annos_candidates[0] == '*, ? -> *'
    op = F.Mul(IRTensor([1,2]),IRTensor([1,2]))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b, a b -> a b'
    op = F.Mul(IRTensor([2,2]),IRTensor([2]))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b, b -> a b'
    op = F.Mul(100,IRTensor([6]))
    assert len(op._annos_candidates) == 2 and op._annos_candidates[1] == '?, * -> *'
    op = F.Mul(torch.tensor([[1, 2], [1, 2]]),100)
    assert torch.equal(torch.mul(torch.tensor([[1, 2], [1, 2]]), 100), op), "The result does not match the expected output"
    op = F.Mul(torch.tensor([[1, 2], [1, 2]]),torch.tensor([[1, 2]]))
    assert torch.equal(torch.mul(torch.tensor([[1, 2], [1, 2]]), torch.tensor([[1, 2]])), op), "The result does not match the expected output"
    op = F.Mul(torch.tensor([1, 2]),IRObject(value=100, is_constant=False), signature='torch.mul')
    assert torch.equal(op.outputs()[0].value, torch.mul(torch.tensor([1, 2]), 100)) and not op.outputs()[0].is_constant  and  torch.equal(op.outputs()[0].value, torch.tensor([100, 200]))


def test_Softmax():
    op = F.Softmax(IRTensor([2, 3, 4]))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b c -> a b c'
    op = F.Softmax(IRTensor([2, 3, 4]), dim=1)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b^ c -> a b^ c'
    op = F.Softmax(IRTensor([2, 3, 4]), dim=2, dtype=torch.float64)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b c^ -> a b c^'
    op = F.Softmax(IRTensor([2, 3, 4]), dtype=torch.float32)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b c -> a b c'


def test_Conv1D():
    op = F.Conv1D(IRTensor([3, 4]), IRTensor([3, 3, 1]))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'iC+ 4, oC iC+ 1 -> oC 4'
    op = F.Conv1D(IRTensor([3, 4]), IRTensor([3, 3, 1]), groups=1,padding="valid")
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'iC+ 4, oC iC+ 1 -> oC 4'
    op = F.Conv1D(input=IRTensor([8, 32]), weight=IRTensor([16, 8, 3]), bias=IRObject(value=16),groups=1,padding="same")
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'iC^ 32, oC iC^ 3, oC -> oC 32'
    op = F.Conv1D(IRTensor([2, 3, 4]), IRTensor([3, 3, 1]))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'n iC+ 4, oC iC+ 1 -> n oC 4'
    op = F.Conv1D(IRTensor([2, 3, 4]), IRTensor([3, 3, 1]), stride=2)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'n iC+ 4, oC iC+ 1 -> n oC 2'
    op = F.Conv1D(IRTensor([2, 3, 4]), IRTensor([3, 3, 1]), padding=1)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'n iC+ 4, oC iC+ 1 -> n oC 6'
    op = F.Conv1D(IRTensor([2, 3, 4]), IRTensor([3, 3, 1]), padding=IRObject(value=1))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'n iC+ 4, oC iC+ 1 -> n oC 6'
    op = F.Conv1D(IRTensor([2, 3, 4]), IRTensor([3, 3, 1]), dilation=2)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'n iC+ 4, oC iC+ 1 -> n oC 4'
    op = F.Conv1D(IRTensor([2, 3, 4]), IRTensor([3, 3, 1]), groups=1)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'n iC+ 4, oC iC+ 1 -> n oC 4'
    op = F.Conv1D(IRTensor([2, 3, 4]), IRTensor([3, 3, 1]), groups=1,padding="valid")
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'n iC+ 4, oC iC+ 1 -> n oC 4'
    op = F.Conv1D(input=IRTensor([4, 8, 32]), weight=IRTensor([16, 8, 3]), bias=IRObject(value=16),groups=1,padding="same")
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'n iC^ 32, oC iC^ 3, oC -> n oC 32'
    op = F.Conv1D(IRTensor([2, 3, 4]), IRTensor([3, 1, 1]), groups=3)
    expected_annotation_for_groups = 'n (g 1) 4, (g 1) 1 1 -> n (g 1) 4'
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == expected_annotation_for_groups
    op = F.Conv1D(IRTensor([2, 3, 4]), IRTensor([3, 3, 1]), bias=IRObject(value=3))
    assert op._annos_candidates[0] == 'n iC^ 4, oC iC^ 1, oC -> n oC 4', "Annotation mismatch."


def test_Arange():
    op = F.Arange(10)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == ' -> 10^' and op.kwargs['dtype'] == torch.int64
    op = F.Arange(1, 10, 2)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == ' -> 5^' and op.kwargs['dtype'] == torch.int64
    op = F.Arange(1, 10, 2, dtype=torch.float)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == ' -> 5^' and op.kwargs['dtype'] == torch.float
    op = F.Arange(1.0, 10.0, 2.0)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == ' -> 5^' and op.kwargs['dtype'] == torch.float
    op = F.Arange(IRObject(value=10))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == ' -> 10^' and op.kwargs['dtype'] == torch.int64
    op = F.Arange(IRObject(value=1), IRObject(value=10.0), 2)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == ' -> 5^' and op.kwargs['dtype'] == torch.float


def test_Flatten():
    op = F.Flatten(IRTensor([2,3,4,5]), 1, 2)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b^ c^ d -> a (b^ c^) d'
    op = F.Flatten(IRTensor([2,3,4,5]), 1)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b^ c^ d^ -> a (b^ c^ d^)'
    op = F.Flatten(IRTensor([2,3,4,5]), end_dim = 2)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a^ b^ c^ d -> (a^ b^ c^) d'


def test_Gather():
    op = F.Gather(IRTensor([2, 5, 3]), 2, IRTensor([2, 5, 1]))
    expected_annotation = 'a b c^, ?, a b f^ -> a b f^'
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == expected_annotation, "Annotation mismatch for Gather."
    op = F.Gather(IRTensor([2, 5, 3]), 2, IRTensor([2, 5, 3]))
    expected_annotation = 'a b c^, ?, a b c^ -> a b c^'
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == expected_annotation, "Annotation mismatch for Gather."
    op = F.Gather(IRTensor([2, 5, 3]), 2, IRTensor([2, 4, 3]))
    expected_annotation = 'a b^ c^, ?, a e^ c^ -> a e^ c^'
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == expected_annotation, "Annotation mismatch for Gather."
    op = F.Gather(IRTensor([2, 5, 3]), 2, IRTensor([1, 3, 1]))
    expected_annotation = 'a^ b^ c^, ?, d^ e^ f^ -> d^ e^ f^'
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == expected_annotation, "Annotation mismatch for Gather."
    op = F.Gather(IRTensor([2, 5, 3]), 1, IRTensor([2, 2, 3]))
    expected_annotation = 'a b^ c, ?, a e^ c -> a e^ c'
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == expected_annotation, "Annotation mismatch for Gather."
    op = F.Gather(IRTensor([2, 5, 3]), 0, IRTensor([1, 5, 3]))
    expected_annotation = 'a^ b c, ?, d^ b c -> d^ b c'
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == expected_annotation, "Annotation mismatch for Gather."
    op = F.Gather(IRTensor([2, 3]), 1, IRTensor([2, 1]))
    expected_annotation = 'a b^, ?, a d^ -> a d^'
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == expected_annotation, "Annotation mismatch for Gather."
    op = F.Gather(IRTensor([2, 3]), 1, IRTensor([1, 1]))
    expected_annotation = 'a^ b^, ?, c^ d^ -> c^ d^'
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == expected_annotation, "Annotation mismatch for Gather."
    op = F.Gather(IRTensor([2, 3]), -1, IRTensor([1, 1]))
    expected_annotation = 'a^ b^, ?, c^ d^ -> c^ d^'
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == expected_annotation, "Annotation mismatch for Gather."


def test_Ceil():
    input_tensor = IRTensor([2, 3])
    op = F.Ceil(input_tensor)
    expected_annotation = '* -> *'
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == expected_annotation, "Annotation mismatch for Ceil."
    input_tensor = IRTensor([2, 3, 4])
    op = F.Ceil(input_tensor)
    expected_annotation = '* -> *'
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == expected_annotation, "Annotation mismatch for Ceil."


def test_Sign():
    input_tensor = IRTensor([2, 3])
    op = F.Sign(input_tensor)
    expected_annotation = '* -> *'
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == expected_annotation, "Annotation mismatch for Sign."
    input_tensor = IRTensor([2, 3, 4])
    op = F.Sign(input_tensor)
    expected_annotation = '* -> *'
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == expected_annotation, "Annotation mismatch for Sign."


def test_Unfold():
    input_tensor = IRTensor([2, 3, 32, 32])
    kernel_size = (3, 3)
    stride = (2, 2)
    padding = (1, 1)
    dilation = (1, 1)
    op = F.Unfold(input_tensor, kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'N C 32 32 -> N (C 9) 256'


def test_Sigmoid():
    op = F.Sigmoid(IRTensor([2, 3, 4]))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == '* -> *'


def test_BitwiseOr():
    op = F.BitwiseOr(IRTensor([8, 10]), IRTensor([10]))
    assert op._annos_candidates[0] == 'a b, b -> a b'


def test_TorchAny():
    op = F.TorchAny(IRTensor([10, 10]))
    assert op._annos_candidates[0] == 'a^ b^ -> 1'

    op = F.TorchAny(IRTensor([10, 10]), dim=1)
    assert op._annos_candidates[0] == 'a^ b^ -> a^'

    op = F.TorchAny(IRTensor([10, 10]), dim=1, keepdim=True)
    assert op._annos_candidates[0] == 'a^ b^ -> a^ 1'


def test_L1Loss():
    op = F.L1Loss(IRTensor([8, 10]), IRTensor([8, 10]), reduction='sum')
    assert op._annos_candidates[0] == 'a+ b+, a+ b+ -> 1'

    op = F.L1Loss(IRTensor([8, 10]), IRTensor([8, 10]), reduction='mean')
    assert op._annos_candidates[0] == 'a^ b^, a^ b^ -> 1'


def test_SVD():
    op = F.SVD(IRTensor([3, 4]))
    assert op._annos_candidates[0] == 'a^ b^ -> a^ a^, a^, b^ a^'

    op = F.SVD(IRTensor([4, 3]))
    assert op._annos_candidates[0] == 'a^ b^ -> a^ b^, b^, b^ b^'

    op = F.SVD(IRTensor([4, 3]), False)
    assert op._annos_candidates[0] == 'a^ b^ -> a^ a^, b^, b^ b^'


def test_Diag():
    op = F.Diag(IRTensor([5, 10]), 0)
    assert op._annos_candidates[0] == '5 10 -> 5'

    op = F.Diag(IRTensor([5, 10]), 5)
    assert op._annos_candidates[0] == '5 10 -> 5'

    op = F.Diag(IRTensor([5, 10]), 7)
    assert op._annos_candidates[0] == '5 10 -> 3'

    op = F.Diag(IRTensor([5, 10]), 10)
    assert op._annos_candidates[0] == '5 10 -> 0'

    op = F.Diag(IRTensor([5, 10]), -1)
    assert op._annos_candidates[0] == '5 10 -> 4'


def test_Conv2D():
    op = F.Conv2D(IRTensor([2, 3, 4, 4]), IRTensor([3, 3, 1, 1]))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'n iC+ 4 4, oC iC+ 1 1 -> n oC 4 4'
    op = F.Conv2D(IRTensor([3, 4, 4]), IRTensor([3, 3, 1, 1]))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'iC+ 4 4, oC iC+ 1 1 -> oC 4 4'
    op = F.Conv2D(IRTensor([3, 4, 4]), IRTensor([3, 3, 1, 1]), groups=1, padding="valid")
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'iC+ 4 4, oC iC+ 1 1 -> oC 4 4'
    op = F.Conv2D(input=IRTensor([8, 32, 32]), weight=IRTensor([16, 8, 3, 3]), bias=IRObject(value=16), groups=1, padding="same")
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'iC^ 32 32, oC iC^ 3 3, oC -> oC 32 32'
    op = F.Conv2D(input=IRTensor([8, 32, 32]), weight=IRTensor([16, 4, 3, 3]), bias=IRObject(value=16), groups=2, padding="same")
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == '(g 4) 32 32, (g 8) 4 3 3, (g 8) -> (g 8) 32 32'
    op = F.Conv2D(IRTensor([2, 3, 4, 4]), IRTensor([3, 3, 1, 1]), stride=2)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'n iC+ 4 4, oC iC+ 1 1 -> n oC 2 2'
    op = F.Conv2D(IRTensor([2, 3, 4, 4]), IRTensor([3, 3, 1, 1]), padding=1)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'n iC+ 4 4, oC iC+ 1 1 -> n oC 6 6'
    op = F.Conv2D(IRTensor([2, 3, 4, 4]), IRTensor([3, 3, 1, 1]), padding=IRObject(value=1))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'n iC+ 4 4, oC iC+ 1 1 -> n oC 6 6'
    op = F.Conv2D(IRTensor([2, 3, 4, 4]), IRTensor([3, 3, 1, 1]), dilation=2)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'n iC+ 4 4, oC iC+ 1 1 -> n oC 4 4'
    op = F.Conv2D(IRTensor([2, 3, 4, 4]), IRTensor([3, 3, 1, 1]), groups=1)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'n iC+ 4 4, oC iC+ 1 1 -> n oC 4 4'
    op = F.Conv2D(IRTensor([2, 3, 4, 4]), IRTensor([3, 3, 1, 1]), groups=1, padding="valid")
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'n iC+ 4 4, oC iC+ 1 1 -> n oC 4 4'
    op = F.Conv2D(input=IRTensor([4, 8, 32, 32]), weight=IRTensor([16, 8, 3, 3]), bias=IRObject(value=16), groups=1, padding="same")
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'n iC^ 32 32, oC iC^ 3 3, oC -> n oC 32 32'
    op = F.Conv2D(input=IRTensor([4, 8, 32, 32]), weight=IRTensor([16, 4, 3, 3]), bias=IRObject(value=16), groups=2, padding="same")
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'n (g 4) 32 32, (g 8) 4 3 3, (g 8) -> n (g 8) 32 32'
    op = F.Conv2D(IRTensor([2, 3, 4, 4]), IRTensor([3, 1, 1, 1]), groups=3)
    expected_annotation_for_groups = 'n (g 1) 4 4, (g 1) 1 1 1 -> n (g 1) 4 4'
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == expected_annotation_for_groups
    op = F.Conv2D(IRTensor([2, 3, 4, 4]), IRTensor([3, 3, 1, 1]), bias=IRObject(value=3))
    assert op._annos_candidates[0] == 'n iC^ 4 4, oC iC^ 1 1, oC -> n oC 4 4', "Annotation mismatch."


def test_ConvTranspose2D():
    op = F.ConvTranspose2D(IRTensor([3, 4, 4]), IRTensor([3, 3, 1, 1]))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'iC+ 4 4, iC+ oC 1 1 -> oC 4 4'
    op = F.ConvTranspose2D(IRTensor([3, 4, 4]), IRTensor([3, 3, 1, 1]), bias=IRObject(value=3))
    assert op._annos_candidates[0] == 'iC+ 4 4, iC+ oC 1 1, oC -> oC 4 4', "Annotation mismatch."
    op = F.ConvTranspose2D(IRTensor([3, 4, 4]), IRTensor([3, 3, 1, 1]), padding=IRObject(value=1))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'iC+ 4 4, iC+ oC 1 1 -> oC 2 2'
    op = F.ConvTranspose2D(IRTensor([2, 3, 4, 4]), IRTensor([3, 3, 1, 1]))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'n iC+ 4 4, iC+ oC 1 1 -> n oC 4 4'
    op = F.ConvTranspose2D(IRTensor([2, 3, 4, 4]), IRTensor([3, 3, 1, 1]), stride=2)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'n iC+ 4 4, iC+ oC 1 1 -> n oC 7 7'
    op = F.ConvTranspose2D(IRTensor([2, 3, 4, 4]), IRTensor([3, 3, 1, 1]), padding=1)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'n iC+ 4 4, iC+ oC 1 1 -> n oC 2 2'
    op = F.ConvTranspose2D(IRTensor([2, 3, 4, 4]), IRTensor([3, 3, 1, 1]), padding=IRObject(value=1))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'n iC+ 4 4, iC+ oC 1 1 -> n oC 2 2'
    op = F.ConvTranspose2D(IRTensor([2, 3, 4, 4]), IRTensor([3, 3, 1, 1]), dilation=2)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'n iC+ 4 4, iC+ oC 1 1 -> n oC 4 4'
    op = F.ConvTranspose2D(IRTensor([2, 3, 4, 4]), IRTensor([3, 3, 1, 1]), groups=1)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'n iC+ 4 4, iC+ oC 1 1 -> n oC 4 4'
    op = F.ConvTranspose2D(IRTensor([2, 3, 4, 4]), IRTensor([3, 1, 1, 1]), groups=3)
    expected_annotation_for_groups = 'n (groups group_size^) 4 4, (groups group_size^) oC 1 1 -> n (groups oC) 4 4'
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == expected_annotation_for_groups
    op = F.ConvTranspose2D(IRTensor([2, 3, 4, 4]), IRTensor([3, 3, 1, 1]), bias=IRObject(value=3))
    assert op._annos_candidates[0] == 'n iC+ 4 4, iC+ oC 1 1, oC -> n oC 4 4', "Annotation mismatch."


def test_ConvTranspose1D():
    op = F.ConvTranspose1D(IRTensor([3, 4]), IRTensor([3, 3, 1]))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'iC+ 4, iC+ oC 1 -> oC 4'
    op = F.ConvTranspose1D(IRTensor([3, 4]), IRTensor([3, 3, 1]), groups=3)
    expected_annotation_for_groups = '(groups group_size^) 4, (groups group_size^) oC 1 -> (groups oC) 4'
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == expected_annotation_for_groups
    op = F.ConvTranspose1D(IRTensor([3, 4]), IRTensor([3, 3, 1]), bias=IRObject(value=3))
    assert op._annos_candidates[0] == 'iC+ 4, iC+ oC 1, oC -> oC 4', "Annotation mismatch."
    op = F.ConvTranspose1D(IRTensor([2, 3, 4]), IRTensor([3, 3, 1]))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'n iC+ 4, iC+ oC 1 -> n oC 4'
    op = F.ConvTranspose1D(IRTensor([2, 3, 4]), IRTensor([3, 3, 1]), stride=2)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'n iC+ 4, iC+ oC 1 -> n oC 7'
    op = F.ConvTranspose1D(IRTensor([2, 3, 4]), IRTensor([3, 3, 1]), padding=1)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'n iC+ 4, iC+ oC 1 -> n oC 2'
    op = F.ConvTranspose1D(IRTensor([2, 3, 4]), IRTensor([3, 3, 1]), padding=IRObject(value=1))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'n iC+ 4, iC+ oC 1 -> n oC 2'
    op = F.ConvTranspose1D(IRTensor([2, 3, 4]), IRTensor([3, 3, 1]), dilation=2)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'n iC+ 4, iC+ oC 1 -> n oC 4'
    op = F.ConvTranspose1D(IRTensor([2, 3, 4]), IRTensor([3, 3, 1]), groups=1)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'n iC+ 4, iC+ oC 1 -> n oC 4'
    op = F.ConvTranspose1D(IRTensor([2, 3, 4]), IRTensor([3, 3, 1]), groups=3)
    expected_annotation_for_groups = 'n (groups group_size^) 4, (groups group_size^) oC 1 -> n (groups oC) 4'
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == expected_annotation_for_groups
    op = F.ConvTranspose1D(IRTensor([2, 3, 4]), IRTensor([3, 3, 1]), bias=IRObject(value=3))
    assert op._annos_candidates[0] == 'n iC+ 4, iC+ oC 1, oC -> n oC 4', "Annotation mismatch."


def test_Pad():
    op = F.Pad(IRTensor([3, 3, 4, 2]), pad=(1, 1))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b c 2 -> a b c 4'
    op = F.Pad(IRTensor([3, 3, 4, 2]), pad=(1, 1, 2, 2))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b 4 2 -> a b 8 4'
    op = F.Pad(IRTensor([3, 3, 4, 2]), pad=(0, 1, 2, 1, 3, 3))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a 3 4 2 -> a 9 7 3'
    op = F.Pad(IRTensor([3, 4, 2]), pad=(0, 1, 2, 1, 3, 3))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == '3 4 2 -> 9 7 3'
    op = F.Pad(IRTensor([3, 3, 4, 2]), pad=(o(1), o(1)))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b c 2 -> a b c 4'


def test_Split():
    op = F.Split(IRTensor([3, 3, 4, 2]), split_size_or_sections=1)
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == '3 b c d -> 1 b c d, 1 b c d, 1 b c d'
    op = F.Split(IRTensor([5, 3, 4, 2]), split_size_or_sections=IRObject(value=2, is_constant=False))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == '5 b c d -> 2 b c d, 2 b c d, 1 b c d'
    op = F.Split(IRTensor([7, 3, 4, 2]), split_size_or_sections=IRObject(value=[2, 2, 3], is_constant=False))
    assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == '7 b c d -> 2 b c d, 2 b c d, 3 b c d'


def factors(n):
    return set(reduce(
        list.__add__,
        ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))


def verify_partition(op: IRDimops):
    anno = op.anno
    inputs = torch.randn(op.inputs()[0].shape)
    outputs = inputs.reshape(**IR.try_unwrap(op.kwargs)).clone() \
        if 'reshape' in op.signature \
        else inputs.view(**IR.try_unwrap(op.kwargs)).clone()

    def _get_anno_ids_map(shape_annos: tuple):
        anno_ids = {}
        for idx, shape in enumerate(shape_annos):
            for eidx, edim in enumerate(shape.dims):
                for identifier in edim.identifiers:
                    if identifier[0].isalpha():
                        anno_ids[(idx, eidx)] = identifier
                        break
        return anno_ids

    # (input_idx, input_dim) -> identifier
    input_anno_ids = _get_anno_ids_map(anno.inputs())
    output_anno_ids = _get_anno_ids_map(anno.outputs())
    # assume each identifier is unique, which is true for reshape/view
    # identifier -> (input_idx, input_dim)
    reverse_input_anno_ids = {v: k for k, v in input_anno_ids.items()}
    reverse_output_anno_ids = {v: k for k, v in output_anno_ids.items()}

    transforms = anno.transform_space()
    transform_rules = {}
    for transform_rule in op.transform_rules:
        transform_rules[
            (transform_rule.inputs()[0].dims[0], transform_rule.outputs()[0].dims[0])
        ] = transform_rule.modifier()

    for transform in transforms:
        input_idx, input_dim = transform
        identifier = input_anno_ids[transform]
        output_idx, output_dim = reverse_output_anno_ids[identifier]

        # only one input/one output for reshape/view
        assert input_idx == 0
        assert output_idx == 0

        dim_size = anno.getlen(identifier)
        for factor in factors(dim_size):
            # simulate the partition process
            # 1. chunk input tensor
            input_chunks = torch.chunk(inputs, factor, dim=input_dim)
            # 2. update kwargs
            kwargs = transform_rules[(input_dim, output_dim)](op.kwargs, 0, input_dim, factor, 0)
            kwargs = IR.try_unwrap(kwargs)
            # 3. reshape/view
            reshaped_input_chunks = [chunk.reshape(**kwargs) for chunk in input_chunks] \
                if 'reshape' in op.signature \
                else [chunk.view(**kwargs) for chunk in input_chunks]
            # 4. compare with actual output
            output_chunks = torch.chunk(outputs, factor, dim=output_dim)
            for i in range(factor):
                assert torch.equal(reshaped_input_chunks[i], output_chunks[i])


def test_reshape_view():
    RVF = {F.Reshape: 'shape', F.View: 'size'}
    for f, kwname in RVF.items():
        query = IRTensor([2048, 1, 2, 512])
        op = f(query, IRObject(value=2048), IRObject(value=1), -1, 32)
        assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a 1 b 512 -> a 1 (b 16) 32'
        assert IR.try_unwrap(op.kwargs[kwname]) == (2048, 1, -1, 32)
        assert [type(x) for x in op.kwargs[kwname]] == [IRObject, IRObject, int, int]
        verify_partition(op)

        query = IRTensor([10, 12, 16, 18])
        op = f(query, 10, 2, 3, 32, 18)
        assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a (b 6) 16 c -> a b 3 32 c'
        verify_partition(op)

        query = IRTensor([2, 16, 32, 32])
        op = f(query, 1, 32, 32, 32)
        assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a 16 b c -> 1 (a 16) b c'
        verify_partition(op)

        query = IRTensor([10, 12, 16, 18])
        op = f(query, 10, 24, 8, 18)
        assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a b 16 c -> a (b 2) 8 c'
        verify_partition(op)

        query = IRTensor([10, 12, 16, 18])
        op = f(query, 10, 16, 12, 18)
        assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a 12 16 b -> a 16 12 b'
        verify_partition(op)

        query = IRTensor([2, 1, 1, 16, 32, 1, 1, 32])
        op = f(query, 1, 32, 32, 1, 32)
        assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a 1 1 16 b 1 1 c -> 1 (a 16) b 1 c'
        verify_partition(op)

        query = IRTensor([10, 1, 1, 5, 1, 7, 1, 1])
        op = f(query, 10, 5, 7, 1)
        assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a 1 1 b 1 c 1 1 -> a b c 1'
        verify_partition(op)

        query = IRTensor([10, 1, 1, 5, 1, 7, 1, 1])
        op = f(query, 10, 5, 7)
        assert len(op._annos_candidates) == 1 and op._annos_candidates[0] == 'a 1 1 b 1 c 1 1 -> a b c'
        verify_partition(op)


def test_make_collection():
    # test non-irobject
    l = [1, 2, 3]
    t = (1, 2, 3)
    s = slice(*l)
    r = F.MakeList(l)
    assert r == l
    r = F.MakeTuple(t)
    assert r == t
    r = F.MakeSlice(*l)
    assert r == s

    # test irobject items
    l = [IRObject(value=1), IRFullTensor([2]), 3]
    t = (IRObject(value=1), IRFullTensor([2]), 3)
    s = slice(*l)
    r = F.MakeList(l)
    assert r == l
    r = F.MakeTuple(t)
    assert r == t
    r = F.MakeSlice(*l)
    assert r == s

    # test whole irobject
    l = IRObject(value=[1, 2, 3])
    t = IRObject(value=(1, 2, 3))
    r = F.MakeList(l, signature='builtins.list')
    assert r.output(0).value == l.value
    r = F.MakeTuple(t, signature='builtins.tuple')
    assert r.output(0).value == t.value
    # MakeSlice is not valid.
    # F.MakeSlice(s)


def test_dict_keys_values_items():
    # normal dict
    d = {'a': 1, 'b': 2, 'c': 3}
    r = F.DictKeys(d)
    assert r.output(0).value == tuple(d.keys())
    r = F.DictValues(d)
    assert r.output(0).value == tuple(d.values())
    r = F.DictItems(d)
    assert r.output(0).value == tuple(d.items())

    d = {'a': IRFullTensor([1]), 'b': IRFullTensor([2]), 'c': IRFullTensor([3])}
    r = F.DictKeys(d)
    assert r.output(0).value == tuple(d.keys())
    r = F.DictValues(d)
    # IRFullTensor will be reconstructed, so their ids are different
    assert all(x.shape == y.shape and x != y for x, y in zip(r.output(0), d.values()))
    r = F.DictItems(d)
    # key will never be wrapped with IRObject
    # IRFullTensor will be reconstructed, so their ids are different
    assert all(x[0] == y[0] and x[1].shape == y[1].shape and x[1] != y[1] for x, y in zip(r.output(0), d.items()))
