#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import torch

import pytest

from nnscaler.ir.cten import IRObject, IR
from nnscaler.ir.tensor import IRFullTensor, IRSubTensor
from nnscaler.graph.parser.parser import TensorMetadata, DICT_VALUES_TYPE, DICT_ITEMS_TYPE


@pytest.mark.parametrize('tosub', [True, False])
@pytest.mark.parametrize('requires_grad', [True, False, None])
def test_from_complex(tosub, requires_grad):
    tensor_type = IRSubTensor if tosub else IRFullTensor
    rg = requires_grad
    if rg is None:
        rg = False
    rgt = requires_grad
    if rgt is None:
        rgt = True
    obj = IR.new('n', 1, tosub=tosub, requires_grad=requires_grad)
    assert type(obj) == IRObject and obj.value == 1 and not obj.is_constant and obj.name == 'n'

    obj = IR.new('n', [1, 2], tosub=tosub, requires_grad=requires_grad)
    assert type(obj) == IRObject and obj.value == [1, 2] and not obj.is_constant and obj.name == 'n'

    obj = IR.new('n', {'a': 1, 'b': 2}, tosub=tosub, requires_grad=requires_grad)
    assert type(obj) == IRObject and obj.value == {'a': 1, 'b': 2} and not obj.is_constant and obj.name == 'n'

    obj = IR.new('n', {'a': {'c': [3, 4], 'd': [4, 5]}, 'b': [1,2]}, tosub=tosub, requires_grad=requires_grad)
    assert type(obj) == IRObject and obj.value == {'a': {'c': [3, 4], 'd': [4, 5]}, 'b': [1,2]} and not obj.is_constant and obj.name == 'n'

    t1 = torch.tensor(1.0)
    t2 = torch.tensor([2.0, 3.0], requires_grad=True)

    obj = IR.new('n', t1, tosub=tosub, requires_grad=requires_grad)
    assert type(obj) == tensor_type and id(obj.value) == id(t1) \
        and obj.shape == (1,) and obj.origin_shape == () and obj.dtype == torch.float \
        and obj.requires_grad == rg and not obj.is_constant \
        and obj.name == 'n'

    obj = IR.new('n', [t1, t2, 1], tosub=tosub, requires_grad=requires_grad)
    assert type(obj) == list and len(obj) == 3
    assert type(obj[0]) == tensor_type and id(obj[0].value) == id(t1) \
        and obj[0].shape == (1,) and obj[0].origin_shape == () and obj[0].dtype == torch.float \
        and obj[0].requires_grad == rg and not obj[0].is_constant \
        and obj[0].name == 'n'
    assert type(obj[1]) == tensor_type and id(obj[1].value) == id(t2) \
        and obj[1].shape == (2,) and obj[1].origin_shape == (2,) and obj[1].dtype == torch.float \
        and obj[1].requires_grad == rgt and not obj[1].is_constant \
        and obj[1].name == 'n'
    assert type(obj[2]) == IRObject and obj[2].value == 1 and not obj[2].is_constant and obj[2].name == 'n'

    obj = IR.new('n', {'a': [1, 2, t1], 'b': 2}, tosub=tosub, requires_grad=requires_grad)
    assert type(obj) == dict and len(obj) == 2
    x = obj['a']
    assert type(x) == list and len(x) == 3
    assert type(x[0]) == IRObject and x[0].value == 1 and not x[0].is_constant and x[0].name == 'n'
    assert type(x[1]) == IRObject and x[1].value == 2 and not x[1].is_constant and x[1].name == 'n'
    assert type(x[2]) == tensor_type and id(x[2].value) == id(t1) \
        and x[2].shape == (1,) and x[2].origin_shape == () and x[2].dtype == torch.float \
        and x[2].requires_grad == rg and not x[2].is_constant \
        and x[2].name == 'n'
    y = obj['b']
    assert type(y) == IRObject and y.value == 2 and not y.is_constant and y.name == 'n'

    x = [t1, t2, 1]
    obj = IR.new('n', x, tosub=tosub, tensor_types=(), requires_grad=requires_grad)
    assert type(obj) == IRObject and id(obj.value) == id(x) and not obj.is_constant and obj.name == 'n'

    x_set = {1, t1, t2} # set is not supported
    obj = IR.new('n', x_set, tosub=tosub, requires_grad=requires_grad)
    assert type(obj) == IRObject and id(obj.value) == id(x_set) and not obj.is_constant and obj.name == 'n'

    obj = IR.new('n', [t1, [1, 2, {'a': 3}], (4, 5, {'b': 6, 'c': t2})], tosub=tosub, requires_grad=requires_grad)
    assert type(obj) == list and len(obj) == 3
    assert type(obj[0]) == tensor_type and id(obj[0].value) == id(t1) \
        and obj[0].shape == (1,) and obj[0].origin_shape == () and obj[0].dtype == torch.float \
        and obj[0].requires_grad == rg and not obj[0].is_constant \
        and obj[0].name == 'n'
    assert type(obj[1]) == IRObject and obj[1].value == [1, 2, {'a': 3}] and not obj[1].is_constant and obj[1].name == 'n'
    x = obj[2]
    assert type(x) == tuple and len(x) == 3
    assert type(x[0]) == IRObject and x[0].value == 4 and not x[0].is_constant and x[0].name == 'n'
    assert type(x[1]) == IRObject and x[1].value == 5 and not x[1].is_constant and x[1].name == 'n'
    y = x[2]
    assert type(y) == dict and len(y) == 2
    assert type(y['b']) == IRObject and y['b'].value == 6 and not y['b'].is_constant and y['b'].name == 'n'
    assert type(y['c']) == tensor_type and id(y['c'].value) == id(t2) \
        and y['c'].shape == (2,) and y['c'].origin_shape == (2,) and y['c'].dtype == torch.float \
        and y['c'].requires_grad == rgt and not y['c'].is_constant \
        and y['c'].name == 'n'

    obj_item = IRObject('obj_item', value=1, is_constant=False)
    obj_tensor_item = IRFullTensor(t1.shape, 'obj_tensor_item')

    obj = IR.new('n', slice(obj_item, 2, obj_item))
    assert type(obj) == IRObject and obj.value == slice(1, 2, 1)

    obj = IR.new('n', [obj_item, 2], tosub=tosub, requires_grad=requires_grad)
    assert type(obj) == IRObject and obj.value == [1, 2]

    obj = IR.new('n', [obj_item, 2, obj_tensor_item], tosub=tosub, requires_grad=requires_grad)
    assert type(obj) == list and len(obj) == 3
    assert obj[0].value == 1 and obj[0].tid != obj_item.tid
    assert obj[1].value == 2
    assert type(obj[2]) == tensor_type and obj[2].parent.tid != obj_tensor_item.tid

    obj = IR.new('n', [t1, obj_item], tosub=tosub, requires_grad=requires_grad)
    assert type(obj) == list and len(obj) == 2
    assert type(obj[0]) == tensor_type and id(obj[0].value) == id(t1)
    assert type(obj[1]) == IRObject and obj[1].value == 1 and obj[1].tid != obj_item.tid

    obj = IR.new('n', [t1, obj_item, obj_tensor_item], tosub=tosub, requires_grad=requires_grad)
    assert type(obj) == list and len(obj) == 3
    assert type(obj[0]) == tensor_type and id(obj[0].value) == id(t1)
    assert type(obj[1]) == IRObject and obj[1].value == 1 and obj[1].tid != obj_item.tid
    assert type(obj[2]) == tensor_type and obj[2].parent.tid != obj_tensor_item.tid

    t1 = TensorMetadata(shape=(), dtype=torch.float, requires_grad=False,
        stride=None, memory_format=None, is_quantized=None, qparams=None)
    t2 = TensorMetadata(shape=(2,), dtype=torch.float, requires_grad=True,
        stride=None, memory_format=None, is_quantized=None, qparams=None)

    obj = IR.new('n', {'a': t1, 'b': t2}.values(),
        tensor_types=(TensorMetadata,),
        tosub=tosub, requires_grad=requires_grad
    )
    assert type(obj) == tuple and len(obj) == 2
    x = obj[0]
    assert type(x) == tensor_type and id(x.value) == id(t1) \
        and x.shape == (1,) and x.origin_shape == () and x.dtype == torch.float \
        and x.requires_grad == rg and not x.is_constant \
        and x.name == 'n'
    y = obj[1]
    assert type(y) == tensor_type and id(y.value) == id(t2) \
        and y.shape == (2,) and y.origin_shape == (2,) and y.dtype == torch.float \
        and y.requires_grad == rgt and not y.is_constant \
        and y.name == 'n'

    obj = IR.new('n', {'a': t1, 'b': t2}.items(),
        tensor_types=(TensorMetadata,),
        tosub=tosub, requires_grad=requires_grad
    )
    assert type(obj) == tuple and len(obj) == 2
    x = obj[0]
    assert x[0] == 'a'
    x = x[1]
    assert type(x) == tensor_type and id(x.value) == id(t1) \
        and x.shape == (1,) and x.origin_shape == () and x.dtype == torch.float \
        and x.requires_grad == rg and not x.is_constant \
        and x.name == 'n'
    y = obj[1]
    assert y[0] == 'b'
    y = y[1]
    assert type(y) == tensor_type and id(y.value) == id(t2) \
        and y.shape == (2,) and y.origin_shape == (2,) and y.dtype == torch.float \
        and y.requires_grad == rgt and not y.is_constant \
        and y.name == 'n'
