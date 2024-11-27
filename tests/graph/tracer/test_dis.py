#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import sys

import pytest

from nnscaler.graph.tracer.frame_utils import get_last_instruction, get_instructions


class A:
    def __init__(self) -> None:
        self.caller_inst = None
        self.len_caller_inst = None
    def __iter__(self):
        self.caller_inst = get_last_instruction()
        return iter([1, 2, 3])
    def __len__(self):
        self.len_caller_inst = get_last_instruction()
        return 3


class B:
    def __init__(self) -> None:
        self.value = {'1':2, '3':4, '5':6}
        self.caller_inst = None
        self.getitem_count = 0
    def __iter__(self):
        return iter(self.value)
    def __getitem__(self, key):
        self.getitem_count += 1
        return self.value[key]
    def __len__(self):
        return len(self.value)
    def keys(self):
        self.caller_inst = get_last_instruction()
        return self.value.keys()
    def values(self):
        return self.value.values()


class C:
    def __init__(self) -> None:
        self.caller_inst = None
    def __bool__(self):
        self.caller_inst = get_last_instruction()
        return True


def func0(*args, **kwargs):
    pass


def test_for():
    a = A()
    for _ in a:
        break
    assert a.caller_inst.opname == 'GET_ITER'
    assert a.len_caller_inst is None


def test_single_starargs():
    a = A()
    func0(*a)
    assert a.caller_inst.opname == 'CALL_FUNCTION_EX'
    assert a.len_caller_inst.opname == 'CALL_FUNCTION_EX'


@pytest.mark.skipif(sys.version_info < (3, 9), reason='behavior is different in python3.8')
def test_multi_starargs():
    # in <= python 3.8
    # the opname will be BUILD_TUPLE_UNPACK_WITH_CALL
    # in >= python 3.9
    # the opname will be LIST_EXTEND
    a = A()
    func0(*[1,2], *a)
    assert a.caller_inst.opname == 'LIST_EXTEND'
    assert a.len_caller_inst.opname == 'LIST_EXTEND'

    a = A()
    func0(*a, *[1,2])
    assert a.caller_inst.opname == 'LIST_EXTEND'
    assert a.len_caller_inst.opname == 'LIST_EXTEND'


@pytest.mark.skipif(sys.version_info < (3, 9), reason='behavior is different in python3.8')
def test_normal_item_with_starargs():
    # in <= python 3.8
    # the opname will be BUILD_LIST_UNPACK
    # in >= python 3.9
    # the opname will be LIST_EXTEND
    a = A()
    [1,2, *a]
    assert a.caller_inst.opname == 'LIST_EXTEND'
    assert a.len_caller_inst.opname == 'LIST_EXTEND'


@pytest.mark.skipif(sys.version_info < (3, 9), reason='behavior is different in python3.8')
def test_normal_item_with_starargs2():
    # in <= python 3.8
    # the opname will be BUILD_TUPLE_UNPACK
    # in >= python 3.9
    # the opname will be LIST_EXTEND
    a = A()
    (1,2, *a)
    assert a.caller_inst.opname == 'LIST_EXTEND'
    assert a.len_caller_inst.opname == 'LIST_EXTEND'


@pytest.mark.skipif(sys.version_info < (3, 9), reason='behavior is different in python3.8')
def test_extend():
    a = A()
    [1,2].extend(a)
    # in <= python 3.10, opname is CALL_METHOD
    # in >= python 3.11, opname is CALL
    if sys.version_info.minor <= 10:
        assert a.caller_inst.opname == 'CALL_METHOD'
        assert a.len_caller_inst.opname == 'CALL_METHOD'
    else:
        assert a.caller_inst.opname == 'CALL'
        assert a.len_caller_inst.opname == 'CALL'

    [1, *a]
    assert a.caller_inst.opname == 'LIST_EXTEND' # BUILD_LIST_UNPACK in python 3.8
    assert a.len_caller_inst.opname == 'LIST_EXTEND'

    (1, *a)
    assert a.caller_inst.opname == 'LIST_EXTEND' # BUILD_TUPLE_UNPACK in python 3.8
    assert a.len_caller_inst.opname == 'LIST_EXTEND'


def test_unpack():
    a = A()
    x, y, z = a
    assert a.caller_inst.opname == 'UNPACK_SEQUENCE'
    assert a.len_caller_inst is None


def test_dict_keys1(mocker):
    b = B()
    mock1 = mocker.patch.object(b, '__iter__')
    mock2 = mocker.patch.object(b, '__len__')
    mock3 = mocker.patch.object(b, 'keys', side_effect=b.keys)
    mock4 = mocker.patch.object(b, 'values')

    func0(**b)
    assert mock1.call_count == 0
    assert mock2.call_count == 0
    assert mock3.call_count == 1
    assert mock4.call_count == 0
    assert b.getitem_count == 3  # 3 times of __getitem__


@pytest.mark.skipif(sys.version_info < (3, 9), reason='behavior is different in python3.8')
def test_dict_key2():
    b = B()
    func0(**b)
    assert b.caller_inst.opname == 'DICT_MERGE' # CALL_FUNCTION_EX in python 3.8

    b = B()
    func0(**b, **{'a': 1})
    assert b.caller_inst.opname == 'DICT_MERGE'  # BUILD_MAP_UNPACK_WITH_CALL in python 3.8


@pytest.mark.skipif(sys.version_info < (3, 9), reason='behavior is different in python3.8')
def test_dict_key3():
    b = B()
    {'a': 1, **b}
    assert b.caller_inst.opname == 'DICT_UPDATE'  # BUILD_MAP_UNPACK in python 3.8
    b.caller_inst = None

    {**b, **{'a': 1}}
    assert b.caller_inst.opname == 'DICT_UPDATE'  # BUILD_MAP_UNPACK in python 3.8


def test_bool():
    c0 = 1
    c1 = 0
    c = C()
    not c  # UNARY_NOT
    assert c.caller_inst.opname == 'UNARY_NOT'
    c.caller_inst = None

    x = {c: c}
    bool(x[c])  # CALL_FUNCTION
    # in <= python 3.10, opname is CALL_FUNCTION
    # in >= python 3.11, opname is CALL
    if sys.version_info.minor <= 10:
        assert c.caller_inst.opname == 'CALL_FUNCTION'
    else:
        assert c.caller_inst.opname == 'CALL'

    c and 1 # JUMP_IF_FALSE_OR_POP
    assert c.caller_inst.opname == 'JUMP_IF_FALSE_OR_POP'
    c.caller_inst = None

    c or 1  # JUMP_IF_TRUE_OR_POP
    assert c.caller_inst.opname == 'JUMP_IF_TRUE_OR_POP'
    c.caller_inst = None

    bool(c)  # CALL_FUNCTION
    # in <= python 3.10, opname is CALL_FUNCTION
    # in >= python 3.11, opname is CALL
    if sys.version_info.minor <= 10:
        assert c.caller_inst.opname == 'CALL_FUNCTION'
    else:
        assert c.caller_inst.opname == 'CALL'
    c.caller_inst = None

    if c:
        pass
    if sys.version_info.minor != 11:
        assert c.caller_inst.opname == 'POP_JUMP_IF_FALSE'
    else:
        assert c.caller_inst.opname == 'POP_JUMP_FORWARD_IF_FALSE'
    c.caller_inst = None

    if not c:  # POP_JUMP_IF_TRUE
        pass
    if sys.version_info.minor != 11:
        assert c.caller_inst.opname == 'POP_JUMP_IF_TRUE'
    else:
        assert c.caller_inst.opname == 'POP_JUMP_FORWARD_IF_TRUE'
    c.caller_inst = None

    if bool(c):  # CALL_FUNCTION
        pass
    # in <= python 3.10, opname is CALL_FUNCTION
    # in >= python 3.11, opname is CALL
    if sys.version_info.minor <= 10:
        assert c.caller_inst.opname == 'CALL_FUNCTION'
    else:
        assert c.caller_inst.opname == 'CALL'
    c.caller_inst = None

    x = 1 if c else 0  # POP_JUMP_IF_FALSE
    if sys.version_info.minor != 11:
        assert c.caller_inst.opname == 'POP_JUMP_IF_FALSE'
    else:
        assert c.caller_inst.opname == 'POP_JUMP_FORWARD_IF_FALSE'
    c.caller_inst = None
