#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from dataclasses import asdict, dataclass, field
from typing import List, Optional, Tuple, Dict, Any, Union
import sys

import pytest

from nnscaler.cli.arg_parser import (
    _KeyNotFoundError, parse_args, deserialize_dataclass, _fix_type,
    resolve_args, merge_args
)


def test_parse_args():
    assert parse_args(['--a-good=1', '--b', '2', '--c.d=3', '--c.e', '4', '--f.g.h=5']) == {
        'a_good': '1',
        'b': '2',
        'c': {'d': '3', 'e': '4'},
        'f': {'g': {'h': '5'}}
    }
    parse_args(['--a=1', '--b', '--c.d=3', '--c.e', '4', '--f.g.h=5']) == {
        'a': '1',
        'b': None,
        'c': {'d': '3', 'e': '4'},
        'f': {'g': {'h': '5'}}
    }

    parse_args(['--a=1', '--b', '[1,2,3,4]']) == {
        'a': '1',
        'b': [1,2,3,4],
    }

    with pytest.raises(ValueError):
        parse_args(['--a=1', 'e', '--b', '--c.d=3', '--c.e', '4', '--f.g.h=5'])


def test_fix_type():
    assert _fix_type(int) == int
    assert _fix_type(None) == None
    assert _fix_type(Any) == None
    assert _fix_type(Optional[bool]) == bool
    assert _fix_type(Union[bool, None]) == bool
    assert _fix_type(List[str], False) == List[str]
    assert _fix_type(Optional[List[str]], False) == List[str]

    with pytest.raises(ValueError):
        _fix_type(List[str], True)

    assert _fix_type(Union[bool, int]) == None
    assert _fix_type(Union[bool, int, None]) == None


@pytest.mark.skipif(sys.version_info < (3, 10), reason='| is not available as union type for python < 3.10')
def test_fix_type2():
    assert _fix_type(bool|None) == bool
    assert _fix_type(list[str], False) == list[str]
    assert _fix_type(Optional[list[str]], False) == list[str]
    assert _fix_type(list[str]|None, False) == list[str]

    with pytest.raises(ValueError):
        _fix_type(list[str], True)

    assert _fix_type(bool|int) == None
    assert _fix_type(bool|int|None) == None


@dataclass
class GConfig:
    h: int


def test_deserialize():
    @dataclass
    class C:
        d: int
        e: int

    @dataclass
    class G:
        h: int

    @dataclass
    class F:
        g: G

    @dataclass
    class A:
        a: int
        b: bool
        c: C
        f: F
        h: Tuple[int, ...] = None
        g: List[str] = None
        k: List[int] = None
        w: Dict[str, int] = None
        v: Dict[str, int] = None
        x: Dict[str, Any] = None
        y: List[F] = None
        z: Dict[str, Any] = None

    x = parse_args(['--a=1', '--b', '--c.d=3', '--c.e', '4', '--f.g.h=5', '--v.a=10', '--v.b=20', '--k=[10,12]'])
    y = deserialize_dataclass(x, A)
    assert y == A(a=1, b=True, c=C(d=3, e=4), f=F(g=G(h=5)), k=[10, 12], v={'a': 10, 'b': 20})

    x = parse_args(['--a=1', '--b', 'False', '--c.d=3', '--c.e', '4', '--f.g.h=5', '--v.a=10', '--v.b=20', '--k=[10,12]'])
    y = deserialize_dataclass(x, A)
    assert y == A(a=1, b=False, c=C(d=3, e=4), f=F(g=G(h=5)), k=[10, 12], v={'a': 10, 'b': 20})

    x = parse_args(['--a=1', '--b', 'False', '--c.d=3', '--c.e', '4', '--f.g.unknown=5', '--v.a=10', '--v.b=20', '--k=[10,12]'])
    with pytest.raises(ValueError):
        y = deserialize_dataclass(x, A)

    x = parse_args(['--unknowna=1', '--b', 'False', '--c.d=3', '--c.e', '4', '--f.g.h=5', '--v.a=10', '--v.b=20', '--k=[10,12]'])
    with pytest.raises(ValueError):
        y = deserialize_dataclass(x, A)

    x = parse_args(['--a=1', '--b', '0', '--c.d=3', '--c.e', '4', '--f.g.h=5',
                    '--v.a=10', '--v.b=20',
                    '--z.__type=tests.cli.test_arg_parser.GConfig',
                    '--z.h=6', '--z.y=hello',
                    '--z.x=True',
                    '--z.array=[1,2,3,4,5]',
                    '--z.badarry=[1,b]',
                    '--z.dict={"a": 1, "b": 2}',
                    '--z.baddict={a:1,b:2}',
                    '--z.nest_dict.__type=tests.cli.test_arg_parser.GConfig',
                    '--z.nest_dict.h=7',
    ])
    y = deserialize_dataclass(x, A)
    assert y == A(a=1, b=False, c=C(d=3, e=4), f=F(g=G(h=5)), v={'a': 10, 'b': 20},
                  z={
                      'h': 6,
                      '__type': 'tests.cli.test_arg_parser.GConfig',
                      'y': 'hello',
                      'x': True,
                      'array': [1, 2, 3, 4, 5],
                      'badarry': '[1,b]',
                      'dict': {'a': 1, 'b': 2},
                      'baddict': '{a:1,b:2}',
                      'nest_dict': {
                            'h': 7,
                            '__type': 'tests.cli.test_arg_parser.GConfig'
                      }
                }
    )
    assert deserialize_dataclass(asdict(y), A) == y


def test_deserialize_list():
    @dataclass
    class A:
        a: List[int] = field(default_factory=list)
        b: List[GConfig] = field(default_factory=list)
        c: Tuple[int, ...] = None


    x = parse_args(['--a.0=1', '--a.1=2', '--b.0.h=3', '--b.1.h=4', '--c.1=4'])
    y = deserialize_dataclass(x, A)
    assert y == A(a=[1, 2], b=[GConfig(h=3), GConfig(h=4)], c=(None, 4))


def test_deserialize_union():
    @dataclass
    class A:
        p: Union[str, Dict[str, str], None] = None

    x = parse_args(['--p=hello'])
    y = deserialize_dataclass(x, A)
    assert y.p == 'hello'

    x = parse_args(['--p.a=a', '--p.b=b'])
    y = deserialize_dataclass(x, A)
    assert y.p == {'a': 'a', 'b': 'b'}

    x = parse_args(['--p.a=1', '--p.b=b'])
    y = deserialize_dataclass(x, A)
    assert y.p == {'a': 1, 'b': 'b'}  # Dict[str, str] is ignored. so '1' will be converted to int


def test_deserialize_value_type():
    @dataclass
    class A:
        p: Any = None

    x = parse_args(['--p.__value_type=int', '--p.value=1'])
    y = deserialize_dataclass(x, A)
    assert y.p == {'__value_type': 'int', 'value': '1'}

    x = parse_args(['--p.value=1'])
    y = deserialize_dataclass(x, A)
    assert y.p == {'value': 1}


def test_deserialize_union_type():
    # union annotation is actually ignored.
    # Here the correct result depends on `_guess_deserialize_object` function.
    @dataclass
    class A:
        p: Union[int, str] = None

    x = parse_args(['--p.value=1'])
    y = deserialize_dataclass(x, A)
    assert y.p == {'value': 1}

    x = parse_args(['--p.value=auto'])
    y = deserialize_dataclass(x, A)
    assert y.p == {'value': 'auto'}


def test_merge_list():
    @dataclass
    class A:
        a: List[int] = field(default_factory=list)
        b: List[GConfig] = field(default_factory=list)

    x = parse_args(['--a.0=1', '--a.1=2', '--b.1.h=4'])
    y = deserialize_dataclass(x, A)
    assert y == A(a=[1, 2], b=[None, GConfig(h=4)])
    z = asdict(y)
    import copy
    z0 = copy.deepcopy(z)
    z1 = copy.deepcopy(z)
    merge_args(z, ['--a.2=3','--b.0.h=3', '--b.2.h=5'])
    assert z == {
        'a': [1, 2, '3'],
        'b': [{'h': '3'}, {'h': 4}, {'h': '5'}],
    }
    y = deserialize_dataclass(z, A)
    assert y == A(a=[1, 2, 3], b=[GConfig(h=3), GConfig(h=4), GConfig(h=5)])

    merge_args(z0, ['--a.3=4', '--b=6'])
    assert z0 == {
        'a': [1, 2, None, '4'],
        'b': '6',
    }

    merge_args(z1, ['--b.1.g=5'])
    assert z1 == {
        'a': [1, 2],
        'b': [None, {'h': 4, 'g': '5'}],
    }


def test_remove():
    @dataclass
    class A:
        a: List[int] = field(default_factory=list)
        b: List[GConfig] = field(default_factory=list)

    x = parse_args(['--a.0=1', '--a.1=2', '--b.1.h=4'])
    y = deserialize_dataclass(x, A)
    assert y == A(a=[1, 2], b=[None, GConfig(h=4)])
    z = asdict(y)
    import copy
    z0 = copy.deepcopy(z)
    z1 = copy.deepcopy(z)
    z2 = copy.deepcopy(z)
    z3 = copy.deepcopy(z)
    z4 = copy.deepcopy(z)
    merge_args(z, ['--a.2=3','--b.0.h=3', '--b.1.h!', '--b.2.h!'])
    assert z == {
        'a': [1, 2, '3'],
        'b': [{'h': '3'}, {}, {}],
    }
    merge_args(z0, ['--a.2=3','--b!', '--b.2.h!'])
    assert z0 == {
        'a': [1, 2, '3'],
        'b': {'2': {}}
    }
    merge_args(z1, ['--a.2=3','--b!', '--b.2.h=3'])
    assert z1 == {
        'a': [1, 2, '3'],
        'b': {'2': {'h': '3'}}
    }
    y = deserialize_dataclass(z1, A)
    assert y == A(a=[1, 2, 3], b=[None, None, GConfig(h=3)])

    merge_args(z2, ['--a.2=3','--b.1.h!', '--b.1.h=3'])
    assert z2 == {
        'a': [1, 2, '3'],
        'b': [None, {'h': '3'}],
    }

    merge_args(z3, ['--a.2=3','--b.0.h=3', '--b.1.h!', '--b.2.h=3', '--b.2.h!', '--b.2.g=3'])
    assert z3 == {
        'a': [1, 2, '3'],
        'b': [{'h': '3'}, {}, {'g': '3'}],
    }

    merge_args(z4, ['--a.2=3','--b.1!', '--b.2!', '--b.3.h=3', '--b.4!'])
    assert z4 == {
        'a': [1, 2, '3'],
        'b': [None, None, None, {'h': '3'}],
    }


def test_resolve_args():
    import os
    os.environ['TEST_ENV_VAR_test_resolve_args'] = '20'
    data = {
        'a': 1,
        'b': True,
        'c': {'d': 3, 'e': 4},
        'f': {'g': {'h': 5}},
        'v': {'a': 10, 'b': '$(TEST_ENV_VAR_test_resolve_args)'},
        'x': {'y': '$(c.d)', 'z': '$(k.2.y)'},
        'k': ['$(f.g.h)', {'x': 20}, {'y': '$(c.e)'}],
        'l': '${c}',
    }
    resolve_args(data)
    assert data == {
        'a': 1,
        'b': True,
        'c': {'d': 3, 'e': 4},
        'f': {'g': {'h': 5}},
        'v': {'a': 10, 'b': '20'},
        'x': {'y': 3, 'z': 4},
        'k': [5, {'x': 20}, {'y': 4}],
        'l': {'d': 3, 'e': 4},
    }
    assert id(data['l']) != id(data['c'])


def test_resolve_args2():
    data = {
        'a': 1,
        'b': '$(k)',
        'c': {'d': 3, 'e': 4},
        'f': {'g': {'h': 5}},
        'v': {'a': 10, 'b': 20},
        'x': {'y': '$(c.d)', 'z': '$(k.2.y)'},
        'k': ['$(f.g.h)', {'x': 20}, {'y': '$(c.e)'}],
        'm': '${k.0}1',
        'n': '${m}2$(x.y)',
        'o': r'$\{k.0}1$(x.y)2$\(x.y)',
    }
    resolve_args(data)
    assert data == {
        'a': 1,
        'b': [5, {'x': 20}, {'y': 4}],
        'c': {'d': 3, 'e': 4},
        'f': {'g': {'h': 5}},
        'v': {'a': 10, 'b': 20},
        'x': {'y': 3, 'z': 4},
        'k': [5, {'x': 20}, {'y': 4}],
        'm': '51',
        'n': '5123',
        'o': '${k.0}132$(x.y)',
    }


def test_circular_resolve_args():
    data = {
        'a': 1,
        'b': True,
        'c': {'d': 3, 'e': '$(k.2.y)'},
        'f': {'g': {'h': 5}},
        'v': {'a': 10, 'b': 20},
        'x': {'y': '$(c.d)', 'z': '$(k.2.y)'},
        'k': [10, {'x': 20}, {'y': '$(c.e)'}],
    }
    with pytest.raises(ValueError):
        resolve_args(data)

    data = {
        'a': 1,
        'b': True,
        'c': {'d': 3, 'e': '$(c.e)'},
        'f': {'g': {'h': 5}},
        'v': {'a': 10, 'b': 20},
    }
    with pytest.raises(ValueError):
        resolve_args(data)


def test_missing_resolve_args():
    data = {
        'a': 1,
        'b': True,
        'c': {'d': 3, 'e': '$(k.2.y)'},
    }
    with pytest.raises(_KeyNotFoundError):
        resolve_args(data)
