#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import ast
from textwrap import dedent
import sys

import pytest

from nnscaler.graph.tracer.operator_patcher import (
    OperatorTransformer,
    SuperTransformer,
    ProxyCallTransformer,
    transform
)


@pytest.mark.skipif(sys.version_info < (3, 9), reason='ast.unparse is not available in python3.8')
def test_ifexpr_transfomer():
    # x = ast.parse('nnscaler.runtime.ifexpr(1, 2, 3)')

    tree = ast.parse(dedent('''
        x = 0.1 if self.training else 0.2
    ''').strip())
    transformers = [OperatorTransformer()]
    modified, new_ast  = transform(tree, transformers)
    assert modified
    assert '\n'.join(line for line in ast.unparse(new_ast).split('\n') if line.strip()) == dedent('''
        x = nnscaler.runtime.function.ifexpr(self.training, 0.1, 0.2)
        ''').strip()

    tree = ast.parse(dedent('''
        x = x.p if self.training else 0.2 + 0.3
    ''').strip())
    transformers = [OperatorTransformer()]
    modified, new_ast  = transform(tree, transformers)
    assert modified
    assert '\n'.join(line for line in ast.unparse(new_ast).split('\n') if line.strip()) == dedent('''
        x = nnscaler.runtime.function.ifexpr(self.training, x.p, 0.2 + 0.3)
        ''').strip()

    tree = ast.parse(dedent('''
        x = x.p if self.training else 0.2 + f(0.3)
    ''').strip())
    transformers = [OperatorTransformer()]
    modified, new_ast  = transform(tree, transformers)
    assert not modified

    tree = ast.parse(dedent('''
        x = f(x) if self.training else 0.2
    ''').strip())
    transformers = [OperatorTransformer()]
    modified, new_ast  = transform(tree, transformers)
    assert not modified

    tree = ast.parse(dedent('''
        x = 0.1 if self.training else f(0.2)
    ''').strip())
    transformers = [OperatorTransformer()]
    modified, new_ast  = transform(tree, transformers)
    assert not modified

    tree = ast.parse(dedent('''
        x = f(0.1) if self.training else f(0.2)
    ''').strip())
    transformers = [OperatorTransformer()]
    modified, new_ast  = transform(tree, transformers)
    assert not modified


@pytest.mark.skipif(sys.version_info < (3, 9), reason='ast.unparse is not available in python3.8')
def test_op_transfomer():
    tree = ast.parse(dedent('''
        x = True
        y = not x
        y1 = x and not y
        y2 = x is None
        y3 = x is not None
        y4 = x in ()
        y5 = x not in ()
        if x and not y:
            pass
        y6 = y1 is None or y2 is not None
        y7 = y1 if y2 is None else y3
        while x and not y:
            pass
    ''').strip())
    transformers = [OperatorTransformer()]
    modified, new_ast  = transform(tree, transformers)
    assert modified
    assert '\n'.join(line for line in ast.unparse(new_ast).split('\n') if line.strip()) == dedent('''
        x = True
        y = not_(x)
        y1 = x and not_(y)
        y2 = is_(x, None)
        y3 = is_not(x, None)
        y4 = contains((), x)
        y5 = not_(contains((), x))
        if x and not_(y):
            pass
        y6 = is_(y1, None) or is_not(y2, None)
        y7 = y1 if is_(y2, None) else y3
        while x and not_(y):
            pass
        ''').strip()


@pytest.mark.skipif(sys.version_info < (3, 9), reason='ast.unparse is not available in python3.8')
def test_super_transform():
    tree = ast.parse(dedent('''
        class A:
            def __init__(self) -> None:
                super().__init__()
            def f(self):
                super(A, self).f()
            @staticmethod
            def g():
                pass
            @classmthod
            def h(cls):
                pass
            def i(self):
                pass
        ''').strip())

    transfomers = [SuperTransformer()]
    modified, new_ast  = transform(tree, transfomers)
    assert modified
    assert '\n'.join(line for line in ast.unparse(new_ast).split('\n') if line.strip()) == dedent('''
        class A:
            def __init__(self) -> None:
                super(self.__class__, self).__init__()
            def f(self):
                super(A, self).f()
            @staticmethod
            def g():
                pass
            @classmthod
            def h(cls):
                pass
            def i(self):
                pass
        ''').strip()


@pytest.mark.skipif(sys.version_info < (3, 9), reason='ast.unparse is not available in python3.8')
def test_proxy_call_transform():
    # the `(x+y)(a, b)` statement below just demonstrates
    # AST doesn't care about the real meaning of the expression.
    # It looks like a function call, so it will be treated as function call in AST.
    # And for us, we also patch it just like a function call.
    tree = ast.parse(dedent('''
        def f(func_name, type: int, /, *args, **kwargs):
            return func_name(type, *args, **kwargs)
        def g():
            return (x + y)(a, b)
        class A:
            def f(self) -> None:
                super().f()
        ''').strip())

    transfomers = [ProxyCallTransformer('patched_run')]
    modified, new_ast  = transform(tree, transfomers)
    assert modified
    assert '\n'.join(line for line in ast.unparse(new_ast).split('\n') if line.strip()) == dedent('''
        def f(func_name, type: int, /, *args, **kwargs):
            return patched_run(func_name, type, *args, **kwargs)
        def g():
            return patched_run(x + y, a, b)
        class A:
            def f(self) -> None:
                patched_run(patched_run(super).f)
        ''').strip()


@pytest.mark.skipif(sys.version_info < (3, 9), reason='ast.unparse is not available in python3.8')
def test_transform_combine():
    tree = ast.parse(dedent('''
        x = not True
        def f(func_name, type: int, /, *args, **kwargs):
            return func_name(type, *args, **kwargs)
        class A:
            def __init__(self) -> None:
                super().__init__()
        ''').strip())

    transfomers = [OperatorTransformer(), SuperTransformer(), ProxyCallTransformer('patched_run', ['super'])]
    modified, new_ast  = transform(tree, transfomers)
    assert modified
    assert '\n'.join(line for line in ast.unparse(new_ast).split('\n') if line.strip()) == dedent('''
        x = patched_run(not_, True)
        def f(func_name, type: int, /, *args, **kwargs):
            return patched_run(func_name, type, *args, **kwargs)
        class A:
            def __init__(self) -> None:
                patched_run(super(self.__class__, self).__init__)
        ''').strip()
