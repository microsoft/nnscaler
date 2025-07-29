#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

# CREDITS: This implementation is inspired by PyTorch fx symbolic trace: https://github.com/pytorch/pytorch/blob/main/torch/fx/_symbolic_trace.py

from __future__ import annotations

import dis
import logging
import inspect

from typing import List, Optional, Iterable, Any, Union

import torch
from torch.fx._compatibility import compatibility
from torch.fx.graph import magic_methods, reflectable_magic_methods
from torch.fx.node import Node
from torch.fx.proxy import Proxy
from torch.overrides import is_tensor_method_or_property

from . import concrete_tracer as et
from . import pytree_utils, orig_func, wrap_utils, trace_strategy
from .frame_utils import get_frame_record, get_instructions

_logger = logging.getLogger(__name__)

@compatibility(is_backward_compatible=True)
class ConcreteProxy(Proxy):
    """
    `ConcreteProxy` is a wrapped proxy carried the real intermediate value.
    We can use it to trace a more compatible model, and pass the branches.
    """

    # some jump ops have not find practical examples, add them because they are in python doc,
    # TODO: after finding specific cases, need to add them to the unit tests.
    jump_opnames = (
        'JUMP_IF_NOT_EXC_MATCH', # <= python 3.10
        'JUMP_IF_FALSE_OR_POP',  # <= python 3.11
        'JUMP_IF_TRUE_OR_POP',  # <= python 3.11
        'POP_JUMP_IF_FALSE',  # != python 3.11
        'POP_JUMP_IF_TRUE',  # != python 3.11
        'POP_JUMP_FORWARD_IF_FALSE',  # == python 3.11
        'POP_JUMP_FORWARD_IF_TRUE',  # == python 3.11
        'POP_JUMP_FORWARD_IF_NOT_NONE',  # == python 3.11, not included in unit test
        'POP_JUMP_FORWARD_IF_NONE',  # == python 3.11, not included in unit test
        'POP_JUMP_IF_NOT_NONE',  # >= python 3.12, not included in unit test
        'POP_JUMP_IF_NONE',  # >= python 3.12, not included in unit test
    )
    jump_opcodes = orig_func.tuple(dis.opmap[name] for name in jump_opnames if name in dis.opmap)
    op_compare = dis.opmap['COMPARE_OP']
    op_extended_arg = dis.opmap['EXTENDED_ARG']
    op_call_ex = dis.opmap['CALL_FUNCTION_EX']
    op_not = dis.opmap['UNARY_NOT']
    op_unpack_sequence = dis.opmap['UNPACK_SEQUENCE']
    op_dict_merge = dis.opmap.get('DICT_MERGE', None)  # DICT_MERGE is new in python 3.9
    jump_before_opcodes = (op_compare, op_not)

    # occurred in different python versions
    op_list_extend = dis.opmap['LIST_EXTEND'] if 'LIST_EXTEND' in dis.opmap else None
    op_tuple_unpack_call = dis.opmap['BUILD_TUPLE_UNPACK_WITH_CALL'] if 'BUILD_TUPLE_UNPACK_WITH_CALL' in dis.opmap else None

    def __init__(self, node: Node, value: Any, tracer: Optional[et.ConcreteTracer] = None):
        if tracer is None:
            # This allows you to create a ConcreteProxy object around a raw Node
            tracer = et.GraphAppendingConcreteTracer(node.graph)
        self.tracer = tracer
        self.value = value
        self.node = node

    def __repr__(self) -> str:
        return f'ConcreteProxy({self.node.name}, {self.value})'

    def __getattr__(self, k) -> ConcreteProxy:
        # if the proxy is a wrapped module, forward this call to the torch.nn.Module.__getattribute__
        if orig_func.isinstance(self.value, torch.nn.Module):
            return torch.nn.Module.__getattribute__(self.value, k)
        return ConcreteAttrProxy(self, k)

    def __call__(self, *args, **kwargs) -> ConcreteProxy:
        # If it is a module proxy, we should not create a `call_method` node for this case.
        # What we need is to trace this module or the internals of this module,
        # so here we directly call the `__call__` to trigger `create_proxy` inner the `__call__`.
        if isinstance(self.value, torch.nn.Module):
            return self.value.__call__(*args, **kwargs)
        return self.tracer.create_proxy('call_method', '__call__', (self,) + args, kwargs)

    def __iter__(self) -> Union[Iterable, ConcreteProxy]:
        insts, cur = get_instructions(1)

        if insts[cur].opcode == self.op_call_ex:
            # in executing func(..., *proxy)
            # todo: don't know the func has type_guard or not
            return ConcreteUnpackIterProxy(self)
        elif insts[cur].opcode == self.op_tuple_unpack_call:
            # in executing func(*..., *proxy)
            # todo: don't know the func has type_guard or not
            # <= python 3.8
            return ConcreteUnpackIterProxy(self)
        elif insts[cur].opcode == self.op_list_extend:
            # in executing x.extend(proxy) or [x, *proxy]
            # >= python 3.9
            return ConcreteUnpackIterProxy(self)
        elif insts[cur].opcode == self.op_unpack_sequence:
            # in executing `a, b, c = atuple`
            return ConcreteUnpackIterProxy(self)
        elif insts[cur].opname == 'GET_ITER' and insts[cur + 1].opname == 'FOR_ITER' and orig_func.isinstance(self.value, orig_func.range):
            # in executing `for i in range(...)`
            return iter(self.value)
        # elif insts[cur].opname == 'CONTAINS_OP':
        #     # in executing `for i in range(...)`
        #     return iter(self.value)
        else:
            return self.tracer.create_proxy('call_function', iter, (self,), {})

    def __next__(self) -> ConcreteProxy:
        return self.tracer.create_proxy('call_function', next, (self,), {})

    def __len__(self) -> Union[int, ConcreteProxy]:
        insts, cur = get_instructions(1)

        if insts[cur].opcode == self.op_call_ex:
            # in executing func(..., *proxy)
            return orig_func.len(self.value)
        elif insts[cur].opcode == self.op_tuple_unpack_call:
            # in executing func(*..., *proxy)
            # <= python 3.8
            return orig_func.len(self.value)
        elif insts[cur].opcode == self.op_list_extend:
            # in executing x.extend(*proxy) or [x, *proxy]
            # >= python 3.9
            return orig_func.len(self.value)
        else:
            return self.tracer.create_proxy('call_function', orig_func.len, (self,), {})

    def __getitem__(self, *args, **kwargs) -> ConcreteProxy:
        return self.tracer.create_proxy('call_function', orig_func.getitem, (self,) + args, kwargs)

    def __setitem__(self, *args, **kwargs) -> ConcreteProxy:
        return self.tracer.create_proxy('call_function', orig_func.setitem, (self,) + args, kwargs)

    def __bool__(self) -> Union[bool, ConcreteProxy]:
        insts, cur = get_instructions(1)

        if insts[cur].opcode in self.jump_opcodes or (
            insts[cur].opcode in self.jump_before_opcodes and insts[cur + 1].opcode in self.jump_opcodes):
            # in executing branch condition
            return orig_func.bool(self.value)
        elif insts[cur].opname == 'CONTAINS_OP':
            # in executing 'in'
            return orig_func.bool(self.value)
        elif insts[cur].opname == 'BINARY_SUBSCR':
            # in executing slice or index, my_list[index] or my_dict[key]
            return orig_func.bool(self.value)
        elif insts[cur].opcode == self.op_call_ex:
            # in executing func(..., *proxy)
            return orig_func.bool(self.value)
        elif insts[cur].opcode == self.op_not:
            # We cannot return a proxy because 'UNARY_NOT' op will check the type.
            _logger.warning('please use the function patcher, or use "x = operator.not_(y)" instead of "x = not y",'
                            'otherwise the traced graph may be wrong')
            return orig_func.bool(self.value)
        else:
            return self.tracer.create_proxy('call_function', orig_func.bool, (self,), {})

    def __index__(self) -> Union[int, ConcreteProxy]:
        # should only be in list/tuple getitem
        return orig_func.index(self.value)

    def __hash__(self) -> Union[int, ConcreteProxy]:
        # should only be in dict getitem
        return hash(self.value)

    def __contains__(self, item) -> bool:
        # should only be in iterable
        return self.value.__contains__(item)

    def __enter__(self):
        if getattr(self.value.__class__.__enter__, "__fx_already_patched", False):
            return self.value.__enter__()
        else:
            return self.value.__class__.__enter__(self)

    def __exit__(self, exc_type, exc_value, traceback):
        if getattr(self.value.__class__.__exit__, "__fx_already_patched", False):
            return self.value.__exit__(exc_type, exc_value, traceback)
        else:
            return self.value.__class__.__exit__(self, exc_type, exc_value, traceback)

    @compatibility(is_backward_compatible=True)
    def keys(self):
        insts, cur = get_instructions(1)

        if insts[cur].opcode == self.op_call_ex or insts[cur].opcode == self.op_dict_merge:
            # in executing `**proxy`
            return self.value.keys()
        else:
            return self.tracer.create_proxy('call_method', 'keys', (self,), {})

    @compatibility(is_backward_compatible=True)
    @property
    def values(self):
        if callable(self.value.values):
            def _values():
                return self.tracer.create_proxy('call_method', 'values', (self,), {})
            return _values
        else:
            return ConcreteAttrProxy(self, 'values')

    @compatibility(is_backward_compatible=True)
    def items(self):
        return self.tracer.create_proxy('call_method', 'items', (self,), {})

    @classmethod
    def __torch_function__(cls, orig_method, types, args=None, kwargs=None):
        # to wrap all the functions/methods with tensor inputs in the namespace 'torch.*'.
        # actually a simple way to do wrap, but may get wrong in functions with no tensor inputs.
        # NOTE: now for most functions in torch namespace, we do wrap directly and not use __torch_function__
        _logger.warning(f"{orig_method} is not wrapped by tracer, which is not expected, please consider to register this function.")

        args = args if args else ()
        kwargs = kwargs if kwargs else {}

        with wrap_utils.do_temp_call_origin():
            tracers = orig_func.set()

            def detect_tracer(obj):
                if isinstance(obj, ConcreteProxy):
                    tracers.add(obj.tracer)

            pytree_utils.tree_map(detect_tracer, args)
            pytree_utils.tree_map(detect_tracer, kwargs)

            if len(tracers) > 1:
                raise Exception('more than 1 tracer detected. please report the issue')

            tracer = None if len(tracers) == 0 else tracers.pop()

        if tracer is None:
            raise RuntimeError(f"no proxy detected in the inputs of {orig_method}, please wrap this function for trace completeness.")
        else:
            if isinstance(orig_method, torch._C.ScriptMethod):
                args = (orig_method.owner,) + args
                return tracer.create_proxy('call_method', orig_method.name, args, kwargs)
            if is_tensor_method_or_property(orig_method):
                return tracer.create_proxy('call_method', orig_method.__name__, args, kwargs)
            else:
                return tracer.create_proxy('call_function', orig_method, args, kwargs,
                                           name=tracer.graph._target_to_str(orig_method.__name__))


@compatibility(is_backward_compatible=True)
class ConcreteAttrProxy(ConcreteProxy):
    """
    A more understandable way to deal with sub-field like 'x.y'.
    """

    @compatibility(is_backward_compatible=True)
    def __init__(self, root: ConcreteProxy, attr: str):
        self.root = root
        self.attr = attr
        self.tracer = root.tracer
        # In previous version, the node creation is done lazily.
        # But when we support scoped context,
        # Lazy creation of node will cause the node to be created in the wrong context.
        # Please note unused nodes can still be removed by DCE later.
        self._node: Node = self.tracer.create_proxy(
                'call_function', orig_func.getattr, (self.root, self.attr), {}).node
        if orig_func.isinstance(root.value, torch.Tensor) and attr == 'is_cuda':
            self.value = True
        elif orig_func.isinstance(root.value, torch.Tensor) and attr == 'device':
            self.value = torch.device('cuda')
            warning_msg = "operation <tensor>.device is detected, it will always return torch.device('cuda') during trace, " + \
                          "please make sure don't manually change the tensor device in the code.\n" + \
                          f"\t{get_frame_record()}"
            _logger.warning(warning_msg)
        else:
            self.value = orig_func.getattr(root.value, attr)

    def __repr__(self) -> str:
        calling_frame_name = inspect.stack()[1][1]
        if calling_frame_name.endswith('pydevd_exe2.py') or calling_frame_name.endswith('pydevd_safe_repr.py'):
            return f'ConcreteAttrProxy({self.node.name})'
        return repr(self.value)

    @property
    def node(self):
        return self._node

    def __call__(self, *args, **kwargs):
        return self.tracer.create_proxy('call_method', self.attr, (self.root,) + args, kwargs)


@compatibility(is_backward_compatible=True)
class ConcreteUnpackIterProxy(ConcreteProxy):
    """
    A more understandable way to deal with iterables.
    Only support 'tuple' and 'list'. Will transfer un-subscriptables such as 'set', to 'tuple'.
    todo: support for 'zip'

    examples:
        1. `a, b = c` =>
            ori:
                iter1 = c.__iter__()
                a = iter1.__next__()
                b = iter1.__next__()
            new:
                a = c[0]
                b = c[1]

        2. `y = [x, *proxy]` =>
            ori:
                iter1 = c.__iter__()
                a = iter1.__next__()
                b = iter1.__next__()
                y = [x, a, b]
            new:
                a = proxy[0]
                b = proxy[1]
                y = [x, a, b]
    """

    @staticmethod
    def try_create(root: Any):
        if isinstance(root, ConcreteProxy):
            return ConcreteUnpackIterProxy(root)
        else:
            return iter(root)

    @compatibility(is_backward_compatible=True)
    def __init__(self, root: ConcreteProxy):
        if not hasattr(root.value, '__getitem__'):
            # transfer 'set' to 'tuple'
            # it's tuple not orig_func.tuple!
            # root = tuple(root)
            root = root.tracer.create_proxy('call_function', orig_func.tuple, (root,), {})
        self.root = root
        self.tracer = root.tracer
        self._node: Optional[Node] = None
        self._value: List[Any] = []
        self.index = -1
        self.len = orig_func.len(root.value)

    def __repr__(self) -> str:
        return f'ConcreteUnpackIterProxy({self.node.name})'

    @property
    def node(self):
        # the node for attributes is added lazily, since most will just be method calls
        # which do not rely on the getitem call
        if self._node is None:
            self._node = self.tracer.create_proxy(
                'call_function', iter, (self.root,), {}).node
        return self._node

    @property
    def value(self):
        # the node for attributes is added lazily, since most will just be method calls
        # which do not rely on the getitem call
        if orig_func.len(self._value) == 0:
            self._value.append(iter(self.root.value))
        return self._value[0]

    def __next__(self):
        self.index += 1
        if self.index == self.len:
            raise StopIteration()
        return self.tracer.create_proxy('call_function', orig_func.getitem, (self.root, self.index), {})

@compatibility(is_backward_compatible=True)
def map_aggregate_not_proxy(a, fn):
    """
    Apply fn to each Node appearing arg. arg may be a list, tuple, slice, or dict with string keys.
    """
    if orig_func.isinstance(a, ConcreteProxy):
        return fn(a)
    elif orig_func.isinstance(a, orig_func.tuple):
        t = orig_func.tuple(map_aggregate_not_proxy(elem, fn) for elem in a)
        # Support NamedTuple (if it has `_fields`) by repacking into original type.
        return t if not hasattr(a, '_fields') else orig_func.type(a)(*t)
    elif orig_func.type(a) == orig_func.list:
        return orig_func.list(map_aggregate_not_proxy(elem, fn) for elem in a)
    elif orig_func.isinstance(a, orig_func.dict):
        return orig_func.dict((k, map_aggregate_not_proxy(v, fn)) for k, v in a.items())
    elif orig_func.isinstance(a, orig_func.slice):
        return orig_func.slice(map_aggregate_not_proxy(a.start, fn), map_aggregate_not_proxy(a.stop, fn), map_aggregate_not_proxy(a.step, fn))
    else:
        return fn(a)

# register or wrap common methods on 'ConcreteProxy'
# for method in magic_methods:
# torch.fx.graph.inplace_methods may not exist on some verion of pytorch
inplace_methods = {
    'iadd': '{} += {}',
    'iand': '{} &= {}',
    'ifloordiv': '{} //= {}',
    'ilshift': '{} <<= {}',
    'imod': '{} %= {}',
    'imul': '{} *= {}',
    'imatmul': '{} @= {}',
    'ior': '{} |= {}',
    'ipow': '{} **= {}',
    'irshift': '{} >>= {}',
    'isub': '{} -= {}',
    'itruediv': '{} /= {}',
    'ixor': '{} ^= {}',
    'setitem': '{}[{}] = {}',
}
for method in {**magic_methods, **inplace_methods}:
    def _scope(method):
        def impl(*args, **kwargs):
            tracer = args[0].tracer
            target = orig_func.getattr(orig_func, method)
            return tracer.create_proxy('call_function', target, args, kwargs)
        impl.__name__ = method
        as_magic = f'__{method.strip("_")}__'
        setattr(ConcreteProxy, as_magic, impl)
    _scope(method)


def _define_reflectable(orig_method_name):
    method_name = f'__r{orig_method_name.strip("_")}__'

    def impl(self, rhs):
        target = orig_func.getattr(orig_func, orig_method_name)
        return self.tracer.create_proxy('call_function', target, (rhs, self), {})
    impl.__name__ = method_name
    impl.__qualname__ = method_name
    setattr(ConcreteProxy, method_name, impl)


for orig_method_name in reflectable_magic_methods:
    _define_reflectable(orig_method_name)
