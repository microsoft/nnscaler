#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import builtins
from contextlib import contextmanager
from dataclasses import dataclass, field
import functools
import operator
import importlib

from types import MethodType, ModuleType
from typing import Any, Dict, Optional, Type, List, Callable, Union, TYPE_CHECKING, Tuple

import math
import torch
from torch.fx.proxy import Scope, ScopeContextManager

import nnscaler.graph.tracer as cct
from . import pytree_utils, orig_func, operator_patcher
if TYPE_CHECKING:
    from .concrete_tracer import ConcreteTracer

import logging
_logger = logging.getLogger(__name__)

# global variable to control if the wrapped function should only execute the original logic
TEMP_CALL_ORIGIN = False


@contextmanager
def do_temp_call_origin():
    """
    Under this context, the wrapped functon will directly execute the original logic.
    """
    global TEMP_CALL_ORIGIN
    temp_call_origin = TEMP_CALL_ORIGIN
    TEMP_CALL_ORIGIN = True
    try:
        yield
    finally:
        TEMP_CALL_ORIGIN = temp_call_origin


@dataclass
class Location:
    """
    The place a function/class locates.
    Please note one function/class can be in multiple places.
    Take `torch.meshgrid` for example, there are `torch.meshgrid`, 'torch.functional.meshgrid', 'torch._C._VariableFunctions.meshgrid',
    """
    ns: Union[Type, ModuleType, Any]  # the namespace of the name. It can be a class/module, etc.
    name: str


@dataclass
class LeafWrapInfo:
    """
    extra_locs: The place the function is imported.
    is_force_trace: If set to false, the function will only be traced if inputs include proxy.
        Such as 'torch.rand', we should trace it even if it doesn't have proxy as input, so it should be force traced.
    replacement: If not `None`, we will use it to replace the original function/class in traced code.
        Such as ModuleList.__getitem__, we can use operator.getitem to replace it.
    """
    extra_locs: List[Location] = field(default_factory=list)
    is_force_trace: bool = False
    replacement: Union[None, Callable, Type] = None


default_autowrap_leaf_function: Dict[Any, LeafWrapInfo] = {
    # wrap widely used builtins functions that can be applied on torch.Tensor
    builtins.len:                  LeafWrapInfo([], False, None),
    builtins.abs:                  LeafWrapInfo([], False, None),
    builtins.all:                  LeafWrapInfo([], False, None),
    builtins.any:                  LeafWrapInfo([], False, None),
    builtins.min:                  LeafWrapInfo([], False, None),
    builtins.max:                  LeafWrapInfo([], False, None),

    # force-traced function (the factory functions of tensor creation)
    torch.arange:                  LeafWrapInfo([], True, None),
    torch.empty:                   LeafWrapInfo([], True, None),
    torch.eye:                     LeafWrapInfo([], True, None),
    torch.full:                    LeafWrapInfo([], True, None),
    torch.linspace:                LeafWrapInfo([], True, None),
    torch.logspace:                LeafWrapInfo([], True, None),
    torch.ones:                    LeafWrapInfo([], True, None),
    torch.rand:                    LeafWrapInfo([], True, None),
    torch.randint:                 LeafWrapInfo([], True, None),
    torch.randn:                   LeafWrapInfo([], True, None),
    torch.randperm:                LeafWrapInfo([], True, None),
    torch.tensor:                  LeafWrapInfo([], True, None),
    torch.zeros:                   LeafWrapInfo([], True, None),

    # method
    torch.nn.Sequential.__getitem__:     LeafWrapInfo([], False, operator.getitem),
    torch.nn.Sequential.__len__:         LeafWrapInfo([], False, builtins.len),
    torch.nn.Sequential.__iter__:        LeafWrapInfo([], False, builtins.iter),

    torch.nn.ModuleList.__getitem__:     LeafWrapInfo([], False, operator.getitem),
    torch.nn.ModuleList.__len__:         LeafWrapInfo([], False, builtins.len),
    torch.nn.ModuleList.__iter__:        LeafWrapInfo([], False, builtins.iter),

    torch.nn.ModuleDict.__getitem__:     LeafWrapInfo([], False, operator.getitem),
    torch.nn.ModuleDict.__len__:         LeafWrapInfo([], False, builtins.len),
    torch.nn.ModuleDict.__iter__:        LeafWrapInfo([], False, builtins.iter),
    torch.nn.ModuleDict.__contains__:    LeafWrapInfo([], False, operator.contains),

    torch.nn.ParameterList.__getitem__:  LeafWrapInfo([], False, operator.getitem),
    torch.nn.ParameterList.__len__:      LeafWrapInfo([], False, builtins.len),
    torch.nn.ParameterList.__iter__:     LeafWrapInfo([], False, builtins.iter),

    torch.nn.ParameterDict.__getitem__:  LeafWrapInfo([], False, operator.getitem),
    torch.nn.ParameterDict.__len__:      LeafWrapInfo([], False, builtins.len),
    torch.nn.ParameterDict.__iter__:     LeafWrapInfo([], False, builtins.iter),
    torch.nn.ParameterDict.__contains__: LeafWrapInfo([], False, operator.contains),

    torch.autocast.__enter__:            LeafWrapInfo([], False, None),
    torch.autocast.__exit__:             LeafWrapInfo([], False, None),
}


def _functions_in_module(module: ModuleType):
    """
    Detect all the callable functions in the module, exclude the functions start with `_` or typing related
    """
    for name in module.__dir__():
        # get all callable except private function
        if not name.startswith('_') and callable(getattr(module, name)):
            op = getattr(module, name)
            # exclude the typing related
            if not isinstance(op, Type) and (hasattr(op, '__module__') and op.__module__ not in ('typing,')):
                yield op, name


# the functions that should never be wrapped
# TODO:
# currently we only have import_module, and should add more if needed
# Putting these functions as leaf functions doesn't work
# because
#  1. We only wrap function calls via `ProxyCallTransformer`
#  2. But some functions can be triggered by getattr (e.g. torch._dynamo)
#  Two ways to fix this:
#  1. Handle popular functions in `default_never_wrap_function` in a special way.
#  2. Refine the `ProxyCallTransformer` to handle getattr as well.
#  The second way is more general, but it's more complex and may introduce potential bugs.
#  For now, we choose the first way as a quick fix.
default_never_wrap_function: Dict[Callable, LeafWrapInfo] = {
    orig_func.import_module: LeafWrapInfo([Location(importlib, 'import_module')], False, None)
}


# get all functions in the default_autowrap_modules and add them to default_autowrap_leaf_function
default_autowrap_modules = (operator, math, torch, torch.functional, torch.nn.functional)
for module in default_autowrap_modules:
    for func, func_name in _functions_in_module(module):
        if func in default_autowrap_leaf_function:
            default_autowrap_leaf_function[func].extra_locs.append(Location(module, func_name))
        else:
            default_autowrap_leaf_function[func] = LeafWrapInfo([Location(module, func_name)], False, None)


default_autowrap_leaf_class: Dict[Type, LeafWrapInfo] = {
    # class
    builtins.bool:                 LeafWrapInfo([], False),
    builtins.int:                  LeafWrapInfo([], False),
    builtins.float:                LeafWrapInfo([], False),

    # iterable class
    builtins.tuple:                LeafWrapInfo([], False),
    builtins.list:                 LeafWrapInfo([], False),
    builtins.set:                  LeafWrapInfo([], False),
    builtins.frozenset:            LeafWrapInfo([], False),
    builtins.dict:                 LeafWrapInfo([], False),
    builtins.reversed:             LeafWrapInfo([], False),

    torch.Size:                    LeafWrapInfo([], False),
    torch.finfo:                   LeafWrapInfo([], False),
}


# all wrapped classes should add to this mapping, use to track the original class, used by isinstance wrapper
# {class_wrapper: original_class}
wrapped_cls_to_orig_cls: Dict[Type, Type] = {}


def create_wrapped_leaf_func(func: Callable, *, replace_func: Optional[Callable]=None, default_tracer: Optional['ConcreteTracer']=None,
                             is_method: bool=False, method_name: Optional[str]=None):
    """
    Create a wrapped function/method that will generate a call_function/call_method node when call `func` if there has proxy in the inputs.

    Args:
        func (Callable) : the original function.
        replace_func (Optional[Callable]) : forward the call to another function.
        default_tracer (Tracer) : if the tracer is set, then use this tracer to create a node, no matter there has proxy in the inputs.
        is_method (bool): if the functionl is a bound method.
        method_name (str): use to identify the method name, if the function is a bound method.
    """
    @functools.wraps(func)
    def func_wrapper(*args, **kwargs):
        global TEMP_CALL_ORIGIN
        if TEMP_CALL_ORIGIN:
            return func(*args, **kwargs)
        else:
            with do_temp_call_origin():
                tracers = set()
                if default_tracer is not None:
                    tracers.add(default_tracer)

                def detect_tracer(obj):
                    if isinstance(obj, cct.ConcreteProxy):
                        tracers.add(obj.tracer)

                pytree_utils.tree_map(detect_tracer, args)
                pytree_utils.tree_map(detect_tracer, kwargs)

                if len(tracers) > 1:
                    raise Exception('more than 1 tracer detected. please report the issue')

                tracer = None if len(tracers) == 0 else tracers.pop()

            if tracer is None:
                return func(*args, **kwargs)
            else:
                if replace_func is None:
                    if is_method:
                        return tracer.create_proxy('call_method', method_name, args, kwargs)
                    else:
                        return tracer.create_proxy('call_function', func, args, kwargs)
                else:
                    return tracer.create_proxy('call_function', replace_func, args, kwargs)

    return func_wrapper


def create_wrapped_leaf_class(clz, *, replace_cls: Optional[Callable]=None, default_tracer: Optional['ConcreteTracer']=None):
    """
    Wrap a class as a tracable class, we usually wrap some classes that can be seen as creation functions.
    For example, we can prevent the trace be interrupted by wrap ```int``` in the following case:

        ...
        # x is a scalar
        x_value = int(x)
        new_x = torch.tensor([x_value, x_value])
        ...

    Args:
        clz : the original class.
        replace_cls : forward the call to another function.
        default_tracer (Tracer) : if the tracer is set, then use this tracer to create a node, no matter there has proxy in the inputs.
        is_method (bool): if the functionl is a bound method.
        method_name (str): use to identify the method name, if the function is a bound method.
    """
    class clz_wrapper_clz:
        # used to track the original class
        _fx_wrapped_ori_clz = clz

        def __new__(cls, *args, **kwargs):
            global TEMP_CALL_ORIGIN
            if TEMP_CALL_ORIGIN:
                return clz(*args, **kwargs)
            else:
                with do_temp_call_origin():
                    tracers = set()
                    if default_tracer is not None:
                        tracers.add(default_tracer)

                    def detect_tracer(obj):
                        if isinstance(obj, cct.ConcreteProxy):
                            tracers.add(obj.tracer)

                    pytree_utils.tree_map(detect_tracer, args)
                    pytree_utils.tree_map(detect_tracer, kwargs)

                    if len(tracers) > 1:
                        raise Exception('more than 1 tracer detected. please report the issue')

                    tracer = None if len(tracers) == 0 else tracers.pop()

                if tracer is None:
                    return clz(*args, **kwargs)
                else:
                    if replace_cls is None:
                        return tracer.create_proxy('call_function', clz, args, kwargs)
                    else:
                        return tracer.create_proxy('call_function', replace_cls, args, kwargs)

        def __eq__(self, __o: object) -> bool:
            return id(__o) in (id(self), id(clz))

        def __hash__(self):
            return id(self)

    with do_temp_call_origin():
        for name in dir(clz):
            attr = getattr(clz, name)
            # '__getitem__', '__setitem__', '__iter__', '__len__' means this class can be iterable
            # then we should wrap these methods to keep the graph preserved
            if not name.startswith('_') or name in ('__getitem__', '__setitem__', '__iter__', '__len__'):
                if isinstance(attr, Callable):
                    wrapped_method = create_wrapped_leaf_func(attr, default_tracer=default_tracer, is_method=True, method_name=name)
                    setattr(clz_wrapper_clz, name, wrapped_method)
                else:
                    setattr(clz_wrapper_clz, name, attr)

        # to support subscriptable type hint like func(x: dict[str, str])
        if hasattr(clz, '__class_getitem__'):
            setattr(clz_wrapper_clz, '__class_getitem__', clz.__class_getitem__)

    wrapped_cls_to_orig_cls[clz_wrapper_clz] = clz
    return clz_wrapper_clz


def create_wrapped_module_getattribute(tracer: 'ConcreteTracer'):
    @functools.wraps(orig_func.torch_module_getattribute)
    def module_getattribute_wrapper(mod, attr):
        global TEMP_CALL_ORIGIN
        if TEMP_CALL_ORIGIN:
            try:
                return orig_func.torch_module_getattribute(mod, attr)
            except AttributeError:
                return orig_func.torch_module_getattr(mod, attr)
        with do_temp_call_origin():
            try:
                attr_val = orig_func.torch_module_getattribute(mod, attr)
            except AttributeError:
                attr_val = orig_func.torch_module_getattr(mod, attr)
        if orig_func.isinstance(attr_val, cct.ConcreteProxy):
            warn_msg = f'Detected {tracer.get_path_of_module(mod)}.{attr} is a ConcreteProxy, ' + \
                'this is usually caused by directly assigning the return value of some leaf function to the attribute of the module. ' + \
                'Please note that this writing method may cause some trace errors.'
            _logger.warning(warn_msg)
            return attr_val
        # using isinstance instead of _orig_isinstance to judge whether
        # the ConcreteProxy.value is the following three types if the attr_val is a ConcreteProxy
        elif isinstance(attr_val, (orig_func.tuple, orig_func.list)):
            if tracer.get_path_of_module(mod) == '':
                return tracer.create_proxy('get_attr', f'{attr}', (), {})
            else:
                return tracer.create_proxy('get_attr', f'{tracer.get_path_of_module(mod)}.{attr}', (), {})
        elif attr in tracer.default_module_getattr:
            if tracer.get_path_of_module(mod) == '':
                return tracer.create_proxy('get_attr', f'{attr}', (), {})
            else:
                return tracer.create_proxy('get_attr', f'{tracer.get_path_of_module(mod)}.{attr}', (), {})
        elif id(attr_val) in tracer.path_of_parameter:
            return tracer.create_proxy('get_attr', tracer.path_of_parameter[id(attr_val)], (), {})
        elif id(attr_val) in tracer.path_of_buffer:
            return tracer.create_proxy('get_attr', tracer.path_of_buffer[id(attr_val)], (), {})
        return attr_val
    return module_getattribute_wrapper


def create_wrapped_module_call(tracer: 'ConcreteTracer'):
    @functools.wraps(orig_func.torch_module_call)
    def module_call_wrapper(mod, *args, **kwargs):
        global TEMP_CALL_ORIGIN
        if TEMP_CALL_ORIGIN:
            return orig_func.torch_module_call(mod, *args, **kwargs)
        else:
            # codes below corresponds to symbolic tracer's call_module
            module_qualified_name = tracer.get_path_of_module(mod)
            with ScopeContextManager(tracer.scope, Scope(module_qualified_name, type(mod))) as _scope:
                tracer.module_stack[_scope.module_path] = _scope.module_type
                if not tracer.is_leaf_module(mod, module_qualified_name):
                    autowrap_check(tracer, mod.__dict__)
                    ret_val = orig_func.torch_module_call(mod, *args, **kwargs)
                else:
                    ret_val = tracer.create_proxy('call_module', module_qualified_name, args, kwargs)
                key, _ = tracer.module_stack.popitem(last=True)
                assert key == _scope.module_path, f" Unexpected key {key}"
            return ret_val
    return module_call_wrapper


def create_wrapped_nn_module_func(tracer: 'ConcreteTracer', mod: torch.nn.Module, name: str):
    orig_fn = orig_func.getattr(mod, name)
    if not orig_func.isinstance(orig_fn, MethodType):
        raise RuntimeError(f'{tracer.get_path_of_module(mod)}.{name} is not a bound method, only support wrap bound method.')

    @functools.wraps(orig_fn)
    def wrapped(*args, **kwargs):
        module_qualified_name = tracer.get_path_of_module(mod)
        with ScopeContextManager(tracer.scope, Scope(module_qualified_name, type(mod))) as _scope:
            need_pop = False
            if _scope.module_path not in tracer.module_stack:
                need_pop = True
                tracer.module_stack[_scope.module_path] = _scope.module_type
            elif _scope.module_path != list(tracer.module_stack)[-1]:
                raise RuntimeError(f'Scope not match: {_scope.module_path} vs {list(tracer.module_stack)[-1]}')
            # has tracer means in tracing progress
            if operator_patcher.OperatorPatcherContext.ctx_tracer and operator_patcher.OperatorPatcherContext.ctx_patcher:
                autowrap_check(tracer, orig_fn.__globals__)
                # `patch_run` is needed because this function will be patched by fx patcher,
                # which means it will have `__fx_already_patched` flag, and operator patcher will not patch it again,
                # so directly call `patch_run` here to avoid the `orig_fn is not patched by the operator patcher.
                result = operator_patcher.OperatorPatcherContext.patch_run(orig_fn, *args, **kwargs)
            else:
                result = orig_fn(*args, **kwargs)
            if need_pop:
                key, _ = tracer.module_stack.popitem(last=True)
                assert key == _scope.module_path, f" Unexpected key {key}"
        return result

    return wrapped


def is_autograd_apply(func) -> bool:
    return getattr(func, '__name__', None) == 'apply' \
        and orig_func.isinstance(getattr(func, '__self__', None), Type) and issubclass(func.__self__, torch.autograd.Function)


def create_wrapped_autograd_apply(default_tracer: 'ConcreteTracer'):
    @classmethod
    @functools.wraps(orig_func.torch_agfunc_apply)
    def agfunc_apply_wrapper(clz, *args, **kwargs):
        if clz not in default_tracer.autograd_functions_mapping:
            default_tracer.autograd_functions_mapping[clz] = torch._C._FunctionBase.__dict__['apply'].__get__(None, clz)
        global TEMP_CALL_ORIGIN
        if TEMP_CALL_ORIGIN:
            return default_tracer.autograd_functions_mapping[clz](*args, **kwargs)
        with do_temp_call_origin():
            tracers = set()

            def detect_tracer(obj):
                if isinstance(obj, cct.ConcreteProxy):
                    tracers.add(obj.tracer)

            pytree_utils.tree_map(detect_tracer, args)
            pytree_utils.tree_map(detect_tracer, kwargs)

            if len(tracers) > 1:
                raise Exception('more than 1 tracer detected. please report the issue')

            tracer = None if len(tracers) == 0 else tracers.pop()
        if tracer is None:
            return default_tracer.autograd_functions_mapping[clz](*args, **kwargs)
        else:
            assert tracer == default_tracer
            return default_tracer.create_proxy('call_function', default_tracer.autograd_functions_mapping[clz], args, kwargs)
    return agfunc_apply_wrapper


class map_wrapper_clz:
    # used to track the original class
    _fx_wrapped_ori_clz = orig_func.map

    def __new__(cls, the_func, *iterables: Any):
        global TEMP_CALL_ORIGIN
        if TEMP_CALL_ORIGIN:
            return orig_func.map(the_func, *iterables)
        else:
            # get the result first
            results = orig_func.list()
            for args in zip(*iterables):
                results.append(the_func(*args))
            # if there contains proxy in results, then create a proxy with tuple as target
            with do_temp_call_origin():
                tracers = set()

                def detect_tracer(obj):
                    if isinstance(obj, cct.ConcreteProxy):
                        tracers.add(obj.tracer)

                pytree_utils.tree_map(detect_tracer, results)

                if len(tracers) > 1:
                    raise Exception('more than 1 tracer detected. please report the issue')
                elif len(tracers) == 1:
                    return next(iter(tracers)).create_proxy('call_function', orig_func.tuple, (results,), {})

            return orig_func.tuple(results)

    def __eq__(self, __o: object) -> bool:
        return id(__o) in (id(self), id(orig_func.map))

    def __hash__(self):
        return id(self)

wrapped_cls_to_orig_cls[map_wrapper_clz] = orig_func.map


class range_wrapper_clz:
    # used to track the original class
    _fx_wrapped_ori_clz = orig_func.range

    def __new__(cls, *args):
        assert 1 <= orig_func.len(args) <= 3
        args = (arg.value if orig_func.isinstance(arg, cct.ConcreteProxy) else arg for arg in args)
        return orig_func.range(*args)

    def __eq__(self, __o: object) -> bool:
        return id(__o) in (id(self), id(orig_func.range))

    def __hash__(self):
        return id(self)

wrapped_cls_to_orig_cls[range_wrapper_clz] = orig_func.range


class enumerate_wrapper_clz:
    # used to track the original class
    _fx_wrapped_ori_clz = orig_func.enumerate

    def __new__(cls, iterable, start=0):
        count = start
        for elem in iterable:
            if orig_func.isinstance(elem, cct.ConcreteProxy) and orig_func.isinstance(elem.value, (orig_func.int, str)):
                yield count, elem.value
            else:
                yield count, elem
            count += 1

    def __eq__(self, __o: object) -> bool:
        return id(__o) in (id(self), id(orig_func.enumerate))

    def __hash__(self):
        return id(self)

wrapped_cls_to_orig_cls[enumerate_wrapper_clz] = orig_func.enumerate


class type_wrapper_clz:
    # used to track the original class
    _fx_wrapped_ori_clz = orig_func.type

    def __new__(cls, obj_or_name, *args):
        # case 1: class type(name, bases, dict, **kwds)
        if orig_func.len(args) > 0:
            assert orig_func.len(args) == 2
            base_cls, cls_dict = args[0], args[1]
            # if it is a wrapped class, replace it to the original one
            base_cls = orig_func.tuple(bs._fx_wrapped_ori_clz if hasattr(bs, '_fx_wrapped_ori_clz') else bs for bs in base_cls)
            return orig_func.type(obj_or_name, base_cls, cls_dict)
        # case 2: class type(object)
        else:
            orig_type = orig_func.type(obj_or_name)
            if issubclass(orig_type, cct.ConcreteProxy):
                return orig_func.type(obj_or_name.value)
            else:
                return orig_type

    def __eq__(self, __o: object) -> bool:
        return id(__o) in (id(self), id(orig_func.type))

    def __hash__(self):
        return id(self)

wrapped_cls_to_orig_cls[type_wrapper_clz] = orig_func.type


# wrap autocast to make it support proxy input and the related node will be DCE in DCE stage.
class torch_autocast_wrapper_clz:
    # used to track the original class
    _fx_wrapped_ori_clz = orig_func.torch_autocast

    def __new__(cls, *args, **kwargs):
        return orig_func.torch_autocast(*args, **kwargs)

    def __eq__(self, __o: object) -> bool:
        return id(__o) in (id(self), id(orig_func.type))

    def __hash__(self):
        return id(self)

wrapped_cls_to_orig_cls[torch_autocast_wrapper_clz] = orig_func.torch_autocast


@functools.wraps(orig_func.torch_assert)
def torch_assert_wrapper(condition, message):
    if orig_func.isinstance(condition, cct.ConcreteProxy):
        condition = condition.value
    return orig_func.isinstance(condition, message)


@functools.wraps(orig_func.isinstance)
def isinstance_wrapper(instance, clz):
    if orig_func.type(clz) in (slice, tuple, list, orig_func.slice, orig_func.tuple, orig_func.list):
        clz_wrapped = []
        for wrapped_type, orig_type in wrapped_cls_to_orig_cls.items():
            if wrapped_type in clz:
                clz_wrapped.append(orig_type)
        clz = (*clz_wrapped, *(aclz for aclz in clz if aclz not in wrapped_cls_to_orig_cls))
        # use _orig_isinstance(clz, Iterable) will cause an endless recursive loop
        for cls in (object, cct.ConcreteProxy):
            if cls in clz and orig_func.isinstance(instance, cls):
                return True
        if orig_func.isinstance(instance, cct.ConcreteProxy):
            return orig_func.isinstance(instance.value, clz)
        else:
            return orig_func.isinstance(instance, clz)
    else:
        if clz in (object, cct.ConcreteProxy):
            return orig_func.isinstance(instance, clz)
        if clz in wrapped_cls_to_orig_cls:
            clz = wrapped_cls_to_orig_cls[clz]
        if orig_func.isinstance(instance, cct.ConcreteProxy):
            instance = instance.value
        return orig_func.isinstance(instance, clz)


@functools.wraps(orig_func.issubclass)
def issubclass_wrapper(subclass, clz):
    if orig_func.type(clz) in (slice, tuple, list, orig_func.slice, orig_func.tuple, orig_func.list):
        clz_wrapped = []
        for wrapped_type, orig_type in wrapped_cls_to_orig_cls.items():
            if wrapped_type in clz:
                clz_wrapped.append(orig_type)
        clz = (*clz_wrapped, *(aclz for aclz in clz if aclz not in wrapped_cls_to_orig_cls))
        return orig_func.issubclass(subclass, clz)
    else:
        if clz in wrapped_cls_to_orig_cls:
            clz = wrapped_cls_to_orig_cls[clz]
        return orig_func.issubclass(subclass, clz)


@functools.wraps(orig_func.getattr)
def getattr_wrapper(obj, *args):
    if not 1 <= orig_func.len(args) <= 2:
        raise Exception()
    args = orig_func.list(args)
    if orig_func.isinstance(args[0], cct.ConcreteProxy):
        args[0] = args[0].value
    return orig_func.getattr(obj, *args)


# NOTE: not in used, still need some test
@functools.wraps(orig_func.id)
def id_wrapper(obj):
    if hasattr(obj, '_fx_wrapped_ori_clz'):
        return orig_func.id(orig_func.getattr(obj, '_fx_wrapped_ori_clz'))
    else:
        return orig_func.id(obj)


def autowrap_check(tracer: 'ConcreteTracer', frame_dict : Dict[str, Any]):
    """
    Some methods, like `math.sqrt` are common enough we want to automatically wrap them as we see them.
    This method searches a scope for them and patches them if found.
    """
    if tracer.patcher.visit_once(frame_dict):
        for name, value in frame_dict.items():
            if callable(value) and not name.startswith('_') and (getattr(orig_func, name, None) is not value):
                if value in tracer.wrapped_leaf:
                    tracer.patcher.patch(frame_dict, name, tracer.wrapped_leaf[value][1])
                if is_autograd_apply(value):
                    if value.__self__ not in tracer.autograd_functions_mapping:
                        tracer.autograd_functions_mapping[value.__self__] = create_wrapped_leaf_func(value)
                    tracer.patcher.patch(frame_dict, name, tracer.autograd_functions_mapping[value.__self__])
