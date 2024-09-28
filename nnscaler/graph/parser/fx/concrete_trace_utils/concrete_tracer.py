#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

# CREDITS: This implementation is inspired by PyTorch fx symbolic trace: https://github.com/pytorch/pytorch/blob/main/torch/fx/_symbolic_trace.py

from __future__ import annotations

import collections
import copy
import sys
import inspect
import logging
import builtins

from types import FunctionType, MethodDescriptorType, MethodType, MethodWrapperType, ModuleType
from typing import Any, Dict, Optional, Set, Tuple, Type, List, Callable, Union, Literal
from contextlib import contextmanager

import torch
from torch._C import ScriptObject

import torch.fx
from torch.fx import GraphModule
from torch.fx._compatibility import compatibility
from torch.fx._symbolic_trace import _proxyable_classes
from torch.fx.graph import Graph
from torch.fx.node import Target, Node, Argument
from torch.fx.proxy import TracerBase, Scope
from torch.fx.operator_schemas import check_for_mutable_operation

dict_keys_type = type(dict().keys())
dict_values_type = type(dict().values())
dict_items_type = type(dict().items())

from . import concrete_proxy as ep
from . import pytree_utils, orig_func, wrap_utils
from .frame_utils import get_frame_record
from .function_patcher import FunctionPatcher
from .metadata import EmptyResult, extract_results_metadata
from .operator_patcher import OperatorPatcherContext
from .torch_fx_patcher import TorchFXPatcher, ExtraSEFPatcher, side_effectful_inplace_ops
from .trace_strategy import TRACE_STRATEGY

# pyright: reportGeneralTypeIssues=false
_logger = logging.getLogger(__name__)
HAS_VARSTUFF = inspect.CO_VARARGS | inspect.CO_VARKEYWORDS


@compatibility(is_backward_compatible=True)
class ConcreteTracer(TracerBase):
    """
    A model tracer similar to _symbolic_trace.Tracer, but with concrete execution and real value so we can pass complex conditions
    and go into correct brunches.
    """

    default_module_getattr = (
        'training',
    )

    @compatibility(is_backward_compatible=True)
    def __init__(self, strategy, record_frames = False):
        """
        similar to _symbolic_trace.Tracer.__init__.
        remove the 'param_shapes_constant' because we can get real shape when executing.

        Args:
            strategy (Literal['cpu', 'cuda', 'meta', 'cuda_run_cpu_offload', 'reuse_cache']):
                The device placement strategy for intermediate results and module parameters/buffer, and run target.
                The following strategies are supported:
                    'cpu': Execute all functions on cpu, model weights and intermediate results are on cpu.
                    `cuda': Execute all functions on cuda, model weights and intermediate results are on cuda.
                        This strategy is recommended if the model can inference on single gpu.
                    'meta': Execute all functions on meta, model weights are on cpu and intermediate results are on meta.
                    'cuda_run_cpu_offload': Try to execute all functions on cuda, and retry to execute the function on cpu as backup if meet OOM error,
                        model weights and intermediate results are on cpu. This strategy is recommanded for most case if the model is too large to inference on single gpu.
                    'reuse_cache': Similar to `cuda_run_cpu_offload` strategy, additional add a buffer to cache all the intermediate results with different function signatures on cpu,
                        function with same signature exist in cache directly take the cached result as this time function execution to save time.
                        Same signature means the funtions are the same and have almost the same inputs
                        (for tensor type input, just check if they have same tensor meta data[shape, dtyep, requires_grad, stride, memory_format, ...], and don't check the value).
                        This strategy is an experimental strategy to speedup the large-model-large-input case,
                        and have risk to trace an incorrect graph if the signature defined here can not distinguish the differnet functions used in the model,
                        for example, torch.nonzero will always return the same result if the input have same meta data but different value.
                        We have plan to continue improve this strategy to handle most these kind of data dependence cases, but please note that the risk is still inevitable.

            record_frames (bool): If set to True, will add frame information to node.meta['frame_record']. Note this will cost additional trace time.
        """
        super().__init__()
        self.scope = Scope("", None)
        self.module_stack = collections.OrderedDict()
        self.node_name_to_scope = {}
        self.strategy = TRACE_STRATEGY[strategy](self)
        self.record_frames = record_frames
        self.patcher = FunctionPatcher()

        # When we concrete executing some functions,
        # we need revert all the patched function to the unpatched version to ensure the correctness of some underlying code.
        # For most functions, disable_call is sufficient, but it is necessary when executing, for example, a triton function.
        # Here we put all user wrapped function into the set, and unpatch all the patched functions when executing the user function.
        self.need_revert_functions = set()
        self.need_revert_wrapped_functions = set()

        self.temp_call_origin = False

    def add_need_revert_function(self, func, wrapped_func):
        self.need_revert_functions.add(func)
        self.need_revert_wrapped_functions.add(wrapped_func)

    def need_revert(self, func):
        return func in self.need_revert_functions or func in self.need_revert_wrapped_functions

    @contextmanager
    def do_temp_call_origin(self):
        temp_call_origin = self.temp_call_origin
        self.temp_call_origin = True
        try:
            yield
        finally:
            self.temp_call_origin = temp_call_origin

    @compatibility(is_backward_compatible=True)
    def fetch_attr(self, target: str) -> Any:
        """
        to get the attr in self.root. only for execution of 'call_module' nodes.
        """
        with wrap_utils.do_temp_call_origin():
            target_atoms = target.split('.')
            attr_itr = self.root
            for i, atom in orig_func.enumerate(target_atoms):
                # if atom == '':
                #     continue
                if not hasattr(attr_itr, atom):
                    raise RuntimeError(f"Node referenced nonexistent target \'{'.'.join(target_atoms[:i])}\'")
                attr_itr = orig_func.getattr(attr_itr, atom)
            return attr_itr

    @compatibility(is_backward_compatible=True)
    def create_node(self, kind : str, target : Target,
                    args : Tuple[Argument, ...], kwargs : Dict[str, Argument], name : Optional[str] = None,
                    type_expr : Optional[Any] = None, node_result: Any = EmptyResult) -> Node:
        """
        This method is almost the same as the one in `TracerBase` class of Pytorch2.0.
        Add it here because this method of Pytorch1.13 and older version
        doesn't have the part related to `module_stack` and `node_name_to_scope`.
        If we don't add it here, we can not use these two attributes in Pytorch1.13 and older version.
        """
        if kind == 'call_function' and self.check_mutable_operations:
            check_for_mutable_operation(target, args, kwargs)

        node = self.graph.create_node(kind, target, args, kwargs, name, type_expr)
        # TODO node_name_to_scope will be depricated in favor of
        # node.meta['nn_module_stack']
        self.node_name_to_scope[node.name] = (
            self.scope.module_path,
            self.scope.module_type,
        )
        if self.module_stack:
            node.meta['nn_module_stack'] = copy.copy(self.module_stack)
        else:
            node.meta['nn_module_stack'] = collections.OrderedDict()

        def unwrap_nested_proxy(proxy: ep.ConcreteProxy):
                return pytree_utils.tree_map_only(ep.ConcreteProxy, unwrap_nested_proxy, proxy.value)

        # unwrap all proxy in the node result here, because no proxy should be record in the tensor metadata
        node_result = pytree_utils.tree_map_only(ep.ConcreteProxy, unwrap_nested_proxy, node_result)
        extract_results_metadata(node_result, node)
        return node

    @compatibility(is_backward_compatible=True)
    def proxy(self, value: Any, node: Node) -> ep.ConcreteProxy:
        """
        overloaded to use custom 'proxy'.
        """
        return ep.ConcreteProxy(node, value, self)

    @compatibility(is_backward_compatible=True)
    def create_proxy(self, kind: str, target: Target, args: Tuple[Any, ...], kwargs: Dict[str, Any],
                    name: Optional[str] = None, type_expr: Optional[Any] = None,
                    proxy_factory_fn: Optional[Callable[[Node], Any]] = None):
        """
        similar to _symbolic_trace.Tracer.create_proxy.
        use the 'run_target' to actually execute the code, and store the value in 'value' field.
        create the nodes for the target and the input of the target (if the target is one of call_method, call_function, call_module).
        """
        with wrap_utils.do_temp_call_origin():
            def unwrap_nested_proxy(proxy: ep.ConcreteProxy):
                return pytree_utils.tree_map_only(ep.ConcreteProxy, unwrap_nested_proxy, proxy.value)

            args_unwrapped = pytree_utils.tree_map_only(ep.ConcreteProxy, unwrap_nested_proxy, args)
            kwargs_unwrapped = pytree_utils.tree_map_only(ep.ConcreteProxy, unwrap_nested_proxy, kwargs)

            if self.need_revert(target):
                with self.patcher.revert():
                    value_unwrapped, args_run, kwargs_run = self.strategy.run_target(kind, target, args_unwrapped, kwargs_unwrapped)
            else:
                value_unwrapped, args_run, kwargs_run = self.strategy.run_target(kind, target, args_unwrapped, kwargs_unwrapped)

            # because setitem is an inplace operation and will not return the obj, so here is a workaound to record node result
            node_result = args_run[0] if kind == "call_function" and target == orig_func.setitem else value_unwrapped
            # here update the origin args/kwargs to prevent inplace operator to the input
            args = update_tree_proxy_value(args, args_run)
            kwargs = update_tree_proxy_value(kwargs, kwargs_run)

            args_ = self.create_arg(args)
            kwargs_ = self.create_arg(kwargs)
            assert isinstance(args_, tuple)
            assert isinstance(kwargs_, dict)

        node = self.create_node(kind, target, args_, kwargs_, name, type_expr, node_result)

        if self.record_frames and kind != 'placeholder':
            with wrap_utils.do_temp_call_origin():
                node.meta['frame_record'] = get_frame_record()

        proxy = self.proxy(value_unwrapped, node)
        return proxy

    @compatibility(is_backward_compatible=True)
    def create_arg(self, a: Any) -> Union[Node, Any]:
        """
        similar to _symbolic_trace.Tracer.create_arg
        move the base case to the top in case the wrapping of the function 'isinstance'
        """
        # base case: we unwrap the Proxy object
        if isinstance(a, ep.ConcreteProxy):
            return a.node

        if isinstance(a, torch.nn.Parameter):
            for n, p in self.root.named_parameters():
                if a is p:
                    return self.create_node('get_attr', n, (), {}, node_result=a)
            raise NameError('parameter is not a member of this module')
        elif isinstance(a, torch.Tensor):
            for n_, p_ in self.root.named_buffers():
                if a is p_:
                    return self.create_node('get_attr', n_, (), {}, node_result=a)
        elif isinstance(a, torch.nn.Module):
            for n_, p_ in self.root.named_modules():
                if a is p_:
                    return self.create_node('get_attr', n_, (), {}, node_result=a)
        # for slice
        if isinstance(a, slice):
            start = self.create_arg(a.start)
            stop = self.create_arg(a.stop)
            step = self.create_arg(a.step)
            if orig_func.isinstance(start, Node)\
                or orig_func.isinstance(stop, Node)\
                or orig_func.isinstance(step, Node):
                return self.create_node('call_function', orig_func.slice, (start, stop, step), {}, node_result=a)
            else:
                return a
        # For NamedTuple instances that appear literally as args, we emit
        # a node to construct the NamedTuple and use that Node as the argument.
        if isinstance(a, tuple) and hasattr(a, '_fields'):
            args = tuple(self.create_arg(elem) for elem in a)
            return self.create_node('call_function', a.__class__, args, {}, node_result=a)

        # Tensors do not have a reliable string repr() from which they can be
        # constructed (and we probably don't want to rely on that, either), so
        # for any constant Tensor values we encounter, first search for if they
        # are an attribute of some module in the module hierarchy. If so, emit
        # a get_attr to retrieve that tensor. Otherwise, we'll store away the
        # tensor value into a special attribute on the Module s.t. we can
        # retrieve it with a get_attr.
        if isinstance(a, (torch.Tensor, ScriptObject)):
            qualname: Optional[str] = self.tensor_attrs.get(a)

            # Tensor was not found in the Module hierarchy, stow it away in a
            # TODO: warning for the not found tensor
            if not qualname:
                i = 0
                while True:
                    qualname = f'_tensor_constant{i}'
                    if not hasattr(self.root, qualname):
                        break
                    i += 1
                self.tensor_attrs[a] = qualname
                setattr(self.root, qualname, a)

            return self.create_node('get_attr', qualname, (), {}, node_result=a)

        if orig_func.type(a) in _proxyable_classes:
            # This is an instance of a proxyable class for which we did not
            # witness its construction. Intern this as a constant attribute

            # TODO: binary search
            i = 0
            while True:
                qualname = f'_{a.__class__.__name__}_constant_{i}'
                if not hasattr(self.root, qualname):
                    break
                i += 1
            setattr(self.root, qualname, a)

            return self.create_node('get_attr', qualname, (), {}, node_result=a)

        if isinstance(a, (torch.autograd.function.Function, torch.autograd.function.FunctionMeta)):
            return a

        if isinstance(a, (dict_keys_type, dict_values_type, dict_items_type)):
            # here we directly flat all values as a list,
            # for the create_arg do not support (dict_keys_type, dict_values_type, dict_items_type)
            a = list(a)

        return super().create_arg(a)

    @compatibility(is_backward_compatible=True)
    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        """
        in nnscaler, will unpack all module to functions, so always return False
        """
        return False

    def get_path_of_module(self, mod: torch.nn.Module):
        if id(mod) in self.path_of_module:
            return self.path_of_module[id(mod)]
        else:
            # if the module id does not exsit in the self.path_of_module, that means this module is not in the orginal root model,
            # may be created somewhere outside of the root model, e.g., created on the fly in the forward computation,
            # in the following example, a new CrossEntropyLoss module will be created during forward:
            #
            #   def forward(self, x, y):
            #       loss = torch.nn.CrossEntropyLoss()
            #       return loss(x, y)
            #
            # in this case, we create a `_module_constants` field on root model to save these module for the completeness.
            if not hasattr(self.root, '_module_constants'):
                self.root._module_constants = torch.nn.ModuleList()
            module_constants = self.root._module_constants
            assert isinstance(module_constants, torch.nn.ModuleList)
            sub_path = str(orig_func.len(module_constants))
            if not hasattr(module_constants, sub_path):
                module_constants.add_module(sub_path, mod)
            path = '_module_constants.%s' % sub_path
            self.path_of_module[id(mod)] = path
            return path

    # This method will be refactored
    @compatibility(is_backward_compatible=False)
    def create_args_for_root(self, root_fn, is_module, concrete_args: Union[Dict[str, Any], Tuple]) -> Tuple[Any, list, Any, Any]:
        """
        for wrapping all the parameters of the function with dummy_input.
        in concrete tracer, we need all the parameters input by users.

        todo: this function should be refactored after the same function in torch.fx be refactored.
        """
        # In some cases, a function or method has been decorated with a wrapper
        # defined via ``functools.wraps``. In this case, the outer code object
        # will likely not contain the actual parameters we care about, so unwrap
        # the function to get to the innermost callable.
        # TODO: keyward-only arguments are not supported now
        fn_for_analysis = inspect.unwrap(root_fn)
        default_value_list = fn_for_analysis.__defaults__
        if default_value_list is None:
            default_value_list = tuple()
        co = fn_for_analysis.__code__
        total_args = co.co_argcount + co.co_kwonlyargcount
        # orig_args = list(co.co_varnames)
        names_iter = iter(co.co_varnames)
        args: List[Any] = []
        more_args = []
        kwargs = {}
        skip_arg_idx = 0
        if is_module:
            if total_args == 0:
                raise RuntimeError('``self`` argument cannot be part of *args expansion!')
            skip_arg_idx = 1
            next(names_iter)  # skip self
            args.append(self.root)

        cnt = 0
        self.placeholder_dict = {}
        arg_names = [next(names_iter) for idx in range(skip_arg_idx, total_args)]
        diff_len = orig_func.len(arg_names) - orig_func.len(default_value_list)
        default_args = {arg_names[idx + diff_len]: default_value_list[idx] for idx in range(len(default_value_list))}
        if isinstance(concrete_args, tuple):
            if orig_func.len(arg_names) != orig_func.len(concrete_args):
                raise RuntimeError(f"Tracing expected {len(arg_names)} arguments but got {len(concrete_args)} concrete arguments")
            concrete_args = {name: val for name, val in zip(arg_names, concrete_args)}
        def proxy_placeholder(name: str):
            nonlocal cnt
            cnt += 1

            default_arg = ()
            if name in default_args and not name.startswith('*'):
                default_arg = (default_args[name],)

            if name in concrete_args:
                self.placeholder_dict[name] = concrete_args[name]
            else:
                # TODO: better infomation
                assert name in default_args
                self.placeholder_dict[name] = default_args[name]
            return self.create_proxy('placeholder', name, default_arg, {})
        args.extend(proxy_placeholder(names) for names in arg_names)


        if hasattr(co, 'co_kwonlyargcount') and (
            co.co_kwonlyargcount > 0 or co.co_flags & HAS_VARSTUFF):
            # TODO: type annotations for *args and **kwargs
            if co.co_flags & inspect.CO_VARARGS:
                name = '*' + next(names_iter)
                default_args[name] = ()
                more_args = proxy_placeholder(name)
            if co.co_flags & inspect.CO_VARKEYWORDS:
                name = '**' + next(names_iter)
                default_args[name] = {}
                kwargs = proxy_placeholder(name)

        return root_fn, args, more_args, kwargs

    def get_wrapped_leaves(self, leaf_functions: Dict[Callable, wrap_utils.LeafWrapInfo], leaf_class: Dict[ModuleType, wrap_utils.LeafWrapInfo]):
        wrapped_leaf_leaves = {}
        for func, wrap_info in leaf_functions.items():
            locations = tuple(wrap_info.extra_locs)
            if wrap_utils.is_autograd_apply(func):
                # torch.autograd.function
                assert wrap_info.replacement == None, '<subclass of torch.autograd.Function>.apply should set to_func to None!'
                if func.__self__ not in self.autograd_functions_mapping:
                    self.autograd_functions_mapping[func.__self__] = wrap_utils.create_wrapped_leaf_func(func)
                wrapped = self.autograd_functions_mapping[func.__self__]
            elif isinstance(func, torch._C.ScriptFunction):
                # if it is a script function,
                # here will wrap the origin function location and forward the script function to the origin one.
                # _torchdynamo_inline is introduced in pytorch 2.0, it is the original function of the script function.
                inner_func = func._torchdynamo_inline
                # some `func.__module__` may have additional `_` compare with its import path in user code,
                # for example, `operator.add.__module__` is `_operator` and `_operator` is a built-in module and we don't want to touch it,
                # we assume user won't import function from module named with prefix `_`,
                # here we only wrap the function under no prefix `_` module, i.e. functions under `operator`.
                if inner_func.__module__.startswith('_') and inner_func.__module__ != '__main__':
                    path = sys.modules.get(inner_func.__module__[1:], sys.modules[inner_func.__module__])
                else:
                    path = sys.modules[inner_func.__module__]
                locations = (*locations, wrap_utils.Location(path, inner_func.__name__))
                wrapped = wrap_utils.create_wrapped_leaf_func(
                    func,
                    replace_func=inner_func,
                    default_tracer=self if wrap_info.is_force_trace else None,
                )
            else:
                # 'TensorBase': torch >= 2.3, '_TensorBase': torch < 2.3
                if func.__qualname__.startswith('_TensorBase') or func.__qualname__.startswith('TensorBase'):
                    locations = (*locations, wrap_utils.Location(torch.Tensor, func.__name__))
                    wrapped = wrap_utils.create_wrapped_leaf_func(
                        getattr(torch.Tensor, func.__name__),
                        replace_func=wrap_info.replacement,
                        default_tracer=self if wrap_info.is_force_trace else None,
                        is_method=True,
                        method_name=func.__name__,
                    )
                elif func.__qualname__.startswith('_VariableFunctionsClass'):
                    if hasattr(torch, func.__name__) and getattr(torch, func.__name__) == func:
                        # avoid bad attr like 'unique_dim'
                        locations = (*locations, wrap_utils.Location(torch, func.__name__))
                    wrapped = wrap_utils.create_wrapped_leaf_func(
                        func,
                        replace_func=wrap_info.replacement,
                        default_tracer=self if wrap_info.is_force_trace else None,
                    )
                elif isinstance(func, (MethodDescriptorType, MethodWrapperType)):
                    wrapped = wrap_utils.create_wrapped_leaf_func(
                        func,
                        replace_func=wrap_info.replacement,
                        default_tracer=self if wrap_info.is_force_trace else None,
                        is_method=True,
                        method_name=func.__name__,
                    )
                elif func.__name__ != func.__qualname__ and func.__qualname__ != 'boolean_dispatch.<locals>.fn' \
                    and not func.__qualname__.startswith('PyCapsule'):
                    # method
                    # in torch >= 2.2, we found two functions under torch._C has no __module__:
                    #   <built-in method _make_subclass of torch._C._TensorMeta>
                    #   <built-in method _make_wrapper_subclass of torch._C._TensorMeta>
                    if func.__module__ is not None and func.__module__ in sys.modules:
                        if func.__module__.startswith('_') and func.__module__ != '__main__':
                            path = sys.modules.get(func.__module__[1:], sys.modules[func.__module__])
                        else:
                            path = sys.modules[func.__module__]
                        path = getattr(path, func.__qualname__.split('.')[0])
                        locations = (*locations, wrap_utils.Location(path, func.__name__))
                    if len(locations) == 0:
                        _logger.warning(f'Can not find location of {func}, skip wrap it.')
                        continue
                    wrapped = wrap_utils.create_wrapped_leaf_func(
                        func,
                        replace_func=wrap_info.replacement,
                        default_tracer=self if wrap_info.is_force_trace else None,
                        is_method=True,
                        method_name=func.__name__,
                    )
                else:
                    # common function
                    # in torch >= 2.2, we found two functions under torch._C has no __module__:
                    #   <built-in method _make_subclass of torch._C._TensorMeta>
                    #   <built-in method _make_wrapper_subclass of torch._C._TensorMeta>
                    if func.__module__ is not None and func.__module__ in sys.modules:
                        if func.__module__.startswith('_') and func.__module__ != '__main__':
                            path = sys.modules.get(func.__module__[1:], sys.modules[func.__module__])
                        else:
                            path = sys.modules[func.__module__]
                        locations = (*locations, wrap_utils.Location(path, func.__name__))
                    if len(locations) == 0:
                        _logger.warning(f'Can not find location of {func}, skip wrap it.')
                        continue
                    wrapped = wrap_utils.create_wrapped_leaf_func(
                        func,
                        replace_func=wrap_info.replacement,
                        default_tracer=self if wrap_info.is_force_trace else None,
                    )
            wrapped_leaf_leaves[func] = (locations, wrapped)

        for clz, wrap_info in leaf_class.items():
            if clz.__module__.startswith('_') and clz.__module__ != '__main__':
                path = sys.modules.get(func.__module__[1:], sys.modules[func.__module__])
            else:
                path = sys.modules[clz.__module__]
            wrapped = wrap_utils.create_wrapped_leaf_class(
                clz,
                replace_cls=wrap_info.replacement,
                default_tracer=self if wrap_info.is_force_trace else None,
            )
            locations = (*wrap_info.extra_locs, wrap_utils.Location(path, clz.__name__))
            wrapped_leaf_leaves[clz] = (locations, wrapped)

        return wrapped_leaf_leaves

    @compatibility(is_backward_compatible=True)
    def trace(self, root: Union[torch.nn.Module, Callable[..., Any]], *,
              autowrap_leaf_function: Optional[Dict[Any, wrap_utils.LeafWrapInfo]] = None,
              autowrap_leaf_class: Optional[Dict[Type, wrap_utils.LeafWrapInfo]] = None,
              concrete_args: Union[Dict[str, Any], Tuple],
              use_operator_patch: bool = True,
              operator_patch_backlist: List[str] | None = None,
              forward_function_name: str = 'forward') -> Graph:
        """
        similar to _symbolic_trace.Tracer.trace
        different args:
            use_operator_patch:
                the operators 'not/is/is not/in/not in' cannot be wrapped after
                    compiled. so we re-parse the functions, replace these operators
                    with functions 'operator.not_/is_/is_not/contains', then we
                    could wrap and trace these.
                for example: in ``if x is None:``, if x is a proxy, the tracer will
                    never go into the branch, even x is a proxy with value 'None'.
                values:
                true: before executing a func, the func will be patched if the func
                    is not in operator_patch_backlist
                false: before executing a func, the func will be patched if the func
                    is in operator_patch_backlist

            operator_patch_backlist:
                such as '__main__.FooModel' or '__main__.bar_func'. the namespace is
                always needed.
        """
        if not isinstance(root, torch.nn.Module):
            # TODO: support trace any callable function by add the fill default values logic.
            raise RuntimeError('Only support trace a torch.nn.Module instance now.')

        self.root = self.strategy.place_model(root)

        # fill default values
        args = inspect.getfullargspec(getattr(root, forward_function_name)).args[1:]
        defaults = inspect.getfullargspec(getattr(root, forward_function_name)).defaults
        defaults = tuple() if defaults is None else defaults
        if isinstance(concrete_args, (tuple, list)):
            concrete_args, _ = self.strategy.place_inputs(concrete_args, {})
            concrete_args = (*concrete_args, *defaults[len(concrete_args) + len(defaults) - len(args):])
        else:
            _, concrete_args = self.strategy.place_inputs((), concrete_args)
            kv_default = {k: v for k, v in zip(args[-len(defaults):], defaults)}
            concrete_args = {
                **concrete_args,
                **{n: kv_default[n] for n in args if n not in concrete_args}
            }

        # preprocess arguments
        autowrap_leaf_function = autowrap_leaf_function if autowrap_leaf_function is not None else {}
        autowrap_leaf_class = autowrap_leaf_class if autowrap_leaf_class is not None else {}
        operator_patch_backlist = operator_patch_backlist if operator_patch_backlist is not None else []

        self.autowrap_leaf_function = {**autowrap_leaf_function, **wrap_utils.default_autowrap_leaf_function}
        self.autowrap_leaf_class = {**autowrap_leaf_class, **wrap_utils.default_autowrap_leaf_class}
        if isinstance(root, torch.nn.Module):
            self.root = root

            # TODO: better infomation
            assert hasattr(
                root, forward_function_name
            ), f"traced_func_name={forward_function_name} doesn't exist in {orig_func.type(root).__name__}"

            fn = getattr(root, forward_function_name)
        else:
            self.root = torch.nn.Module()
            fn = root

        tracer_cls = getattr(self, '__class__', None)
        self.graph = Graph(tracer_cls=tracer_cls)

        # When we encounter a Tensor value that's not a parameter, we look if it
        # is some other attribute on the model. Construct a dict mapping Tensor
        # values to the qualified name here for efficiency. This is used downstream
        # in create_arg
        self.tensor_attrs: Dict[Union[torch.Tensor, ScriptObject], str] = {}

        def collect_tensor_attrs(m: torch.nn.Module, prefix_atoms: List[str]):
            for k, v in m.__dict__.items():
                if isinstance(v, (torch.Tensor, ScriptObject)):
                    self.tensor_attrs[v] = '.'.join(prefix_atoms + [k])
            for k, v in m.named_children():
                collect_tensor_attrs(v, prefix_atoms + [k])

        collect_tensor_attrs(self.root, [])

        if isinstance(fn, MethodType):
            fn = fn.__func__
        assert isinstance(fn, FunctionType)

        fn_globals = fn.__globals__  # run before it gets patched

        fn, args, more_args, kwargs = self.create_args_for_root(fn, isinstance(root, torch.nn.Module), concrete_args)

        self.path_of_module = {id(v): k for k, v in self.root.named_modules()}
        self.path_of_parameter = {id(v): k for k, v in self.root.named_parameters()}
        self.path_of_buffer = {id(v): k for k, v in self.root.named_buffers()}

        # use to track the autograd function classes with the wrapped apply method
        # {autograd_function_class: wrapped_autograd_function_apply}
        self.autograd_functions_mapping: dict[Type, Any] = {}
        self.wrapped_leaf: Dict[Any, Tuple[Tuple[wrap_utils.Location,...], Any]] = self.get_wrapped_leaves(self.autowrap_leaf_function, self.autowrap_leaf_class)

        # for the customized functions, we need to revert all the wrapped function to the original one to run it
        # for the functions default wrapped, we don't revert to save time
        for func in autowrap_leaf_function:
            self.add_need_revert_function(func, self.wrapped_leaf.get(func, (None, None))[1])

        # wrap all forward in the submodule to trace the module stack
        # NOTE: temp disable the forward wrap, will add back later
        for mod in self.root.modules():
            wrapped = wrap_utils.create_wrapped_nn_module_func(self, mod, forward_function_name)
            self.wrapped_leaf[mod.forward] = ((wrap_utils.Location(mod, forward_function_name),), wrapped)

        try:
            with self.patcher:
                # allow duplicate patches to support the case of nested calls
                self.patcher.patch_method(torch.nn.Module, "__getattribute__", wrap_utils.create_wrapped_module_getattribute(self), deduplicate=False)

                self.patcher.patch_method(torch.nn.Module, "__call__", wrap_utils.create_wrapped_module_call(self), deduplicate=False)
                # for cuda versions of pytorch, autograd.Function.apply should be reverted by delattr
                self.patcher.patch_method(torch.autograd.Function, "apply", wrap_utils.create_wrapped_autograd_apply(self), deduplicate=False, revert_by_del=True)
                self.patcher.patch_method(torch, "_assert", wrap_utils.torch_assert_wrapper, deduplicate=False)

                self.patcher.patch_method(builtins, "map", wrap_utils.map_wrapper_clz, deduplicate=False)
                self.patcher.patch_method(builtins, "enumerate", wrap_utils.enumerate_wrapper_clz, deduplicate=False)
                self.patcher.patch_method(builtins, "range", wrap_utils.range_wrapper_clz, deduplicate=False)
                self.patcher.patch_method(builtins, "type", wrap_utils.type_wrapper_clz, deduplicate=False)
                self.patcher.patch_method(builtins, "isinstance", wrap_utils.isinstance_wrapper, deduplicate=False)
                self.patcher.patch_method(builtins, "issubclass", wrap_utils.issubclass_wrapper, deduplicate=False)
                self.patcher.patch_method(builtins, "getattr", wrap_utils.getattr_wrapper, deduplicate=False)

                for obj, (positions, wrapped) in self.wrapped_leaf.items():
                    for loc in positions:
                        self.patcher.patch_method(loc.ns, loc.name, wrapped, deduplicate=False)
                
                wrap_utils.autowrap_check(self, fn_globals)

                with OperatorPatcherContext(self, use_operator_patch, operator_patch_backlist):
                    results = OperatorPatcherContext.patch_run(fn, *args, *more_args, **kwargs)
                    # we should unwrap proxy to the original value in the results when we record it to node.meta['tensor_meta']
                    with wrap_utils.do_temp_call_origin():
                        def unwrap_nested_proxy(proxy: ep.ConcreteProxy):
                            return pytree_utils.tree_map_only(ep.ConcreteProxy, unwrap_nested_proxy, proxy.value)

                        node_result = pytree_utils.tree_map_only(ep.ConcreteProxy, unwrap_nested_proxy, results)
                    self.create_node('output', 'output', (self.create_arg(results),),
                                     {}, type_expr=fn.__annotations__.get('return', None), node_result=node_result)
        finally:
            _retain_weight_consistency(self.root)

        return self.graph


def update_tree_proxy_value(dst_pytree, src_pytree):
    """
    copy the value from src_pytree to dst_pytree with the dst_pytree spec,
    if the leaf is proxy, only replace the proxy.value, not replace the proxy.
    """
    # consider about this case:
    #   dst_pytree: {'a': [1, 2, 3]}
    #   src_pytree: {'a': [1, 2, 3, 4]}
    # then the public spec is {'a': *}, we don't want to flatten the list here.
    common_spec = pytree_utils.get_common_spec(pytree_utils.tree_structure(dst_pytree), pytree_utils.tree_structure(src_pytree))

    def update_proxy_value(a, b):
        if isinstance(a, ep.ConcreteProxy):
            a.value = update_tree_proxy_value(a.value, b)
            return a
        else:
            return b

    flat_dst_leaves = pytree_utils.tree_leaves_with_spec(dst_pytree, common_spec)
    flat_src_leaves = pytree_utils.tree_leaves_with_spec(src_pytree, common_spec)
    new_leaves = [update_proxy_value(dst_leaf, src_leaf) for dst_leaf, src_leaf in zip(flat_dst_leaves, flat_src_leaves)]
    return pytree_utils.tree_unflatten(new_leaves, common_spec)


@compatibility(is_backward_compatible=True)
class GraphAppendingConcreteTracer(ConcreteTracer):
    def __init__(self, graph: Graph):
        super().__init__()
        self.graph = graph


def _retain_weight_consistency(root: torch.nn.Module):
    _flag = 0
    for module in root.modules():
        for name, param in module.named_parameters():
            if orig_func.isinstance(param, ep.ConcreteProxy):
                param: ep.ConcreteProxy
                _logger.warning(f'Parameter {name} of {module} is a ConcreteProxy. Some weight may be modified inplace within forward().')
                setattr(module, name, param.value)
                _flag |= 1
        for name, buffer in module.named_buffers():
            if orig_func.isinstance(buffer, ep.ConcreteProxy):
                buffer: ep.ConcreteProxy
                _logger.warning(f'Buffer {name} of {module} is a ConcreteProxy. Some buffer may be modified inplace within forward().')
                setattr(module, name, buffer.value)
                _flag |= 1
    if _flag:
        _logger.warning('Some weight or buffer is modified inplace within forward(). This may cause unexpected behavior.'
                        ' ``concrete_trace`` may not guarantee the consistency of the traced graph.')
    return root


def concrete_trace(root : Union[torch.nn.Module, Callable[..., Any]],
                   concrete_args: Union[Dict[str, Any], Tuple],
                   *,
                   use_operator_patch: bool = True,
                   operator_patch_backlist: List[str] | None = None,
                   forward_function_name: str = 'forward',
                   check_args: Optional[Dict[str, Any]] = None,
                   autowrap_leaf_function: Optional[Dict[Any, wrap_utils.LeafWrapInfo]] = None,
                   autowrap_leaf_class: Optional[Dict[Type, wrap_utils.LeafWrapInfo]] = None,
                   dce: bool = True,
                   dce_ignored_function: Set[Callable] | None = None,
                   strategy: Literal['cpu', 'cuda', 'meta', 'cuda_run_cpu_offload', 'reuse_cache'] = 'cuda_run_cpu_offload',
                   trace_twice: bool = False,
                   record_frames: bool = False,
                   ) -> GraphModule:
    """
    Concrete tracing API

    Given an ``nn.Module`` or function instance ``root`` and a dummy input `concrete_args`, this function will return a ``GraphModule``
    constructed by recording operations seen while tracing through ``root``.

    It has solved many problems compared to fx.symbolic_trace, and can execute on many third-party models.

    For example::

        def f(a, b):
            return a + b

        traced_f = concrete_trace(f, concrete_args={'a': 1, 'b': 2})
        # or `traced_f = concrete_trace(f, (1, 2))`
        assert traced_f(3, 4) == 7

        def f(x):
            out1, out2 = 0, 0
            for k, v in x.items():
                out1 += k
                out2 += v
            return out1, out2
        traced_f = concrete_trace(f, ({1: 1, 2: 2}, ))
        assert traced_f({2: 3, 4: 5}) == (6, 8)

    Note that we can only record static structure, so all the branches such as if-else or loop will be flattened::

        def f(x):
            out1, out2 = 0, 0
            for k, v in x.items():
                out1 += k
                out2 += v
            return out1, out2
        traced_f = concrete_trace(f, ({1: 1, 2: 2}, ))
        assert traced_f({2: 3, 4: 5, 6:7}) == (6, 8) # not (12, 15)

        # traced code like:
        def traced_f(self, x):
            out1, out2 = 0, 0
            items = x.items()

            # for loop
            iter = iter(items)

            # first loop content
            items0 = next(iter)
            out1 += items0[0]
            out2 += items0[1]

            # second loop content
            items1 = next(iter)
            out1 += items1[0]
            out2 += items1[1]

            return (out1, out2)

    If you want to trace 'is', 'is not', 'in' or 'not in' in your module, you can set use_function_patch to True::

        def f(x, y):
            if x is None:
                return y
            else:
                return x - y
        # traced_f = concrete_trace(f, (None, 1)) # bad
        traced_f = concrete_trace(f, (None, 1), use_function_patch=True) # f should exist in a file.

    If you have a function/method that should be treated as a leaf function but not trace into it, use autowrap_leaf_function to mark it::

        def leaf_op(x, y, z):
            # if not treated as a leaf function, then only 1 branch will exist.
            if x > 0:
                return y + z
            else:
                return y - z

        def f(x):
            return leaf_op(x, 3, 2)

        traced_f = concrete_trace(f, (1, ), autowrap_leaf_function = {
            leaf_op: ([], False, None), **ConcreteTracer.default_autowrap_leaf_function})
        assert traced_f(1) == 5 and traced_f(-1) == 1

    If you have a class that should be treated as a leaf class, use autowrap_leaf_class to mark it::

        class leaf_clz:
            def __init__(self, a, b):
                self.c = a + b

        def f(x, y):
            return leaf_clz(x, y)

        traced_f = concrete_trace(f, (1, 2), autowrap_leaf_class = {
            leaf_clz: ([], False), **ConcreteTracer.default_autowrap_leaf_class})
        assert isinstance(traced_f(3, 4), leaf_clz) and traced_f(3, 4).c == 7

    Args:
        root (Union[torch.nn.Module, Callable]): Module or function to be traced and converted into a Graph representation.
        concrete_args (Union[Dict[str, Any], Tuple]): Dummy inputs to do concrete trace.

        use_function_patch (bool): Use operator patcher recursively on function calls. Operator patcher will re-compile the function and
            translate '{} is {}' into 'operator.is_({}, {})', then we can treat 'is', 'is not', 'in' and 'not in' as function calls.

        operator_patch_backlist (List[str]): Blacklist of the operator patcher.

        autowrap_leaf_function (Dict[Any, LeafFnWrapInfo]): Leaf function dict,
            such as 'add' or 'torch.xxx'. You can add your own leaf functions.

        autowrap_leaf_class: (Dict[Type, LeafClassWrapInfo]): Leaf class dict, such as 'int',
            'range' or 'zip'. You can add your own leaf functions such as 'modeling_outputs.SequenceClassifierOutput'.

        dce (bool): If set to True, dead code eliminatation will be applied on the graph.

        dce_ignored_function (Set[Callable]): The node that its target in this set will not be removed from the graph during dce.

        strategy (Literal['cpu', 'cuda', 'meta', 'cuda_run_cpu_offload', 'reuse_cache']):
            The device placement strategy for intermediate results and module parameters/buffer, and run target.
            The following strategies are supported:
                'cpu': Execute all functions on cpu, model weights and intermediate results are on cpu.
                `cuda': Execute all functions on cuda, model weights and intermediate results are on cuda.
                    This strategy is recommended if the model can inference on single gpu.
                'meta': Execute all functions on meta, model weights are on cpu and intermediate results are on meta.
                'cuda_run_cpu_offload': Try to execute all functions on cuda, and retry to execute the function on cpu as backup if meet OOM error,
                    model weights and intermediate results are on cpu. This strategy is recommanded for most case if the model is too large to inference on single gpu.
                'reuse_cache': Similar to `cuda_run_cpu_offload` strategy, additional add a buffer to cache all the intermediate results with different function signatures on cpu,
                    function with same signature exist in cache directly take the cached result as this time function execution to save time.
                    Same signature means the funtions are the same and have almost the same inputs
                    (for tensor type input, just check if they have same tensor meta data[shape, dtyep, requires_grad, stride, memory_format, ...], and don't check the value).
                    This strategy is an experimental strategy to speedup the large-model-large-input case,
                    and have risk to trace an incorrect graph if the signature defined here can not distinguish the differnet functions used in the model,
                    for example, torch.nonzero will always return the same result if the input have same meta data but different value.
                    We have plan to continue improve this strategy to handle most these kind of data dependence cases, but please note that the risk is still inevitable.

        trace_twice (bool): If set to True, a second trace will be performed, and the two obtained graphs will be checked for consistency.

        record_frames (bool): If set to True, will add frame information to node.meta['frame_record']. Note this will cost additional trace time.

    Returns:
        fx.GraphModule: a Module created from the recorded operations from ``root``.
    """
    dce_ignored_function = dce_ignored_function if isinstance(dce_ignored_function, set) else set()
    assert all(callable(ignore_func) for ignore_func in dce_ignored_function)

    tracer = ConcreteTracer(strategy = strategy, record_frames = record_frames)

    graph = tracer.trace(root,
        autowrap_leaf_function = autowrap_leaf_function,
        autowrap_leaf_class = autowrap_leaf_class,
        concrete_args = concrete_args,
        use_operator_patch = use_operator_patch,
        operator_patch_backlist = operator_patch_backlist,
        forward_function_name = forward_function_name,
    )

    if trace_twice:
        graph_check = tracer.trace(root,
            autowrap_leaf_function = autowrap_leaf_function,
            autowrap_leaf_class = autowrap_leaf_class,
            concrete_args = concrete_args,
            use_operator_patch = use_operator_patch,
            operator_patch_backlist = operator_patch_backlist,
            forward_function_name = forward_function_name,
        )
        # compare to check equal
        assert len(graph.nodes) == len(graph_check.nodes), f'number nodes: {len(graph.nodes)} vs {len(graph_check.nodes)}'
        for node_a, node_b in zip(graph.nodes, graph_check.nodes):
            node_a: Node
            node_b: Node
            target_a = node_a.target
            target_b = node_b.target
            if node_a.op == 'get_attr' and node_a.name.startswith('_tensor_constant'):
                assert node_b.op == 'get_attr' and node_b.name.startswith('_tensor_constant')
                assert torch.equal(getattr(root, node_a.name), getattr(root, node_b.name))
            elif node_a.op == 'call_function' and wrap_utils.is_autograd_apply(target_a):
                assert node_b.op == 'call_function' and wrap_utils.is_autograd_apply(target_b)
            else:
                assert node_a.op == node_b.op and target_a == target_b, f'op: {node_a.op} vs {node_b.op}, target: {target_a} vs {target_b}'

    with TorchFXPatcher():
        name = root.__class__.__name__ if isinstance(root, torch.nn.Module) else root.__name__
        traced = GraphModule(tracer.root, graph, name)

        if dce:
            # some side effectful functions that should not be deleted during dead code elimination
            # there may be more than listed here
            default_extra_side_effectful_functions = {
                builtins.next,
                *side_effectful_inplace_ops
            }
            extra_side_effectful_functions = default_extra_side_effectful_functions | dce_ignored_function
            with ExtraSEFPatcher(extra_side_effectful_functions):
                traced.graph.eliminate_dead_code()
            traced.recompile()  # this need to be done in TorchFXPatcher context

    # TODO: better infomation
    if check_args is not None:
        assert root(**check_args) == traced(**check_args)

    return traced
