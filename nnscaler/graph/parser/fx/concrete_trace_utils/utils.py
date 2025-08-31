# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import builtins
from collections import namedtuple
from dataclasses import dataclass
import importlib
import operator
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, NamedTuple, Optional, Set, Tuple, Type, List

import torch
import torch.utils._pytree as torch_pytree
from torch.fx.node import Node, map_aggregate, _side_effectful_functions
from torch.utils._pytree import tree_flatten, tree_unflatten, LeafSpec, TreeSpec, SUPPORTED_NODES

from . import concrete_proxy as ep

DICT_KEYS_TYPE = type({}.keys())
DICT_VALUES_TYPE= type({}.values())
DICT_ITEMS_TYPE= type({}.items())


# These need to run in global scope to handle nested calls correctly
_orig_module_call: Callable = torch.nn.Module.__call__
_orig_module_getattr: Callable = torch.nn.Module.__getattr__
_orig_module_getattribute: Callable = torch.nn.Module.__getattribute__

_orig_agfunc_apply: Callable = torch.autograd.function.Function.apply
_orig_torch_assert: Callable = torch._assert

_orig_type: Callable = builtins.type
_orig_isinstance: Callable = builtins.isinstance
_orig_issubclass: Callable = builtins.issubclass
_orig_getattr: Callable = builtins.getattr

_orig_range: Type[Any] = builtins.range
_orig_int: Type[Any] = builtins.int
_orig_float: Type[Any] = builtins.float
_orig_bool: Type[Any] = builtins.bool
_orig_tuple: Type[Any] = builtins.tuple
_orig_list: Type[Any] = builtins.list
_orig_set: Type[Any] = builtins.set
_orig_frozenset: Type[Any] = builtins.frozenset
_orig_dict: Type[Any] = builtins.dict
_orig_map: Type[Any] = builtins.map
_orig_zip: Type[Any] = builtins.zip
_orig_enumerate: Type[Any] = builtins.enumerate
_orig_slice: Type[Any] = builtins.slice
_orig_reversed: Type[Any] = builtins.reversed

_orig_torch_size: Type[Any] = torch.Size
_orig_torch_finfo: Type[Any] = torch.finfo

_orig_len: Callable = builtins.len
_orig_not: Callable = operator.not_
_orig_is: Callable = operator.is_
_orig_is_not: Callable = operator.is_not
_orig_contains: Callable = operator.contains
_orig_index: Callable = operator.index

_orig_all: Callable = builtins.all
_orig_min: Callable = builtins.min
_orig_max: Callable = builtins.max

_orig_node_is_impure: Callable = Node.is_impure

side_effectful_inplace_ops = {
    operator.iadd, operator.isub, operator.imul, operator.itruediv, operator.ifloordiv,
    operator.iand, operator.ior, operator.ixor, operator.ilshift, operator.irshift,
    operator.imod, operator.ipow,
    # operator.imatmul is not implemented in torch
    # so let's ignore it now
    operator.setitem,
}


def map_recursive(fn: Callable, arg) -> Any:
    """
    Apply fn to each Node appearing arg. arg may be a list, tuple, slice, or dict with string keys.
    """
    if _orig_type(arg) != torch.Size and _orig_isinstance(arg, _orig_tuple):
        t = _orig_tuple(map_recursive(fn, elem) for elem in arg)
        # Support NamedTuple (if it has `_fields`) by repacking into original type.
        return t if not hasattr(arg, '_fields') else _orig_type(arg)(*t)
    elif _orig_isinstance(arg, _orig_list):
        return _orig_list(map_recursive(fn, elem) for elem in arg)
    elif _orig_isinstance(arg, _orig_dict):
        return {k: map_recursive(fn, v) for k, v in arg.items()}
    else:
        return fn(arg)


def _get_node_type(pytree: Any) -> Any:
    if isinstance(pytree, ep.ConcreteProxy):
        return _orig_type(pytree)
    if torch_pytree._is_namedtuple_instance(pytree):
        return namedtuple
    return type(pytree)

torch_pytree._get_node_type = _get_node_type


def get_common_spec(dst_spec: TreeSpec, src_sepc: TreeSpec) -> TreeSpec:
    """
    Return the common part of two treespec.
    For example:
        dst_spec is {'a': [*,], 'b': [*, *]}
        src_sepc is {'a': [*,], 'b': [*, *, *]}
        common spec is {'a': [*,], 'b': *}
    """
    if isinstance(dst_spec, LeafSpec) or isinstance(src_sepc, LeafSpec):
        return LeafSpec()
    if dst_spec.type == src_sepc.type and dst_spec.context == src_sepc.context:
        if len(dst_spec.children_specs) == len(src_sepc.children_specs):
            children_specs = [get_common_spec(dst, src) for dst, src in zip(dst_spec.children_specs, src_sepc.children_specs)]
            return TreeSpec(type=dst_spec.type, context=dst_spec.context, children_specs=children_specs)
    return LeafSpec()


def flatten_trees_with_func(fn, pytrees) -> Tuple[List[Any], TreeSpec]:
    """
    Each pytree in pytrees should have the same structure.

    Example:

        pytrees = [
            [1, 2, (3, 4)], # pytree 1
            [5, 6, (7, 8)], # pytree 2
        ]

        # the returned value is
        [fn(1, 5), fn(2, 6), fn(3, 7), fn(4, 8)], [*, *, (*, *)]
    """
    flat_trees = [tree_flatten(pytree) for pytree in pytrees]
    flat_args = [v[0] for v in flat_trees]
    specs = [v[1] for v in flat_trees]

    if not all(len(flat_arg) == len(flat_args[0]) for flat_arg in flat_args):
        raise RuntimeError('the element number of pytrees are not equal')
    if not all(str(spec) == str(specs[0]) for spec in specs):
        raise RuntimeError('the structure of pytrees are not equal')

    return [fn(*vals) for vals in zip(*flat_args)], specs[0]


def map_trees_with_func(fn, pytrees):
    """
    Each pytree in pytrees should have the same structure.
    The returned value has the same structure with pytree in pytrees.

    Example:

        pytrees = [
            [1, 2, (3, 4)], # pytree 1
            [5, 6, (7, 8)], # pytree 2
        ]

        # the returned value is
        [fn(1, 5), fn(2, 6), (fn(3, 7), fn(4, 8))]
    """
    flat_args, spec = flatten_trees_with_func(fn, pytrees)
    return tree_unflatten([i for i in flat_args], spec)


def flatten_tree_with_spec(pytree, spec: TreeSpec) -> List:
    """
    Flat a pytree with a given spec.

    Example:

        pytree = [1, (2, {3: 4})]
        spec = TreeSpec([*, (*, *)])
    
        # the returned value is
        [1, 2, {3: 4}]
    """
    assert isinstance(spec, TreeSpec)

    if isinstance(spec, LeafSpec):
        return [pytree]

    flatten_fn = SUPPORTED_NODES[spec.type].flatten_fn
    child_pytrees, _ = flatten_fn(pytree)

    if len(child_pytrees) != len(spec.children_specs):
        raise RuntimeError(f'The number of pytree children is not equal to the give specs.')

    result = []
    for child, child_spec in zip(child_pytrees, spec.children_specs):
        flat = flatten_tree_with_spec(child, child_spec)
        result += flat

    return result


def flatten_trees_with_func_and_spec(fn, pytrees, spec):
    """
    Example:

        pytrees = [
            [1, (2, {3: 4})],
            [5, (6, 7)]
        ]
        spec = [*, (*, *)]

        # the returned value is
        [fn(1, 5), fn(2, 6), fn({3: 4}, 7)]
    """
    flat_args = [flatten_tree_with_spec(pytree, spec) for pytree in pytrees]
    if not all(len(flat_arg) == len(flat_args[0]) for flat_arg in flat_args):
        raise RuntimeError('the element number of pytrees are not equal')
    return [fn(*vals) for vals in zip(*flat_args)]


class ExtraSEFPatcher:
    def __init__(self, extra_side_effectful_functions: Set[Callable]):
        self.extra_side_effectful_functions = extra_side_effectful_functions
        self.incontext_funcs = set()

    def __enter__(self):
        self.incontext_funcs = self.extra_side_effectful_functions - _side_effectful_functions
        _side_effectful_functions.update(self.incontext_funcs)

    def __exit__(self, exc_type, exc_val, exc_tb):
        _side_effectful_functions.difference_update(self.incontext_funcs)


class TensorMetadata(NamedTuple):
    # TensorMetadata is a structure containing pertinent information
    # about a tensor within a PyTorch program.

    # General Tensor metadata
    shape : torch.Size
    dtype : torch.dtype
    requires_grad : bool
    stride : Tuple[int]
    memory_format : Optional[torch.memory_format]

    # Quantization metadata
    is_quantized : bool
    qparams: Dict[str, Any]


def _extract_tensor_metadata(result: torch.Tensor) -> TensorMetadata:
    """
    Extract a TensorMetadata NamedTuple describing `result`.
    """
    shape = result.shape
    dtype = result.dtype
    requires_grad = result.requires_grad
    stride = result.stride()

    memory_formats = {
        torch.contiguous_format,
        torch.channels_last,
        torch.channels_last_3d,
    }

    memory_format = None

    for query_format in memory_formats:
        if result.is_contiguous(memory_format=query_format):
            memory_format = query_format
            break

    is_quantized = result.is_quantized
    qparams: Dict[str, Any] = {}
    if is_quantized:
        qscheme = result.qscheme()
        qparams["qscheme"] = qscheme
        if qscheme in {torch.per_tensor_affine, torch.per_tensor_symmetric}:
            qparams["scale"] = result.q_scale()  # type: ignore[assignment]
            qparams["zero_point"] = result.q_zero_point()  # type: ignore[assignment]
        elif qscheme in {torch.per_channel_affine, torch.per_channel_affine_float_qparams, torch.per_channel_symmetric}:
            # In this branch, scale and zero_point are expected to be tensors,
            # we store the values as immutable_list in TensorMetadata for
            # easier serialization downstream
            qparams["scale"] = result.q_per_channel_scales().tolist()  # type: ignore[assignment]
            qparams["zero_point"] = result.q_per_channel_zero_points().tolist()  # type: ignore[assignment]
            qparams["axis"] = result.q_per_channel_axis()  # type: ignore[assignment]

    return TensorMetadata(
        shape, dtype, requires_grad, stride, memory_format, is_quantized, qparams)


def extract_tensor_metadata(obj: Any):
    if isinstance(obj, torch.Tensor):
        return _extract_tensor_metadata(obj)
    else:
        return obj


def extract_results_metadata(results: Any, node: Node):
    if results is not EmptyResult:
        res = tuple(results) if isinstance(results, (DICT_KEYS_TYPE, DICT_VALUES_TYPE, DICT_ITEMS_TYPE)) else results
        meta = map_aggregate(res, extract_tensor_metadata)
        # we should get the meta info of the inner element of these type obj
        if isinstance(results, DICT_KEYS_TYPE):
            meta = {i: m for i, m in enumerate(meta)}.keys()
        if isinstance(results, DICT_VALUES_TYPE):
            meta = {i: m for i, m in enumerate(meta)}.values()
        if isinstance(results, DICT_ITEMS_TYPE):
            meta = {i: m for i, m in meta}.items()
        node.meta['tensor_meta'] = meta
        node.meta['type'] = type(results)


class EmptyResult:
    """Used for identification no results.
    """
    pass


@dataclass
class FrameRecord:
    filename: str
    lineno: str
    line: str
    # the name of the frame is the function name
    name: str

    def __repr__(self) -> str:
        if self.filename:
            return f'File "{self.filename}", line {self.lineno}, in {self.name},  {self.line}'
        else:
            return ''


def get_frame_record() -> Optional[FrameRecord]:
    # record code frame, include filename, line number, and function name
    frame_record = None
    cube_path = str(Path(importlib.util.find_spec('nnscaler').origin).parent) + '/'  # the cube path
    torch_path = str(Path(importlib.util.find_spec('torch').origin).parent) + '/'  # the torch path
    ignore_dirs = [cube_path, torch_path]
    # the last frame is the current frame [get_frame_record], so we need to skip it
    for frame in traceback.extract_stack()[-2::-1]:
        if any(p in frame.filename for p in ignore_dirs):
            continue
        frame_record = FrameRecord(frame.filename, frame.lineno, frame.line, frame.name)
        break
    return frame_record
