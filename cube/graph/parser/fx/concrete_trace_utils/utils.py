# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import builtins
from dataclasses import dataclass
import operator
from typing import Any, Callable, Dict, NamedTuple, Optional, Set, Tuple, Type
import functools

import torch
from torch.fx.node import Node, map_aggregate, _side_effectful_functions

# These need to run in global scope to handle nested calls correctly
_orig_module_call: Callable = torch.nn.Module.__call__
_orig_module_getattr: Callable = torch.nn.Module.__getattr__
_orig_module_getattribute: Callable = torch.nn.Module.__getattribute__

_orig_agfunc_apply: Callable = torch.autograd.function.Function.apply
_orig_torch_assert: Callable = torch._assert
_orig_torch_no_grad: Callable = torch.no_grad
_orig_torch_no_grad_enter: Callable = torch.no_grad.__enter__
_orig_torch_no_grad_exit: Callable = torch.no_grad.__exit__

_orig_type: Callable = builtins.type
_orig_isinstance: Callable = builtins.isinstance
_orig_issubclass: Callable = builtins.issubclass
_orig_getattr: Callable = builtins.getattr

_orig_range: Type[Any] = builtins.range
_orig_int: Type[Any] = builtins.int
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


def run_onlyif_instance(cond_type: Type[Any], return_orig: bool = True, return_const: Any = None):
    def helper(fn):
        if return_orig:
            @functools.wraps(fn)
            def wrapper_orig(*args):
                if _orig_isinstance(args[-1], cond_type):
                    return fn(*args)
                return args[-1]
            return wrapper_orig
        else:
            @functools.wraps(fn)
            def wrapper_const(*args):
                if _orig_isinstance(args[-1], cond_type):
                    return fn(*args)
                return return_const
            return wrapper_const
    return helper

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

def map_recursive_zip(fn: Callable, arg0, *args) -> Any:
    """
    Apply fn to each Node appearing arg. arg may be a list, tuple, slice, or dict with string keys.
    """
    if _orig_type(arg0) != torch.Size and _orig_isinstance(arg0, _orig_tuple):
        for arg in args:
            assert (not _orig_isinstance(arg, torch.Size)) and _orig_isinstance(arg, _orig_tuple)
            assert len(arg0) == len(arg)
        return _orig_tuple(map_recursive_zip(fn, *sub_args) for sub_args in _orig_zip(arg0, *args))
    elif _orig_isinstance(arg0, _orig_list):
        for arg in args:
            assert _orig_isinstance(arg, _orig_list)
            assert len(arg0) == len(arg)
        return _orig_list(map_recursive_zip(fn, *sub_args) for sub_args in _orig_zip(arg0, *args))
    elif _orig_isinstance(arg0, _orig_dict):
        keys = _orig_set(arg0.keys())
        for arg in args:
            assert _orig_isinstance(arg, _orig_dict) and len(keys.symmetric_difference(arg.keys())) == 0
        return {k: map_recursive_zip(fn, arg0[k], *(arg[k] for arg in args)) for k in keys}
    else:
        # assert not _orig_isinstance(arg0, slice)
        return fn(arg0, *args)


@dataclass
class FrameRecord:
    filename: str
    lineno: str
    line: str
    name: str

    def __repr__(self) -> str:
        if self.filename:
            return f'File "{self.filename}", line {self.lineno}, in {self.name},  {self.line}'
        else:
            return ''


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
        meta = map_aggregate(results, extract_tensor_metadata)
        node.meta['tensor_meta'] = meta
        node.meta['type'] = type(results)


class EmptyResult:
    """Used for identification no results.
    """
    pass
