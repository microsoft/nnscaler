#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from typing import Any, Dict, NamedTuple, Optional, Tuple
from dataclasses import dataclass, asdict

import torch
from torch.fx.node import Node

from . import pytree_utils

DICT_KEYS_TYPE = type({}.keys())
DICT_VALUES_TYPE= type({}.values())
DICT_ITEMS_TYPE= type({}.items())


class EmptyResult:
    """
    Used for identification no results.
    """
    pass


@dataclass
class GradMode:
    grad_mode: bool
    no_grad_mode: bool
    inference_mode: bool

    @classmethod
    def from_context(cls):
        return cls(torch.is_grad_enabled(), not torch.is_grad_enabled(), torch.is_inference_mode_enabled())


@dataclass
class AutocastInfo:
    # the nesting number of autocast context, if =0, means it is not under autocast context
    # torch use this field to determine whether the cache needs to be cleaned
    # nnscaler use this field to determine whether generating autocast context manager in code
    nesting: int

    cache_enabled: bool
    cpu_enabled: bool
    cpu_dtype: torch.dtype
    cuda_enabled: bool
    cuda_dtype: torch.dtype
    # NOTE: not care about "xpu" and "hpu" now

    @classmethod
    def from_context(cls):
        # use function pair [torch.autocast_increment_nesting, torch.autocast_decrement_nesting] to get the nesting number
        torch.autocast_increment_nesting()
        return cls(torch.autocast_decrement_nesting(),  torch.is_autocast_cache_enabled(),
                   torch.is_autocast_cpu_enabled(), torch.get_autocast_cpu_dtype(),
                   torch.is_autocast_enabled(), torch.get_autocast_gpu_dtype())


@dataclass
class OpContext:
    """
    OpContext is a dataclass that holds the context of an operation.

    Args:
        constant_folding: Whether constant folding is enabled.
        Please note we will not unfold/fold inputs
            when we enter the code block with different constant folding setting.
    """
    constant_folding: Optional[bool] = None


_GLOBAL_OP_CONTEXT = OpContext()


def get_op_context() -> OpContext:
    """
    Get op context information.
    Please note that current only tracked context managers that modify the tensor properties, for example, modify the requires_grad, dtype,
    so that nnscaler can generate context manager code for them safety.
    """
    return asdict(_GLOBAL_OP_CONTEXT) | {'grad_mode': asdict(GradMode.from_context()), 'autocast_info': asdict(AutocastInfo.from_context())}


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

    return TensorMetadata(shape, dtype, requires_grad, stride, memory_format, is_quantized, qparams)


def extract_metadata(results: Any, node: Node):
    if results is not EmptyResult:
        res = tuple(results) if isinstance(results, (DICT_KEYS_TYPE, DICT_VALUES_TYPE, DICT_ITEMS_TYPE)) else results
        meta = pytree_utils.tree_map_only(torch.Tensor, _extract_tensor_metadata, res)
        # we should get the meta info of the inner element of these type obj
        if isinstance(results, DICT_KEYS_TYPE):
            meta = {m: i for i, m in enumerate(meta)}.keys()
        if isinstance(results, DICT_VALUES_TYPE):
            meta = {i: m for i, m in enumerate(meta)}.values()
        if isinstance(results, DICT_ITEMS_TYPE):
            meta = {i: m for i, m in meta}.items()
        node.meta['tensor_meta'] = meta
        node.meta['type'] = type(results)

    node.meta['op_context'] = get_op_context()
