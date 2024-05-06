# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Any, Callable, List, Tuple, Dict, Union, Iterable
import string
import copy
import torch
import operator
import numpy as np
import math
import logging
from collections.abc import Iterable

from cube.ir.cten import IRTensor, IRObject
from cube.ir.tensor import IRSubTensor, IRFullTensor
from cube.graph.function.pyfunc import IRPyFunc
from cube.graph.function.dimops import DimopSplit, ShapeAnno, OpAnno, IRDimops, TransformRule
from cube.graph.function.conv import IRPad, IRConv2D, IRConv3D
from cube.graph.function.anchor import IRGraphAnchor

_logger = logging.getLogger(__name__)


def Identity(tensor: IRObject, signature = None):
    signature = 'cube.runtime.function.identity'
    eshape = ShapeAnno.create_shape_str(tensor.shape)
    anno = OpAnno.create_op_str([eshape], [eshape])
    return IRDimops(Identity, 'identity', signature, [anno], [tensor])


def MultiRef(tensor: IRTensor, times: int, signature = None):
    """
    cube.runtime.function.multiref(itensor: torch.Tensor, times: int) -> Tuple[torch.Tensor]
    """
    signature = 'cube.runtime.function.multiref'
    assert isinstance(tensor, IRTensor), "require all inputs to be IRSubTensor"
    assert isinstance(times, int), "require int for second input"
    anno = '* -> ' + ', '.join('*' for _ in range(times))
    node = IRDimops(MultiRef, 'multiref', signature, [anno], [tensor], times=times)
    return node


def Accum(*inputs, signature = None):
    """
    tensor = cube.runtime.function.accum(tensors)
    """
    assert all(isinstance(t, IRTensor) for t in inputs)
    signature = 'cube.runtime.function.accum'
    iannos = [ShapeAnno.create_shape_str(t.shape) for t in inputs]
    oannos = [copy.copy(iannos[0])]
    anno = OpAnno.create_op_str(iannos, oannos)
    return IRDimops(Cat, 'accum', signature, [anno], inputs)


def Linear(input, weight, bias=None, signature = None):
    signature = 'torch.nn.functional.linear'
    if bias is None:
        annos = ['b * k+, n k+ -> b * n']
        return IRDimops(Linear, 'linear', signature, annos, [input, weight], bias=None)
    else:
        annos = ['b * k^, n k^, n -> b * n']
        _logger.warning(
            'detected a linear operator has bias, the partition on reduction dimension is disabled.')
        return IRDimops(Linear, 'linear', signature, annos, [input, weight, bias])


def BatchLinear(input, mat2, *, out=None, signature = None):
    assert out is None
    annos = ['b m k+, b k+ n -> b m n']
    return IRDimops(BatchLinear, 'bmm', signature, annos, [input, mat2])


def BMMAdd(input, batch1, batch2, *, beta=1, alpha=1, out=None, signature = None):
    """
    torch.baddbmm(input, batch1, batch2, *, beta=1, alpha=1, out=None)
    """
    assert out is None
    in_dims = ['b', 'm', 'n']
    assert len(input.shape) == 3
    for i, size in enumerate(input.shape):
        if size == 1:
            in_dims[i] = '1'
    in_anno = ' '.join(in_dims)
    anno = f'{in_anno}, b m k^, b k^ n -> b m n'
    return IRDimops(BMMAdd, 'baddbmm', signature, [anno], [input, batch1, batch2], alpha=alpha, beta=beta)


def CubeEinSum(*operands, equation=None, signature = None):
    assert isinstance(equation, str)
    signature = 'cube.runtime.function.einsum'
    lhs, rhs = equation.split('->')
    assert ',' not in rhs
    lhs_dims = set(lhs.replace(',', ' ').split(' '))
    for dim in lhs_dims:
        if dim not in rhs:
            lhs = lhs.replace(dim, f'{dim}+')
    anno = f'{lhs} -> {rhs}'
    return IRDimops(CubeEinSum, 'einsum', signature, [anno], operands, equation=equation)

def EinSum(equation: str, *operands, signature = None):
    return CubeEinSum(*operands, equation=equation, signature=signature)


def Matmul(input, other, *, out=None, signature=None):
    """
    torch.matmul
    _operator.matmul
    """
    signature = 'torch.matmul'
    assert out is None
    annos = [
        'm k+, k+ n -> m n',
        'k+, k+ n -> n',
        'm k+, k+ -> m',
        '* m k+, k+ n -> * m n',
        '* m k+, * k+ n -> * m n'  # TODO: broadcast
    ]
    if len(input.shape) > 2 and len(other.shape) > 2:
        assert tuple(input.shape[:-2]) == tuple(other.shape[:-2]), "broadcast of matmul (bmm) is not supported"
    return IRDimops(Matmul, 'matmul', signature, annos, [input, other])


# =============================================== creators ==========================================

def _get_creator_anno_rules(size: Tuple[int], partitionable: bool) -> str:
    """
    Create annotation and transformation rules for creator
    """
    eshape = [str(dimlen) + ('' if partitionable else '^') for dimlen in size]
    anno = OpAnno.create_op_str([], [eshape])
    rules = []
    if partitionable:
        for dim in range(len(size)):
            def creator_modifier(kwargs: Dict, idx, dim, num: int) -> Dict:
                kwargs = dict(**kwargs)
                size = list(kwargs['size'])
                size[dim] = size[dim] // num
                kwargs['size'] = tuple(size)
                return kwargs
            rules.append(TransformRule([], [DimopSplit.D(dim)], creator_modifier))
    return anno, rules


def CubeArange(start: Union[int, IRObject], end: Union[int, IRObject], step: Union[int, IRObject],
               dtype=None, requires_grad=False, signature=None):
    dtype = dtype if dtype is not None else torch.get_default_dtype()
    assert isinstance(dtype, torch.dtype), f"only supports torch.dtype but got {dtype}"
    signature = 'cube.runtime.function.arange'
    kwargs = {'start': start, 'end': end, 'step': step,
              'dtype': dtype, 'requires_grad': requires_grad}
    start_val = start.value if isinstance(start, IRObject) else start
    end_val = end.value if isinstance(end, IRObject) else end
    step_val = step.value if isinstance(step, IRObject) else step
    size = (math.ceil((end_val-start_val)/step_val),)
    anno, rules = _get_creator_anno_rules(
        tuple(dim.value if isinstance(dim, IRObject) else dim for dim in size), False)
    dimop = IRDimops(CubeArange, 'arange', signature, [anno], [], rules, **kwargs)
    dimop.output(0).parent.dtype = dtype
    return dimop


def Arange(*args, out=None, dtype=None, layout=None,
           device=None, requires_grad=False, signature=None):
    """
    torch.arange(start=0, end, step=1, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
    """
    assert layout is None
    if len(args) == 1:
        start, end, step = 0, args[0], 1
    elif len(args) == 2:
        start, end, step = args[0], args[1], 1
    elif len(args) == 3:
        start, end, step = args
    else:
        raise RuntimeError(f'Invalid number {len(args)} of args in Arange.')
    return CubeArange(start, end, step, dtype, requires_grad=requires_grad)


def Empty(size, *arg_size, out=None, dtype=None, layout=None, device=None, requires_grad=False,
          pin_memory=False, memory_format=None, signature=None):
    # note: device is ignored
    assert layout in (None, torch.strided) and memory_format is None, f"Not support for non-default memory_format and layout"
    dtype = dtype if dtype is not None else torch.get_default_dtype()
    assert isinstance(dtype, torch.dtype), f"only supports torch.dtype but got {dtype}"
    signature = 'cube.runtime.function.empty'
    size = (size,) if isinstance(size, (int, IRObject)) else tuple(size)
    size: Tuple[Union[int, IRObject]] = size + arg_size
    kwargs = {'size': size, 'requires_grad': requires_grad,
              'dtype': dtype, 'pin_memory': pin_memory}
    anno, rules = _get_creator_anno_rules(
        tuple(dim.value if isinstance(dim, IRObject) else dim for dim in size), True)
    dimop = IRDimops(Empty, 'empty', signature, [anno], [], rules, **kwargs)
    dimop.output(0).parent.dtype = dtype
    return dimop


def Zeros(size, *arg_size, out=None, dtype=None, layout=None,
          device=None, requires_grad=False, signature=None):
    # note: device is ignored
    assert layout in (None, torch.strided), f"Not support for non-strided layout, get {layout}"
    dtype = dtype if dtype is not None else torch.get_default_dtype()
    assert isinstance(dtype, torch.dtype), f"only supports torch.dtype but got {dtype}"
    signature = 'cube.runtime.function.zeros'
    size = (size,) if isinstance(size, (int, IRObject)) else tuple(size)
    size: Tuple[Union[int, IRObject]] = size + arg_size
    kwargs = {'size': size, 'requires_grad': requires_grad, 'dtype': dtype}
    anno, rules = _get_creator_anno_rules(
        tuple(dim.value if isinstance(dim, IRObject) else dim for dim in size), True)
    dimop = IRDimops(Zeros, 'zeros', signature, [anno], [], rules, **kwargs)
    dimop.output(0).parent.dtype = dtype
    return dimop


def Ones(size, *arg_size, out=None, dtype=None, layout=None,
         device=None, requires_grad=False, signature=None):
    # note: device is ignored
    assert layout in (None, torch.strided), f"Not support for non-strided layout, get {layout}"
    dtype = dtype if dtype is not None else torch.get_default_dtype()
    assert isinstance(dtype, torch.dtype), f"only supports torch.dtype but got {dtype}"
    signature = 'cube.runtime.function.ones'
    size = (size,) if isinstance(size, (int, IRObject)) else tuple(size)
    size: Tuple[Union[int, IRObject]] = size + arg_size
    kwargs = {'size': size, 'requires_grad': requires_grad, 'dtype': dtype}
    anno, rules = _get_creator_anno_rules(
        tuple(dim.value if isinstance(dim, IRObject) else dim for dim in size), True)
    dimop = IRDimops(Ones, 'ones', signature, [anno], [], rules, **kwargs)
    dimop.output(0).parent.dtype = dtype
    return dimop


def Rand(size, *arg_size, out=None, dtype=None, layout=None, device=None, requires_grad=False,
         pin_memory=False, memory_format=None, signature=None):
    # note: device is ignored
    assert layout in (None, torch.strided) and memory_format is None, f"Not support for non-default memory_format and layout"
    dtype = dtype if dtype is not None else torch.get_default_dtype()
    assert isinstance(dtype, torch.dtype), f"only supports torch.dtype but got {dtype}"
    signature = 'cube.runtime.function.rand'
    size = (size,) if isinstance(size, (int, IRObject)) else tuple(size)
    size: Tuple[Union[int, IRObject]] = size + arg_size
    kwargs = {'size': size, 'requires_grad': requires_grad,
              'dtype': dtype, 'pin_memory': pin_memory}
    anno, rules = _get_creator_anno_rules(
        tuple(dim.value if isinstance(dim, IRObject) else dim for dim in size), True)
    dimop = IRDimops(Rand, 'rand', signature, [anno], [], rules, **kwargs)
    dimop.output(0).parent.dtype = dtype
    return dimop


def Full(size, fill_value, *, out=None, dtype=None, layout=None,
         device=None, requires_grad=False, signature=None):
    """
    torch.full(size, fill_value, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
    """
    assert layout in (None, torch.strided), f"Not support for non-default layout"
    dtype = dtype if dtype is not None else torch.get_default_dtype()
    signature = 'cube.runtime.function.full'
    # cube treat scalar as size (1,) tensor now, scalar support will in another pr if necessary
    size = tuple(size) if size else (1,)
    anno, rules = _get_creator_anno_rules(
        tuple(dim.value if isinstance(dim, IRObject) else dim for dim in size), True)
    dimop = IRDimops(Full, 'full', signature, [anno], [], rules,
                     size=size, fill_value=fill_value, dtype=dtype, requires_grad=requires_grad)
    dimop.output(0).parent.dtype = dtype
    return dimop


def NewTensor(data, *, dtype=None, device=None,
              requires_grad=False, pin_memory=False, signature=None):
    # note: device is ignored
    dtype = dtype if dtype is not None else torch.get_default_dtype()
    signature = 'cube.runtime.function.tensor'
    size = tuple(np.array(data).shape) if np.array(data).shape else (1,)  # (1,) means it is a scalar
    kwargs = {'size': size, 'requires_grad': requires_grad,
              'dtype': dtype, 'pin_memory': pin_memory}
    anno, rules = _get_creator_anno_rules(size, True)
    dimop = IRDimops(NewTensor, 'tensor', signature, [anno], [], rules, **kwargs)
    dimop.output(0).parent.dtype = dtype
    return dimop


def _handle_broadcast(lhs: IRTensor, rhs: IRTensor) -> Tuple[List[str]]:
    """Create shape annotations for element wise operator following broadcastable rules:
    https://pytorch.org/docs/stable/notes/broadcasting.html

    Args:
        lhs IRTensor: the lhs input tensor
        rhs IRTensor: the rhs input tensor

    Returns:
        lhs_anno List[str]: lhs shape annotation
        rhs_anno List[str]: rhs shape annotation
        out_anno List[str]: output shape annotation
    """
    ins_anno, out_anno = _handle_broadcast_multi([lhs, rhs])
    assert len(ins_anno) == 2
    return ins_anno[0], ins_anno[1], out_anno


def _handle_broadcast_multi(ins_list: List[IRTensor]) -> Tuple[Tuple[List[str]], List[str]]:
    """Similar to ``_handle_broadcast``, handle broadcast for more than two input tensors.

    Create shape annotations for element wise operator following broadcastable rules:
    https://pytorch.org/docs/stable/notes/broadcasting.html

    Args:
        ins_list List[IRTensor]: the list of input tensors

    Returns:
        ins_anno (Tuple[List[str]]): a list of input tensors annotation
        out_anno (List[str]): output shape annotation
    """
    assert len(ins_list) >= 2, 'at least two tensor require for broadcast'
    ins_ndims = [len(inp.shape) for inp in ins_list]
    # init annotation string
    maxlen_shape = ins_list[ins_ndims.index(max(ins_ndims))].shape
    shape_anno = ShapeAnno.create_shape_str(maxlen_shape)
    ins_anno = [shape_anno[-ndims:] for ndims in ins_ndims]
    # expand dimensions for empty dimensions
    ins_ofst = [max(ins_ndims) - ndims for ndims in ins_ndims]
    ins_shape = [[1] * ins_ofst[idx] + list(inp.shape) for idx, inp in enumerate(ins_list)]
    # init out_shape
    out_anno = []
    for dim in range(len(maxlen_shape)):
        dim_annos = [None if dim - ins_ofst[idx] < 0 else anno[dim-ins_ofst[idx]] for idx, anno in enumerate(ins_anno)]
        not_none_annos = [anno for anno in dim_annos if anno is not None]
        assert len(not_none_annos) > 0
        not_none_anno = not_none_annos[0]
        if all(x[dim] == ins_shape[0][dim] for x in ins_shape):
            out_anno.append(not_none_anno)
        else:
            not_one_shape = [shape[dim] for shape in ins_shape if shape[dim] != 1]
            if len(not_one_shape) > 0 and not all(s == not_one_shape[0] for s in not_one_shape):
                raise ValueError(f"cannot broadcast tensor list: {ins_list}")
            else:
                out_anno.append(not_none_anno)
                for idx, anno in enumerate(dim_annos):
                    if anno is not None and ins_shape[idx][dim] == 1:
                        ins_anno[idx][dim-ins_ofst[idx]] = '1'
    return ins_anno, out_anno


def Expand(input, *sizes, size = None, signature = None):
    """
    torch.Tensor.expand(*sizes)

    The reason of add ``size`` to this function argument is:
    1. ``sizes`` need to reuse in IRDimops.new(), but it is a ``non-keyword arguments``,
    and can not put it into keyword arguments (something like Expand(input, sizes=[1, 2, 3])) is not work,
    to support IRDimops.new API, here add a ``size`` to workaround.

    2. in torch._C.expand API, it has:
        def expand(self, size: Sequence[Union[_int, SymInt]], *, implicit: _bool=False) -> Tensor: ...
      so add ``size`` can also solve user using something like:
        torch.rand(3, 1).expand(size=(3, 3))
    """
    signature = 'torch.Tensor.expand'
    if size is not None:
        assert len(sizes) == 0
        sizes = size
    ori_len, exp_len = len(input.shape), len(sizes)
    assert ori_len <= exp_len
    assert all(dim == expand_dim or dim == 1 or expand_dim == -1 for dim, expand_dim in zip(input.shape, sizes[-ori_len:]))
    edim_ou = ShapeAnno.create_shape_str(sizes)
    edim_in = copy.copy(edim_ou[-ori_len:])
    new_size = [-1] * len(sizes)
    for idx, (dim, expand_dim) in enumerate(zip(input.shape, sizes[-len(input.shape):])):
        if dim == 1 and dim != expand_dim and expand_dim != -1:
            edim_in[idx] += '^'
            edim_ou[exp_len - ori_len + idx] = str(expand_dim)
            new_size[exp_len - ori_len + idx] = expand_dim
    for idx in range(exp_len - ori_len):
        edim_ou[idx] = str(sizes[idx])
        new_size[idx] = sizes[idx]
    anno = OpAnno.create_op_str([edim_in], [edim_ou])
    return IRDimops(Expand, 'expand', signature, [anno], [input], size=new_size)


def ExpandAs(input, other, signature = None):
    return Expand(input, *other.shape, signature = signature)


def Clone(input, *, memory_format=None, signature = None):
    """
    torch.clone(input, *, memory_format=torch.preserve_format)
    """
    assert memory_format is None, f"Not supported for a specific memory format"
    annos = ['* -> *']
    return IRDimops(Clone, 'clone', signature, annos, [input])


def BitwiseOr(input, other, *, out=None, signature=None):
    """
    torch.bitwise_or(input, other, *, out=None) → Tensor
    """
    assert out is None
    if (not isinstance(input, IRObject)) and (not isinstance(other, IRObject)):
        return input | other
    assert isinstance(input, IRTensor) and isinstance(other, IRTensor)
    annos = ['*, * -> *']
    return IRDimops(BitwiseOr, 'bitwise_or', signature, annos, [input, other])


def BitwiseNot(input, *, out=None, signature=None):
    assert out is None
    if not isinstance(input, IRObject):
        return ~input
    assert isinstance(input, IRTensor)
    annos = ['* -> *']
    return IRDimops(BitwiseNot, 'bitwise_not', signature, annos, [input])


def Add(input, other, alpha=1, *, out=None, signature = None):
    assert out is None
    if (not isinstance(input, IRObject)) and (not isinstance(other, IRObject)):
        return input + alpha * other
    if (not isinstance(input, IRTensor)) and (not isinstance(other, IRTensor)):
        iv = input.value if isinstance(input, IRObject) else input
        ov = other.value if isinstance(other, IRObject) else other
        return IRPyFunc(signature, [input, other], [IRObject(name='add', value=iv+ov)])
    signature = 'torch.add'
    annos = ['*, ? -> *', '?, * -> *',]
    if isinstance(input, IRTensor) and isinstance(other, IRTensor):
        lshape, rshape, oshape = _handle_broadcast(input, other)
        annos = [OpAnno.create_op_str([lshape, rshape], [oshape])]
    return IRDimops(Add, 'add', signature, annos, [input, other], alpha=alpha)


def Sub(input, other, alpha=1, *, out=None, signature = None):
    assert out is None
    signature = 'torch.sub'
    if (not isinstance(input, IRObject)) and (not isinstance(other, IRObject)):
        return input - alpha * other
    if (not isinstance(input, IRTensor)) and (not isinstance(other, IRTensor)):
        iv = input.value if isinstance(input, IRObject) else input
        ov = other.value if isinstance(other, IRObject) else other
        return IRPyFunc(signature, [input, other], [IRObject(name='sub', value=iv-ov)])
    annos = ['*, ? -> *', '?, * -> *',]
    if isinstance(input, IRTensor) and isinstance(other, IRTensor):
        lshape, rshape, oshape = _handle_broadcast(input, other)
        annos = [OpAnno.create_op_str([lshape, rshape], [oshape])]
    return IRDimops(Sub, 'sub', signature, annos, [input, other], alpha=alpha)


def Mul(input, other, *, out=None, signature = None):
    assert out is None
    if (not isinstance(input, IRObject)) and (not isinstance(other, IRObject)):
        return input * other
    if (not isinstance(input, IRTensor)) and (not isinstance(other, IRTensor)):
        iv = input.value if isinstance(input, IRObject) else input
        ov = other.value if isinstance(other, IRObject) else other
        return IRPyFunc(signature, [input, other], [IRObject(name='mul', value=iv*ov)])
    signature = 'torch.mul'
    annos = ['*, ? -> *', '?, * -> *',]
    if isinstance(input, IRTensor) and isinstance(other, IRTensor):
        lshape, rshape, oshape = _handle_broadcast(input, other)
        annos = [OpAnno.create_op_str([lshape, rshape], [oshape])]
    return IRDimops(Mul, 'mul', signature, annos, [input, other])


def Mod(input, other, *, out = None, signature = None):
    assert out is None
    if (not isinstance(input, IRObject)) and (not isinstance(other, IRObject)):
        return input % other
    signature = 'torch.fmod'
    annos = ['*, ? -> *']
    if isinstance(input, IRTensor) and isinstance(other, IRTensor):
        lshape, rshape, oshape = _handle_broadcast(input, other)
        annos = [OpAnno.create_op_str([lshape, rshape], [oshape])]
    return IRDimops(Mod, 'mod', signature, annos, [input, other])


def Div(input, other, *, rounding_mode=None, out=None, signature = None):
    assert rounding_mode is None and out is None
    if (not isinstance(input, IRObject)) and (not isinstance(other, IRObject)):
        return input / other
    if (not isinstance(input, IRTensor)) and (not isinstance(other, IRTensor)):
        iv = input.value if isinstance(input, IRObject) else input
        ov = other.value if isinstance(other, IRObject) else other
        return IRPyFunc(signature, [input, other], [IRObject(name='div', value=iv/ov)])
    signature = 'torch.div'
    annos = ['*, ? -> *', '?, * -> *',]
    if isinstance(input, IRTensor) and isinstance(other, IRTensor):
        lshape, rshape, oshape = _handle_broadcast(input, other)
        annos = [OpAnno.create_op_str([lshape, rshape], [oshape])]
    return IRDimops(Div, 'div', signature, annos, [input, other], rounding_mode=rounding_mode)


def Exp(input, *, out=None, signature=None):
    """
    torch.exp(input, *, out=None)
    """
    assert out is None
    if not isinstance(input, IRObject):
        return torch.exp(input)
    if not isinstance(input, IRTensor):
        assert input.value is not None
        return IRPyFunc(signature, [input], [IRObject(name='exp', value=torch.exp(input.value))])
    shape = ShapeAnno.create_shape_str(input.shape)
    annos = [OpAnno.create_op_str([shape], [shape])]
    return IRDimops(Exp, 'exp', signature, annos, [input])


def Sqrt(input, *, out=None, signature=None):
    """
    torch.sqrt(input, *, out=None)
    """
    assert out is None
    if not isinstance(input, IRObject):
        return torch.sqrt(input)
    if not isinstance(input, IRTensor):
        iv = input.value if isinstance(input, IRObject) else input
        return IRPyFunc(signature, [input], [IRObject(name='sqrt', value=torch.sqrt(iv))])
    shape = ShapeAnno.create_shape_str(input.shape)
    annos = [OpAnno.create_op_str([shape], [shape])]
    return IRDimops(Sqrt, 'sqrt', signature, annos, [input])


def RSqrt(input, *, out=None, signature=None):
    assert out is None
    if not isinstance(input, IRObject):
        return torch.rsqrt(input)
    if not isinstance(input, IRTensor):
        iv = input.value if isinstance(input, IRObject) else input
        return IRPyFunc(signature, [input], [IRObject(name='rsqrt', value=torch.rsqrt(iv))])
    shape = ShapeAnno.create_shape_str(input.shape)
    annos = [OpAnno.create_op_str([shape], [shape])]
    return IRDimops(RSqrt, 'rsqrt', signature, annos, [input])


def FloorDiv(input, other, *, out=None, signature = None):
    assert out is None
    if (not isinstance(input, IRObject)) and (not isinstance(other, IRObject)):
        return input // other
    if (not isinstance(input, IRTensor)) and (not isinstance(other, IRTensor)):
        iv = input.value if isinstance(input, IRObject) else input
        ov = other.value if isinstance(other, IRObject) else other
        return IRPyFunc(signature, [input, other], [IRObject(name='fdiv', value=iv//ov)])
    annos = ['*, ? -> *', '?, * -> *',]
    if isinstance(input, IRTensor) and isinstance(other, IRTensor):
        lshape, rshape, oshape = _handle_broadcast(input, other)
        annos = [OpAnno.create_op_str([lshape, rshape], [oshape])]
    return IRDimops(FloorDiv, 'floordiv', signature, annos, [input, other])


def Pow(input, exponent, *, out=None, signature = None):
    assert out is None
    if (not isinstance(input, IRObject)) and (not isinstance(exponent, IRObject)):
        return input ** exponent
    if (not isinstance(input, IRTensor)) and (not isinstance(exponent, IRTensor)):
        iv = input.value if isinstance(input, IRObject) else input
        ev = exponent.value if isinstance(exponent, IRObject) else exponent
        return IRPyFunc(signature, [input, exponent], [IRObject(name='pow', value=iv**ev)])
    annos = ['*, ? -> *', '?, * -> *',]
    if isinstance(input, IRTensor) and isinstance(exponent, IRTensor):
        lshape, rshape, oshape = _handle_broadcast(input, exponent)
        annos = [OpAnno.create_op_str([lshape, rshape], [oshape])]
    return IRDimops(Pow, 'pow', signature, annos, [input, exponent])


def Neg(input, *, out=None, signature = None):
    assert out is None
    if not isinstance(input, IRObject): return -1 * input
    if not isinstance(input, IRTensor):
        iv = input.value if isinstance(input, IRObject) else input
        return IRPyFunc(signature, [input], [IRObject(name='neg', value=-iv)])
    annos = ['* -> *']
    return IRDimops(Neg, 'neg', signature, annos, [input])


def Sin(input, *, out=None, signature = None):
    assert out is None
    annos = ['* -> *']
    return IRDimops(Sin, 'sin', signature, annos, [input])


def Cos(input, *, out=None, signature = None):
    assert out is None
    annos = ['* -> *']
    return IRDimops(Cos, 'cos', signature, annos, [input])


def Tanh(input, *, out=None, signature = None):
    assert out is None
    annos = ['* -> *']
    return IRDimops(Tanh, 'tanh', signature, annos, [input])


def GeLU(input, approximate='none', signature = None):
    annos = ['* -> *']
    signature = 'torch.nn.functional.gelu'
    return IRDimops(GeLU, 'gelu', signature, annos, [input], approximate=approximate)


def SiLU(input, inplace=False, signature = None):
    annos = ['* -> *']
    signature = 'torch.nn.functional.silu'
    return IRDimops(SiLU, 'silu', signature, annos, [input], inplace=inplace)


def ReLU(input, inplace=False, signature = None):
    annos = ['* -> *']
    signature = 'torch.nn.functional.relu'
    return IRDimops(ReLU, 'relu', signature, annos, [input], inplace=inplace)


def Abs(input, *, out=None, signature = None):
    assert out is None
    annos = ['* -> *']
    return IRDimops(Abs, 'abs', signature, annos, [input])


def Clamp(input, min=None, max=None, *, out=None, signature = None):
    assert out is None
    annos = ['* -> *']
    return IRDimops(Clamp, 'clamp', signature, annos, [input], min=min, max=max)


def ClampMin(input, min, *, out=None, signature = None):
    return Clamp(input, min=min, out=out, signature='torch.clamp')


def Softmax(input, dim=None, _stacklevel=3, dtype=None, signature = None):
    """
    torch.nn.functional.softmax(input, dim=None, _stacklevel=3, dtype=None)
    """
    edim_in = ShapeAnno.create_shape_str(input.shape)
    edim_ou = copy.copy(edim_in)
    if dim is not None:
        edim_in[dim] += '^'
        edim_ou[dim] += '^'
    anno = OpAnno.create_op_str([edim_in], [edim_ou])
    return IRDimops(Softmax, 'softmax', signature, [anno], [input],
                    dim=dim, _stacklevel=_stacklevel, dtype=dtype)


def LogSoftmax(input, dim=None, _stacklevel=3, dtype=None, signature=None):
    """
    torch.nn.functional.log_softmax(input, dim=None, _stacklevel=3, dtype=None)
    """
    edim_in = ShapeAnno.create_shape_str(input.shape)
    edim_ou = copy.copy(edim_in)
    if dim is not None:
        edim_in[dim] += '^'
        edim_ou[dim] += '^'
    anno = OpAnno.create_op_str([edim_in], [edim_ou])
    return IRDimops(LogSoftmax, 'log_softmax', signature, [anno], [input],
                    dim=dim, _stacklevel=_stacklevel, dtype=dtype)


def Dropout(input, p=0.5, training=True, inplace=False, signature = None):
    """
    torch.nn.functional.dropout(input, p=0.5, training=True, inplace=False)
    """
    annos = ['* -> *']
    return IRDimops(Dropout, 'dropout', signature, annos, [input],
                    p=p, training='self.training', inplace=inplace)


def nnDropout(input, p=0.5, inplace=False, signature=None):
    """
    torch.nn.Dropout(p=0.5, inplace=False)
    """
    signature = 'cube.runtime.function.nndropout'
    annos = ['* -> *']
    return IRDimops(nnDropout, 'Dropout', signature, annos, [input],
                    p=p, inplace=inplace)


def Detach(input, signature = None):
    """
    torch.Tensor.detach(input)
    """
    annos = ['* -> *']
    return IRDimops(Detach, 'detach', signature, annos, [input])


def NanToNum(input, nan=0.0, posinf=None, neginf=None, *, out=None, signature = None):
    assert out is None
    annos = ['* -> *']
    return IRDimops(NanToNum, 'nan_to_num', signature, annos, [input], nan=nan, posinf=posinf, neginf=neginf)


def Long(input, memory_format=None, signature = None):
    """
    torch.Tensor.long(memory_format=torch.preserve_format)
    """
    assert memory_format is None
    annos = ['* -> *']
    return IRDimops(Long, 'long', signature, annos, [input])


def Int(input, memory_format=None, signature = None):
    """
    Tensor.int(memory_format=torch.preserve_format) → Tensor
    """
    assert memory_format is None
    annos = ['* -> *']
    return IRDimops(Int, 'int', signature, annos, [input])


def Float(input, memory_format=None, signature = None):
    """
    Tensor.float(memory_format=torch.preserve_format) → Tensor
    """
    assert memory_format is None
    annos = ['* -> *']
    return IRDimops(Float, 'float', signature, annos, [input])


def Bool(input, memory_format=None, signature = None):
    """
    torch.Tensor.bool(memory_format=torch.preserve_format)
    """
    assert memory_format is None
    annos = ['* -> *']
    return IRDimops(Bool, 'bool', signature, annos, [input])


def Fill(input, value, signature = None):
    """
    torch.Tensor.fill_(value)
    """
    edim_in = ShapeAnno.create_shape_str(input.shape)
    edim_ou = copy.copy(edim_in)
    anno = OpAnno.create_op_str([edim_in], [edim_ou])
    return IRDimops(Fill, 'fill', signature, [anno], [input], value=value)


def MaskedFill(input, mask, value, signature = None):
    """
    torch.Tensor.masked_fill_(mask, value)
    """
    edim_in0 = ShapeAnno.create_shape_str(input.shape)
    edim_in1 = ShapeAnno.create_shape_str(mask.shape)
    edim_ou = copy.copy(edim_in0)
    #TODO: add broadcast rule
    for idx, (lhs, rhs) in enumerate(zip(input.shape, mask.shape)):
        if lhs != rhs and rhs == 1:
            edim_in1[idx] = '1'
    anno = OpAnno.create_op_str([edim_in0, edim_in1], [edim_ou])
    return IRDimops(MaskedFill, 'masked_fill', signature, [anno], [input, mask], value=value)


def Where(condition, input, other, *, out=None, signature = None):
    """
    torch.where
    """
    assert isinstance(condition, IRTensor)
    if isinstance(input, IRTensor) and isinstance(other, IRTensor):
        (edim_in0, edim_in1, edim_in2), edim_out = _handle_broadcast_multi([condition, input, other])
    elif isinstance(input, IRTensor) and len(input.shape) > 0 and not (len(input.shape) == 1 and input.shape[0] == 1):
        edim_in0, edim_in1, edim_out = _handle_broadcast(condition, input)
        edim_in2 = ['?']
    elif isinstance(other, IRTensor) and len(other.shape) > 0 and not (len(other.shape) == 1 and other.shape[0] == 1):
        edim_in0, edim_in2, edim_out = _handle_broadcast(condition, other)
        edim_in1 = ['?']
    else:
        edim_in0 = ShapeAnno.create_shape_str(condition.shape)
        edim_in1, edim_in2 = ['?'], ['?']
        edim_out = copy.copy(edim_in0)

    annos = [OpAnno.create_op_str([edim_in0, edim_in1, edim_in2], [edim_out])]
    dimop = IRDimops(Where, 'where', signature, annos, [condition, input, other])
    return dimop

def CubeLayerNorm(input, weight=None, bias=None, normalized_shape=None, eps=1e-05, signature = None):
    """
    cube.runtime.function.layer_norm(input, weight, bias, normliazed_shape, eps)
    """
    signature = 'cube.runtime.function.layer_norm'
    assert not (weight is None and bias is not None), f"Not support for None of weight and parameter of bias"
    letters = iter(string.ascii_lowercase)
    einput = ShapeAnno.create_shape_str(input.shape, iterator=letters)
    eoutput = copy.copy(einput)
    ndims = len(input.shape)
    for dim in range(len(normalized_shape)):
        einput[ndims-1-dim] += '^'
        eoutput[ndims-1-dim] += '^'
    einputs, inputs = [einput], [input]
    kwargs = {}
    if weight is not None:
        eweight = ShapeAnno.create_shape_str(weight.shape, reduction='^', iterator=letters)
        einputs.append(eweight)
        inputs.append(weight)
    else:
        kwargs['weight'] = weight
    if bias is not None:
        ebias = ShapeAnno.create_shape_str(bias.shape, reduction='^', iterator=letters)
        einputs.append(ebias)
        inputs.append(bias)
    else:
        kwargs['bias'] = bias
    anno = OpAnno.create_op_str(einputs, [eoutput])
    kwargs['normalized_shape'] = normalized_shape
    kwargs['eps'] = eps
    return IRDimops(CubeLayerNorm, 'layernorm', signature, [anno], inputs, **kwargs)


def LayerNorm(input, normalized_shape, weight=None, bias=None, eps=1e-05, signature = None):
    """
    torch.nn.functional.layer_norm(input, normliazed_shape, weight=None, bias=None, eps)
    """
    return CubeLayerNorm(input, weight, bias, normalized_shape, eps, signature=signature)


def Norm(input, p='fro', dim=None, keepdim=False, out=None, dtype=None, signature=None):
    assert dtype is None, "Currently Norm only support dtype=None"
    einput = ShapeAnno.create_shape_str(input.shape)
    eoutput = copy.copy(einput)
    kwargs = {
        'p': p,
        'dim': dim,
        'keepdim': keepdim,
        'out': out,
        'dtype': dtype,
    }
    if dim is None:
        einput = [edim + '^' for edim in einput]
        anno = OpAnno.create_op_str([einput], ['1'])
        return IRDimops(Norm, 'norm', signature, [anno], [input], **kwargs)
    else:
        dim = (dim,) if isinstance(dim, int) else dim
        for dimidx in dim:
            einput[dimidx] += '^'
        if keepdim:
            for dimidx in dim:
                eoutput[dimidx] = '1'
        else:
            sort_dim = list(dim)
            sort_dim.sort()
            for dimidx in sort_dim[::-1]:
                eoutput.pop(dimidx)
        anno = OpAnno.create_op_str([einput], [eoutput])
        return IRDimops(Norm, 'norm', signature, [anno], [input], **kwargs)


def Sum(input, dim=None, keepdim=False, *, dtype=None, signature = None):
    """
    torch.sum(input, *, dtype=None) -> Tensor
    torch.sum(input, dim, keepdim=False, *, dtype=None) -> Tensor

    @note troch.sum is overrided by two signatures, which may lead mismatch in torch.jit.script:
        may get (input, dtype) as input
    """
    assert dtype is None, "Currently Sum only support dtype=None"
    einput = ShapeAnno.create_shape_str(input.shape)
    eoutput = copy.copy(einput)
    if dim is None:
        einput = [edim + '+' for edim in einput]
        anno = OpAnno.create_op_str([einput], ['1'])
        return IRDimops(Sum, 'sum', signature, [anno], [input])
    else:
        dim = (dim,) if isinstance(dim, int) else dim
        for dimidx in dim:
            einput[dimidx] += '+'
        if keepdim:
            for dimidx in dim:
                eoutput[dimidx] = '1'
        else:
            sort_dim = list(dim)
            sort_dim.sort()
            for dimidx in sort_dim[::-1]:
                eoutput.pop(dimidx)
        anno = OpAnno.create_op_str([einput], [eoutput])
        return IRDimops(Sum, 'sum', signature, [anno], [input], dim=dim, keepdim=keepdim)


def Mean(input, dim=None, keepdim=False, *, dtype=None, signature = None):
    """
    torch.mean(input, *, dtype=None)
    torch.mean(input, dim=None, keepdim=False, *, dtype=None)
    torch.Tensor.mean(input, dim=None, keepdim=False, *, dtype=None)
    """
    assert dtype is None
    einput = ShapeAnno.create_shape_str(input.shape)
    eoutput = copy.copy(einput)
    if dim is not None:
        dims = (dim,) if isinstance(dim, int) else tuple(dim)
        dims = tuple(dim % len(input.shape) for dim in dims)
        for dim in sorted(dims, reverse=True):
            einput[dim] += '^'
            if keepdim:
                eoutput[dim] = '1'
            else:
                eoutput.pop(dim)
        dim = dims
    else:
        eoutput = ['1']
        einput = [edim + '^' for edim in einput]
    anno = OpAnno.create_op_str([einput], [eoutput])
    if dim is not None:
        return IRDimops(Mean, 'mean', signature, [anno], [input], dim=dim, keepdim=keepdim)
    else:
        return IRDimops(Mean, 'mean', signature, [anno], [input])


def Transpose(input, dim0, dim1, signature = None):
    """
    out = torch.transpose(tensor, dim0, dim1)
    """
    edim_in = ShapeAnno.create_shape_str(input.shape)
    edim_ou = copy.copy(edim_in)
    edim_ou[dim0], edim_ou[dim1] = edim_ou[dim1], edim_ou[dim0]
    anno = OpAnno.create_op_str([edim_in], [edim_ou])
    return IRDimops(Transpose, 'transpose', signature, [anno], [input],
                    dim0=dim0, dim1=dim1)


def Split(tensor, split_size_or_sections, dim = 0, signature = None):
    """
    torch.functional.split(tensor, split_size_or_sections, dim=0) -> List[Tensor]
    """
    if isinstance(split_size_or_sections, int):
        sections = [split_size_or_sections for _ in range(tensor.shape[dim] // split_size_or_sections)]
        if tensor.shape[dim] % split_size_or_sections != 0:
            sections.append(tensor.shape[dim] % split_size_or_sections)
    else:
        sections = split_size_or_sections
    assert sum(sections) == tensor.shape[dim]
    edim_in = ShapeAnno.create_shape_str(tensor.shape)
    edim_ous = [copy.copy(edim_in) for _ in sections]
    edim_in[dim] = str(tensor.shape[dim])
    for edim_ou, dimlen in zip(edim_ous, sections):
        edim_ou[dim] = str(dimlen)
    anno = OpAnno.create_op_str([edim_in], edim_ous)
    return IRDimops(Split, 'split', signature, [anno], [tensor], split_size_or_sections=split_size_or_sections, dim=dim)


def Contiguous(input, memory_format = None, signature = None):
    """
    torch.Tensor.contiguous(Tensor self) -> Tensor
    """
    assert memory_format is None
    anno = ['* -> *']
    signature = 'torch.Tensor.contiguous'
    return IRDimops(Contiguous, 'contiguous', signature, anno, [input])


def _reshape_anno(in_shape: List[int], ou_shape: List[int], kwarg_name: str) -> Tuple[str, List[TransformRule]]:
    """
    reshape / view annotation and transformation rule generator

    Args:
        in_shape List[int]: input shape
        ou_shape List[int]: output shape
        kwarg_name str: kwarg name of reshape / view op

    Returns:
        str: annotation string
        List[TransformRule]: transformation rules
    """
    def nele(shape, nele=1):
        for dimlen in shape: nele *= dimlen
        return nele

    # infer -1
    cnt = nele(in_shape)
    if -1 in ou_shape:
        idx = ou_shape.index(-1)
        ou_shape[idx] = cnt // (-nele(ou_shape))
    assert nele(in_shape) == nele(ou_shape), f"shape mismatch: {in_shape}, {ou_shape}"

    # generate annotation
    rest_inshape = [dimlen for dimlen in in_shape]
    rest_oushape = [dimlen for dimlen in ou_shape]
    chain = []
    can_bucket = True
    while len(rest_inshape) != 0 or len(rest_oushape) != 0:
        if len(rest_inshape) == 0:
            chain = chain + rest_oushape
            rest_oushape = []
        elif len(rest_oushape) == 0:
            chain = chain + rest_inshape
            rest_inshape = []
        else:
            dimlen = min(rest_inshape[0], rest_oushape[0])
            if max(rest_inshape[0], rest_oushape[0]) % dimlen == 0:
                chain.append(dimlen)
                if dimlen == rest_inshape[0]:
                    rest_inshape.pop(0)
                else:
                    rest_inshape[0] = rest_inshape[0] // dimlen
                if dimlen == rest_oushape[0]:
                    rest_oushape.pop(0)
                else:
                    rest_oushape[0] = rest_oushape[0] // dimlen
            else:
                can_bucket = False
                break

    letters = iter(string.ascii_lowercase)
    if can_bucket:
        inchain = ouchain = chain
        inedims = ouedims = edims = [next(letters) for _ in chain]
    else:
        inchain, ouchain = in_shape, ou_shape
        inedims = [str(dimlen) for dimlen in in_shape]
        ouedims = [str(dimlen) for dimlen in ou_shape]
        chain = inchain + ouchain
        edims = inedims + ouedims
    shape_map: Dict[str, int] = {edim: eshape for (edim, eshape) in zip(edims, chain)}

    # generate input and output shape annotations
    # greedy fuse suffix number
    def buckets(shape: List[int], chain: List[int], edims: List[int]) -> List[List[str]]:
        anno = []
        dimidx = 0
        for idx, dimlen in enumerate(shape):
            elements, bracket = 1, []
            maxele = len(chain) - dimidx - (len(shape) - 1 - idx)
            while True:
                if len(bracket) == maxele:
                    assert elements == dimlen, f"internal match error1: {bracket}"
                    break
                if dimidx >= len(chain) or elements * chain[dimidx] > dimlen:
                    assert elements == dimlen, f"internal match error2: {bracket}"
                    break
                else:
                    elements *= chain[dimidx]
                    bracket.append(edims[dimidx])
                    dimidx += 1
            # fetch as many 1^ as possible from tail of the previous bracket
            if len(bracket) == 0:
                assert dimlen == 1, f"internal match error3: dimlen={dimlen}"
                back = 0
                for edim in anno[-1][1:][::-1]:
                    if chain[edims.index(edim)] != 1:
                        break
                    back += 1
                assert back > 0, f"internal match error4: dimlen={dimlen}"
                bracket = anno[-1][-back:]
                anno[-1] = anno[-1][:-back]
            assert len(bracket) > 0, f"got a dimension with no edim"
            anno.append(bracket)
        return anno

    in_anno = buckets(in_shape, inchain, inedims)
    ou_anno = buckets(ou_shape, ouchain, ouedims)

    # postprocess on dimlen == 1
    shape_map['1'] = 1
    for bracket in in_anno + ou_anno:
        for subdim, edim in enumerate(bracket):
            if shape_map[edim] == 1:
                bracket[subdim] = str(shape_map[edim])

    # find out the axis that can be partitioned
    ispatial, ifirst = set(), []
    for bracket in in_anno:
        sdim = None
        for hdim in range(len(bracket)):
            if bracket[hdim] == '1' or shape_map[bracket[hdim]] == 1: continue
            sdim = bracket[hdim]
            break
        if sdim is not None:
            ispatial.add(sdim)
        ifirst.append(sdim)

    ospatial, ofirst = set(), []
    for bracket in ou_anno:
        sdim = None
        for hdim in range(len(bracket)):
            if bracket[hdim] == '1' or shape_map[bracket[hdim]] == 1: continue
            sdim = bracket[hdim]
            break
        if sdim is not None:
            ospatial.add(sdim)
        ofirst.append(sdim)

    # intersection for spatial partitioned dimensions
    spatial = ispatial.intersection(ospatial)

    # set dimension cannot be partitioned
    for bracket in in_anno + ou_anno:
        for hdim in range(len(bracket)):
            if bracket[hdim] not in spatial:
                bracket[hdim] = str(shape_map[bracket[hdim]])

    def modifier(kwargs: Dict, idx, dim, num: int) -> Dict:
        kwargs = dict(**kwargs)
        identifier = ifirst[dim]
        oidx = ofirst.index(identifier)
        if isinstance(kwargs[kwarg_name], IRObject):
            _logger.warning(f'partition size in IRObject: {kwargs[kwarg_name]}')
            size = list(kwargs[kwarg_name].value)
        else:
            size = list(kwargs[kwarg_name])
        if isinstance(size[oidx], IRObject):
            _logger.warning(f'partition dim size in IRObject: {size[oidx]}')
            size[oidx] = size[oidx].value
        size[oidx] = size[oidx] // num
        kwargs[kwarg_name] = tuple(size)
        return kwargs

    # special rules: to change output size argument
    rules: TransformRule = []
    for identifier in spatial:
        iidx = ifirst.index(identifier)
        oidx = ofirst.index(identifier)
        rules.append(
            TransformRule([DimopSplit.D(iidx)], [DimopSplit.D(oidx)], modifier)
        )

    anno = OpAnno.create_op_str([in_anno], [ou_anno])
    return anno, rules


def View(input, size: Tuple[int], *arg_size, signature = None):
    """
    out = torch.Tensor.view(tensor: torch.Tensor, *size)
    """
    in_shape = list(input.shape)
    if isinstance(size, IRObject):
        assert size.value is not None, f"shape should have a reference value but got: {size}"
        if isinstance(size.value, int):
            size = (size,) + arg_size
            ou_shape = [d.value if isinstance(d, IRObject) else d for d in size]
        else:  # tuple[int] / list[int]
            assert len(arg_size) == 0, f"already got a tuple of int shape"
            ou_shape = list(size.value)
    else:  # int / tuple[int]
        size = ((size,) if isinstance(size, int) else tuple(size)) + arg_size
        ou_shape = [d.value if isinstance(d, IRObject) else d for d in size]
    assert all(isinstance(d, int) for d in ou_shape), f"but got {ou_shape}"

    anno, rules = _reshape_anno(in_shape, ou_shape, kwarg_name='size')
    signature = 'torch.Tensor.view'
    return IRDimops(View, 'view', signature, [anno], [input], rules, size=size)


def Reshape(input, shape: Tuple[int], *arg_shape, signature = None):
    """
    torch.reshape(Tensor self, int[] shape) -> Tensor
    """
    in_shape = list(input.shape)
    if isinstance(shape, IRObject):
        assert shape.value is not None, f"shape should have a reference value but got: {shape}"
        if isinstance(shape.value, int):
            shape = (shape,) + arg_shape
            ou_shape = [d.value if isinstance(d, IRObject) else d for d in shape]
        else:  # tuple[int] / list[int]
            assert len(arg_shape) == 0, f"already got a tuple of int shape"
            ou_shape = list(shape.value)
    else:  # int / tuple[int]
        shape = ((shape,) if isinstance(shape, int) else tuple(shape)) + arg_shape
        ou_shape = [d.value if isinstance(d, IRObject) else d for d in shape]
    assert all(isinstance(d, int) for d in ou_shape), f"but got {ou_shape}"

    anno, rules = _reshape_anno(in_shape, ou_shape, kwarg_name='shape')
    signature = 'torch.Tensor.reshape'
    return IRDimops(Reshape, 'reshape', signature, [anno], [input], rules, shape=shape)


def Permute(input, dims: Tuple[int], *arg_dims, signature = None):
    """
    torch.Tensor.permute(input, *dims)
    torch.permute(input, dims: Tuple[int])
    """
    dims = (dims,) if isinstance(dims, int) else tuple(dims)
    dims = dims + arg_dims
    assert all(isinstance(dim, int) for dim in dims), f"but got {dims}"
    edim_in = ShapeAnno.create_shape_str(input.shape)
    edim_ou = [copy.copy(edim_in[dim]) for dim in dims]
    anno = OpAnno.create_op_str([edim_in], [edim_ou])
    return IRDimops(Permute, 'permute', signature, [anno], [input], dims=dims)


def Squeeze(input, dim=None, signature = None):
    """
    out = torch.squeeze(tensor)
    """
    if isinstance(dim, int):
        dim = (dim,)
    if dim is not None:
        dim = tuple(d if d >= 0 else d + len(input.shape) for d in dim)
    edim_in = ShapeAnno.create_shape_str(input.shape)
    assert len(edim_in) == len(input.shape)
    edim_ou = []
    for idx in range(len(input.shape)):
        if dim is None or idx in dim:
            if input.shape[idx] != 1:
                # If this dimension is not 1, then we should never partation it
                # Otherwise, it could be squeezed mistakenly if the dimension after partition is 1
                # For example, a tensor with shape（2，4）
                # 1. for single gpu，
                # after calling squeeze(t, 0) the shape is still (2， 4）
                # 2. for 2 gpus, if we partition dim 0, then the tensor shape in each gpu will be (1,4)
                # after calling squeeze(t, 0), the shape becomes (4,) in each gpu
                # which is not correct
                edim_in[idx] += '^'
                edim_ou.append(edim_in[idx])
            # else remove this dimension in edim_out
        else:
            edim_ou.append(edim_in[idx])
    anno = OpAnno.create_op_str([edim_in], [edim_ou])
    return IRDimops(Squeeze, 'squeeze', signature, [anno], [input], dim=dim)


def Unsqueeze(input, dim, signature = None):
    """
    out = torch.unsqueeze(tensor, dim)
    A dim value within the range [-input.dim() - 1, input.dim() + 1) can be used.
    Negative dim will correspond to unsqueeze() applied at dim = dim + input.dim() + 1.
    """
    edim_in = ShapeAnno.create_shape_str(input.shape)
    edim_ou = copy.copy(edim_in)
    if dim < 0:
        dim += len(edim_ou) + 1
    edim_ou.insert(dim, '1')
    anno = OpAnno.create_op_str([edim_in], [edim_ou])
    return IRDimops(Unsqueeze, 'unsqueeze', signature, [anno], [input],dim=dim)


def TypeAs(input: IRTensor, tensor: IRTensor, signature = None):
    """
    translate to To
    """
    return To(input, tensor)


def Triu(input, diagonal=0, *, out=None, signature = None):
    """
    out = torch.triu(tensor, diagonal)
    """
    edim_in = ShapeAnno.create_shape_str(input.shape)
    assert len(edim_in) >= 2
    edim_in[-1] += '^'
    edim_in[-2] += '^'
    edim_ou = copy.copy(edim_in)
    anno = OpAnno.create_op_str([edim_in], [edim_ou])
    return IRDimops(Triu, 'triu', signature, [anno], [input], diagonal=diagonal)


def Tril(input, diagonal=0, *, out=None, signature=None):
    """
    torch.tril(input, diagonal=0, *, out=None)
    """
    assert out is None
    assert isinstance(input, IRTensor)
    edim_in = ShapeAnno.create_shape_str(input.shape)
    assert len(edim_in) >= 2
    edim_in[-1] += '^'
    edim_in[-2] += '^'
    edim_ou = copy.copy(edim_in)
    anno = OpAnno.create_op_str([edim_in], [edim_ou])
    return IRDimops(Tril, 'tril', signature, [anno], [input],
                    diagonal=diagonal)


def CumSum(tensor, dim, signature = None):
    """
    out = torch.cumsum(tensor, dim)
    """
    edim_in = ShapeAnno.create_shape_str(tensor.shape)
    edim_in[dim] += '^'
    edim_ou = copy.copy(edim_in)
    anno = OpAnno.create_op_str([edim_in], [edim_ou])
    return IRDimops(CumSum, 'cumsum', signature, [anno], [tensor], dim=dim)


# def Pad(signature, inputs):
#     """
#     torch.nn.functional.pad(input: torch.Tensor, pad: List[int], mode='constant', value=0.0)
#     """
#     signature = 'torch.nn.functional.pad'
#     tensor, pad, mode, value = inputs
#     ianno = ShapeAnno.create_shape_str(tensor.shape)
#     oanno = []
#     ndims = len(pad) // 2
#     for dim in range(ndims):
#         pad_left, pad_right = pad[2 * dim], pad[2 * dim + 1]
#         if pad_left == 0 and pad_right == 0:
#             oanno.insert(0, ianno[-1-dim])
#         else:
#             ianno[-1-dim] = str(tensor.shape[-1-dim])
#             oanno.insert(0, str(tensor.shape[-1-dim] + pad_left + pad_right))
#     oanno = copy.copy(ianno[:len(tensor.shape) - ndims]) + oanno
#     anno = OpAnno.create_op_str([ianno], [oanno])
#     return IRDimops(Pad, 'pad', signature, [anno], [tensor], pad=pad, mode=mode, value=value)


def Pad(input, pad, mode='constant', value=0.0, signature = None):
    """
    torch.nn.functional.pad(input, pad, mode='constant', value=0.0)
    """
    return IRPad(signature, [input], 'pad', pad=pad, mode=mode, value=value)


# def Conv2D(signature, inputs):
#     """
#     torch.conv2d(input, weight, bias, stride, padding, dialation, groups)
#     https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html?highlight=torch%20conv2d#torch.nn.functional.conv2d
#     """
#     def adapt(anno: OpAnno, node: IRDimops) -> OpAnno:
#         iH, iW = node.input(0).shape[2:4]
#         stride = node.kwargs['stride']
#         padding = node.kwargs['padding']
#         dilation = node.kwargs['dilation']
#         dH = node.input(1).shape[2]
#         dW = node.input(1).shape[3]
#         oH = (iH + 2 * padding[0] - dilation[0] * (dH - 1) - 1) // stride[0] + 1
#         oW = (iW + 2 * padding[1] - dilation[1] * (dW - 1) - 1) // stride[1] + 1
#         anno.outputs[0][2] = DimAnno([str(oH)])
#         anno.outputs[0][3] = DimAnno([str(oW)])
#         return anno
#     annos = [
#         ('N iC+ H^ W^, oC iC+ dH^ dW^, oC -> N oC oH^ oW^', adapt),
#         ('N iC+ H^ W^, oC iC+ dH^ dW^ -> N oC oH^ oW^', adapt),
#     ]
#     tensors = inputs[0:3]
#     if tensors[-1] is None:
#         tensors = inputs[0:2]
#     stride, padding, dilation, groups = inputs[3:]
#     return IRDimops(signature, annos, tensors, 'conv2d',
#                     stride=stride, padding=padding, dilation=dilation, groups=groups)


def Conv2D(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, signature = None):
    """
    torch.nn.functional.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
    """
    if isinstance(padding, int):
        padding = [padding] * 4
    elif len(padding) == 2:
        padH, padW = padding
        padding = [padH, padH, padW, padW]
    return IRConv2D(signature, [input, weight, bias], 'conv2d',
                    stride=stride, padding=padding, dilation=dilation, groups=groups)


def Conv3D(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, signature = None):
    """
    torch.nn.functional.conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
    """
    if isinstance(padding, int):
        padding = [padding] * 4
    elif len(padding) == 2:
        padH, padW = padding
        padding = [padH, padH, padW, padW]
    return IRConv3D(signature, [input, weight, bias], 'conv3d',
                    stride=stride, padding=padding, dilation=dilation, groups=groups)


def CubeCat(*tensors, dim=0, signature = None):
    """
    torch.cat(tensors, dim=0, *, out=None)
    """
    # REMARK: IRFwOperation doesn't support taking a list of IRTensors.
    # Therefore, the argument interface is adapted to take unpacked tensors
    # with dimension. dim=None is for the support of kwarg inputs from torchfx
    assert all(isinstance(tensor, IRTensor) for tensor in tensors)
    assert isinstance(dim, int)
    signature = 'cube.runtime.function.cat'
    iannos = [ShapeAnno.create_shape_str(t.shape) for t in tensors]
    dimlens = [t.shape[dim] for t in tensors]
    for ashape, dimlen in zip(iannos, dimlens):
        ashape[dim] = str(dimlen)
    oannos = [copy.copy(iannos[-1])]
    oannos[0][dim] = str(sum(dimlens))
    anno = OpAnno.create_op_str(iannos, oannos)
    return IRDimops(CubeCat, 'cat', signature, [anno], tensors, dim=dim)


def Cat(tensors, dim=0, out=None, signature=None):
    """
    torch.cat(tensors, dim=0, *, out=None)
    """
    assert out is None
    return CubeCat(*tensors, dim=dim, signature=signature)


def CubeStack(*tensors, dim=0, signature=None):
    # REMARK: IRFwOperation doesn't support taking a list of IRTensors.
    # Therefore, the argument interface is adapted to take unpacked tensors
    # with dimension.
    assert all(isinstance(tensor, IRTensor) for tensor in tensors), f'but got {tensors}'
    assert isinstance(dim, int), f"but not {dim}"
    signature = 'cube.runtime.function.stack'
    iannos = [ShapeAnno.create_shape_str(t.shape) for t in tensors]
    oanno = [None for i in range(len(tensors[0].shape) + 1)]
    oanno[dim] = f'{len(tensors)}^'
    offset = 0
    for i in range(len(oanno)):
        if oanno[i] is None:
            oanno[i] = copy.copy(iannos[-1][offset])
            offset += 1
    anno = OpAnno.create_op_str(iannos, [oanno])
    return IRDimops(CubeStack, 'stack', signature, [anno], tensors, dim=dim)


def Stack(tensors, dim=0, out=None, signature = None):
    """
    torch.stack(tensors, dim=0, *, out=None)
    It needs CubeStack and runtime.function.stack, because
        (i) if the tensors are packed in a list or tuple, it is treated as a whole tensor which is not aligned
            with tensor partitioning;
        (ii) if the tensors are not packed in a list or tuple, torch.stack cannot receive unpacked tensors.

    """
    assert out is None
    return CubeStack(*tensors, dim=dim, signature=signature)


def Chunk(input, chunks, dim=0, signature = None):
    """
    torch.chunk(input, chunks, dim=0)
    """
    assert input.shape[dim] % chunks == 0
    iannos = [ShapeAnno.create_shape_str(input.shape)]
    oannos = [copy.copy(iannos[0]) for _ in range(chunks)]
    iannos[0][dim] = str(input.shape[dim])
    for oanno in oannos:
        oanno[dim] = str(input.shape[dim] // chunks)
    anno = OpAnno.create_op_str(iannos, oannos)
    return IRDimops(Chunk, 'chunk', signature, [anno], [input], chunks=chunks, dim=dim)


def Select(input, dim, index, signature = None):
    """
    torch.select(self:Tensor, dim:int, index:int) -> Tensor
    """
    ianno = ShapeAnno.create_shape_str(input.shape)
    oanno = copy.copy(ianno)
    ianno[dim] += '^'
    oanno.pop(dim)
    anno = OpAnno.create_op_str([ianno], [oanno])
    return IRDimops(Select, 'select', signature, [anno], [input], dim=dim, index=index)


def CubeIndexSelect(input: torch.Tensor, index: torch.Tensor, dim: int, signature = None):
    signature = 'cube.runtime.function.index_select'
    edim_in = ShapeAnno.create_shape_str(input.shape)
    edim_in[dim] += '^'
    idx_anno = chr(ord(edim_in[-1]) + 1)
    edim_ou = copy.copy(edim_in)
    edim_ou[dim] = copy.copy(idx_anno)
    anno = OpAnno.create_op_str([edim_in, idx_anno], [edim_ou])
    # FIXME: runtime function support
    return IRDimops(CubeIndexSelect, 'index_select', signature, [anno], [input, index], dim=dim)


def IndexSelect(input: torch.Tensor, dim: int, index: torch.Tensor, *, out=None, signature = None):
    assert out is None
    return CubeIndexSelect(input, index, dim, signature=signature)


def FullSlice(tensor: IRTensor, slicers: Tuple[Union[None, slice, int]], signature=None):
    """
    Examples:
        >>> a = torch.randn((4,2))
        >>> a[(2,)], a[2]                          # shape [2]
        >>> a[2:3], a[2:3,:]                       # shape [1,2]
        >>> a[(2, slice(None, None, None))]        # shape [2]
        >>> a[(2, None)]                           # shape [1,2]
        >>> a[(2, slice(None, None, None)), None]  # shape [2,1]
        >>> a[(2, None, slice(None, None, None))]  # shape [1,2]
    """
    signature = 'cube.runtime.function.fullslice'
    slicers = tuple(slicers)

    # deal with ... in slice
    if any(slicer is Ellipsis for slicer in slicers):
        front_slicers, back_slicers, ellipsis_flag = [], [], False
        for slicer in slicers:
            if not slicer is Ellipsis:
                front_slicers.append(slicer) if not ellipsis_flag else back_slicers.append(slicer)
            else:
                ellipsis_flag = True
        front_count = len([slicer for slicer in front_slicers if slicer is not None])
        back_count = len([slicer for slicer in back_slicers if slicer is not None])
        assert front_count + back_count <= len(tensor.shape)
        mid_slicers = [slice(None, None, None) for _ in range(len(tensor.shape) - front_count - back_count)]
        slicers = tuple(front_slicers + mid_slicers + back_slicers)

    edim_in = ShapeAnno.create_shape_str(tensor.shape)
    edim_ou = []
    in_idx = 0
    def obj_helper(obj):
        if isinstance(obj, IRObject):
            return obj.value
        else:
            return obj
    for slicer in slicers:
        if slicer is None:
            edim_ou.append('1')
        elif isinstance(slicer, int):
            edim_in[in_idx] += '^'
            in_idx += 1
        elif isinstance(slicer, slice):
            if slicer != slice(None, None, None):
                edim_in[in_idx] += '^'
            _start, _stop, _step = obj_helper(slicer.start), obj_helper(slicer.stop), obj_helper(slicer.step)
            start = 0 if _start is None else _start + tensor.shape[in_idx] if _start < 0 else _start
            stop = tensor.shape[in_idx] if _stop is None else _stop + tensor.shape[in_idx] if _stop < 0 else _stop
            start, stop = min(start, tensor.shape[in_idx]), min(stop, tensor.shape[in_idx])
            step = 1 if _step is None else _step
            dimlen = len(range(start, stop, step))
            if dimlen == tensor.shape[in_idx]:
                edim_ou.append(edim_in[in_idx])
            else:
                edim_ou.append(str(dimlen))
            in_idx += 1
        else:
            raise RuntimeError(f"Unsupported slicer {slicer}")
    edim_ou += edim_in[in_idx:]
    # special case for scalar = torch.Tensor([1,2,3])[0]
    if len(edim_ou) == 0:
        edim_ou.append('1')
    anno = OpAnno.create_op_str([edim_in], [edim_ou])
    return IRDimops(FullSlice, 'fullslice', signature, [anno], [tensor], slicers=slicers)


def Slice(tensor: torch.Tensor, dim, start, end, step, signature = None):
    """
    aten::slice(input:Tensor, dim:int, start:Optional[int], end:Optional[int], step:int) -> Tensor
    """
    signature = 'torch.ops.aten.slice'
    ianno = ShapeAnno.create_shape_str(tensor.shape)
    oanno = copy.copy(ianno)
    ianno[dim] = str(tensor.shape[dim])

    def clip(ofst):
        ofst = ofst + tensor.shape[dim] if ofst < 0 else ofst
        return min(tensor.shape[dim], max(0, ofst))

    # set start and end to possitive itegers
    start = 0 if start is None else start
    end = tensor.shape[dim] if end is None else end
    start, end = clip(start), clip(end)

    oanno[dim] = str(len(range(start, end, step)))
    anno = OpAnno.create_op_str([ianno], [oanno])
    return IRDimops(Slice, 'slice', signature, [anno], [tensor], dim=dim, start=start, end=end, step=step)


def SelectScatter(self: torch.Tensor, input: torch.Tensor, dim: int, index: int, signature = None):
    """
    torch.select_scatter(self:Tensor, input:Tensor, dim:int, index:int) -> Tensor
    """
    # 'torch.select_scatter' isn't supported by Torch2ONNX yet.
    signature = 'cube.runtime.function.select_scatter'
    # shape check
    self_shape, input_shape = self.shape, input.shape
    self_shape.pop(dim)
    assert tuple(self_shape) == tuple(input_shape)
    in1_anno = ShapeAnno.create_shape_str(self.shape)
    in2_anno = in1_anno.copy()
    in2_anno.pop(dim)
    in1_anno[dim] = str(self.shape[dim])
    out_anno = in1_anno.copy()
    anno = OpAnno.create_op_str([in1_anno, in2_anno], [out_anno])
    return IRDimops(SelectScatter, 'select_scatter', signature,
                    [anno], [self, input], dim=dim, index=index)


# If the type is IROject, then value should be type of int, Tuple[int], List[int]
# If the type is Tuple[IROject] or List[IRObject], then the value of each element should be type of int
_VariadicInt = Union[int, Tuple[int, ...], List[int], IRObject, Tuple[IRObject, ...], List[IRObject]]

def extract_variadic(v: _VariadicInt) -> Tuple[List[int], List[bool]]:
    if isinstance(v, int):
        if isinstance(v, bool):
            raise ValueError("Unsupported type: bool")
        return [v], [False]
    elif isinstance(v, IRObject):
        r = extract_variadic(v.value)
        return r[0], [True] * len(r[0]) # because all elements are from IRObject
    elif isinstance(v, (tuple, list)):
        r = [extract_variadic(e) for e in v]
        if any(len(x[0]) != 1 for x in r):
            raise ValueError("tuple/list can't be nested")
        return [x[0][0] for x in r], [x[1][0] for x in r]
    else:
        raise ValueError(f"Unsupported type: {type(v)}")


def Repeat(tensor, repeats: _VariadicInt, *arg_repeats, signature = None):
    """
    torch.Tensor.repeat(*sizes)
    """
    signature = 'torch.ops.aten.repeat'
    if isinstance(repeats, (list, tuple)) or (
        isinstance(repeats, IRObject) and isinstance(repeats.value, (list, tuple))
    ):
        # follow the behavior of torch.Tensor.repeat,
        # ignore arg_repeats in this case
        complete_repeats = repeats
    else:
        complete_repeats = (repeats,) + arg_repeats
    repeats, repeats_is_ir = extract_variadic(complete_repeats)

    in_shape = list(tensor.shape)
    if len(in_shape) > len(repeats):
        raise ValueError("Number of dimensions of repeat dims can not be smaller than number of dimensions of tensor")
    expand = len(repeats) - len(tensor.shape)
    in_shape = [1] * expand + in_shape
    ou_shape = [dimlen * repeat for dimlen, repeat in zip(in_shape, repeats)]
    ianno, oanno = ShapeAnno.create_shape_str(in_shape), []
    for dim, dimlen in enumerate(ou_shape):
        if dim < expand:
            oanno.append(str(dimlen))
        else:
            if repeats[dim] != 1:
                ianno[dim] += '^'
                dim_anno = [str(repeats[dim]), ianno[dim]]
            elif repeats_is_ir[dim]:  # for dynamic repeat, don't split the dimension
                ianno[dim] += '^'
                dim_anno = ianno[dim]
            else:
                dim_anno = ianno[dim]
            oanno.append(dim_anno)
    anno = OpAnno.create_op_str([ianno[expand:]], [oanno])
    return IRDimops(Repeat, 'repeat', signature, [anno], [tensor], repeats=complete_repeats)


def CubeEmbedding(input, weight, padding_idx, signature = None, **kwargs):
    """
    cube.runtime.function.embedding(input, weight, padding_idx, start, stop)
    """
    signature = 'cube.runtime.function.embedding'
    if isinstance(weight, IRSubTensor):
        start, stop = weight.indmap[0]
    else:
        start, stop = 0, weight.shape[0]
    annos = ['*, n+ e -> * e']
    return IRDimops(CubeEmbedding, 'embedding', signature, annos, [input, weight],
                    padding_idx=padding_idx, start=start, stop=stop)


def Embedding(input, weight, padding_idx=None, max_norm=None, norm_type=2.0,
              scale_grad_by_freq=False, sparse=False, signature = None):
    """
    torch.nn.functional.embedding(input, weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False)
    """
    assert max_norm is None and norm_type == 2.0 and (not scale_grad_by_freq) and (not sparse)
    return CubeEmbedding(input, weight, padding_idx, signature=signature)


def Flatten(input, start_dim=0, end_dim=-1, signature = None):
    start_dim = len(input.shape) + start_dim if start_dim < 0 else start_dim
    end_dim = len(input.shape) + end_dim if end_dim < 0 else end_dim
    ishape = ShapeAnno.create_shape_str(input.shape)
    for dim in range(start_dim, end_dim+1):
        ishape[dim] += '^'
    oshape = ishape[:start_dim]
    oshape.append(ishape[start_dim:end_dim+1])
    anno = OpAnno.create_op_str([ishape], [oshape])
    return IRDimops(Flatten, 'flatten', signature, [anno], [input],
                    start_dim=start_dim, end_dim=end_dim)


def Roll(input, shifts: Union[int, Tuple[int]], dims=None, signature = None):
    shifts = (shifts,) if isinstance(shifts, int) else shifts
    ishape = ShapeAnno.create_shape_str(input.shape)
    for dim in range(len(ishape)):
        if dims is None or dim in dims:
            ishape[dim] += '^'
    anno = OpAnno.create_op_str([ishape], [ishape])
    return IRDimops(Roll, 'roll', signature, [anno], [input], shifts=shifts, dims=dims)


def Inverse(input, *, out=None, signature=None):
    """
    torch.inverse(input, *, out=None) → Tensor
    """
    ishape = ShapeAnno.create_shape_str(input.shape)
    ishape = [i + '^' for i in ishape]
    oshape = copy.copy(ishape)
    anno = OpAnno.create_op_str([ishape], [oshape])
    return IRDimops(Inverse, 'inverse', signature, [anno], [input])


def AdaptiveAvgPool1d(input, output_size, signature = None):
    """
    torch.nn.functional.adaptive_avg_pool2d(input, output_size)
    """
    ishape = ShapeAnno.create_shape_str(input.shape)
    ishape[-1] += '^'
    oshape = ishape[:-1] + [str(size) for size in output_size]
    anno = OpAnno.create_op_str([ishape], [oshape])
    return IRDimops(AdaptiveAvgPool1d, 'adaptive_avg_pool1d', signature, [anno], [input], output_size=output_size)


def CrossEntropy(input, target, weight=None,
                 size_average=None, ignore_index=- 100, reduce=None,
                 reduction='mean', label_smoothing=0.0, signature = None):
    """
    torch.nn.functional.cross_entropy(
        input, target, weight=None,
        size_average=None, ignore_index=- 100, reduce=None,
        reduction='mean', label_smoothing=0.0)
    """
    if reduction == 'sum':
        annos = [
            'C^, N -> 1',
            'N+ C^, N+ -> 1',
            'N+ C^ *, N+ * -> 1'
        ]
    else:
        annos = [
            'C^, N -> 1',
            'N^ C^, N^ -> 1',
            'N^ C^ *, N^ * -> 1'
        ]
    return IRDimops(
        CrossEntropy, 'cross_entropy',
        signature, annos, [input, target],
        weight=weight, size_average=size_average, ignore_index=ignore_index,
        reduce=reduce, reduction=reduction, label_smoothing=label_smoothing
    )


def GraphAnchor(name: str, signature = None):
    """
    cube.runtime.function.anchor() -> None
    """
    node = IRGraphAnchor(signature, name)
    return node


def _comparison(creator: Callable, f: Callable, name: str, signature: str,
                input, other):
    """
    if both operands are scalars, returns bool.
    if one operand is a tensor, returns a broadcasted tensor with dtype being bool.

    @param creator Callable: the outside creation function
    @param f Callable: (Scalar, Scalar) -> bools
    """
    # case 0: return constant
    if (not isinstance(input, IRObject)) and (not isinstance(other, IRObject)):
        return f(input, other)
    # case1: torch.equal(tensor1, tensor2)
    if isinstance(input, IRTensor) and isinstance(other, IRTensor):
        lshape, rshape, oshape = _handle_broadcast(input, other)
        annos = [OpAnno.create_op_str([lshape, rshape], [oshape])]
        dimop = IRDimops(creator, name, signature, annos, [input, other])
        dimop.output(0).parent.dtype = torch.bool
        return dimop
    # case2: torch.equal(tensor1, obj2) / torch.equal(obj1, tensor2)
    if isinstance(input, IRTensor) or isinstance(other, IRTensor):
        annos = ['*, ? -> *', '?, * -> *',]
        dimop = IRDimops(creator, name, signature, annos, [input, other])
        dimop.output(0).parent.dtype = torch.bool
        return dimop
    # case3: torch.equal(obj1, obj2)
    else:
        return IRPyFunc(signature, [input, other], [IRObject()])


def CompareGT(input, other, *, out=None, signature = None):
    """
    torch.gt(input, other, *, out=None) -> Tensor
    """
    return _comparison(CompareGT, operator.gt, 'gt', signature, input, other)


def CompareLT(input, other, *, out=None, signature = None):
    """
    torch.lt(input, other, *, out=None) -> Tensor
    """
    return _comparison(CompareLT, operator.lt, 'lt', signature, input, other)


def CompareGE(input, other, *, out=None, signature = None):
    """
    torch.ge(input, other, *, out=None) -> Tensor
    """
    return _comparison(CompareGE, operator.ge, 'ge', signature, input, other)


def CompareLE(input, other, *, out=None, signature = None):
    """
    torch.gt(input, other, *, out=None) -> Tensor
    """
    return _comparison(CompareLE, operator.le, 'le', signature, input, other)


def CompareEQ(input, other, *, out=None, signature = None):
    """
    torch.eq(input, other, *, out=None)
    """
    return _comparison(CompareEQ, operator.eq, 'eq', signature, input, other)


def CompareNE(input, other, *, out=None, signature = None):
    """
    torch.ne(input, other, *, out=None)
    """
    return _comparison(CompareNE, operator.eq, 'ne', signature, input, other)


def Max(input, other_or_dim=None, out_or_keepdim=None, *, out=None, signature = None):
    """
    torch.max(input)
    torch.max(input, dim, keepdim=False, *, out=None)
    torch.max(input, other, *, out=None)
    """
    signature = 'cube.runtime.function.max_'
    if other_or_dim is None:
        edim_in = [s + '^' for s in ShapeAnno.create_shape_str(input.shape)]
        annos = [OpAnno.create_op_str([edim_in], ['1'])]
        return IRDimops(Max, 'max', signature, annos, [input])
    elif isinstance(other_or_dim, IRTensor):
        lshape, rshape, oshape = _handle_broadcast(input, other_or_dim)
        annos = [OpAnno.create_op_str([lshape, rshape], [oshape])]
        return IRDimops(Max, 'max', signature, annos, [input, other_or_dim])
    else:
        assert isinstance(other_or_dim, int) and isinstance(out_or_keepdim, bool)
        edim_in = ShapeAnno.create_shape_str(input.shape)
        edim_in[other_or_dim] += '^'
        edim_out = copy.copy(edim_in)
        if out_or_keepdim:
            edim_out[other_or_dim] = '1'
        else:
            edim_out.pop(other_or_dim)
        annos = [OpAnno.create_op_str([edim_in], [edim_out, edim_out])]
        return IRDimops(Max, 'max', signature, annos, [input], other_or_dim=other_or_dim, out_or_keepdim=out_or_keepdim)


def ShapeAsTensor(input: IRTensor, signature = None):
    """
    torch._shape_as_tensor
    """
    _logger.warning(
        'shape_as_tensor is interpreted as an IRPyFunc '
        'and generate an IRObject instead of IRTensor')
    signature = 'torch._shape_as_tensor'
    return IRPyFunc(signature, [input], [IRObject(name='shape', value=input.shape)])
    edim_in = ShapeAnno.create_shape_str(input.shape)
    edim_ou = [str(len(input.shape))]
    anno = OpAnno.create_op_str([edim_in], [edim_ou])
    return IRDimops(ShapeAsTensor, '_shape_as_tensor', signature, [anno], [input])


# ================== Non-autograd Function Space =================

def Size(tensor, dim=None, signature = None) -> Union[List[int], IRPyFunc]:
    """
    torch.Tensor.size(tensor, dim=None)
    """
    assert isinstance(tensor, IRTensor)
    val = tensor.shape[dim] if isinstance(dim, int) else tensor.shape
    assert val is not None
    if dim is None:
        return IRPyFunc(signature, [tensor], [IRObject(name='size', value=val)])
    else:
        return IRPyFunc(signature, [tensor], [IRObject(name='size', value=val)], dim=dim)


def Dim(tensor, signature=None) -> Union[List[int], IRPyFunc]:
    """
    torch.Tensor.dim(tensor)
    """
    assert isinstance(tensor, IRTensor)
    # constant
    return len(tensor.shape)


def To(tensor: IRTensor, dtype_or_device=None, *, device=None, dtype=None, out=None, signature = None):
    """
    torch.Tensor.to(*args, **kwargs) → Tensor
    """
    assert out is None
    # FIXME: support full version of torch.Tensor.to
    dtype_or_device = dtype if dtype is not None else dtype_or_device
    dtype_or_device = device if dtype_or_device is None else dtype_or_device
    # create "to" in cube runtime functions because dtype if not kwarg in torch.Tensor.to
    signature = 'cube.runtime.function.to'
    annos = ['* -> *']
    if isinstance(dtype_or_device, torch.device):
        # skip device movement as policy can determine device for the tensor.
        return Identity(tensor)
    elif isinstance(dtype_or_device, torch.dtype):
        return IRDimops(To, 'to', signature, annos, [tensor], dtype_or_device=dtype_or_device)
    elif isinstance(dtype_or_device, IRTensor):
        dtype = dtype_or_device.dtype
        return IRDimops(To, 'to', signature, annos, [tensor], dtype_or_device=dtype)
    else:
        raise RuntimeError(f'function.To with unknown arg: {dtype_or_device}')


def GetItem(a: Any, b: Any, signature = None) -> Union[Any, IRPyFunc]:
    """_operator.getitem(a, b): return a[b]"""
    assert not isinstance(b, IRObject)
    obj, index = a, b
    # tensor slice
    if isinstance(obj, IRTensor):
        # note `None` will always
        index = (index,) if isinstance(index, (int, slice)) else tuple(index)
        return FullSlice(obj, index)
    # object slice
    if isinstance(obj, IRObject):
        assert obj.value is not None
        if isinstance(obj.value[index], IRTensor):
            out = obj.value[index]
        else:
            out = IRObject(name='getitem', value=obj.value[index])
        return IRPyFunc(signature, [obj, index], [out])
    return obj[index]


def GetAttr(instance: object, field: str, signature = None) -> Union[List[int], IRPyFunc]:
    """
    builtins.getattr(object, name[, default])
    NOTE: only deal with the attr "shape" of IRFullTensor, because other type of object may not
    have instantiated object or the attr is not simple value.
    """
    obj, name = instance, field
    if isinstance(obj, IRTensor):
        if name == 'shape':
            assert isinstance(obj, IRFullTensor), f"type {type(obj)} is not supported"
            shape = IRObject('shape', value=obj.shape)
            return IRPyFunc(signature, [instance, field], [shape])
        if name == 'dtype':
            assert isinstance(obj, IRFullTensor), f"type {type(obj)} is not supported"
            assert hasattr(obj, name), f"attr {name} is not existed in {obj}"
            return getattr(obj, name)
        if name == 'device':
            assert isinstance(obj, IRFullTensor), f"type {type(obj)} is not supported"
            # FIXME: this is hack, IRFullTensor does not have attribute "device"
            return torch.device('cpu')
        if name == 'layout':
            assert isinstance(obj, IRFullTensor), f"type {type(obj)} is not supported"
            _logger.warning("getattr of 'layout' will always return torch.strided")
            return torch.strided
    if isinstance(obj, torch.finfo):
        return getattr(obj, name)
    return IRPyFunc(signature, [instance, field], [IRObject()])


def FInfo(dtype: torch.dtype, signature = None) -> torch.finfo:
    assert isinstance(dtype, torch.dtype)
    return torch.finfo(dtype)


def NLLLoss(input, target, weight=None, size_average=None,
            ignore_index=-100, reduce=None, reduction='mean',
            signature=None):
    """
    torch.nn.functional.nll_loss(input, target, weight=None, size_average=None,
                                 ignore_index=-100, reduce=None, reduction='mean')
    """
    assert weight is None
    annos = [
        'C^, N -> 1',
        'N+ C, N+ -> 1',
        'N+ C *, N+ * -> 1'
    ]
    return IRDimops(
        NLLLoss, 'nll_loss',
        signature, annos, [input, target],
        weight=weight, size_average=size_average, ignore_index=ignore_index,
        reduce=reduce, reduction=reduction)


def MakeTuple(inputs: Iterable, signature=None):
    return tuple(inputs)


def MakeList(inputs: Iterable, signature=None):
    if isinstance(inputs, Iterable):
        return list(inputs)
    else:
        return IRPyFunc(signature, [inputs], [IRObject(value=list(inputs.value))])


def MakeSlice(*inputs: Iterable, signature=None):
    return slice(*inputs)


def Is(input, other, signature=None):
    assert not isinstance(input, IRObject) and not isinstance(other, IRObject)
    return input is other


def IsNot(input, other, signature=None):
    assert not isinstance(input, IRObject) and not isinstance(other, IRObject)
    return input is not other
