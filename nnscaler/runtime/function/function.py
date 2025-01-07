#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from contextlib import contextmanager
from typing import Optional, List, Tuple, Union, Any
import torch
import torch.nn.functional as TorchF
import operator
import datetime
from nnscaler.flags import CompileFlag


def identity(tensor: torch.Tensor) -> torch.Tensor:
    """
    identity forward
    """
    return tensor


def ifexpr(cond: bool, true_value: Any, false_value: Any) -> Any:
    """
    if expression
    Please note there is no short-circuit evaluation in this function.
    """
    return true_value if cond else false_value


def anchor(name: str):
    """
    anchor operation for graph navigation
    """
    return None


@contextmanager
def constant_folding(constant_folding: bool = True):
    """
    Context manager to enable/disable constant folding.
    You can put it inside your forward function to control the constant folding behavior.
    Please note as we don't set it as leaf function in tracer,
    it will not be present in the traced graph.
    """
    from nnscaler.graph.tracer.metadata import _GLOBAL_OP_CONTEXT

    old_constant_folding = _GLOBAL_OP_CONTEXT.constant_folding
    _GLOBAL_OP_CONTEXT.constant_folding = constant_folding
    try:
        yield
    finally:
        _GLOBAL_OP_CONTEXT.constant_folding = old_constant_folding


def no_constant_folding():
    """
    Context manager to disable constant folding.
    """
    return constant_folding(constant_folding=False)


def fold_constant(a: Any) -> Any:
    """
    Fold a constant(non-tensor) if constant folding is enabled.

    Please note this should be only used in `constant_folding` block
    to make sure the input to a `constant_folding` block is not wrapped in an IRObject in the graph.

    Example:
    ```
    a = some_func()  # the value is wrapped in IRObject in graph
    with constant_folding():
        a = fold_constant(a)  # unwrap value
        torch.add(t, a)       #  in graph a is a constant
    ```
    """
    return a


def multiref(tensor: torch.Tensor, times: int) -> Tuple[torch.Tensor]:
    """
    identity forward. Create multiple same tensor.
    """
    return tensor if times == 1 else tuple([tensor] * times)


def to(tensor: torch.Tensor, dtype_or_device: Union[torch.device, torch.dtype]) -> torch.Tensor:
    # deprecated
    # keep it only for backward compatibility
    return tensor.to(dtype_or_device)


def accum(*tensors: Tuple[torch.Tensor]) -> torch.Tensor:
    """
    accumulate tensors in to one tensor
    """
    if len(tensors) == 2:
        return tensors[0] + tensors[1]
    else:
        return torch.sum(torch.stack(tensors, dim=0), dim=0)


def fullslice(input: torch.Tensor, *slicers: Union[None, slice, int, torch.Tensor]):
    """Slice tensors

    Note:
    1) `None` will always extend a dimension at current position.
    2) `slice(None, None, None)` equals to `:`,
        meaning select every element at its dimension.

    Args:
        input (torch.Tensor): input tensor
        slicers (Union[None | slicer | int | torch.Tensor]): slicers for input

    Returns:
        torch.Tensor: sliced tensor
    """
    return input[tuple(slicers)]


def conv2d(input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor],
           stride: int, padding: List[int], dilation, groups: int = 1):
    """
    input:  N  iC H  W
    weight: oC iC dH dW
    bias:   oC
    padding: List[int, int, int, int]: [Htop, Hbottom, Wtop, Wbottom] or
             List[int, int]: [Hside, Wside]
    """
    # switch H and W to match torch.nn.functional.pad
    padding = padding[len(padding) // 2:] + padding[0:len(padding) // 2]
    input = TorchF.pad(input, padding, 'constant', 0)
    return TorchF.conv2d(input, weight, bias, stride=stride, dilation=dilation, groups=groups)


def conv3d(input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor],
           stride: int, padding: List[int], dilation, groups: int = 1):
    """
    input:  N iC D H W,
    weight: oC iC dH dW, oC
    bias:   oC
    padding: List[int, int, int, int]: [Htop, Hbottom, Wtop, Wbottom] or
             List[int, int]: [Hside, Wside]

    output: N oC oD oH oW
    """
    # switch D, H and W to match torch.nn.functional.pad
    pad_padding = [padding[-1 - (i // 2)] for i in range(len(padding) * 2)]
    input = TorchF.pad(input, pad_padding, 'constant', 0)
    return TorchF.conv3d(input, weight, bias, stride=stride, dilation=dilation, groups=groups)


def embedding(input: torch.Tensor, weight: torch.Tensor, padding_idx: Optional[int], start: int, stop: int):
    """
    add start/stop to make vocab dim partitionable.

    for example, if the vocab size is 100, and partition the weigth on vocab dim to 4 part,
    then on each part, it will have different start/stop:
        1: [start=0, stop=25]
        2: [start=25, stop=50]
        3: [start=50, stop=75]
        4: [start=75, stop=100]
    before do embedding, the input index outside the range will be masked,
    and directly assign 0.0 to the masked position on the output.

    If vocab dim is partitioned, the results are summed to ensure the correctness of the final result.

    Inputs:
        input: torch.Tensor [*]
        weight: [vocab size, embed size]
        start: int, the weight split start index on vocab dim
        stop: int, the weight split stop index on vocab dim

    Outputs:
        output: [*, embed_size]
    """
    input = input.long()
    input_mask = (input < start) | (input >= stop)
    # make the range of value in the input to [0, stop-start)
    # note that the embedding is implemented like a look up table.
    masked_input = input.clone() - start
    masked_input[input_mask] = 0
    # if padding_idx is inside [start, stop), should map it to [0, stop-start)
    # if padding_idx is outside [start, stop), directly make it None
    if padding_idx is not None and start <= padding_idx < stop:
        padding_idx -= start
    else:
        padding_idx = None
    output = TorchF.embedding(
        masked_input, weight, padding_idx,
        None, 2.0, False, False
    )
    output[input_mask, :] = 0.0
    return output


def layer_norm(input: torch.Tensor,
               weight: Optional[torch.Tensor], bias: Optional[torch.Tensor],
               normalized_shape: List[int], eps: float = 1e-05) -> torch.Tensor:
    """
    LayerNorm
    """
    return torch.nn.functional.layer_norm(input, normalized_shape, weight, bias, eps)


# 'torch.select_scatter' isn't supported by Torch2ONNX yet.
# Implement it with 'torch.masked_scatter' which is supported with ONNX opset=11.
def select_scatter(input:torch.Tensor, src:torch.Tensor, dim:int, index:int):
    # e.g. [..., 1, -1, 1, ...]
    shape = [1] * input.ndim
    shape[dim] = -1

    d = input.shape[dim]
    mask = torch.zeros([d], dtype=torch.bool, device=input.device)
    mask[index] = True
    mask = mask.reshape(shape)

    return torch.masked_scatter(input, mask, src)


def tensor(data, *, dtype=None, requires_grad=False, pin_memory=False):
    """
    force set the device to torch.cuda.current_device()
    """
    return torch.tensor(
        data, dtype=dtype,
        device=torch.cuda.current_device(),
        requires_grad=requires_grad, pin_memory=pin_memory
    )


def empty(size: Tuple[int], dtype=None, requires_grad=False, pin_memory=False):
    """
    force set the device to torch.cuda.current_device()
    """
    return torch.empty(
        size, dtype=torch.get_default_dtype() if dtype is None else dtype,
        device=torch.cuda.current_device(),
        requires_grad=requires_grad, pin_memory=pin_memory
    )


def zeros(size: Tuple[int], dtype=None, requires_grad=False):
    """
    force set the device to torch.cuda.current_device()
    """
    return torch.zeros(
        size, dtype=torch.get_default_dtype() if dtype is None else dtype,
        device=torch.cuda.current_device(),
        requires_grad=requires_grad
    )


def ones(size: Tuple[int], dtype=None, requires_grad=False):
    """
    force set the device to torch.cuda.current_device()
    """
    return torch.ones(
        size, dtype=torch.get_default_dtype() if dtype is None else dtype,
        device=torch.cuda.current_device(),
        requires_grad=requires_grad
    )


def rand(size: Tuple[int], dtype=None, requires_grad=False, pin_memory=False):
    """
    force set the device to torch.cuda.current_device()
    """
    return torch.rand(
        size, dtype=torch.get_default_dtype() if dtype is None else dtype,
        device=torch.cuda.current_device(),
        pin_memory=pin_memory,
        requires_grad=requires_grad,
    )


def randn(size: Tuple[int], dtype=None, requires_grad=False, pin_memory=False):
    """
    force set the device to torch.cuda.current_device()
    """
    return torch.randn(
        size, dtype=torch.get_default_dtype() if dtype is None else dtype,
        device=torch.cuda.current_device(),
        pin_memory=pin_memory,
        requires_grad=requires_grad,
    )


def full(size: Tuple[int], fill_value, dtype=None, requires_grad=False):
    """
    force set the device to torch.cuda.current_device()
    """
    return torch.full(
        size, fill_value, dtype=dtype, requires_grad=requires_grad,
        device=torch.cuda.current_device()
    )


def arange(start: int, end: int, step: int, dtype: torch.dtype, requires_grad=False):
    return torch.arange(start=start, end=end, step=step,
                        dtype=dtype, requires_grad=requires_grad,
                        device=torch.cuda.current_device())


def linspace(start: Union[int, torch.Tensor], end: Union[int, torch.Tensor],
             steps: int, dtype: torch.dtype, requires_grad=False):
    return torch.linspace(start, end, steps, dtype=dtype, requires_grad=requires_grad,
                          device=torch.cuda.current_device())


def index_select(input: torch.Tensor, index: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.index_select(input, dim, index)


def einsum(*operands, equation=None) -> torch.Tensor:
    return torch.einsum(equation, *operands)


def stack(*tensors, dim=0) -> torch.Tensor:
    return torch.stack(tensors, dim)


def cat(*tensors, dim=0) -> torch.Tensor:
    return torch.cat(tensors, dim)


def nndropout(input: torch.Tensor, p=0.5, inplace=False):
    return torch.nn.Dropout(p, inplace)(input)


def setitem(__a, *__bc):
    """
    If __bc has more than 2 elements, that means idxs are flatten becasue idxs contains tensor.
    In this runtime function, idxs will be structured as a tuple if they are flatten,
    and return __a to make this inplace operation trackable.
    """
    if len(__bc) < 2:
        raise ValueError(f'at least two arguments needed, but get __bc={__bc}')
    elif len(__bc) == 2:
        __b, __c = __bc[0], __bc[1]
    else:
        __b, __c = __bc[:-1], __bc[-1]
    operator.setitem(__a, __b, __c)
    return __a


def dict_keys(d: dict):
    return tuple(d.keys())


def dict_values(d: dict):
    return tuple(d.values())


def dict_items(d: dict):
    return tuple(d.items())


def print_time(content: str):
    if not CompileFlag.line_timer:
        return
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else -1
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f"line timer: {rank} - {datetime.datetime.now()} - {content}")