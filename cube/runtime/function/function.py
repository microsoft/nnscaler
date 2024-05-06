# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Optional, List, Tuple, Union
import torch
import torch.nn.functional as TorchF


def identity(tensor: torch.Tensor) -> torch.Tensor:
    """
    identity forward
    """
    return tensor


def anchor(name: str):
    """
    anchor operation for graph navigation
    """
    return None


def multiref(tensor: torch.Tensor, times: int) -> Tuple[torch.Tensor]:
    """
    identity forward. Create multiple same tensor.
    """
    return tensor if times == 1 else tuple([tensor] * times)


def to(tensor: torch.Tensor, dtype_or_device: Union[torch.device, torch.dtype]) -> torch.Tensor:
    return tensor.to(dtype_or_device)


def max_(input: torch.Tensor, other_or_dim: Union[torch.Tensor, int, None]=None, out_or_keepdim: Optional[bool]=None) -> torch.Tensor:
    if other_or_dim is None:
        return torch.max(input)
    elif isinstance(other_or_dim, int):
        return torch.max(input, other_or_dim, out_or_keepdim)
    else:
        assert isinstance(other_or_dim, torch.Tensor)
        return torch.max(input, other_or_dim)


def accum(*tensors: Tuple[torch.Tensor]) -> torch.Tensor:
    """
    accumulate tensors in to one tensor
    """
    if len(tensors) == 2:
        return tensors[0] + tensors[1]
    else:
        return torch.sum(torch.stack(tensors, dim=0), dim=0)


def fullslice(input: torch.Tensor, slicers: Tuple[Union[None, slice, int]]):
    """Slice tensors

    Note:
    1) `None` will always extend a dimension at current position.
    2) `slice(None, None, None)` equals to `:`,
        meaning select every element at its dimension.
    
    Args:
        input (torch.Tensor): input tensor
        slicers (Tuple[None | slicer | int]): slicer tuple


    Returns:
        torch.Tensor: sliced tensor
    """
    return input[slicers]


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
    Embedding

    Inputs:
        input: torch.Tensor [*]
        weight: [vocab size, embed size]
        start: int
        stop: int

    Outputs:
        output: [*, embed_size]
    """
    input = input.long()
    input_mask = (input < start) | (input >= stop)
    masked_input = input.clone() - start
    masked_input[input_mask] = 0
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


def empty(size: Tuple[int], dtype=None, requires_grad=False, pin_memory=False):
    return torch.empty(
        size, dtype=torch.get_default_dtype() if dtype is None else dtype,
        device=torch.cuda.current_device(),
        requires_grad=requires_grad, pin_memory=pin_memory
    )


def zeros(size: Tuple[int], dtype=None, requires_grad=False):
    return torch.zeros(
        size, dtype=torch.get_default_dtype() if dtype is None else dtype,
        device=torch.cuda.current_device(),
        requires_grad=requires_grad
    )


def ones(size: Tuple[int], dtype=None, requires_grad=False):
    return torch.ones(
        size, dtype=torch.get_default_dtype() if dtype is None else dtype,
        device=torch.cuda.current_device(),
        requires_grad=requires_grad
    )


def rand(size: Tuple[int], dtype=None, requires_grad=False):
    return torch.rand(
        size, dtype=torch.get_default_dtype() if dtype is None else dtype,
        device=torch.cuda.current_device(),
        requires_grad=requires_grad
    )

def full(size: Tuple[int], fill_value, dtype=None, requires_grad=False):
    return torch.full(
        size, fill_value, dtype=dtype, requires_grad=requires_grad, 
        device=torch.cuda.current_device()
    )


def arange(start: int, end: int, step: int, dtype: torch.dtype, requires_grad=False):
    return torch.arange(start=start, end=end, step=step, 
                        dtype=dtype, requires_grad=requires_grad,
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
