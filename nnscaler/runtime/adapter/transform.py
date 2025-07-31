#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""
Adapter: Tensor Transformation
"""

from typing import List, Tuple
import torch


def identity(tensor: torch.Tensor):
    """
    identity 
    """
    return tensor


def select(tensor: torch.Tensor,
           indmap: Tuple[slice], valmap: int) -> torch.Tensor:
    """
    Select a part of tensor spatially and numerically.
    """
    with torch.no_grad():
        sub_tensor = tensor[indmap]
        if valmap != 1:
            sub_tensor = sub_tensor / valmap
        sub_tensor = sub_tensor.detach()
    return sub_tensor


def smerge(tensors: List[torch.Tensor], dim: int) -> torch.Tensor:
    """
    Runtime primitive of spatial merge.
    Concatenate the tensors along a dimension

    Args:
        tensors: a list of torch tensor
        dim: the dimension to concatenate.
    """
    with torch.no_grad():
        out = torch.concat(tuple(tensors), dim)
    return out


def vmerge(tensors: List[torch.Tensor]) -> torch.Tensor:
    """
    Runtime primitives of numerical merge.
    Sum the tensors.

    Args:
        tensors: a list of torch tensor
    """
    with torch.no_grad():
        out = tensors[0]
        for tensor in tensors[1:]:
            out = out + tensor
    return out
