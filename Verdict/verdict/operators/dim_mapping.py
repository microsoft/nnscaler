#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from typing import List, Tuple


def resolve_reshape_shape(input_shape: List[int], target_shape: List[int]) -> List[int]:
    """Resolve -1 in reshape target to concrete shape"""
    input_numel = 1
    for dim in input_shape:
        input_numel *= dim

    known_product = 1
    unknown_index = -1
    for i, dim in enumerate(target_shape):
        if dim == -1:
            if unknown_index != -1:
                raise ValueError("Only one dimension can be -1")
            unknown_index = i
        else:
            known_product *= dim

    if unknown_index != -1:
        if input_numel % known_product != 0:
            raise ValueError("Invalid reshape: cannot infer dimension size")
        target_shape = target_shape[:]
        target_shape[unknown_index] = input_numel // known_product

    return target_shape


def get_dim_mapping(
    input_shape: List[int], target_shape: List[int]
) -> List[Tuple[List[int], List[int]]]:
    """Returns a mapping from input dims to target dims by tracking reshape"""

    target_shape = list(target_shape).copy()
    resolved_target = resolve_reshape_shape(input_shape, target_shape)
    input_sizes = input_shape[:]
    target_sizes = resolved_target[:]

    # flatten dimension products
    input_ptr = 0
    target_ptr = 0
    mapping = []

    while input_ptr < len(input_sizes) and target_ptr < len(target_sizes):
        in_start = input_ptr
        out_start = target_ptr
        in_acc = input_sizes[input_ptr]
        out_acc = target_sizes[target_ptr]

        while in_acc != out_acc:
            if in_acc < out_acc:
                input_ptr += 1
                if input_ptr >= len(input_sizes):
                    break
                in_acc *= input_sizes[input_ptr]
            else:
                target_ptr += 1
                if target_ptr >= len(target_sizes):
                    break
                out_acc *= target_sizes[target_ptr]
        # record the group
        mapping.append(
            (
                list(range(in_start, input_ptr + 1)),
                list(range(out_start, target_ptr + 1)),
            )
        )
        input_ptr += 1
        target_ptr += 1

    return mapping
