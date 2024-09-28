#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from typing import List, Any
import torch


class DTypeInfo:
    """Tensor dtype information

    Attributes:
        bytes (Dict[Any, int]): data type -> btye size.
        priority (List[torch.dtype]): the priority of dtypes for promotion
    """
    bytes = {
        torch.complex128: 128,
        torch.complex64: 64,
        torch.complex32: 32,
        torch.float64: 8,
        torch.float32: 4,
        torch.bfloat16: 2,
        torch.float16: 2,
        torch.int64: 8,
        torch.int32: 4,
        torch.int16: 2,
        torch.int8: 1,
        torch.uint8: 1,
        torch.bool: 1,
    }

    priority = [
        torch.float64, torch.float32, torch.bfloat16, torch.float16,
        torch.int64, torch.int32, torch.int16, torch.int8, torch.bool
    ]

    @staticmethod
    def get_byte_size(dtype: Any) -> int:
        """Get dtype btye size"""
        if dtype not in DTypeInfo.bytes:
            raise NotImplementedError(f'Unknown dtype {dtype}')
        return DTypeInfo.bytes[dtype]
    
    @staticmethod
    def promote(dtypes: List[torch.dtype]) -> torch.dtype:
        """Infer the promoted dtype according to dtypes.

        This will follow the dtype promotion rule, which is same with PyTorch.

        Reference:
        https://pytorch.org/docs/stable/tensor_attributes.html#type-promotion-doc

        priority: torch.float64 > torch.float32 > torch.bfloat16 > torch.float16 >
                  torch.int64 > torch.int32 > torch.int16 > torch.int8 > torch.bool

        Args:
            dtypes List[torch.dtype]: a list of dtypes

        Returns:
            the promoted dtype
        """
        if not all(dtype in DTypeInfo.priority for dtype in dtypes):
            raise NotImplementedError(
                f"Fail to promote dtypes because one dtype "
                f"in {dtypes} doesn't appear in priority list.")
        dtype = None
        for dtype in DTypeInfo.priority:
            if dtype in dtypes: break
        return dtype
