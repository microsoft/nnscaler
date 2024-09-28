#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from typing import Tuple, List, Dict
import torch
from torch import Tensor
import torch.distributed

from nnscaler.graph.parser.register import register_op
from nnscaler.ir.operator import IRFwOperation
from core.zigzag_attn_implementation import ZigZagRingFlashAttnFunc
from flash_attn import flash_attn_func

import torch.distributed as dist
from nnscaler.runtime.device import DeviceGroup

def wrap_zigzag_attn_func(q: Tensor, k: Tensor, v: Tensor, softmax_scale: Tensor=None,
                          dropout_p: float=0.0, causal: bool=True, window_size: Tuple[int]=(-1, -1),
                          alibi_slopes: Tensor=None, deterministic: bool=False,
                          return_attn_probs: bool=False,
                          process_group: Tuple[int]=None) -> Tensor:
    '''
    wrap the zigzag_attn_func to support the distributed training in nnScaler.
    most of the arguments are the same as the original flash_attn_func.
    `process_group` should be none in the user code since nnScaler accepts the
    program defined for the single device and will automatically generate the
    required communications.
    '''

    if process_group is None or len(process_group) == 1:
        # there is an additional checker for the `softmax_scale`, which is equivalent
        # to the behavior of the original flash_attn_func.
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        output = flash_attn_func(q, k, v, 0.0, softmax_scale, causal)
        return output

    assert causal == True, "zigzag_ring is meaningless for causal=False"
    assert len(q.shape) == 4, "q must have shape [bs, ql, qh, dim]"
    assert len(k.shape) == 4, "k must have shape [bs, kl, kh, dim]"
    assert len(v.shape) == 4, "v must have shape [bs, vl, vh, dim]"
    qbsz, qlen, qheads, qdim = q.shape
    kbsz, klen, kheads, kdim = k.shape
    vbsz, vlen, vheads, vdim = v.shape
    assert qbsz == kbsz == vbsz, "batch size must be the same"
    assert qlen == klen == vlen, "sequence length must be the same"
    assert kheads == vheads, "number of k and v heads must be the same"
    assert qheads % kheads == 0, "number of q heads must be a multiple of k heads"
    assert qdim == kdim == vdim, "dimension must be the same"

    local_process_group = DeviceGroup().get_group(process_group)

    output = ZigZagRingFlashAttnFunc.apply(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        local_process_group,
    ).contiguous()

    return output

def emit_zigzag(node: IRFwOperation, args: List[str], kwargs: Dict[str, str], runtime_devid: int, plan_ndevs: int, runtime_ndevs: int) -> str:
    """Special rule to generate zigzag_attn node"""

    signature = node.signature

    offset = (runtime_devid // plan_ndevs) * plan_ndevs
    scale_unit_dev_ids = [local_rank + offset for local_rank in range(plan_ndevs)]

    kw_pairs = list()
    for key, val in kwargs.items():
        code = f'{key}={val}'
        kw_pairs.append(code)

    sub_input = node.inputs()[0]
    full_input = sub_input.parent
    partition_dims = [i for i, (s, f) in enumerate(zip(sub_input.shape, full_input.shape)) if s != f]
    assert len(partition_dims) <= 1, f"support no more than one partition dim, but got {partition_dims}"
    if not partition_dims:
        kw_pairs.append("process_group=None")
    else:
        # if the 'process_group' is None, we will use the local attention (flash_attn_func)
        if partition_dims[0] == 0: # partition on batch dim
            # partition the bsz dim, use local flash_attn_func
            kw_pairs.append("process_group=None")
        elif partition_dims[0] == 1: # partition on sequence dim
            # the synchronization should occur across scaleunits
            kw_pairs.append(f"process_group={scale_unit_dev_ids}")
        elif partition_dims[0] == 2:
            # partition on num_head dim
            kw_pairs.append("process_group=None")
        else:
            raise ValueError(f'unsupported partition dim: {partition_dims[0]}')
                
    args = ", ".join(list(args) + kw_pairs)
    return f"{signature}({args})"

register_op('bs l h dim^, bs l h dim^, bs l h dim^ -> bs l h dim^', emit_fn=emit_zigzag)(wrap_zigzag_attn_func)
