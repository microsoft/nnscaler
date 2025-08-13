#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from typing import Tuple, List, Dict
from torch import Tensor
import torch.distributed as dist

from nnscaler.graph.parser.register import register_op
from nnscaler.ir.operator import IRFwOperation
from flash_attn import flash_attn_varlen_func
from .core.ring_attn_varlen_implementation import llama3_flash_attn_prepare_cu_seqlens, llama3_flash_attn_varlen_func
from .core.utils import gen_head_anno

from nnscaler.runtime.device import DeviceGroup


def wrap_ring_attn_varlen_func(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        cu_seqlens_q: Tensor,
        cu_seqlens_k: Tensor,
        dropout_p: float = 0.0,
        softmax_scale: Tensor = None,
        causal: bool = False,
        window_size: Tuple[int] = (-1, -1),
        alibi_slopes: Tensor = None,
        deterministic: bool = False,
        return_attn_probs: bool = False,
        process_group: Tuple[int] = None,
):
    '''
    wrap the ring_attn_varlen_func to support the distributed training in nnScaler.
    most of the arguments are the same as the original flash_attn_varlen_func.
    `process_group` should be none in the user code since nnScaler accepts the
    program defined for the single device and will automatically generate the
    required communications.
    '''
    assert not return_attn_probs, "return_attn_probs is not supported in ring-attention"
    assert alibi_slopes is None, "alibi_slopes is not supported in ring-attention"
    max_seqlen_q = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item()
    max_seqlen_k = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).max().item()

    if process_group is None or len(process_group) == 1:
        output = flash_attn_varlen_func(
            q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_attn_probs=False,
        )
        return output

    assert len(q.shape) == 3, "q must have shape [total_q, qh, dim]"
    assert len(k.shape) == 3, "k must have shape [total_k, kh, dim]"
    assert len(v.shape) == 3, "v must have shape [total_k, vh, dim]"
    total_q, qheads, qdim = q.shape
    total_k, kheads, kdim = k.shape
    total_v, vheads, vdim = v.shape
    assert total_q == total_k == total_v, "total_q, total_k and total_v must be the same"
    assert kheads == vheads, "number of k and v heads must be the same"
    assert qheads % kheads == 0, "number of q heads must be a multiple of k heads"
    assert qdim == kdim == vdim, "dimension must be the same"

    local_process_group = DeviceGroup().get_group(process_group)
    local_rank = dist.get_rank(local_process_group)
    local_world_size = dist.get_world_size(local_process_group)

    (
        local_cu_seqlens_q,
        local_cu_seqlens_k,
        local_max_seqlen_q,
        local_max_seqlen_k,
        local_k_slice,
    ) = llama3_flash_attn_prepare_cu_seqlens(
        cu_seqlens_q,
        causal=causal,
        rank=local_rank,
        world_size=local_world_size,
    )

    output = llama3_flash_attn_varlen_func(
        q,
        k,
        v,
        local_cu_seqlens_q,
        local_cu_seqlens_k,
        local_max_seqlen_q,
        local_max_seqlen_k,
        heads_k_stride=1,
        local_k_slice=local_k_slice,
        dropout_p=dropout_p,
        causal=causal,
        window_size=window_size,
        alibi_slopes=None,
        deterministic=deterministic,
        return_attn_probs=False,
        group=local_process_group,
    )

    return output


def emit_ring(node: IRFwOperation, args: List[str], kwargs: Dict[str, str], runtime_devid: int, plan_ndevs: int, runtime_ndevs: int) -> str:
    """Special rule to generate ring_attn node"""

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
        if partition_dims[0] == 0: # partition on sequence dim
            # the synchronization should occur across scaleunits
            kw_pairs.append(f"process_group={scale_unit_dev_ids}")
        elif partition_dims[0] == 1:
            # partition the head dim, use local flash_attn_func
            kw_pairs.append("process_group=None")
        else:
            raise ValueError(f'unsupported partition dim: {partition_dims[0]}')
                
    args = ", ".join(list(args) + kw_pairs)
    return f"{signature}({args})"


def flash_attention_anno(query_states, key_states, value_states, *args, **kwargs) -> str:
    q_anno, kv_anno = gen_head_anno(query_states, key_states, value_states)
    return f'l {q_anno} hd^, l {kv_anno} hd^, l {kv_anno} vd^, e^, e^ -> l {q_anno} vd^'


register_op(flash_attention_anno, emit_fn=emit_ring)(wrap_ring_attn_varlen_func)
