#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import torch
import torch.distributed as dist
from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward
from .utils import shuffle_input, recover_output, GlobalMemoryBuffer, get_default_args


_GLOBAL_MEMORY_BUFFER = GlobalMemoryBuffer()


def ring_flash_attn_forward(
    process_group,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
):
    def forward(q, k, v, causal):
        params = get_default_args(_flash_attn_forward).copy()
        params.update(
            {
                "q": q,
                "k": k,
                "v": v,
                "dropout_p": dropout_p,
                "softmax_scale": softmax_scale,
                "causal": causal,
                "alibi_slopes": alibi_slopes,
                "return_softmax": True and dropout_p > 0,
            }
        )
        if "window_size" in params:
            params.update({"window_size": window_size})
        else:
            params.update(
                {
                    "window_size_left": window_size[0],
                    "window_size_right": window_size[1],
                }
            )
        outputs = _flash_attn_forward(**params)
        if len(outputs) == 8:
            block_out, _, _, _, _, block_lse, _, _ = outputs
        else:
            assert len(outputs) == 4
            block_out, block_lse, _, _ = outputs
        return block_out, block_lse

    block_len = q.size(1) // 2
    curr_rank = dist.get_rank(process_group)
    world_size = dist.get_world_size(process_group)
    keep_idx = 2 * curr_rank
    dual_rank = world_size - curr_rank - 1
    dual_send_idx = 2 * dual_rank + 1
    up_rank = min(keep_idx, dual_send_idx)
    down_rank = max(keep_idx, dual_send_idx)

    up_q = q[:, :block_len]
    if causal:
        up_k = k[:, :(up_rank + 1) * block_len]
        up_v = v[:, :(up_rank + 1) * block_len]
    else:
        up_k, up_v = k, v

    up_out, up_lse = forward(up_q, up_k, up_v, causal)

    down_q = q[:, block_len:]
    if causal:
        down_k = k[:, :(down_rank + 1) * block_len]
        down_v = v[:, :(down_rank + 1) * block_len]
    else:
        down_k, down_v = k, v
    down_out, down_lse = forward(down_q, down_k, down_v, causal)

    out = torch.cat([up_out, down_out], dim=1)
    return out, up_lse, down_lse


def ring_flash_attn_backward(
    process_group,
    dout,
    q,
    k,
    v,
    out,
    up_lse,
    down_lse,
    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
):
    block_len = q.size(1) // 2
    curr_rank = dist.get_rank(process_group)
    world_size = dist.get_world_size(process_group)
    keep_idx = 2 * curr_rank
    dual_rank = world_size - curr_rank - 1
    dual_send_idx = 2 * dual_rank + 1
    up_rank = min(keep_idx, dual_send_idx)
    down_rank = max(keep_idx, dual_send_idx)

    dq = torch.zeros_like(q)
    dk_buffer = _GLOBAL_MEMORY_BUFFER.get_tensor(k.size(), k.dtype, "bwd_dk")
    dk_buffer.zero_()
    dv_buffer = _GLOBAL_MEMORY_BUFFER.get_tensor(v.size(), v.dtype, "bwd_dv")
    dv_buffer.zero_()

    up_q = q[:, :block_len]
    up_out = out[:, :block_len]
    up_dout = dout[:, :block_len]
    if causal:
        up_k = k[:, :(up_rank + 1) * block_len]
        up_v = v[:, :(up_rank + 1) * block_len]
    else:
        up_k, up_v = k, v

    params = get_default_args(_flash_attn_backward).copy()
    params.update(
        {
            "dout": up_dout,
            "q": up_q,
            "k": up_k,
            "v": up_v,
            "out": up_out,
            "softmax_lse": up_lse,
            "dq": dq[:, :block_len],
            "dk": dk_buffer[:, :(up_rank + 1) * block_len],
            "dv": dv_buffer[:, :(up_rank + 1) * block_len],
            "dropout_p": dropout_p,
            "softmax_scale": softmax_scale,
            "causal": causal,
            "alibi_slopes": alibi_slopes,
            "deterministic": deterministic,
        }
    )
    if "window_size" in params:
        params.update({"window_size": window_size})
    else:
        params.update(
            {
                "window_size_left": window_size[0],
                "window_size_right": window_size[1],
            }
        )
    _flash_attn_backward(**params)

    down_q = q[:, block_len:]
    down_out = out[:, block_len:]
    down_dout = dout[:, block_len:]
    # TODO: optimize the buffer allocation
    down_dk_buffer = _GLOBAL_MEMORY_BUFFER.get_tensor(k.size(), k.dtype, "bwd_down_dk")
    down_dk_buffer.zero_()
    down_dv_buffer = _GLOBAL_MEMORY_BUFFER.get_tensor(v.size(), v.dtype, "bwd_down_dv")
    down_dv_buffer.zero_()
    if causal:
        down_k = k[:, :(down_rank + 1) * block_len]
        down_v = v[:, :(down_rank + 1) * block_len]
    else:
        down_k, down_v = k, v

    params = get_default_args(_flash_attn_backward).copy()
    params.update(
        {
            "dout": down_dout,
            "q": down_q,
            "k": down_k,
            "v": down_v,
            "out": down_out,
            "softmax_lse": down_lse,
            "dq": dq[:, block_len:],
            "dk": down_dk_buffer[:, :(down_rank + 1) * block_len],
            "dv": down_dv_buffer[:, :(down_rank + 1) * block_len],
            "dropout_p": dropout_p,
            "softmax_scale": softmax_scale,
            "causal": causal,
            "alibi_slopes": alibi_slopes,
            "deterministic": deterministic,
        }
    )
    if "window_size" in params:
        params.update({"window_size": window_size})
    else:
        params.update(
            {
                "window_size_left": window_size[0],
                "window_size_right": window_size[1],
            }
        )
    _flash_attn_backward(**params)
    dk_buffer.add_(down_dk_buffer)
    dv_buffer.add_(down_dv_buffer)

    dim_size = list(k.size())
    dim_size[1] = dim_size[1] // world_size
    dk = torch.empty(dim_size, dtype=k.dtype, device=k.device)
    dv = torch.empty(dim_size, dtype=v.dtype, device=v.device)
    dist.reduce_scatter_tensor(dk, dk_buffer, group=process_group)
    dist.reduce_scatter_tensor(dv, dv_buffer, group=process_group)
    
    return dq, dk, dv


'''
In nnscaler, sequence are stored in the initial order, e.g., [0 1 2 3 4 5 6 7].
However, ring flash attention requires the sequence to be in the order of [0 7 2 5 3 4 1 6].
As a result:
- in forward, we need to shuffle q, all gather k, v and recover the out
- in backward, we need to shuffle dout and recover the dq, reduce scatter dk, dv
'''
class RingFlashAttnFunc(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_softmax,
        group,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        assert alibi_slopes is None

        q = shuffle_input(to_send=q, process_group=group)
        world_size = dist.get_world_size(group)
        dim_size = list(k.size())
        dim_size[1] = dim_size[1] * world_size
        k_buffer = _GLOBAL_MEMORY_BUFFER.get_tensor(dim_size, k.dtype, "fwd_k")
        v_buffer = _GLOBAL_MEMORY_BUFFER.get_tensor(dim_size, v.dtype, "fwd_v")
        torch.distributed.all_gather_into_tensor(k_buffer, k, group=group)
        torch.distributed.all_gather_into_tensor(v_buffer, v, group=group)

        out, up_lse, down_lse = ring_flash_attn_forward(
            group,
            q,
            k_buffer,
            v_buffer,
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=False,
        )
        # this should be out_padded
        ctx.save_for_backward(q, k, v, out, up_lse, down_lse)
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        ctx.group = group
        out = recover_output(out, process_group=group)
        return out

    @staticmethod
    def backward(ctx, dout, *args):
        dout = shuffle_input(to_send=dout, process_group=ctx.group)
        q, k, v, out, up_lse, down_lse = ctx.saved_tensors
        world_size = dist.get_world_size(ctx.group)
        dim_size = list(k.size())
        dim_size[1] = dim_size[1] * world_size
        k_buffer = _GLOBAL_MEMORY_BUFFER.get_tensor(dim_size, k.dtype, "fwd_k")
        v_buffer = _GLOBAL_MEMORY_BUFFER.get_tensor(dim_size, v.dtype, "fwd_v")
        torch.distributed.all_gather_into_tensor(k_buffer, k, group=ctx.group)
        torch.distributed.all_gather_into_tensor(v_buffer, v, group=ctx.group)

        dq, dk, dv = ring_flash_attn_backward(
            ctx.group,
            dout,
            q,
            k_buffer,
            v_buffer,
            out,
            up_lse,
            down_lse,
            softmax_scale=ctx.softmax_scale,
            dropout_p=ctx.dropout_p,
            causal=ctx.causal,
            window_size=ctx.window_size,
            alibi_slopes=ctx.alibi_slopes,
            deterministic=ctx.deterministic,
        )
        dq = recover_output(dq, ctx.group)
        return dq, dk, dv, None, None, None, None, None, None, None, None
