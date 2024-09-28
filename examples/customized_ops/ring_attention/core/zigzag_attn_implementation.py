#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

# Credits: This logger implementation is inspired by project https://github.com/zhuzilin/ring-flash-attention

import torch
import torch.distributed as dist
from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward
from .utils import RingComm, update_out_and_lse, shuffle_input, recover_output

'''
Assume we have 4 GPUs A, B, C, D.
The sequence is represented as [0 1 2 3 4 5 6 7].

The P2P communication ring is A -> D -> B -> C -> A
The initial status of the attention computation is
X
X X
X X X
X X X X
X X X X X
X X X X X X
X X X X X X X
X X X X X X X X
Note:
- the computation in the diagonal is `causal=True`
- the computation in the off-diagonal is `causal=False`
We consider a `X` with `causal=True` as a unit computation block.
In this example, there are 4 steps. Each device is responsible for 2 unit computation blocks in each step.

q status is same across all steps (q is not transmitted):
GPU A: [0 7]
GPU B: [2 5]
GPU C: [3 4]
GPU D: [1 6]

Step 0, kv status:
GPU A: [0 7]
GPU B: [2 5]
GPU C: [3 4]
GPU D: [1 6]
Computation status:
A
X D
X X B
X X X C
X X X C C
X X B X X B
X D X X X X D
A X X X X X X A

Step 1, kv status:
GPU A: [3 4]
GPU B: [1 6]
GPU C: [2 5]
GPU D: [0 7]
Computation status:
X
D X
X B X
X X C X
X X C X X
X B X X X X
D X X X X X X
X X X A A X X X

Step 2, kv status:
GPU A: [2 5]
GPU B: [0 7]
GPU C: [1 6]
GPU D: [3 4]
Computation status:
X
X X
B X X
X C X X
X C X X X
B X X X X X
X X X D D X X
X X A X X A X X

Step 3, kv status:
GPU A: [1 6]
GPU B: [3 4]
GPU C: [0 7]
GPU D: [2 5]
Computation status:
X
X X
X X X
C X X X
C X X X X
X X X B B X
X X D X X D X
X A X X X X A X

From this example, we can conclude the key insight of zigzag ring flash attention is:
- split the sequence into fine-grained blocks to achieve balance across steps and gpus
- schedule the computation in a zigzag pattern to minimize the communication overhead

To be more specific, if the sequence length is L=4n, the total computation cost of flash attention
with causal=True is 1/2  L^2 = 8n^2. Each device needs to compute 4n. Each step needs to compute 2.

Computation task assigned for each GPU:

GPU 0:   (0, 4n-1)
GPU 1:   (2, 4n-3)
...
GPU n-1: (2n-2, 2n+1)
GPU n:   (2n-1, 2n)
GPU n+1: (2n-3, 2n+2)
...
GPU 2n-1: (1, 4n-2)

Dependence of kv (required kv range) for each device:
GPU 0:    [0, 4n-1]
GPU 1:    [0, 4n-3]
...
GPU n-1:  [0, 2n+1]
GPU n:    [0, 2n]
GPU n+1:  [0, 2n+2]
...
GPU 2n-1: [0, 4n-2]

In general, if there are 2n GPUs, the ring is 0 -> 2n-1 -> 1 -> 2n-2 -> ... -> n -> n+1 -> 0

For each device, the 2n steps is divided into 3 parts:
1. compute the local attention with `causal=True`
2. if current step is less or equal to its relative rank in the ring, select the first half
   of the received kv to compute the attention with `causal=False`. In the example above, each
   device computes to `left` of its corresponding rows in the status matrix.
3. if current step is greater than its relative rank in the ring, select the second half of
   local q and full received kv to compute the attention with `causal=False`. In the example
   above, each device fills the remaining part of its lower row in the status matrix.
'''

def zigzag_ring_flash_attn_forward(
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
    assert causal == True, "zigzag ring is meaningless for causal=False"
    comm = RingComm(process_group)

    block_seq_len = q.shape[1] // 2
    q1 = q[:, block_seq_len:]

    out = None
    lse = None
    next_k, next_v = None, None

    def forward(q, k, v, causal):
        block_out, _, _, _, _, block_lse, _, _ = _flash_attn_forward(
            q,
            k,
            v,
            dropout_p,
            softmax_scale,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            return_softmax=True and dropout_p > 0,
        )
        return block_out, block_lse

    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            next_k: torch.Tensor = comm.send_recv(k)
            next_v: torch.Tensor = comm.send_recv(v)
            comm.commit()

        if step == 0:
            block_out, block_lse = forward(q, k, v, causal=True)
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)
        elif step <= comm.revert_rank:
            k0 = k[:, :block_seq_len]
            v0 = v[:, :block_seq_len]
            block_out, block_lse = forward(q, k0, v0, causal=False)
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)
        else:
            block_out, block_lse = forward(q1, k, v, causal=False)
            out, lse = update_out_and_lse(
                out,
                lse,
                block_out,
                block_lse,
                slice_=(slice(None), slice(block_seq_len, None)),
            )

        if step + 1 != comm.world_size:
            comm.wait()
            k = next_k
            v = next_v

    out = out.to(q.dtype)
    lse = lse.squeeze(dim=-1).transpose(1, 2)
    return out, lse


'''
In the backward pass, we assume q, k, v and out are saved in the shuffled order.
In addition, the backward pass requires a shuffled dout as input and generates
a shuffled dq, dk, dv as output. Note that out is a sum of all step outputs, so
we can directly pass dout to each step's backward block to compute the local gradient
according to the differiential chain rule.

Similar to the forward pass, in the backward pass, the 2n steps are divided into 3 parts.

Different from the forward pass, we need to communicate the gradient of kv in a ring as well.
To be more specific, each device calculates the local gradients of dq, dk, dv. In the following
steps, dq will be accumulated in the initial device, while dk and dv will be transmitted to the
next consumer device, then accumulated in the consumer device. In the end, the dk and dv will be
transmitted back to the initial device.

In addition, to be compatible with the flash-attn's interface and reduce the precision loss,
we will accumulate and transmit the gradients in float32. They will be converted back to the
original dtype at the end of the backward pass.
'''
def zigzag_ring_flash_attn_backward(
    process_group,
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
):
    assert causal == True, "zigzag ring is meaningless for causal=False"
    kv_comm = RingComm(process_group)
    d_kv_comm = RingComm(process_group)
    dq, dk, dv = None, None, None
    next_dk, next_dv = None, None
    next_k, next_v = None, None
    dk_comm_buffer, dv_comm_buffer = None, None

    dout1 = dout.chunk(2, dim=1)[1]
    q1 = q.chunk(2, dim=1)[1]
    out1 = out.chunk(2, dim=1)[1]
    softmax_lse1 = softmax_lse.chunk(2, dim=2)[1].contiguous()
    block_seq_len = q.shape[1] // 2

    # repeatly allocating buffer may be slow...
    dq_buffer = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    dk_buffer = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    dv_buffer = torch.empty(v.shape, dtype=v.dtype, device=v.device)

    def backward(dout, q, k, v, out, softmax_lse, causal):
        seqlen_q = q.shape[1]
        seqlen_kv = k.shape[1]
        _flash_attn_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq_buffer[:, :seqlen_q],
            dk_buffer[:, :seqlen_kv],
            dv_buffer[:, :seqlen_kv],
            dropout_p,
            softmax_scale,
            causal,
            window_size,
            alibi_slopes,
            deterministic,
            rng_state=None,
        )

    for step in range(kv_comm.world_size):
        if step + 1 != kv_comm.world_size:
            next_k = kv_comm.send_recv(k)
            next_v = kv_comm.send_recv(v)
            kv_comm.commit()

        if step == 0:
            backward(dout, q, k, v, out, softmax_lse, causal=True)
            dq = dq_buffer.to(torch.float32)
            dk = dk_buffer.to(torch.float32)
            dv = dv_buffer.to(torch.float32)
        else:
            if step <= kv_comm.revert_rank:
                k0 = k[:, :block_seq_len]
                v0 = v[:, :block_seq_len]
                backward(dout, q, k0, v0, out, softmax_lse, causal=False)
                dq += dq_buffer
            else:
                backward(dout1, q1, k, v, out1, softmax_lse1, causal=False)
                # always use the first half in dq_buffer.
                dq[:, block_seq_len:] += dq_buffer[:, :block_seq_len]

            d_kv_comm.wait()
            dk_comm_buffer, dv_comm_buffer = dk, dv
            dk, dv = next_dk, next_dv

            if step <= kv_comm.revert_rank:
                dk[:, :block_seq_len] += dk_buffer[:, :block_seq_len]
                dv[:, :block_seq_len] += dv_buffer[:, :block_seq_len]
            else:
                dk += dk_buffer
                dv += dv_buffer

        if step + 1 != kv_comm.world_size:
            kv_comm.wait()
            k = next_k
            v = next_v

        next_dk = d_kv_comm.send_recv(dk, dk_comm_buffer)
        next_dv = d_kv_comm.send_recv(dv, dv_comm_buffer)
        d_kv_comm.commit()

    d_kv_comm.wait()

    return dq.to(q.dtype), next_dk.to(q.dtype), next_dv.to(q.dtype)

'''
In nnscaler, sequence are stored in the initial order, e.g., [0 1 2 3 4 5 6 7].
However, zigzag ring flash attention requires the sequence to be in the order of [0 7 2 5 3 4 1 6].
As a result:
- in forward, we need to shuffle q, k, v and recover the out
- in backward, we need to shuffle dout and recover the dq, dk, dv
'''
class ZigZagRingFlashAttnFunc(torch.autograd.Function):
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
        k = shuffle_input(to_send=k, process_group=group)
        v = shuffle_input(to_send=v, process_group=group)
        k = k.contiguous()
        v = v.contiguous()
        out, softmax_lse = zigzag_ring_flash_attn_forward(
            group,
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=False,
        )
        # this should be out_padded
        ctx.save_for_backward(q, k, v, out, softmax_lse)
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        ctx.group = group
        out = recover_output(out, process_group=group)
        return out if not return_softmax else (out, softmax_lse, None)

    @staticmethod
    def backward(ctx, dout, *args):
        dout = shuffle_input(to_send=dout, process_group=ctx.group)
        q, k, v, out, softmax_lse = ctx.saved_tensors
        dq, dk, dv = zigzag_ring_flash_attn_backward(
            ctx.group,
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            softmax_scale=ctx.softmax_scale,
            dropout_p=ctx.dropout_p,
            causal=ctx.causal,
            window_size=ctx.window_size,
            alibi_slopes=ctx.alibi_slopes,
            deterministic=ctx.deterministic,
        )
        dq = recover_output(dq, ctx.group)
        dk = recover_output(dk, ctx.group)
        dv = recover_output(dv, ctx.group)
        return dq, dk, dv, None, None, None, None, None, None, None, None


def zigzag_ring_flash_attn_qkvpacked_func(
    qkv,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
):
    return ZigZagRingFlashAttnFunc.apply(
        qkv[:, :, 0],
        qkv[:, :, 1],
        qkv[:, :, 2],
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
    )


def zigzag_ring_flash_attn_kvpacked_func(
    q,
    kv,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
):
    return ZigZagRingFlashAttnFunc.apply(
        q,
        kv[:, :, 0],
        kv[:, :, 1],
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
    )


def zigzag_ring_flash_attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
):
    return ZigZagRingFlashAttnFunc.apply(
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
        group,
    )
