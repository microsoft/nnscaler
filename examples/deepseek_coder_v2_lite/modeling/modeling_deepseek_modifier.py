#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

# This file modifies the official modeling_llama.py file at runtime to
# 1. register the flash attention function to nnscaler and update related code
# 2. replace the un-fused RMSNorm with apex's fused version
# 3. register the MoE routing function to nnscaler
# 4. replace the for loop in MoE forward with grouped gemm implementation

import types
from typing import List, Optional, Tuple, Union

from nnscaler.graph.parser.register import register_op
from nnscaler.ir import IRTensor

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from .modeling_deepseek import DeepseekV2FlashAttention2, ATTENTION_CLASSES, apply_rotary_pos_emb, DeepseekV2RMSNorm, AddAuxiliaryLoss, MoEGate, DeepseekV2MoE, _get_unpad_data
from .moe_utils import moe_gather, moe_scatter, permute, unpermute


if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa


try:
    from apex.normalization.fused_layer_norm import fused_rms_norm_affine
    has_apex = True
except ImportError:
    has_apex = False


try:
    from grouped_gemm.ops import gmm
except ImportError:
    raise ImportError(
        "Grouped GEMM is not available. Please run "
        "`pip install git+https://github.com/fanshiqing/grouped_gemm@v1.0`."
    )


def rmsnorm_fwd(self, hidden_states):
    if has_apex:
        return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
    else:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def moe_gate_fwd(self, hidden_states):
    topk_idx, topk_weight, aux_loss = moe_route(hidden_states, self.weight, self.topk_method, self.top_k, self.n_group, self.n_routed_experts, 
                                                self.topk_group, self.training, self.alpha, self.norm_topk_prob, self.routed_scaling_factor, self.seq_aux)
    return topk_idx, topk_weight, aux_loss


def moe_fwd(self, hidden_states):
    identity = hidden_states
    orig_shape = hidden_states.shape
    topk_idx, topk_weight, aux_loss = self.gate(hidden_states)
    if self.training:
        # gate_projs, up_projs, down_projs are merged after checkpoints are loaded
        y = nnscaler_moe_gmm(hidden_states, topk_idx, topk_weight, aux_loss, self.gate_projs, self.up_projs, self.down_projs, self.config.n_routed_experts, 0, self.config.n_routed_experts)
    else:
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        y = self.moe_infer(hidden_states, topk_idx, topk_weight).view(*orig_shape)
    if self.config.n_shared_experts is not None:
        y = y + self.shared_experts(identity)
    return y


class NNScalerDeepseekFlashAttention2(DeepseekV2FlashAttention2):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # DeepseekV2FlashAttention2 attention does not support output_attentions
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

            # overwrite attention_mask with padding_mask
            attention_mask = kwargs.pop("padding_mask")

        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)  # start signal
        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        kv = (
            self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
            .view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            .transpose(1, 2)
        )

        k_nope, value_states = torch.split(
            kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )
        kv_seq_len = value_states.shape[-2]

        kv_seq_len = value_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

        # query_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
        # query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
        # query_states[:, :, :, self.qk_nope_head_dim :] = q_pe
        query_states = torch.cat([q_nope, q_pe], dim=-1)

        # key_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
        # key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
        # key_states[:, :, :, self.qk_nope_head_dim :] = k_pe
        key_states = torch.cat([k_nope, k_pe.expand(-1, k_nope.size(1), -1, -1)], dim=-1)

        if self.q_head_dim != self.v_head_dim:
            value_states = F.pad(value_states, [0, self.q_head_dim - self.v_head_dim])

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (DeepseekV2RMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            # Handle the case where the model is quantized
            if hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            elif torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            else:
                target_dtype = (
                    self.q_proj.weight.dtype
                    if self.q_lora_rank is None
                    else self.q_a_proj.weight.dtype
                )

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LlamaFlashAttention2 __init__.
            causal = self.is_causal and q_len != 1

        attn_output = nnscaler_flash_attention_forward(
            query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate, causal=causal
        )
        if self.q_head_dim != self.v_head_dim:
            attn_output = attn_output[:, :, :, : self.v_head_dim]

        attn_output = attn_output.reshape(
            bsz, q_len, self.num_heads * self.v_head_dim
        ).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


### register custom functions
def nnscaler_flash_attention_forward(
    query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None, causal=True
):
    # Contains at least one padding token in the sequence
    if attention_mask is not None:
        batch_size = query_states.shape[0]
        query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = nnscaler_upad_input(
            query_states, key_states, value_states, attention_mask, query_length
        )

        cu_seqlens_q, cu_seqlens_k = cu_seq_lens
        max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

        attn_output_unpad = flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_in_batch_q,
            max_seqlen_k=max_seqlen_in_batch_k,
            dropout_p=dropout,
            softmax_scale=softmax_scale,
            causal=causal,
        )

        attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
    else:
        attn_output = flash_attn_func(
            query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
        )
    return attn_output


def nnscaler_upad_input(query_layer, key_layer, value_layer, attention_mask, query_length):
    indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
    batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

    key_layer = index_first_axis(
        key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
    )
    value_layer = index_first_axis(
        value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
    )
    if query_length == kv_seq_len:
        query_layer = index_first_axis(
            query_layer.reshape(batch_size * kv_seq_len, query_layer.shape[-2], head_dim), indices_k
        )
        cu_seqlens_q = cu_seqlens_k
        max_seqlen_in_batch_q = max_seqlen_in_batch_k
        indices_q = indices_k
    elif query_length == 1:
        max_seqlen_in_batch_q = 1
        cu_seqlens_q = torch.arange(
            batch_size + 1, dtype=torch.int32, device=query_layer.device
        )  # There is a memcpy here, that is very bad.
        indices_q = cu_seqlens_q[:-1]
        query_layer = query_layer.squeeze(1)
    else:
        # The -q_len: slice assumes left padding.
        attention_mask = attention_mask[:, -query_length:]
        query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

    return (
        query_layer,
        key_layer,
        value_layer,
        indices_q,
        (cu_seqlens_q, cu_seqlens_k),
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
    )


def moe_route(hidden_states: torch.Tensor, weight: torch.Tensor,
              topk_method: str, top_k: int, n_group: int, n_routed_experts: int, topk_group: int,
              training: bool, alpha: float, norm_topk_prob: bool, routed_scaling_factor: float, seq_aux: bool):
    bsz, seq_len, h = hidden_states.shape
    ### compute gating score
    hidden_states = hidden_states.view(-1, h)
    logits = F.linear(
        hidden_states.type(torch.float32), weight.type(torch.float32), None
    )
    scores = nn.functional.softmax(logits, dim=-1, dtype=torch.float32)

    ### select top-k experts
    if topk_method == "greedy":
        topk_weight, topk_idx = torch.topk(
            scores, k=top_k, dim=-1, sorted=False
        )
    elif topk_method == "group_limited_greedy":
        group_scores = (
            scores.view(bsz * seq_len, n_group, -1).max(dim=-1).values
        )  # [n, n_group]
        group_idx = torch.topk(
            group_scores, k=topk_group, dim=-1, sorted=False
        )[
            1
        ]  # [n, top_k_group]
        group_mask = torch.zeros_like(group_scores)  # [n, n_group]
        group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(
                bsz * seq_len, n_group, n_routed_experts // n_group
            )
            .reshape(bsz * seq_len, -1)
        )  # [n, e]
        tmp_scores = scores.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
        topk_weight, topk_idx = torch.topk(
            tmp_scores, k=top_k, dim=-1, sorted=False
        )

    ### norm gate to sum 1
    if top_k > 1 and norm_topk_prob:
        denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
        topk_weight = topk_weight / denominator
    else:
        topk_weight = topk_weight * routed_scaling_factor
    ### expert-level computation auxiliary loss
    if training and alpha > 0.0:
        scores_for_aux = scores
        aux_topk = top_k
        # always compute aux loss based on the naive greedy topk method
        topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
        if seq_aux:
            scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
            ce = torch.zeros(
                bsz, n_routed_experts, device=hidden_states.device
            )
            ce.scatter_add_(
                1,
                topk_idx_for_aux_loss,
                torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device),
            ).div_(seq_len * aux_topk / n_routed_experts)
            aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(
                dim=1
            ).mean() * alpha
        else:
            mask_ce = F.one_hot(
                topk_idx_for_aux_loss.view(-1), num_classes=n_routed_experts
            )
            ce = mask_ce.float().mean(0)
            Pi = scores_for_aux.mean(0)
            fi = ce * n_routed_experts
            aux_loss = (Pi * fi).sum() * alpha
    else:
        aux_loss = None
    return topk_idx, topk_weight, aux_loss


from nnscaler.ir.operator import IRFwOperation
from nnscaler.graph.function.dimops import DimopSplit, TransformRule


# NOTE: moe_route is replicated intra scale unit, since:
# 1. the computation overhead is small
# 2. the returned aux_loss is summed by mean along the batch dimension, which makes
#    it difficult to handle it correctly without modifying the code
# 3. dispatch by allgather is used currently, which is compatible with the replicated
#    moe_route plan
register_op(f'n^ l^ h^, e^ h^ -> (n^ l^) k^, (n^ l^) k^, 1')(moe_route)


def nnscaler_llama_flash_attention_forward_anno(query_states, key_states, value_states, attention_mask, *args, **kwargs) -> str:
    if query_states.shape[2] != key_states.shape[2]:
        assert query_states.shape[2] % key_states.shape[2] == 0
        group_size = query_states.shape[2] // key_states.shape[2]
        assert query_states.shape[2] == value_states.shape[2] * group_size
        q_anno = f'(group_num {group_size})'
        kv_anno = 'group_num'
    else:
        q_anno = kv_anno = 'num_heads'
    from nnscaler.ir import IRTensor
    if isinstance(attention_mask, IRTensor):
        return f'b l^ {q_anno} hd^, b s^ {kv_anno} hd^, b s^ {kv_anno} vd^, b l^ -> b l^ {q_anno} vd^'
    else:
        return f'b l^ {q_anno} hd^, b s^ {kv_anno} hd^, b s^ {kv_anno} vd^ -> b l^ {q_anno} vd^'


register_op(nnscaler_llama_flash_attention_forward_anno)(nnscaler_flash_attention_forward)


def nnscaler_moe_gmm(
    hidden_states: torch.Tensor, topk_idx: torch.Tensor, topk_weight: torch.Tensor, aux_loss: torch.Tensor, 
    gate_projs: torch.Tensor, up_projs: torch.Tensor, down_projs: torch.Tensor,
    n_routed_experts: int, local_expert_start: int, local_expert_end: int):
    
    orig_shape = hidden_states.shape
    hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
    topk_weight = topk_weight.reshape(-1, topk_weight.shape[-1])

    with torch.no_grad():
        local_mask = (topk_idx >= local_expert_start) & (topk_idx < local_expert_end)
        local_idx = topk_idx.masked_select(local_mask)

    local_prob = topk_weight.masked_select(local_mask)
    local_prob = local_prob.view(-1, 1)
    local_map = local_mask.nonzero()[:, 0]
    local_map = local_map.view(-1, 1).expand(-1, hidden_states.shape[-1])
    local_hidden_states = moe_gather.apply(hidden_states, local_map)

    with torch.no_grad():
        tokens_per_expert = torch.histc(local_idx, bins=local_expert_end - local_expert_start, min=local_expert_start, max=local_expert_end - 1)
        tokens_per_expert = tokens_per_expert.cpu().to(torch.long)

    permuted_inputs, row_id_map = permute(local_hidden_states, local_idx)

    fc1_output = gmm(permuted_inputs, gate_projs, tokens_per_expert, trans_b=True)
    fc2_output = gmm(permuted_inputs, up_projs, tokens_per_expert, trans_b=True)
    intermediate_parallel = torch.nn.functional.silu(fc1_output) * fc2_output
    expert_outs = gmm(intermediate_parallel, down_projs, tokens_per_expert, trans_b=True)

    y = unpermute(expert_outs, row_id_map)
    y = y * local_prob
    y = moe_scatter.apply(y, local_map, hidden_states.shape)

    y = y.to(hidden_states.dtype).view(*orig_shape)
    y = AddAuxiliaryLoss.apply(y, aux_loss)

    return y


def build_ep_transform_rule():
    itransform = [
        DimopSplit.R(),
        DimopSplit.R(),
        DimopSplit.R(),
        DimopSplit.R(),
        DimopSplit.D(0),
        DimopSplit.D(0),
        DimopSplit.D(0),
    ]

    otransform = [
        DimopSplit.V(),
    ]

    def modifier(kwargs, idx, dim, num, pos):
        updated_kwargs = dict(**kwargs)
        expert_num = kwargs['local_expert_end'] - kwargs['local_expert_start']
        updated_kwargs['local_expert_start'] = expert_num // num * pos
        updated_kwargs['local_expert_end'] = expert_num // num * (pos + 1)
        return updated_kwargs

    return TransformRule(itransform, otransform, modifier)


def input_gen_fn(node: IRFwOperation):
    inputs = []
    device = torch.cuda.current_device()
    for i, t in enumerate(node.inputs()):
        if i == 1:
            inputs.append(torch.randint(low=0, high=64, size=t.shape, dtype=torch.int64, device=device, requires_grad=t.requires_grad))
        else:
            inputs.append(torch.rand(t.shape, dtype=t.dtype, device=device, requires_grad=t.requires_grad))
    return tuple(inputs)


register_op(f'n l h^, (n l) 6, (n l) 6, 1, E+ d+ h^, E+ d+ h^, E+ h^ d+ -> n l h^', transform_rules=(build_ep_transform_rule(),), input_gen_fn=input_gen_fn)(nnscaler_moe_gmm)


def nnscaler_deepseek_init():
    ATTENTION_CLASSES['flash_attention_2'] = NNScalerDeepseekFlashAttention2
    DeepseekV2RMSNorm.forward = rmsnorm_fwd
    MoEGate.forward = moe_gate_fwd
    DeepseekV2MoE.forward = moe_fwd
