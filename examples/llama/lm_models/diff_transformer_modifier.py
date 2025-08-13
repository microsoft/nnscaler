#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb
try:
    from apex.normalization import FusedRMSNorm as RMSNorm
except ModuleNotFoundError:
    print("No fused RMSNorm")
    from transformers.models.llama.modeling_llama import LlamaRMSNorm as RMSNorm

import logging
import math

from transformers.utils import is_flash_attn_greater_or_equal_2_10

from .utils import nnscaler_flash_attention_forward
from customized_ops.ring_attention import wrap_ring_attn_func

_logger = logging.getLogger(__name__)


def nnscaler_ring_attn_func(query_states, key_states, value_states, *args, **kwargs):
    return wrap_ring_attn_func(query_states, key_states, value_states)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=1, repeats=n_rep)"""
    bs, n_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )

def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


class NNScalerMultiheadDiffAttn(LlamaAttention):
    """
    Llama attention module using Diff-transformer attention. This module inherits from `LlamaAttention` as the weights 
    of the module stays untouched. The only changes are on attention part using implementation of multihead_diffattn.py,
    original implementation can be refered to https://github.com/microsoft/unilm/blob/master/Diff-Transformer/multihead_diffattn.py
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.head_dim = self.hidden_size // self.num_heads // 2
        self.scaling = self.head_dim ** -0.5

        if (self.head_dim * self.num_heads) != self.hidden_size // 2:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self._init_rope()

        assert self.layer_idx is not None, "layer_idx must be provided for NNScalerMultiheadDiffAttn"
        self.lambda_init = lambda_init_fn(self.layer_idx)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))

        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5)

    # Adapted from LlamaAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, 2 * self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, 2 * self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, 2 * self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # In case static cache is used, it is an instance attribute.
        past_key_value = getattr(self, "past_key_value", past_key_value)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        query_states *= self.scaling
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]
        else:
            causal_mask = torch.triu(torch.zeros([q_len, q_len]).float().fill_(float("-inf")).type_as(query_states), 1)

        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2))
        attn_weights = torch.nan_to_num(attn_weights)
        attn_weights += causal_mask
        attn_weights = F.softmax(attn_weights, dim=-1)

        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1))
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1))
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        attn_weights = attn_weights.view(bsz, self.num_heads, 2, q_len, q_len)
        attn_weights = attn_weights[:, :, 0] - lambda_full * attn_weights[:, :, 1]

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = self.subln(attn_output)
        attn_output = attn_output * (1 - self.lambda_init)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class NNScalerMultiheadDiffFlashAttn(NNScalerMultiheadDiffAttn):
    """
    Llama attention module using Diff-transformer flash attention. This module inherits from `LlamaAttention` as the weights 
    of the module stays untouched. The only changes are on attention part using implementation of multihead_flashdiff_2.py,
    original implementation can be refered to https://github.com/microsoft/unilm/blob/master/Diff-Transformer/multihead_flashdiff_2.py
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()
        self.attn_func = nnscaler_flash_attention_forward
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            _logger.warning("output_attentions is not supported for NNScalerMultiheadDiffFlashAttn.")
        if attention_mask:
            _logger.warning("attention_mask is not supported for NNScalerMultiheadDiffFlashAttn.")

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, 2 * self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, 2 * self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, 2, self.head_dim)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=2)

        # In case static cache is used, it is an instance attribute.
        past_key_value = getattr(self, "past_key_value", past_key_value)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        
        query_states = query_states.reshape(bsz, q_len, self.num_heads, 2, self.head_dim)
        key_states = key_states.reshape(bsz, q_len, self.num_key_value_heads, 2, self.head_dim)
        q1, q2 = query_states[:, :, :, 0], query_states[:, :, :, 1]
        k1, k2 = key_states[:, :, :, 0], key_states[:, :, :, 1]
        v1, v2 = value_states[:, :, :, 0], value_states[:, :, :, 1]

        attn11 = self.attn_func(q1, k1, v1, attention_mask, q_len, causal=True)
        attn12 = self.attn_func(q1, k1, v2, attention_mask, q_len, causal=True)
        attn1 = torch.cat([attn11, attn12], dim=-1)
        
        attn21 = self.attn_func(q2, k2, v1, attention_mask, q_len, causal=True)
        attn22 = self.attn_func(q2, k2, v2, attention_mask, q_len, causal=True)
        attn2 = torch.cat([attn21, attn22], dim=-1)

        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1))
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1))
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        attn_output = attn1 - lambda_full * attn2
        attn_output = self.subln(attn_output)
        attn_output = attn_output * (1 - self.lambda_init)
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * 2 * self.head_dim).contiguous()
        
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


class NNScalerMultiheadDiffRingAttn(NNScalerMultiheadDiffFlashAttn):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attn_func = nnscaler_ring_attn_func
