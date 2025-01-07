#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import torch
from transformers.models.llama.modeling_llama import LLAMA_ATTENTION_CLASSES, LlamaRMSNorm

try:
    from apex.normalization.fused_layer_norm import fused_rms_norm_affine
    has_apex = True
except ImportError:
    has_apex = False


def rmsnorm_fwd(self, hidden_states):
    if has_apex:
        return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
    else:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def nnscaler_lm_init(args):
    if args.enable_diff_attn:
        from .diff_transformer_modifier import NNScalerMultiheadDiffAttn, NNScalerMultiheadDiffFlashAttn
        if args.attn_implementation == "sdpa":
            raise ValueError("sdpa is currently not supported in Diff-Transformer")
        LLAMA_ATTENTION_CLASSES["eager"] = NNScalerMultiheadDiffAttn
        LLAMA_ATTENTION_CLASSES["flash_attention_2"] = NNScalerMultiheadDiffFlashAttn
    else:
        from .llama_modifier import NNScalerLlamaFlashAttention2
        LLAMA_ATTENTION_CLASSES["flash_attention_2"] = NNScalerLlamaFlashAttention2
    LlamaRMSNorm.forward = rmsnorm_fwd