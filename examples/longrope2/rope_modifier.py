#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import torch

from nnscaler.graph.parser.register import register_op

def get_longrope_inv_freq(position_ids, base, head_dim, original_max_position_embeddings, long_factor, short_factor):
    seq_len = torch.max(position_ids) + 1
    if seq_len and seq_len > original_max_position_embeddings:
        ext_factors = torch.tensor(long_factor, dtype=torch.float32, device=position_ids.device)
    else:
        ext_factors = torch.tensor(short_factor, dtype=torch.float32, device=position_ids.device)
    inv_freq_shape = torch.arange(0, head_dim, 2, dtype=torch.int64, device=position_ids.device).float() / head_dim
    inv_freq = 1.0 / (ext_factors * base**inv_freq_shape)
    return inv_freq

register_op("b^ l^ -> ?")(get_longrope_inv_freq)


@torch.no_grad()
def longrope_forward(self, x, position_ids):
    assert self.rope_type == "longrope"
    base = self.config.rope_theta
    head_dim = getattr(self.config, "head_dim", self.config.hidden_size // self.config.num_attention_heads)
    long_factor = self.config.rope_scaling["long_factor"]
    short_factor = self.config.rope_scaling["short_factor"]
    
    if hasattr(self.config, "original_max_position_embeddings"):
        original_max_position_embeddings = self.config.original_max_position_embeddings
    else:
        original_max_position_embeddings = self.config.max_position_embeddings
    inv_freq = get_longrope_inv_freq(position_ids, base, head_dim, original_max_position_embeddings, long_factor, short_factor)

    # Core RoPE block
    inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
    position_ids_expanded = position_ids[:, None, :].float()
    # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
    device_type = x.device.type
    device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()

    # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
    cos = cos * self.attention_scaling
    sin = sin * self.attention_scaling

    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def modify_rope_cls(cls):
    cls.forward = longrope_forward
    return cls
