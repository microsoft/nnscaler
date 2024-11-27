#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import tempfile
import torch
from nnscaler.graph.parser.converter import convert_model

from ...utils import replace_all_device_with


# copy from transformers llama modeling
class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class TestModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rotary_emb = LlamaRotaryEmbedding(128)
        self.fc1 = torch.nn.Linear(128, 128)
        self.fc2 = torch.nn.Linear(128, 128)

    def forward(self, x, position_ids):
        hidden = self.fc1(x)
        cos, sin = self.rotary_emb(hidden, position_ids)
        return self.fc2(hidden * cos * sin)


@replace_all_device_with('cpu')
def test_requires_grad():
    with tempfile.TemporaryDirectory() as tempdir:
        model = TestModule()
        dummy_input = {'x': torch.rand(1, 100, 128), 'position_ids': torch.arange(0, 100, dtype=torch.int64).reshape(1, 100)}
        graph = convert_model(model, dummy_input, tempdir)

    hidden_mul_cos_node = graph.nodes()[15]

    # hidden requires_grad is True
    assert hidden_mul_cos_node.inputs()[0].parent.requires_grad is True
    # cos requires_grad is False
    assert hidden_mul_cos_node.inputs()[1].parent.requires_grad is False
    # output requires_grad is True
    assert hidden_mul_cos_node.outputs()[0].parent.requires_grad is True
