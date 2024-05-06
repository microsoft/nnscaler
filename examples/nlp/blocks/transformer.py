# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from examples.nlp.blocks.attention import MultiHeadSelfAttention
from examples.nlp.blocks.attention import MultiHeadCrossAttention
from examples.nlp.blocks.mlp import MLP


class TransformerLayer(torch.nn.Module):

    def __init__(self, embed_dim: int, num_heads: int,
                 attn_hidden_dim: int, ffn_hidden_dim: int,
                 dropout: float = 0.2, atten_dropout: float = 0.2, activation_dropout: float = 0.2,
                 use_cross_attention: bool = False):
        super().__init__()
        self.self_attn_layer_norm = torch.nn.LayerNorm(embed_dim)
        self.self_attn = MultiHeadSelfAttention(
            embed_dim, num_heads, attn_hidden_dim, atten_dropout
        )

        self.use_cross_attention = use_cross_attention
        if use_cross_attention:
            self.cross_attn_layer_norm = torch.nn.LayerNorm(embed_dim)
            self.cross_attn = MultiHeadCrossAttention(
                embed_dim, num_heads, attn_hidden_dim, atten_dropout
            )

        self.dropout = torch.nn.Dropout(p=dropout)
        self.mlp = MLP(embed_dim, ffn_hidden_dim, activation_dropout)
        self.final_layer_norm = torch.nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, encoder_output = None) -> torch.Tensor:
        # self attention
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(x)
        x = self.dropout(x)
        x = x + residual

        # cross attention
        if self.use_cross_attention:
            residual = x
            x = self.cross_attn_layer_norm(x)
            x = self.cross_attn(x, encoder_output)
            x = self.dropout(x)
            x = x + residual

        # mlp
        residual = x
        x = self.final_layer_norm(x)
        x = self.mlp(x)
        x = self.dropout(x)
        x = x + residual

        return x
