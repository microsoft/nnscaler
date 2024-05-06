# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from dataclasses import dataclass

import cube
from cube.runtime.utils import create_dummy_dataloader

from examples.nlp.blocks.transformer import TransformerLayer


@dataclass
class Config:
    hidden: int = 1024
    layers: int = 8
    heads: int = 16
    ffn_hidden_dim: int = 4096
    num_embeddings: int = 51200
    seqlen: int = 1024
    dropout: float = 0.2
    attn_dropout: float = 0.2
    activation_dropout: float = 0.2


def build_gpt_config(name: str) -> Config:
    if name == 'toy':
        hidden, layers, heads = 1024, 4, 16
    elif name == '350M':
        hidden, layers, heads = 1024, 24, 16
    elif name == '760M':
        hidden, layers, heads = 1536, 24, 16
    elif name == '1.3B':
        hidden, layers, heads = 2048, 24, 32
    elif name == '2.6B':
        hidden, layers, heads = 2560, 32, 32
    elif name == '6.7B':
        hidden, layers, heads = 4096, 32, 32
    elif name == '15B':
        hidden, layers, heads = 5120, 48, 40
    elif name == '39B':
        hidden, layers, heads = 8192, 48, 64
    elif name == '175B':
        hidden, layers, heads = 12288, 96, 96
    else:
        assert False, f'unrecognized name: {name}'
    return Config(hidden, layers, heads, hidden, 4 * hidden)


class GPT(torch.nn.Module):

    def __init__(self, cfg: Config):
        super().__init__()

        # self.embed = torch.nn.Embedding(cfg.num_embeddings, cfg.hidden)
        self.embedw = torch.nn.Parameter(torch.empty(cfg.num_embeddings, cfg.hidden))
        self.position = torch.nn.Embedding(cfg.seqlen, cfg.hidden)
        self.embed_dropout = torch.nn.Dropout()

        self.layers = torch.nn.ModuleList(
            [TransformerLayer(
                cfg.hidden, cfg.heads,
                cfg.hidden, cfg.ffn_hidden_dim,
                cfg.dropout, cfg.attn_dropout, cfg.activation_dropout,
                use_cross_attention=False,
            ) for _ in range(cfg.layers)]
        )
        self.final_layernorm = torch.nn.LayerNorm(cfg.hidden)

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor):

        # embed = self.embed(input_ids)
        embed = torch.nn.functional.embedding(
            input_ids, self.embedw, padding_idx=None,
            max_norm=None, norm_type=2., scale_grad_by_freq=False, sparse=False
        )
        pos_embed = self.position(position_ids)
        embed = embed + pos_embed
        embed = self.embed_dropout(embed)
        enc = embed.transpose(0, 1)

        for layer in self.layers:
            cube.runtime.function.anchor('transformer start')
            enc = layer(enc)
        enc = self.final_layernorm(enc)

        # logits = torch.nn.functional.linear(enc, self.embed.weight)
        logits = torch.nn.functional.linear(enc, self.embedw)
        # simplified
        loss = torch.sum(logits)
        return loss


def get_gpt_dummy_dataloader(batch_size: int, cfg: Config):

    input_ids = torch.randint(
        0, cfg.num_embeddings,
        size=(cfg.seqlen,),
        dtype=torch.int64,
        device=torch.cuda.current_device()
    )
    position_ids = torch.arange(
        0, cfg.seqlen, dtype=torch.int64,
        device=torch.cuda.current_device()
    ).view(cfg.seqlen,)

    return create_dummy_dataloader(
        (input_ids, position_ids), batch_size)
