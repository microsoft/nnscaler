
import torch
import cube

from cube.runtime.utils import create_dummy_dataloader
from examples.gpt.blocks import TransformerLayer
from dataclasses import dataclass


@dataclass
class Config:
    embed_dim: int = 1024
    layers: int = 8
    attention_heads: int = 16
    ffn_hidden_dim: int = 4096
    num_embeddings: int = 51200
    seqlen: int = 1024
    hidden_dropout: float = 0.2
    attn_dropout: float = 0.2
    activation_dropout: float = 0.2


class GPT(torch.nn.Module):

    def __init__(self, cfg: Config):
        super().__init__()

        # self.embed = torch.nn.Embedding(cfg.num_embeddings, cfg.embed_dim)
        self.embedw = torch.nn.Parameter(torch.empty(cfg.num_embeddings, cfg.embed_dim))
        self.position = torch.nn.Embedding(cfg.seqlen, cfg.embed_dim)
        self.embed_dropout = torch.nn.Dropout()

        self.layers = torch.nn.ModuleList(
            [TransformerLayer(
                cfg.embed_dim, cfg.attention_heads,
                cfg.ffn_hidden_dim,
                cfg.hidden_dropout, cfg.attn_dropout, cfg.activation_dropout
            ) for _ in range(cfg.layers)]
        )
        self.final_layernorm = torch.nn.LayerNorm(cfg.embed_dim)

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

        # ====> pretrain setting
        # logits = torch.nn.functional.linear(enc, self.embedw)
        # # simplify
        # loss = torch.sum(logits)

        # ===> finetune setting
        loss = torch.sum(enc)
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
