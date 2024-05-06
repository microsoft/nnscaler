from typing import Tuple
import torch

import cube
from cube.runtime.utils import create_dummy_dataloader

from dataclasses import dataclass
from examples.t5.blocks import EncoderLayer, DecoderLayer


@dataclass
class Config:
    vocab_size: int = 250112
    d_model: int = 512  # size of the encoder layer
    d_kv: int = 64 # size of the key, query, value projections per attention head. d_kv == d_model // num_heads
    d_ff: int = 1024 # size of the intermediate feeadforward layer
    num_layers: int = 8  # number of encoder layers and decoder layers (total layers = 2 * num_layers)
    num_heads: int = 6
    seqlen: int = 1024  # sequence length
    relative_attention_num_buckets: int = 32  # The number of buckets to use for each attention layer.
    relative_attention_max_distance: int = 128  # The maximum distance of the longer sequences for the bucket separation.
    dropout_rate: float = 0.1 # dropout rate
    layer_norm_eps: float = 1e-6 # layer norm epsilon
    feed_forward_proj: str = 'gated-gelu' #  Type of feed forward layer to be used. Should be one of "relu" or "gated-gelu".


@cube.graph.parser.register('* -> *', name='ref')
def ref(x):
    return torch.clone(x)

@cube.graph.parser.register('bs^ *^, bs^ *^ -> (2 bs^) *^')
def pack(x, y):
    return torch.concat((x, y), dim=0)

@cube.graph.parser.register('(2 bs^) *^ -> bs^ *^, bs^ *^')
def unpack(xy):
    x, y = torch.chunk(xy, chunks=2, dim=0)
    return x, y


class T5(torch.nn.Module):

    def __init__(self, cfg: Config) -> None:
        """The recompute flag is only used by FSDP"""
        super().__init__()
        self.cfg = cfg
        self.embedw = torch.nn.Parameter(torch.empty(cfg.vocab_size, cfg.d_model))
        self.encoder_position = torch.nn.Embedding(cfg.seqlen, cfg.d_model)
        self.decoder_position = torch.nn.Embedding(cfg.seqlen, cfg.d_model)
        self.embed_dropout = torch.nn.Dropout(p=0.0)

        self.encoders = torch.nn.ModuleList(
            [EncoderLayer(
                cfg.d_model, cfg.num_heads, cfg.d_ff,
                cfg.dropout_rate, cfg.dropout_rate, cfg.dropout_rate,
                cfg.layer_norm_eps
            ) for _ in range(cfg.num_layers)]
        )

        self.decoders = torch.nn.ModuleList(
            [DecoderLayer(
                cfg.d_model, cfg.num_heads, cfg.d_ff,
                cfg.dropout_rate, cfg.dropout_rate, cfg.dropout_rate,
                cfg.layer_norm_eps
            ) for _ in range(cfg.num_layers)]
        )

        self.final_layernorm = torch.nn.LayerNorm(cfg.d_model)

    def forward(self, 
                input_ids: torch.LongTensor,
                input_position_ids: torch.LongTensor,
                decoder_input_ids: torch.LongTensor,
                decoder_position_ids: torch.LongTensor):

        # encoder / decoder embedding
        enc_embed = torch.nn.functional.embedding(input_ids, self.embedw)
        pos_embed = self.encoder_position(input_position_ids)
        enc_embed = enc_embed + pos_embed
        enc_embed = self.embed_dropout(enc_embed)
        enc = enc_embed.transpose(0, 1)
        
        dec_embed = torch.nn.functional.embedding(decoder_input_ids, self.embedw)
        pos_embed = self.decoder_position(decoder_position_ids)
        dec_embed = dec_embed + pos_embed
        dec_embed = self.embed_dropout(dec_embed)
        dec = dec_embed.transpose(0, 1)

        decenc = pack(dec, enc)

        # encoder
        for encoder in self.encoders:
            cube.runtime.function.anchor('encoder layer')
            dec, enc = unpack(decenc)
            enc = encoder(enc)
            decenc = pack(dec, enc)

        # decoder
        for decoder in self.decoders:
            cube.runtime.function.anchor('decoder layer')
            dec, enc = unpack(decenc)
            dec = decoder(dec, enc)
            decenc = pack(dec, enc)

        cube.runtime.function.anchor('post-process')
        # FIXME: removing this will compile fail
        dec, enc = unpack(decenc)
        # L N E -> N L E
        hidden = dec.transpose(0, 1)
        logits = torch.nn.functional.linear(hidden, self.embedw)
        # simply for evaluation
        loss = torch.sum(logits)

        return loss

    def flops(self, batch_size: int) -> int:
        """Get FLOPs of a single iteration"""
        seqlen, hidden = self.cfg.seqlen, self.cfg.d_model
        nparams = 0
        visited = set()
        for param in self.parameters():
            if id(param) in visited:
                continue
            nparams += param.numel()
            visited.add(id(param))
        flops = 6 * nparams * seqlen * batch_size
        return flops
        

def get_t5_dummy_dataloader(batch_size: int, config: Config):

    encoder_input_ids = torch.randint(
        0, config.vocab_size,
        size=(config.seqlen,),
        dtype=torch.int64, device=torch.cuda.current_device()
    )
    encoder_position_ids = torch.arange(
        0, config.seqlen, dtype=torch.int64,
        device=torch.cuda.current_device()
    )
    decoder_input_ids = torch.randint(
        0, config.vocab_size,
        size=(config.seqlen,),
        dtype=torch.int64, device=torch.cuda.current_device()
    )
    decoder_position_ids = torch.arange(
        0, config.seqlen, dtype=torch.int64,
        device=torch.cuda.current_device()
    )
    return create_dummy_dataloader(
        (encoder_input_ids, encoder_position_ids,
         decoder_input_ids, decoder_position_ids), batch_size)
