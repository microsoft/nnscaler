from typing import List
import torch
import numpy as np
import deepspeed

from examples.t5.blocks import feedforward, self_attention, cross_attention
from cube.runtime.utils import create_dummy_dataloader
from dataclasses import dataclass

class MPU:

    class __MPU:

        def __init__(self, dp: int, tp: int, init_with_cube: bool = False):
            # torch.distributed.init_process_group(backend='nccl')
            assert torch.distributed.is_initialized()
            self.rank = torch.distributed.get_rank()
            self._dp_group = None
            self._dp_rank = None
            self._dp_world_size = None
            self._tp_group = None
            self._tp_group_ranks = None
            self._tp_rank = None
            self._tp_world_size = None
            assert torch.distributed.get_world_size() == tp * dp
            grid = np.arange(dp * tp).reshape((dp, tp))
            # set tp
            for ranks in grid.tolist():
                if init_with_cube:
                    import cube
                    group = cube.runtime.device.DeviceGroup().get_group(list(ranks))
                else:
                    group = torch.distributed.new_group(list(ranks))
                if self.rank in ranks:
                    print(f'> [{self.rank}]: tp group: {ranks}')
                    self._tp_group = group
                    self._tp_group_ranks = ranks
                    self._tp_world_size = len(ranks)
                    self._tp_rank = torch.distributed.get_rank(group=group)
            # set dp
            for ranks in np.transpose(grid, (1, 0)).tolist():
                if init_with_cube:
                    import cube
                    group = cube.runtime.device.DeviceGroup().get_group(list(ranks))
                else:
                    group = torch.distributed.new_group(list(ranks))
                if self.rank in ranks:
                    print(f'> [{self.rank}]: dp group: {ranks}')
                    self._dp_group = group
                    self._dp_world_size = len(ranks)
                    self._dp_rank = torch.distributed.get_rank(group=group)

    instance = None

    def __init__(self, dp=None, tp=None, init_with_cube: bool = False):
        if not MPU.instance:
            assert isinstance(tp, int) and isinstance(dp, int)
            MPU.instance = MPU.__MPU(dp, tp, init_with_cube)
        else:
            assert dp is None and tp is None

    def __getattr__(self, name):
        return getattr(self.instance, name)

    def get_model_parallel_rank(self):
        return self.instance._tp_rank

    def get_model_parallel_group_ranks(self) -> List[int]:
        return self._tp_group_ranks

    def get_model_parallel_world_size(self):
        return self.instance._tp_world_size

    def get_model_parallel_group(self):
        return self.instance._tp_group

    def get_data_parallel_rank(self):
        return self.instance._dp_rank

    def get_data_parallel_world_size(self):
        return self.instance._dp_world_size

    def get_data_parallel_group(self):
        return self.instance._dp_group


class AllreduceIdentity(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_: torch.Tensor):
        group = MPU().get_model_parallel_group()
        world_size = MPU().get_model_parallel_world_size()
        if world_size == 1: return input_
        input_ = input_.contiguous() if not input_.is_contiguous() else input_
        torch.distributed.all_reduce(input_, group=group)
        torch.cuda.synchronize()
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class IdentityAllreduce(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_):
        return input_

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        group = MPU().get_model_parallel_group()
        world_size = MPU().get_model_parallel_world_size()
        if world_size == 1: return grad_output
        grad_output = grad_output.contiguous() if not grad_output.is_contiguous() else grad_output
        torch.distributed.all_reduce(grad_output, group=group)
        torch.cuda.synchronize()
        return grad_output


class MultiHeadSelfAttention(torch.nn.Module):

    def __init__(self, embed_dim: int, num_heads: int, inner_dim: int, dropout: float = 0.0):
        super().__init__()
        self.inner_dim = inner_dim // MPU().get_model_parallel_world_size()
        self.num_heads = num_heads // MPU().get_model_parallel_world_size()
        self.head_dim = inner_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.dropout_p = dropout
        # QKV [(h d 3), E]
        self.qkv_proj = torch.nn.Parameter(torch.empty(3 * self.inner_dim, embed_dim))
        self.qkv_bias = torch.nn.Parameter(torch.empty(3 * self.inner_dim))
        # Out
        self.out_proj = torch.nn.Parameter(torch.empty(embed_dim, self.inner_dim))
        self.out_bias = torch.nn.Parameter(torch.empty(embed_dim))

    def forward(self, query):
        query = IdentityAllreduce.apply(query)
        attn = self_attention(
            query, self.qkv_proj, self.qkv_bias,
            self.out_proj,
            self.num_heads, self.scaling, self.dropout_p, mask=False
        )
        attn = AllreduceIdentity.apply(attn)
        attn = attn + self.out_bias
        return attn


class MultiHeadCrossAttention(torch.nn.Module):

    def __init__(self, embed_dim: int, num_heads: int, inner_dim: int, dropout: float = 0.0):
        super().__init__()
        self.inner_dim = inner_dim // MPU().get_model_parallel_world_size()
        self.num_heads = num_heads // MPU().get_model_parallel_world_size()
        self.head_dim = inner_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.dropout_p = dropout
        # Q
        self.q_proj = torch.nn.Parameter(torch.empty(self.inner_dim, embed_dim))
        self.q_bias = torch.nn.Parameter(torch.empty(self.inner_dim))
        # K
        self.k_proj = torch.nn.Parameter(torch.empty(self.inner_dim, embed_dim))
        self.k_bias = torch.nn.Parameter(torch.empty(self.inner_dim))
        # V
        self.v_proj = torch.nn.Parameter(torch.empty(self.inner_dim, embed_dim))
        self.v_bias = torch.nn.Parameter(torch.empty(self.inner_dim))
        # Out
        self.out_proj = torch.nn.Parameter(torch.empty(embed_dim, self.inner_dim))
        self.out_bias = torch.nn.Parameter(torch.empty(embed_dim))

    def forward(self, query: torch.Tensor, key: torch.Tensor):
        query = IdentityAllreduce.apply(query)
        key = IdentityAllreduce.apply(key)
        attn = cross_attention(
            query, key,
            self.q_proj, self.q_bias,
            self.k_proj, self.k_bias,
            self.v_proj, self.v_bias,
            self.out_proj,
            self.num_heads, self.scaling, self.dropout_p, mask=True
        )
        attn = AllreduceIdentity.apply(attn)
        attn = attn + self.out_bias
        return attn


class MLP(torch.nn.Module):

    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        assert hidden_dim % MPU().get_model_parallel_world_size() == 0
        tp_size = MPU().get_model_parallel_world_size()

        self.proj1 = torch.nn.Parameter(torch.empty((hidden_dim // tp_size, embed_dim)))
        self.proj1_bias = torch.nn.Parameter(torch.empty((hidden_dim // tp_size,)))
        self.proj2 = torch.nn.Parameter(torch.empty((embed_dim, hidden_dim // tp_size)))
        self.proj2_bias = torch.nn.Parameter(torch.empty((embed_dim,)))
        self.dropout = dropout

    def forward(self, x: torch.Tensor):
        x = IdentityAllreduce.apply(x)
        x = feedforward(x, self.proj1, self.proj1_bias,
                        self.proj2, self.dropout, self.training)
        x = AllreduceIdentity.apply(x)
        x = x + self.proj2_bias
        return x


class EncoderLayer(torch.nn.Module):

    def __init__(self, embed_dim: int, num_heads: int, ffn_hidden_dim: int,
                 hidden_dropout: float = 0.0, attn_dropout: float = 0.0, activation_dropout: float = 0.0,
                 layernomr_eps: float = 1e-6):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(
            embed_dim, num_heads, embed_dim, attn_dropout
        )
        self.self_attn_layer_norm = torch.nn.LayerNorm(embed_dim, eps=layernomr_eps)
        self.dropout = torch.nn.Dropout(p=hidden_dropout)
        self.mlp = MLP(embed_dim, ffn_hidden_dim, activation_dropout)
        self.final_layer_norm = torch.nn.LayerNorm(embed_dim, eps=layernomr_eps)

    def forward(self, decenc: torch.Tensor) -> torch.Tensor:
        dec, x = unpack(decenc)
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(x)
        x = self.dropout(x)
        x = x + residual

        residual = x
        x = self.final_layer_norm(x)
        x = self.mlp(x)
        x = self.dropout(x)
        x = x + residual
        return pack(dec, x)


class DecoderLayer(torch.nn.Module):

    def __init__(self, embed_dim: int, num_heads: int, ffn_hidden_dim: int,
                 hidden_dropout: float = 0.0, attn_dropout: float = 0.0, activation_dropout: float = 0.0,
                 layernomr_eps: float = 1e-6):
        super().__init__()

        self.apply_residual_connection_post_layernorm: bool = False
        self.hidden_dropout: float = hidden_dropout

        # input layer norm
        self.input_layernorm = torch.nn.LayerNorm(embed_dim, eps=layernomr_eps)
        
        # self attention
        self.self_attn = MultiHeadSelfAttention(
            embed_dim, num_heads, embed_dim, attn_dropout
        )

        # layer norm on the attention output
        self.post_attention_layernorm = torch.nn.LayerNorm(embed_dim, eps=layernomr_eps)

        # cross attention
        self.inter_attention = MultiHeadCrossAttention(embed_dim, num_heads, embed_dim, attn_dropout)

        # layernomr on the attention output
        self.post_inter_attention_layernorm = torch.nn.LayerNorm(embed_dim, eps=layernomr_eps)

        # MLP
        self.mlp = MLP(embed_dim, ffn_hidden_dim, activation_dropout)

    def forward(self, decenc: torch.Tensor):
        # hidden_states: torch.Tensor, encoder_output: torch.Tensor):
        # hidden states [s, b, h]
        hidden_states, encoder_output = unpack(decenc)
        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)

        # Self attention.
        attention_output = self.self_attn(layernorm_output)

        # Residual connection
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        layernorm_input = residual + torch.nn.functional.dropout(
            attention_output, p=self.hidden_dropout, training=self.training)

        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)

        # Cross attention.
        attention_output = self.inter_attention(layernorm_output, encoder_output)

        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input
        # Dropout-add.
        layernorm_input = residual + torch.nn.functional.dropout(
            attention_output, p=self.hidden_dropout, training=self.training)
        # Layer norm
        layernorm_output = self.post_inter_attention_layernorm(layernorm_input)
        # MLP.
        mlp_output = self.mlp(layernorm_output)

        # Second residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input
        output = residual + torch.nn.functional.dropout(mlp_output)
    
        return pack(output, encoder_output)


class ParallelEmbed(torch.nn.Module):

    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        tp_size = MPU().get_model_parallel_world_size()
        tp_idx = MPU().get_model_parallel_rank()
        assert vocab_size % tp_size == 0
        self.start = vocab_size // tp_size * tp_idx
        self.end = vocab_size // tp_size * (tp_idx + 1)
        self.weight = torch.nn.Parameter(
            torch.empty((vocab_size // tp_size, embed_dim))
        )

    def forward(self, tokens: torch.Tensor):
        if MPU().get_model_parallel_world_size() > 1:
            mask = (tokens < self.start) | (tokens >= self.end)
            tokens = tokens.clone() - self.start
            tokens[mask] = 0
            embed = torch.nn.functional.embedding(tokens, self.weight)
            embed[mask, :] = 0.0
            # print(embed.shape, embed.dtype, embed.device)
            embed = AllreduceIdentity.apply(embed)
        else:
            embed = torch.nn.functional.embedding(tokens, self.weight)
        return embed

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

def ref(x):
    return torch.clone(x)

def pack(x, y):
    return torch.concat((x, y), dim=0)

def unpack(xy):
    x, y = torch.chunk(xy, chunks=2, dim=0)
    return x, y


class T5(torch.nn.Module):

    def __init__(self, cfg: Config, recompute: bool = False) -> None:
        """The recompute flag is only used by FSDP"""
        super().__init__()
        self.cfg = cfg
        self.embed = ParallelEmbed(cfg.vocab_size, cfg.d_model)
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
        enc_embed = self.embed(input_ids)
        pos_embed = self.encoder_position(input_position_ids)
        enc_embed = enc_embed + pos_embed
        enc_embed = self.embed_dropout(enc_embed)
        enc = enc_embed.transpose(0, 1)
        
        dec_embed = self.embed(decoder_input_ids)
        pos_embed = self.decoder_position(decoder_position_ids)
        dec_embed = dec_embed + pos_embed
        dec_embed = self.embed_dropout(dec_embed)
        dec = dec_embed.transpose(0, 1)

        decenc = pack(dec, enc)
        
        # encoder
        for encoder in self.encoders:
            # decenc = encoder(decenc)
            decenc = deepspeed.checkpointing.checkpoint(encoder, decenc)

        # decoder
        for decoder in self.decoders:
            # decenc = decoder(decenc)
            decenc = deepspeed.checkpointing.checkpoint(decoder, decenc)
        
        def recompute(decenc, weight):
            dec, enc = unpack(decenc)
            # L N E -> N L E
            hidden = dec.transpose(0, 1)
            logits = torch.nn.functional.linear(hidden, weight)
            # simply for evaluation
            loss = torch.sum(logits)
            return loss
        
        loss = deepspeed.checkpointing.checkpoint(recompute, decenc, self.embed.weight)
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
        nparams -= self.cfg.vocab_size * hidden
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
