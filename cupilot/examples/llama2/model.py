"""
Llama implementation refers from:

https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
"""
from typing import Optional
import torch
from dataclasses import dataclass

import cube
from examples.llama2.blocks import LlamaDecoderLayer, LlamaRMSNorm, _make_causal_mask, _expand_mask
from cube.runtime.utils import create_dummy_dataloader


@dataclass
class LlamaConfig:
    seqlen: int = 2048
    hidden_size: int = 4096,
    intermediate_size: int = 11008,
    num_hidden_layers: int = 32,
    num_attention_heads: int = 32,
    num_key_value_heads: Optional[int] = None,
    max_position_embeddings: int = 2048,
    vocab_size: int = 32000,
    initializer_range: int = 0.02,
    rms_norm_eps: float = 1e-6,
    pad_token_id: Optional[int] = None,
    bos_token_id: int = 1,
    eos_token_id: int = 2,
    pretraining_tp: int = 1,
    rope_theta: int = 10000.0,
    rope_scaling: Optional[int] = None,


def build_llama_config(model_size: str, seqlen: int, vocab: int = 32000):

    def cal_intermediate_size(hidden: int, multiple_of: int) -> int:
        dim = hidden * 4
        dim = int(2 * dim / 3)
        dim = multiple_of * ((dim + multiple_of - 1) // multiple_of)
        return dim

    defaults = dict(
        initializer_range=0.02,
        rms_norm_eps=0.02,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        rope_theta=10000.0,
        rope_scaling=None,
    )

    if model_size == '1.3b':
        # Llama2 doesn't release 1.3b model, we instead use gpt config in evaluation
        config = LlamaConfig(
            vocab_size=vocab,
            seqlen=seqlen,
            hidden_size=2048,
            intermediate_size=cal_intermediate_size(2048, 256),
            num_hidden_layers=24,
            num_attention_heads=32,
            num_key_value_heads=32,
            max_position_embeddings=seqlen,
            **defaults,
        )
    elif model_size == '7b':
        config = LlamaConfig(
            vocab_size=vocab,
            seqlen=seqlen,
            hidden_size=4096,
            intermediate_size=11008,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=32,
            max_position_embeddings=seqlen,
            **defaults,
        )
    elif model_size == '13b':
        config = LlamaConfig(
            vocab_size=vocab,
            seqlen=seqlen,
            hidden_size=5120,
            intermediate_size=cal_intermediate_size(5120, 256),
            num_hidden_layers=40,
            num_attention_heads=40,
            num_key_value_heads=40,
            max_position_embeddings=seqlen,
            **defaults,
        )
    elif model_size == '34b':
        # code Llama architecture:
        # https://huggingface.co/codellama/CodeLlama-34b-hf/blob/main/config.json
        config = LlamaConfig(
            vocab_size=vocab,
            seqlen=seqlen,
            hidden_size=8192,
            intermediate_size=cal_intermediate_size(8192, 256),  # 22016
            num_hidden_layers=48,
            num_attention_heads=64,
            num_key_value_heads=8,
            max_position_embeddings=seqlen,
            **defaults,
        )
    else:
        raise NotImplementedError(f"Unknown arch: {model_size}")
    return config


def init_weight(model: torch.nn.Module, init_std: float):
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                module.weight.data.normal_(mean=0.0, std=init_std)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, torch.nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=init_std)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
        

class LlamaModel(torch.nn.Module):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = torch.nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx)

        self.layers = torch.nn.ModuleList(
            [LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])

        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Initialize weights and apply final processing
        # init_weight(self, self.config.initializer_range)

    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds
        for decoder_layer in self.layers:
            cube.runtime.function.anchor('decoder starts')
            hidden_states = decoder_layer(hidden_states)
        cube.runtime.function.anchor('post-process starts')
        hidden_states = self.norm(hidden_states)
        return hidden_states


@cube.graph.parser.register('bs+ seqlen+ h^, bs+ seqlen+ -> 1')
@torch.jit.ignore
def compute_loss(logits: torch.Tensor, labels: torch.Tensor, vocab_size: int) -> torch.Tensor:
    logits = logits.float()
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = torch.nn.CrossEntropyLoss()
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)
    return loss


@cube.graph.parser.register('B^ L^ E^, B^ L^, V^ E^, L1+ -> 1')
@torch.jit.ignore
def seq_loss(hidden, labels, weight, position_ids):
    # FIXME: the partition algorithm can be error but the computation cost is almost same
    hidden = hidden[:,:position_ids.size(0),:]
    labels = labels[:,:position_ids.size(0)]
    # B L E, V E -> B L V
    logits = torch.nn.functional.linear(hidden, weight)
    logits = logits.float()
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # flatten the tokens
    loss_fct = torch.nn.CrossEntropyLoss()
    shift_logits = shift_logits.view(-1, weight.size(0))
    shift_labels = shift_labels.view(-1)
    loss = loss_fct(shift_logits, shift_labels)
    return loss


class LlamaForCausalLM(torch.nn.Module):

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # connect
        self.lm_head.weight = self.model.embed_tokens.weight

        self.register_buffer('position_ids',
            torch.arange(config.seqlen, dtype=torch.int64))

    def forward(
        self,
        input_ids: torch.Tensor = None,
        labels: torch.Tensor = None,
    ) -> torch.Tensor:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """
        hidden_states = self.model(input_ids)
        logits = self.lm_head(hidden_states)
        loss = compute_loss(logits, labels, self.vocab_size)

        # loss = seq_loss(hidden_states, labels, self.model.embed_tokens.weight, self.position_ids)
        # loss = loss * 1.0
        
        # loss = torch.sum(hidden_states)
        return loss

    def flops(self, batch_size: int) -> int:
        """Get approximate FLOPs of one iteration"""
        seqlen = self.config.seqlen
        flops = 0
        visited = set()
        nparams = 0
        for p in self.parameters():
            if id(p) in visited:
                continue
            visited.add(id(p))
            nparams += p.numel()
        # effective flops without considering recompute
        flops += 6 * batch_size * seqlen * nparams
        # q@k computation: N h L hd, N h L hd -> N h L L
        flops += 6 * batch_size * \
                 seqlen * seqlen * self.config.hidden_size *\
                 self.config.num_hidden_layers
        return flops


def create_llama_dummy_dataloader(config: LlamaConfig, batch_size: int):
    input_ids = torch.randint(
        0, config.vocab_size,
        size=(config.seqlen,),
        dtype=torch.int64, device=torch.cuda.current_device()
    )
    labels = torch.randint(
        0, config.vocab_size,
        size=(config.seqlen,),
        dtype=torch.int64, device=torch.cuda.current_device(),
    )
    return create_dummy_dataloader(
        (input_ids, labels), batch_size)
