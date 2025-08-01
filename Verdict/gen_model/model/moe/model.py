# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Literal

import torch
import torch.nn.functional as F
from torch import nn
import nnscaler
from typing import List
from .moe_utils import moe_gather, moe_scatter, permute, unpermute, gmm

SEQLEN = 128
DIM = 4096
NHEADS = 32

@dataclass
class ModelArgs:
    dim: int = DIM
    n_layers: int = 32
    n_heads: int = NHEADS
    n_kv_heads: Optional[int] = None
    vocab_size: int = 51200
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000

    max_batch_size: int = 1
    max_seq_len: int = SEQLEN
    use_scaled_rope: bool = True
    
    # moe
    inter_dim: int = 1024
    moe_inter_dim: int = 1024
    n_dense_layers = 1
    # n_routed_experts: int = 64
    n_routed_experts: int = 64
    n_shared_experts: int = 2
    n_activated_experts: int = 6
    n_expert_groups: int = 8
    n_limited_groups: int = 4
    score_func: Literal["softmax", "sigmoid"] = "sigmoid"
    route_scale: float = 2.5

# @dataclass
# class ModelArgs:
#     dim: int = DIM
#     n_layers: int = 32
#     n_heads: int = NHEADS
#     n_kv_heads: Optional[int] = None
#     vocab_size: int = -1
#     multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
#     ffn_dim_multiplier: Optional[float] = None
#     norm_eps: float = 1e-5
#     rope_theta: float = 500000

#     max_batch_size: int = 32
#     max_seq_len: int = SEQLEN
#     use_scaled_rope: bool = True

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


@nnscaler.register_op(f'bsz^ seqlen^ head^ dim^, bsz^ seqlen^ head^ dim^, seqlen^ hidden^ -> bsz^ seqlen^ head^ dim^, bsz^ seqlen^ head^ dim^')
def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


@nnscaler.register_op('N seqlen^, N seqlen^ H^ -> 1 1 seqlen^ seqlen^')
def create_mask(tokens: torch.Tensor, h: torch.Tensor):
    seqlen = tokens.shape[1]
    mask = None
    if seqlen > 1:
        mask = torch.full(
            (1, 1, seqlen, seqlen), float("-inf"), device=tokens.device
        )
        mask = torch.triu(mask, diagonal=1).type_as(h)
    return mask


@nnscaler.register_op('N seqlen *, 1 1 * -> N seqlen *')
def apply_mask(x: torch.Tensor, mask: torch.Tensor):
    return x if mask is None else x + mask


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # model_parallel_size = fs_init.get_model_parallel_world_size()
        model_parallel_size = 1
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        # self.wq = ColumnParallelLinear(
        #     args.dim,
        #     args.n_heads * self.head_dim,
        #     bias=False,
        #     gather_output=False,
        #     init_method=lambda x: x,
        # )
        # self.wk = ColumnParallelLinear(
        #     args.dim,
        #     self.n_kv_heads * self.head_dim,
        #     bias=False,
        #     gather_output=False,
        #     init_method=lambda x: x,
        # )
        # self.wv = ColumnParallelLinear(
        #     args.dim,
        #     self.n_kv_heads * self.head_dim,
        #     bias=False,
        #     gather_output=False,
        #     init_method=lambda x: x,
        # )
        # self.wo = RowParallelLinear(
        #     args.n_heads * self.head_dim,
        #     args.dim,
        #     bias=False,
        #     input_is_parallel=True,
        #     init_method=lambda x: x,
        # )
        self.wq = torch.nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
        )
        self.wk = torch.nn.Linear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
        )
        self.wv = torch.nn.Linear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
        )
        self.wo = torch.nn.Linear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
        )


    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        keys = repeat_kv(xk, self.n_rep)
        values = repeat_kv(xv, self.n_rep)

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        scores = apply_mask(scores, mask)
        
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


# class FeedForward(nn.Module):
#     def __init__(
#         self,
#         dim: int,
#         hidden_dim: int,
#         multiple_of: int,
#         ffn_dim_multiplier: Optional[float],
#     ):
#         super().__init__()
#         hidden_dim = int(2 * hidden_dim / 3)
#         # custom dim factor multiplier
#         if ffn_dim_multiplier is not None:
#             hidden_dim = int(ffn_dim_multiplier * hidden_dim)
#         hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

#         # self.w1 = ColumnParallelLinear(
#         #     dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
#         # )
#         # self.w2 = RowParallelLinear(
#         #     hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x
#         # )
#         # self.w3 = ColumnParallelLinear(
#         #     dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
#         # )
#         self.w1 = torch.nn.Linear(
#             dim, hidden_dim, bias=False
#         )
#         self.w2 = torch.nn.Linear(
#             hidden_dim, dim, bias=False
#         )
#         self.w3 = torch.nn.Linear(
#             dim, hidden_dim, bias=False
#         )

#     def forward(self, x):
#         return self.w2(F.silu(self.w1(x)) * self.w3(x))


def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    assert weight.element_size() > 1
    return F.linear(x, weight, bias)



class Gate(nn.Module):
    """
    Gating mechanism for routing inputs in a mixture-of-experts (MoE) model.

    Attributes:
        dim (int): Dimensionality of input features.
        topk (int): Number of top experts activated for each input.
        n_groups (int): Number of groups for routing.
        topk_groups (int): Number of groups to route inputs to.
        score_func (str): Scoring function ('softmax' or 'sigmoid').
        route_scale (float): Scaling factor for routing weights.
        weight (torch.nn.Parameter): Learnable weights for the gate.
        bias (Optional[torch.nn.Parameter]): Optional bias term for the gate.
    """
    def __init__(self, args: ModelArgs):
        """
        Initializes the Gate module.

        Args:
            args (ModelArgs): Model arguments containing gating parameters.
        """
        super().__init__()
        self.dim = args.dim
        self.topk = args.n_activated_experts
        self.n_groups = args.n_expert_groups
        self.topk_groups = args.n_limited_groups
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
        self.bias = nn.Parameter(torch.empty(args.n_routed_experts)) if self.dim == 7168 else None
        
        self.topk_method = "group_limited_greedy"
        self.n_routed_experts = args.n_routed_experts

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the gating mechanism.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Routing weights and selected expert indices.
        """
        return gate_fw(x, self.weight, self.score_func, self.bias, self.n_groups, self.topk_groups, self.topk, self.route_scale)
        # scores = linear(x, self.weight)
        # if self.score_func == "softmax":
        #     scores = scores.softmax(dim=-1, dtype=torch.float32)
        # else:
        #     scores = scores.sigmoid()
        # original_scores = scores
        # if self.bias is not None:
        #     scores = scores + self.bias
        # if self.n_groups > 1:
        #     scores = scores.view(x.size(0), self.n_groups, -1)
        #     if self.bias is None:
        #         group_scores = scores.amax(dim=-1)
        #     else:
        #         group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
        #     indices = group_scores.topk(self.topk_groups, dim=-1)[1]
        #     mask = scores.new_ones(x.size(0), self.n_groups, dtype=bool).scatter_(1, indices, False)
        #     scores = scores.masked_fill_(mask.unsqueeze(-1), float("-inf")).flatten(1)
        # indices = torch.topk(scores, self.topk, dim=-1)[1]
        # weights = original_scores.gather(1, indices)
        # if self.score_func == "sigmoid":
        #     weights /= weights.sum(dim=-1, keepdim=True)
        # weights *= self.route_scale
        # return weights.type_as(x), indices
    
@nnscaler.register_op(f'a^ h^, e^ h^ -> a^ k^, a^ k^')
def gate_fw(x: torch.Tensor, weight: torch.Tensor, score_func, bias, n_groups, topk_groups, topk, route_scale):
    scores = linear(x, weight)
    if score_func == "softmax":
        scores = scores.softmax(dim=-1, dtype=torch.float32)
    else:
        scores = scores.sigmoid()
    original_scores = scores
    if bias is not None:
        scores = scores + bias
    if n_groups > 1:
        scores = scores.view(x.size(0), n_groups, -1)
        if bias is None:
            group_scores = scores.amax(dim=-1)
        else:
            group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
        indices = group_scores.topk(topk_groups, dim=-1)[1]
        mask = scores.new_ones(x.size(0), n_groups, dtype=bool).scatter_(1, indices, False)
        scores = scores.masked_fill_(mask.unsqueeze(-1), float("-inf")).flatten(1)
    indices = torch.topk(scores, topk, dim=-1)[1]
    weights = original_scores.gather(1, indices)
    if score_func == "sigmoid":
        weights /= weights.sum(dim=-1, keepdim=True)
    weights *= route_scale
    return weights.type_as(x), indices

class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) used as a feed-forward layer.

    Attributes:
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
        w3 (nn.Module): Additional linear layer for feature transformation.
    """
    def __init__(self, dim: int, inter_dim: int):
        """
        Initializes the MLP layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        self.w1 = torch.nn.Linear(dim, inter_dim, bias=False)
        self.w2 = torch.nn.Linear(inter_dim, dim, bias=False)
        self.w3 = torch.nn.Linear(dim, inter_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MLP layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after MLP computation.
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
    
class Expert(nn.Module):
    """
    Expert layer for Mixture-of-Experts (MoE) models.

    Attributes:
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
        w3 (nn.Module): Additional linear layer for feature transformation.
    """
    def __init__(self, dim: int, inter_dim: int):
        """
        Initializes the Expert layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        self.w11 = torch.nn.Linear(dim, inter_dim, bias=False)
        self.w22 = torch.nn.Linear(inter_dim, dim, bias=False)
        self.w33 = torch.nn.Linear(dim, inter_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Expert layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert computation.
        """
        return self.w22(F.silu(self.w11(x)) * self.w33(x))


class MoE(nn.Module):
    """
    Mixture-of-Experts (MoE) module.

    Attributes:
        dim (int): Dimensionality of input features.
        n_routed_experts (int): Total number of experts in the model.
        n_local_experts (int): Number of experts handled locally in distributed systems.
        n_activated_experts (int): Number of experts activated for each input.
        gate (nn.Module): Gating mechanism to route inputs to experts.
        experts (nn.ModuleList): List of expert modules.
        shared_experts (nn.Module): Shared experts applied to all inputs.
    """
    def __init__(self, args: ModelArgs):
        """
        Initializes the MoE module.

        Args:
            args (ModelArgs): Model arguments containing MoE parameters.
        """
        super().__init__()
        # ad hoc set MoE parallel group to be 1
        # in this case, experts are replicated across tp and dp workers
        # we have to do this, as nnScaler does not support manual parallelization in code
        # to enable actual EP, add new policy in nnScaler.policies.py
        world_size = 1
        rank = 0
        self.dim = args.dim
        assert args.n_routed_experts % world_size == 0, f"Number of experts must be divisible by world size (world_size={world_size})"
        self.n_routed_experts = args.n_routed_experts
        self.n_local_experts = args.n_routed_experts // world_size
        self.n_activated_experts = args.n_activated_experts
        self.experts_start_idx = rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.gate = Gate(args)
        self.experts: List[Expert] = nn.ModuleList([Expert(args.dim, args.moe_inter_dim) if self.experts_start_idx <= i < self.experts_end_idx else None
                                      for i in range(self.n_routed_experts)])
        self.w11 = torch.stack([expert.w11.weight for expert in self.experts], dim=0)
        self.w33 = torch.stack([expert.w33.weight for expert in self.experts], dim=0)
        self.w22 = torch.stack([expert.w22.weight for expert in self.experts], dim=0)
        self.shared_experts = MLP(args.dim, args.n_shared_experts * args.moe_inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MoE module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert routing and computation.
        """
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x)
        # y = torch.zeros_like(x)
        # counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        # for i in range(self.experts_start_idx, self.experts_end_idx):
        #     if counts[i] == 0:
        #         continue
        #     expert = self.experts[i]
        #     idx, top = torch.where(indices == i)
        #     y[idx] += expert(x[idx]) * weights[idx, top, None]
        y = nnscaler_moe_gmm(x, indices, weights, self.w11, self.w33, self.w22, self.n_routed_experts, 0, self.n_routed_experts)

        z = self.shared_experts(x)
        # if world_size > 1:
        #     dist.all_reduce(y)
        return (y + z).view(shape)
    
def nnscaler_moe_gmm(
    hidden_states: torch.Tensor, topk_idx: torch.Tensor, topk_weight: torch.Tensor, 
    gate_projs: torch.Tensor, up_projs: torch.Tensor, down_projs: torch.Tensor,
    n_routed_experts: int, local_expert_start: int, local_expert_end: int):
    
    orig_shape = hidden_states.shape
    hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
    topk_weight = topk_weight.reshape(-1, topk_weight.shape[-1])

    with torch.no_grad():
        local_mask = (topk_idx >= local_expert_start) & (topk_idx < local_expert_end)
        local_idx = topk_idx.masked_select(local_mask)

    local_prob = topk_weight.masked_select(local_mask)
    local_prob = local_prob.view(-1, 1)
    local_map = local_mask.nonzero()[:, 0]
    local_map = local_map.view(-1, 1).expand(-1, hidden_states.shape[-1])
    local_hidden_states = moe_gather.apply(hidden_states, local_map)

    with torch.no_grad():
        tokens_per_expert = torch.histc(local_idx, bins=local_expert_end - local_expert_start, min=local_expert_start, max=local_expert_end - 1)
        tokens_per_expert = tokens_per_expert.cpu().to(torch.long)

    permuted_inputs, row_id_map = permute(local_hidden_states, local_idx)

    fc1_output = gmm(permuted_inputs, gate_projs, tokens_per_expert, trans_b=True)
    fc2_output = gmm(permuted_inputs, up_projs, tokens_per_expert, trans_b=True)
    intermediate_parallel = torch.nn.functional.silu(fc1_output) * fc2_output
    expert_outs = gmm(intermediate_parallel, down_projs, tokens_per_expert, trans_b=True)

    y = unpermute(expert_outs, row_id_map)
    y = y * local_prob
    y = moe_scatter.apply(y, local_map, hidden_states.shape)

    y = y.to(hidden_states.dtype).view(*orig_shape)

    return y


from nnscaler.graph.function.dimops import DimopSplit, TransformRule

def build_ep_transform_rule():
    itransform = [
        DimopSplit.R(),
        DimopSplit.R(),
        DimopSplit.R(),
        DimopSplit.D(0),
        DimopSplit.D(0),
        DimopSplit.D(0),
    ]

    otransform = [
        DimopSplit.V(),
    ]

    def modifier(kwargs, idx, dim, num, pos):
        updated_kwargs = dict(**kwargs)
        expert_num = kwargs['local_expert_end'] - kwargs['local_expert_start']
        updated_kwargs['local_expert_start'] = expert_num // num * pos
        updated_kwargs['local_expert_end'] = expert_num // num * (pos + 1)
        return updated_kwargs

    return TransformRule(itransform, otransform, modifier)

def input_gen_fn(node):
    inputs = []
    device = torch.cuda.current_device()
    for i, t in enumerate(node.inputs()):
        if i == 1:
            inputs.append(torch.randint(low=0, high=64, size=t.shape, dtype=torch.int64, device=device, requires_grad=t.requires_grad))
        else:
            inputs.append(torch.rand(t.shape, dtype=t.dtype, device=device, requires_grad=t.requires_grad))
    return tuple(inputs)


nnscaler.register_op(f'a h^, a e, a e, E+ d+ h^, E+ d+ h^, E+ h^ d+ -> a h^', transform_rules=(build_ep_transform_rule(),), input_gen_fn=input_gen_fn)(nnscaler_moe_gmm)


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = MLP(args.dim, args.inter_dim) if layer_id < args.n_dense_layers else MoE(args)
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        # self.tok_embeddings = VocabParallelEmbedding(
        #     params.vocab_size, params.dim, init_method=lambda x: x
        # )
        self.tok_embeddings = torch.nn.Embedding(
            params.vocab_size, params.dim
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        # self.output = ColumnParallelLinear(
        #     params.dim, params.vocab_size, bias=False, init_method=lambda x: x
        # )
        self.output = torch.nn.Linear(
            params.dim, params.vocab_size, bias=False
        )

        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
        )

    # @torch.inference_mode()
    def forward(self, tokens: torch.Tensor):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        # self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[:seqlen]

        mask = create_mask(tokens, h)

        for layer in self.layers:
            h = layer(h, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h).float()
        output = torch.sum(output)
        return output
