import torch
import cube


@cube.graph.parser.register('L^ N E^, (h+ d^ 3) E^, (h+ d^ 3), E^ (h+ d^) -> L^ N E^', name='self_attention')
def self_attention(query: torch.Tensor, 
                   qkv_proj: torch.Tensor, qkv_bias: torch.Tensor,
                   out_proj: torch.Tensor,
                   h: int, scale: float, dropout_p: float, mask: bool = False):
    num_head = h
    L, N = query.size(0), query.size(1)
    dim_head = qkv_proj.size(0) // num_head // 3

    qkv = torch.nn.functional.linear(query, qkv_proj, qkv_bias) # L N E, (h d 3) E -> L N (h d 3)
    qkv = qkv.view(L, N, num_head * dim_head, 3) # L N (h d 3) -> L N (h d) 3
    q, k, v = qkv.chunk(3, dim=-1)  # L N (3 h d) -> L N (h d), L N (h d), L N (h d)
    q = q.contiguous().view(L, (N * num_head), dim_head) # L N (h d) -> L (N h) d
    k = k.contiguous().view(L, (N * num_head), dim_head) # L N (h d) -> L (N h) d
    v = v.contiguous().view(L, (N * num_head), dim_head) # L N (h d) -> L (N h) d

    # preallocating input tensor: (N h) L L
    matmul_input_buffer = torch.empty([N * h, L, L], dtype=query.dtype, device=query.device)
    # L (N h) d, L (N h) d -> (N h) L L
    attn = torch.baddbmm(
        matmul_input_buffer,
        q.transpose(0, 1),  # (N h) L d
        k.transpose(0, 1).transpose(1, 2), # (N h) d L
        beta=0.0, alpha=scale
    )
    # ======== replace the semantic into more efficient implementation ============

    # attention mask
    if mask: # (N h) L L -> (N h) L L
        attn = attn.view(N, num_head, L, L)
        ones = torch.ones((N, L, L), device=attn.device)
        amask = torch.tril(ones)
        amask = amask.view(N, 1, L, L)
        amask = (amask < 0.5)
        attn = attn.masked_fill_(amask, -10000.0)
        attn = attn.view((N * num_head), L, L)

    attn = torch.nn.functional.softmax(attn, dim=-1) # (N h) L L -> (N h) L L
    attn = torch.nn.functional.dropout(attn, dropout_p, True, False) # (N h) L L -> (N h) L L
    v = v.transpose(0, 1)  # L (N h) d -> (N h) L d
    output = torch.bmm(attn, v) # (N h) L L, (N h) L d -> (N h) L d
    output = output.transpose(0, 1).contiguous()     # (N h) L d -> L (N h) d
    output = output.view(L, N, num_head * dim_head)  # (N h) L d -> L N (h d)
    output = torch.nn.functional.linear(output, out_proj) # L N (h d), E E  -> L N E
    return output


class MultiHeadSelfAttention(torch.nn.Module):

    def __init__(self, embed_dim: int, num_heads: int, inner_dim: int, dropout: float = 0.0):
        super().__init__()
        self.inner_dim = inner_dim
        self.num_heads = num_heads
        self.head_dim = inner_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.dropout_p = dropout
        # QKV [(h d 3), E]
        self.qkv_proj = torch.nn.Parameter(torch.empty(3 * inner_dim, embed_dim))
        self.qkv_bias = torch.nn.Parameter(torch.empty(3 * inner_dim))
        # Out
        self.out_proj = torch.nn.Parameter(torch.empty(embed_dim, inner_dim))
        self.out_bias = torch.nn.Parameter(torch.empty(embed_dim))

    def forward(self, query):
        attn = self_attention(
            query, self.qkv_proj, self.qkv_bias,
            self.out_proj,
            self.num_heads, self.scaling, self.dropout_p, mask=False
        )
        attn = attn + self.out_bias
        return attn


@cube.graph.parser.register('L^ N E^, H+ E^, H+, E^ H+ -> L^ N E^', name='feedforward')
def feedforward(x: torch.Tensor,
                proj1: torch.Tensor, proj1_bias: torch.Tensor,
                proj2: torch.Tensor,
                dropout: float,
                is_training: bool = True) -> torch.Tensor:
    x = torch.nn.functional.linear(x, proj1, proj1_bias)
    x = torch.nn.functional.gelu(x)
    x = torch.nn.functional.dropout(x, dropout, is_training, False)
    x = torch.nn.functional.linear(x, proj2, None)
    return x


class MLP(torch.nn.Module):

    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.proj1 = torch.nn.Parameter(torch.empty((hidden_dim, embed_dim)))
        self.proj1_bias = torch.nn.Parameter(torch.empty((hidden_dim,)))
        self.proj2 = torch.nn.Parameter(torch.empty((embed_dim, hidden_dim)))
        self.proj2_bias = torch.nn.Parameter(torch.empty((embed_dim,)))
        self.dropout = dropout

    def forward(self, x: torch.Tensor):
        x = feedforward(x, self.proj1, self.proj1_bias,
                        self.proj2, self.dropout, self.training)
        x = x + self.proj2_bias
        return x


class TransformerLayer(torch.nn.Module):

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        return x
