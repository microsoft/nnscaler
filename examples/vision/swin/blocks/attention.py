#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from typing import Optional
import torch
import nnscaler


# REMARK: as default attention has qkv project weight of (3 head dim_head) C,
# this cannot partition on head dimension
# as the head dimension is a secondary hidden dimension in (3 head dim_head).
# To make partition work (correctness guarantee), the dimension is swapped as (head dim_head 3)
@nnscaler.register_op('B N^ C^, (h+ dh^ 3) C^, (h+ dh^ 3), (wh^ ww^) (wh^ ww^), t^ h+, C^ (h+ dh^), ? -> B N^ C^')
def window_attn(x: torch.Tensor,
                qkv_w: torch.Tensor, qkv_bias: torch.Tensor,
                relative_position_index: torch.Tensor,
                relative_position_bias_table: torch.Tensor,
                dense_w: torch.Tensor,
                mask: Optional[torch.Tensor],
                attn_drop: float,
                h: int, scale: float, wh: int, ww: int,):
    """
    @param h  int: number of head
    @param wh int: window size of height
    @param ww int: window size of width
    """
    B_, N, C = x.shape
    dh = qkv_w.size(0) // 3 // h
    # B N (h+ dh 3)
    qkv = torch.nn.functional.linear(x, qkv_w, qkv_bias)
    # 3 B h N dh
    qkv = qkv.reshape(B_, N, h, dh, 3).permute(4, 0, 2, 1, 3)
    # 3 B h N dh
    # qkv = qkv.reshape(B_, N, 3, h, dh).permute(2, 0, 3, 1, 4)
    # B h N dh
    q, k, v = qkv[0], qkv[1], qkv[2]
    # B h N dh
    q = q * scale
    # B h N dh @ B h dh N -> B h N N
    attn = (q @ k.transpose(-2, -1))
    # (wh ww) (wh * ww) h
    relative_position_bias = relative_position_bias_table[
        relative_position_index.view(-1)
    ].view(wh * ww, wh * ww, -1)
    # h (wh ww) (wh ww)
    relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
    # attn: B h N N
    attn = attn + relative_position_bias.unsqueeze(0)
    if mask is not None:
        nW = mask.shape[0]
        attn = attn.view(B_ // nW, nW, h, N, N) + mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, h, N, N)
        attn = torch.nn.functional.softmax(attn, dim=-1)
    else:
        attn = torch.nn.functional.softmax(attn, dim=-1)
    # attn: B h N N
    attn = torch.nn.functional.dropout(attn, attn_drop, True, False)
    # B h N N @ B h N dh -> B h N dh -> B N h dh -> B N h * dh
    x = (attn @ v).transpose(1, 2).reshape(B_, N, h * dh)
    x = torch.nn.functional.linear(x, dense_w)
    return x


def init_relative_position_index(window_size: int) -> torch.Tensor:
    coords_h = torch.arange(window_size)
    coords_w = torch.arange(window_size)
    coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_coords[:, :, 0] += window_size - 1  # shift to start from 0
    relative_coords[:, :, 1] += window_size - 1
    relative_coords[:, :, 0] *= 2 * window_size - 1
    relative_position_index = relative_coords.sum(-1)  # wh * ww, wh * ww
    return relative_position_index


class WindowAttention(torch.nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.wh = window_size[0]
        self.ww = window_size[1]
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.rp_bias_table = torch.nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # wh * ww, wh * ww
        self.register_buffer('rp_index', relative_position_index)

        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        # qkv
        self.qkv_w = torch.nn.Parameter(torch.empty(dim * 3, dim))
        self.qkv_b = torch.nn.Parameter(torch.empty(dim * 3))

        # out
        self.out_w = torch.nn.Parameter(torch.empty(dim, dim))
        self.out_b = torch.nn.Parameter(torch.empty(dim))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        x = window_attn(
            x, self.qkv_w, self.qkv_b,
            self.rp_index,
            self.rp_bias_table,
            self.out_w, mask,
            self.attn_drop, self.num_heads,
            self.scale, self.wh, self.ww
        )
        x = x + self.out_b
        x = torch.nn.functional.dropout(x, self.proj_drop, True, False)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops