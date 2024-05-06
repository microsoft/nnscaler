from typing import Tuple, Optional
import torch
import torch.nn as nn

import math
import cube


def trunc_normal_(tensor: torch.Tensor, mean: float = 0., std: float = 1., a: float = -2., b: float = 2.):
    with torch.no_grad():
        l = (1. + math.erf((a - mean) / std / math.sqrt(2.))) / 2.
        u = (1. + math.erf((b - mean) / std / math.sqrt(2.))) / 2.
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
    return tensor

def get_position_bias(relative_position_index: torch.Tensor,
                      relative_position_bias_table: torch.Tensor,
                      wh: int, ww: int) -> torch.Tensor:
    # (wh ww) (wh * ww) h
    relative_position_bias = relative_position_bias_table[
        relative_position_index.view(-1)
    ].view(wh * ww, wh * ww, -1)
    # h (wh ww) (wh ww)
    relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
    return relative_position_bias

@cube.graph.parser.register('(B nw^) N^ C^, (h+ dh^ 3) C^, (h+ dh^ 3), (wh^ ww^) (wh^ ww^), wt^ h+, C^ (h+ dh^), ? -> (B nw^) N^ C^')
def window_attn(x: torch.Tensor,
                qkv_w: torch.Tensor,
                qkv_bias: torch.Tensor,
                rp_index: torch.Tensor,
                rp_bias: torch.Tensor,
                dense_w: torch.Tensor,
                mask: Optional[torch.Tensor],
                attn_drop: float,
                h: int, scale: float, wh: int, ww: int, nw: int):
    """Window attention.

    Note:
        default attention has qkv project weight of (3 head dim_head) C, 
        this cannot partition on head dimension as the head dimension is a secondary hidden 
        dimension in (3 head dim_head). To make partition work (correctness guarantee),
        the dimension is swapped as (head dim_head 3)

    Args:
        h  (int): number of head
        wh (int): window size of height
        ww (int): window size of width
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
    # attn: B h N N
    # ============== comment following during profiling ==================
    position_bias = get_position_bias(rp_index, rp_bias, wh, ww)
    attn = attn + position_bias
    attn = attn + position_bias.unsqueeze(0)
    # ====================================================================
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

        # [wh * ww, wh * ww]
        rp_index = init_relative_position_index(self.window_size[0])
        self.register_buffer("rp_index", rp_index)

        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        # qkv
        self.qkv_w = torch.nn.Parameter(torch.empty(dim * 3, dim))
        self.qkv_b = torch.nn.Parameter(torch.empty(dim * 3))

        # out
        self.out_w = torch.nn.Parameter(torch.empty(dim, dim))
        self.out_b = torch.nn.Parameter(torch.empty(dim))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor], nw: int):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
            nw int: number of windows
        """
        # position_bias = get_position_bias(self.rp_index, self.rp_bias_table, self.wh, self.ww)
        # position_bias = self.position_bias
        x = window_attn(
            x, self.qkv_w, self.qkv_b,
            self.rp_index,
            self.rp_bias_table,
            self.out_w, mask,
            self.attn_drop, self.num_heads,
            self.scale, self.wh, self.ww, nw,
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
        flops += N * 2 * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * 2 * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * 2 * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * 2 * self.dim * self.dim
        return flops


@cube.graph.parser.register('B HW^ E^, H+ E^, H+, E^ H+ -> B HW^ E^', name='feedforward')
def feedforward(x: torch.Tensor,
                proj1: torch.Tensor, proj1_bias: torch.Tensor,
                proj2: torch.Tensor, dropout: float) -> torch.Tensor:
    x = torch.nn.functional.linear(x, proj1, proj1_bias)
    x = torch.nn.functional.gelu(x)
    x = torch.nn.functional.dropout(x, dropout, True, False)
    x = torch.nn.functional.linear(x, proj2, None)
    return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_w = torch.nn.Parameter(torch.empty(hidden_features, in_features))
        self.fc1_b = torch.nn.Parameter(torch.empty(hidden_features))
        self.fc2_w = torch.nn.Parameter(torch.empty(out_features, hidden_features))
        self.fc2_b = torch.nn.Parameter(torch.empty(out_features))
        self.drop = drop

    def forward(self, x):
        x = feedforward(x, self.fc1_w, self.fc1_b, self.fc2_w, self.drop)
        x = x + self.fc2_b
        x = torch.nn.functional.dropout(x, self.drop, True, False)
        return x


@cube.graph.parser.register('B (2 h^ 2 w^) C^ -> B (h^ w^) (4 C^)')
def patch_merge(x: torch.Tensor, h: int, w: int):
    B, L, C = x.shape
    H = 2 * h
    W = 2 * w
    assert L == H * W, "input feature has wrong size"
    assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
    x = x.view(B, H, W, C)
    x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
    x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
    x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
    x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
    x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
    x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C
    return x

@cube.graph.parser.register('B ic+ (ps^ w^) (ps^ h^), oc ic+ k^ k^, oc -> B oc w^ h^')
def patch(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, ps: int):
    """
    @param ps int: patch size
    """
    return torch.conv2d(x, w, b, stride=ps)


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution: Tuple[int, int], dim: int, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.H = input_resolution[0]
        self.W = input_resolution[1]
        # self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B (H W) C
        """
        x = patch_merge(x, self.H // 2, self.W // 2)
        x = self.norm(x)
        x = self.reduction(x)
        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        flops = self.H * self.W * self.dim
        flops += (self.H // 2) * (self.W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patches_resolution = [img_size[0] // patch_size, img_size[1] // patch_size]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        # patch_size = (patch_size, patch_size)
        # self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.conv_w = nn.Parameter(torch.empty(embed_dim, in_chans, self.patch_size, self.patch_size))
        self.conv_b = nn.Parameter(torch.empty(embed_dim))
    
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        # x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        x = patch(x, self.conv_w, self.conv_b, self.patch_size).flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size * self.patch_size)
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


@cube.graph.parser.register('* -> *')
def drop_path(x: torch.Tensor, drop_prob: float, training: bool):
    if drop_prob <= 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


@cube.graph.parser.register('B (nh^ ws^) (nw^ ws^) C -> (B nh^ nw^) ws^ ws^ C')
def window_partition(x: torch.Tensor, ws: int):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    window_size = ws
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


@cube.graph.parser.register('(B nh^ nw^) ws^ ws^ C -> B (nh^ ws^) (nw^ ws^) C')
def window_reverse(windows: torch.Tensor, ws: int, nh: int, nw: int):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    window_size = ws
    B = int(windows.shape[0] / (nh * nw))
    x = windows.view(B, nh, nw, window_size, window_size, -1)
    H = nh * ws
    W = nw * ws
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim: int, input_resolution: Tuple[int, int], num_heads: int, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.H = input_resolution[0]
        self.W = input_resolution[1]
        self.nw = self.H * self.W // (window_size * window_size)
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = drop_path
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        # self.attn_mask = attn_mask
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H = self.H
        W = self.W
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, self.attn_mask, self.nw)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H // self.window_size, W // self.window_size)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + drop_path(x, self.drop_path, self.training)
        x = x + drop_path(self.mlp(self.norm2(x)), self.drop_path, self.training)

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += H * W * 2 * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops
