#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from typing import Tuple

import torch
import torch.nn as nn

import nnscaler


@nnscaler.register_op('B (2 h^ 2 w^) C^ -> B (h^ w^) (4 C^)')
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


@nnscaler.register_op('B ic+ (ps^ w^) (ps^ h^), oc ic+ k^ k^, oc -> B oc w^ h^')
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
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding
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
