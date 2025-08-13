#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import torch
import torch.nn as nn

from examples.vision.swin.blocks.utils import trunc_normal_
from examples.vision.swin.blocks.transformer import SwinTransformerBlock
from examples.vision.swin.blocks.patch import PatchEmbed, PatchMerging

import nnscaler


class Config:

    # POC test case
    embed_dim = 192
    depths = [2, 2, 2, 2]
    num_heads = [8, 16, 32, 64]

    # swin-large 201M
    # embed_dim = 192
    # depths = [2, 2, 18, 2]
    # num_heads = [6, 12, 24, 48]

    # swin-huge: 2.5B
    # embed_dim = 512
    # depths = [2, 2, 42, 2]
    # num_heads = [16, 32, 64, 128]

    # 355M
    # embed_dim = 256
    # depths = [2, 2, 18, 2]
    # num_heads = [8, 16, 32, 64]

    # 1.8B
    # embed_dim = 512
    # depths = [2, 2, 26, 2]
    # num_heads = [16, 32, 64, 128]

    # 6.6B
    # embed_dim = 768
    # depths = [2, 2, 42, 2]
    # num_heads = [24, 48, 96, 192]

    mlp_ratio = 4
    drop_path_rate = 0.2
    drop_rate = 0.2
    attn_drop_rate = 0.0

    # dataloader

    # 224 x 224
    # img_size = 224
    # window_size = 7

    # 640 x 640
    img_size = 640
    window_size = 40

    # 1536 x 1536
    # img_size = 1536
    # window_size = 48

    num_classes = 1000


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            nnscaler.runtime.function.anchor('transformer block start')
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class SwinTransformer(nn.Module):

    def __init__(self):
        super().__init__()
        cfg = Config()

        self.num_classes = cfg.num_classes
        self.num_layers = len(cfg.depths)
        self.embed_dim = cfg.embed_dim
        self.num_features = int(cfg.embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = 4.


        self.patch_size = 4

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=cfg.img_size,
            patch_size=self.patch_size,
            in_chans=3, embed_dim=cfg.embed_dim,
            norm_layer=nn.LayerNorm
        )
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.pos_drop = nn.Dropout(p=cfg.drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, cfg.drop_path_rate, sum(cfg.depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(cfg.embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=cfg.depths[i_layer],
                               num_heads=cfg.num_heads[i_layer],
                               window_size=cfg.window_size,
                               mlp_ratio=cfg.mlp_ratio,
                               qkv_bias=True, qk_scale=None,
                               drop=cfg.drop_rate, attn_drop=cfg.attn_drop_rate,
                               drop_path=dpr[sum(cfg.depths[:i_layer]):sum(cfg.depths[:i_layer + 1])],
                               norm_layer=nn.LayerNorm,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None)
            self.layers.append(layer)

        self.norm = nn.LayerNorm(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, cfg.num_classes) if cfg.num_classes > 0 else nn.Identity()
        self.criterion = nn.CrossEntropyLoss()

        torch.manual_seed(0)
        for param in self.parameters():
            if len(param.size()) > 1:
                trunc_normal_(param, std=.02)
            else:
                nn.init.constant_(param, 0)
        # this is to match for the correctness with baseline
        for basic_layer in self.layers:
            for block in basic_layer.blocks:
                with torch.no_grad():
                    w: torch.Tensor = block.attn.qkv_w.view(3, -1, block.attn.qkv_w.size(-1))
                    block.attn.qkv_w.copy_(w.permute(1,0,2).reshape(-1, w.size(-1)))

    def forward(self, x):  # , labels: torch.Tensor):
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)

        x = self.head(x)
        # loss = self.criterion(x, labels)
        loss = torch.sum(x)
        return loss

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops


# =========================== Data Loader =======================

def dummy_data(batch_size: int,
               dtype: torch.dtype, cfg: Config):
    input_ids = torch.randn(
        [batch_size, 3, cfg.img_size, cfg.img_size],
        dtype=dtype, device=torch.cuda.current_device()
    )
    return input_ids
