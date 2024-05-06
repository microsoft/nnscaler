from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np
import deepspeed

from examples.swin.blocks import (
    window_attn, feedforward, window_partition, 
    window_reverse, init_relative_position_index,
    PatchEmbed, PatchMerging, drop_path,
)
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
        tp_size = MPU().get_model_parallel_world_size()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.wh = window_size[0]
        self.ww = window_size[1]
        self.num_heads = num_heads // tp_size
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.rp_bias_table = torch.nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), self.num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # [wh * ww, wh * ww]
        rp_index = init_relative_position_index(self.window_size[0])
        self.register_buffer("rp_index", rp_index)

        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        # qkv
        self.qkv_w = torch.nn.Parameter(torch.empty(dim * 3 // tp_size, dim))
        self.qkv_b = torch.nn.Parameter(torch.empty(dim * 3 // tp_size))

        # out
        self.out_w = torch.nn.Parameter(torch.empty(dim, dim // tp_size))
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
        x = IdentityAllreduce.apply(x)
        x = window_attn(
            x, self.qkv_w, self.qkv_b,
            self.rp_index,
            self.rp_bias_table,
            self.out_w, mask,
            self.attn_drop, self.num_heads,
            self.scale, self.wh, self.ww, nw,
        )
        x = AllreduceIdentity.apply(x)
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


class Mlp(torch.nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        tp_size = MPU().get_model_parallel_world_size()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_w = torch.nn.Parameter(torch.empty(hidden_features // tp_size, in_features))
        self.fc1_b = torch.nn.Parameter(torch.empty(hidden_features // tp_size))
        self.fc2_w = torch.nn.Parameter(torch.empty(out_features, hidden_features // tp_size))
        self.fc2_b = torch.nn.Parameter(torch.empty(out_features))
        self.drop = drop

    def forward(self, x):
        x = IdentityAllreduce.apply(x)
        x = feedforward(x, self.fc1_w, self.fc1_b, self.fc2_w, self.drop)
        x = AllreduceIdentity.apply(x)
        x = x + self.fc2_b
        x = torch.nn.functional.dropout(x, self.drop, True, False)
        return x


class SwinTransformerBlock(torch.nn.Module):
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

@dataclass
class Config:

    # POC test case
    embed_dim: int = 192
    depths: Tuple[int] = (2, 2, 2, 2)
    num_heads: Tuple[int] = (8, 16, 32, 64)

    mlp_ratio: int = 4
    drop_path_rate: float = 0.2
    drop_rate: float = 0.2
    attn_drop_rate: float = 0.0

    # 1536 x 1536
    img_size: int = 1536
    window_size: int = 48

    num_classes: int = 1000


def build_config(num_layers: int, hidden: int, heads: int, img_size: int, window_size: int) -> Config:
    assert (img_size, window_size) in (
        (224, 7),
        (640, 40),
        (1536, 48)
    ), f"image size and window size can only be pairs of {((224, 7), (640, 40), (1536, 48))}"
    assert num_layers > 6, f"layers should be larger than 6"
    config = Config(
        embed_dim = hidden,
        depths = [2, 2, num_layers - 6, 2],
        num_heads = [heads, heads * 2, heads * 4, heads * 8],
        mlp_ratio = 4,
        drop_path_rate = 0.2,
        drop_rate = 0.2,
        attn_drop_rate = 0.0,
        img_size = img_size,
        window_size = window_size,
        num_classes = 1000,
    )
    return config


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
            # x = blk(x)
            x = deepspeed.checkpointing.checkpoint(blk, x)
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

    def __init__(self, cfg: Config):
        super().__init__()
        self.num_classes = cfg.num_classes
        self.num_layers = len(cfg.depths)
        self.embed_dim = cfg.embed_dim
        self.num_features = int(cfg.embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = cfg.mlp_ratio
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

        # torch.manual_seed(0)
        # for param in self.parameters():
        #     if len(param.size()) > 1:
        #         trunc_normal_(param, std=.02)
        #     else:
        #         nn.init.constant_(param, 0)
        # # this is to match for the correctness with baseline
        # for basic_layer in self.layers:
        #     for block in basic_layer.blocks:
        #         with torch.no_grad():
        #             w: torch.Tensor = block.attn.qkv_w.view(3, -1, block.attn.qkv_w.size(-1))
        #             block.attn.qkv_w.copy_(w.permute(1,0,2).reshape(-1, w.size(-1)))

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

    def flops(self, batch_size: int):
        """Total training flops"""
        flops = 0
        flops += self.patch_embed.flops()
        for layer in self.layers:
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        # forward + backward
        flops = flops * 3
        flops = flops * batch_size
        return flops


# =========================== Data Loader =======================

def create_swin_dummy_dataloader(cfg: Config, batch_size: int, dtype: torch.dtype):
    images = torch.randn(
        [3, cfg.img_size, cfg.img_size],
        dtype=dtype, device=torch.cuda.current_device()
    )
    return create_dummy_dataloader(images, batch_size)
