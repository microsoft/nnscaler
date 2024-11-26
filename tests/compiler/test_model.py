#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import os
import logging
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F

import pytest

import nnscaler
from nnscaler.parallel import parallelize, ComputeConfig, ReuseType

from ..launch_torchrun import launch_torchrun
from ..utils import assert_parity


def upfirdn2d_native(input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1):
    _, minor, in_h, in_w = input.shape
    kernel_h, kernel_w = kernel.shape

    out = input.view(-1, minor, in_h, 1, in_w, 1)
    out = F.pad(out, [0, up_x - 1, 0, 0, 0, up_y - 1, 0, 0])
    out = out.view(-1, minor, in_h * up_y, in_w * up_x)

    out = F.pad(out, [max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)])
    out = out[
        :,
        :,
        max(-pad_y0, 0) : \
            out.shape[2] - max(-pad_y1, 0),
        max(-pad_x0, 0) : \
            out.shape[3] - max(-pad_x1, 0),
    ]

    out = out.permute(0, 3, 1, 2)
    out = out.reshape([-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1])
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = F.conv2d(out, w)
    out = out.reshape(
        -1,
        minor,
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
    )
    out = out.permute(0, 2, 3, 1)

    return out[:, :, ::down_y, ::down_x]


def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    return upfirdn2d_native(input, kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1])


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)
    if k.ndim == 1:
        k = k[None, :] * k[:, None]
    k /= k.sum()
    return k


class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()
        self.factor = factor
        kernel = make_kernel(kernel) * (factor**2)
        self.register_buffer("kernel", kernel)
        p = kernel.shape[0] - factor
        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2
        self.pad = (pad0, pad1)

    def forward(self, input):
        return upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)


def baseline():
    model = Upsample([1, 3, 3, 1], factor=2)
    torch.manual_seed(0)
    input = torch.rand(1, 3, 64, 64)

    # single gpu execution
    model = model.cuda()
    single_out = model(input.cuda())
    single_out = single_out.to('cpu')
    return single_out


def parallelize_run(tmp_path):
    model = Upsample([1, 3, 3, 1], factor=2)
    torch.manual_seed(0)
    input = torch.rand(1, 3, 64, 64)

    # multiple gpu execution
    nnscaler.init()
    nnscaler.utils.set_default_logger_level(logging.INFO)
    compute_config = ComputeConfig(
        2, 2, constant_folding=True, use_zero=True,
        pas_config={'update_freq': 1, '_batch_size': 1}
    )
    para_model = parallelize(
        model,
        {'input': input,},
        'autodist',
        compute_config,
        reuse=ReuseType.OVERRIDE,
        gen_savedir=tmp_path
    )
    para_model = para_model.cuda()
    para_out = para_model(input.cuda())
    para_out = para_out.to('cpu')
    return para_out


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason='lack of gpu devices')
def test_complex_model(tmp_path):
    """
    The simplified model `Upsample` has complex operations,
    such as, slice, max, permute, reshape, view, pad, flip, conv2d, etc.
    """

    launch_torchrun(2, assert_parity, baseline, partial(parallelize_run, tmp_path))



@nnscaler.register_op('m n -> m n, m n, ?')
def func_multi_outputs(x):
    return x, x, 3


# NOTE: "x" can be partitioned because "?" has no dependency on `x`
@nnscaler.register_op('m n -> m n, ?')
def func_output_complex_dict(x, factor=1):
    x = x * factor
    return x, {'y': 10}


# NOTE: "x" can be partitioned because "?" has no dependency on `x`
@nnscaler.register_op('m n -> m n, ?')
def func_output_complex_slice(x, factor=1):
    x = x * factor
    return x, slice(0, 10, factor)


# NOTE: "x" cannot be partitioned because there is "?" in output annotation
# and has dependency on `x`
@nnscaler.register_op('m^ n^ -> ?')
def func_output_list(x, factor=1):
    x = x * factor
    return [x, x]


class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = func_multi_outputs(x)
        y, _, scalar = x
        (sz, _) = y.shape
        sz = sz + scalar
        out = func_output_list(y, factor=sz)
        out = out[0]
        out, out_dict = func_output_complex_dict(out, factor=scalar)
        out, out_slice = func_output_complex_slice(out, factor=out_dict['y'])
        return {'out': out, 'slice_start': out_slice.start}


def single_run():
    torch.manual_seed(0)
    dummy_input = torch.randn(4, 4)
    module = MyModule()
    module = module.cuda()
    out = module(dummy_input.cuda())
    out = {'out': [each.to('cpu') for each in out['out']], 'slice_start': out['slice_start']}
    return out


def two_gpu_run(tmp_path):
    torch.manual_seed(0)
    dummy_input = torch.randn(4, 4)
    module = MyModule()
    module = module.cuda()
    nnscaler.init()
    nnscaler.utils.set_default_logger_level(logging.INFO)
    compute_config = ComputeConfig(
        2, 2, constant_folding=False, use_zero=True,
        pas_config={'update_freq': 1, '_batch_size': 1}
    )
    para_module = parallelize(
        module,
        {'x': dummy_input.cuda()},
        'tp',
        compute_config, reuse=ReuseType.OVERRIDE,
        gen_savedir=tmp_path
    )
    para_module = para_module.cuda()
    out = para_module(dummy_input.cuda())
    out = {'out': [each.to('cpu') for each in out['out']], 'slice_start': out['slice_start']}
    return out


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason='lack of gpu devices')
def test_complex_outputs(tmp_path):
    launch_torchrun(2, assert_parity,
        single_run,
        partial(two_gpu_run, tmp_path)
    )
