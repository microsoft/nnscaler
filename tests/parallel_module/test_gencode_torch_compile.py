#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import tempfile
import pytest
import torch
import math
import torch.nn.functional as F

from nnscaler import parallelize, ComputeConfig, register_op

from tests.utils import replace_all_device_with

from .test_gencode import _gencode_contains, print_gencode


class ActQuant(torch.autograd.Function):

    @staticmethod
    @torch.compile
    def forward(ctx, x):
        dtype = x.dtype
        x = x.float()
        s = 127 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
        x = (x * s).round().clamp(-128, 127) / s
        return x.to(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class ActQuantInt4(torch.autograd.Function):

    @staticmethod
    @torch.compile
    def forward(ctx, x):
        dtype = x.dtype
        x = x.float()
        s = math.sqrt(7) / x.abs().mean(dim=-1, keepdim=True).clamp_(min=1e-5)
        x = (x * s).round().clamp(-8, 7) / s
        return x.to(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class ActQuantInt2(torch.autograd.Function):

    @staticmethod
    @torch.compile
    def forward(ctx, x):
        dtype = x.dtype
        x = x.float()
        s = math.sqrt(3) / x.abs().mean(dim=-1, keepdim=True).clamp_(min=1e-5)
        x = (x * s).round().clamp(-4, 3) / s
        return x.to(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input

# Two ways to register autograd functions:
# 1. Use `@register_op` decorator
# 2. Use `register_op` function directly, and pass `Function` or `Function.apply`.

@register_op('*^ -> *^')
class WeightQuant(torch.autograd.Function):

    @staticmethod
    @torch.compile
    def forward(ctx, x):
        dtype = x.dtype
        x = x.float()
        s = 1.0 / x.abs().mean().clamp_(min=1e-5)
        x = (x * s).round().clamp(-1, 1) / s
        return x.to(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


register_op('*^ -> *^')(ActQuant)
register_op('*^ -> *^')(ActQuantInt2.apply)
register_op('*^ -> *^')(ActQuantInt4.apply)


class BitLinear(torch.nn.Linear):
    def __init__(self, in_features: int, out_features: int, split_size: list[int], bias: bool = True, act_bits: int = 8):
        super(BitLinear, self).__init__(in_features, out_features, bias)
        self.split_size = split_size
        self.act_bits = act_bits
        assert sum(split_size) == out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(self.split_size) == 1:
            weight = WeightQuant.apply(self.weight)
        else:
            weight = torch.split(self.weight, self.split_size, dim=0)
            weight = [WeightQuant.apply(w) for w in weight]
            weight = torch.cat(weight, dim=0)
        if self.act_bits == 8:
            input = ActQuant.apply(x)
        elif self.act_bits == 4:
            input = ActQuantInt4.apply(x)
        elif self.act_bits == 2:
            input = ActQuantInt2.apply(x)
        else:
            raise ValueError(f"Unsupported act_bits: {self.act_bits}")
        return F.linear(input, weight, self.bias)


# looks torch.compile needs gpu, not sure why.
@pytest.mark.skipif(not torch.cuda.is_available(), reason='lack of gpu devices')
def test_codegen_compile_apply(tmp_path):
    m = BitLinear(64, 128, [64, 64], bias=False)
    m.train()
    parallelize(
        m,
        {'x': torch.randn(128, 64)},
        'dp',
        ComputeConfig(1, 1),
        gen_savedir=tmp_path,
        load_module=False,
        reuse='override',
    )
    assert True


@register_op('* -> *')
@torch.compile
def f(x):
    return x * 2


class Module1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)

    def forward(self, x):
        return self.linear(x) + f(x)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='lack of gpu devices')
def test_codegen_compile_f():
    with tempfile.TemporaryDirectory() as tempdir:
        parallelize(
            Module1(),
            {'x': torch.randn(3, 3)},
            'dp',
            ComputeConfig(1, 1),
            gen_savedir=tempdir,
            load_module=False
        )
        # parallelize will succeed.
        assert True


@torch.compile
def g(x):
    # g is not registered in nnscaler
    # RuntimeError will be raised
    # when parallelize is called.
    return x * 2


class Module2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)

    def forward(self, x):
        return self.linear(x) + g(x)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='lack of gpu devices')
def test_codegen_compile_failed_g():
    with pytest.raises(RuntimeError), tempfile.TemporaryDirectory() as tempdir:
        parallelize(
            Module2(),
            {'x': torch.randn(3, 3)},
            'dp',
            ComputeConfig(1, 1),
            gen_savedir=tempdir,
            load_module=False
        )
        # parallelize will succeed.
        assert True
