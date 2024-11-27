#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import torch
from types import MethodType
from nnscaler.graph.tracer.operator_patcher import OperatorPatcher


def test_patch_func_or_module():
    op_patcher = OperatorPatcher(True, [])

    # case 1: normal function
    def normal_func(a, b):
        return a + b
    new_func = op_patcher.patch_func_or_module(normal_func)
    assert normal_func == new_func

    def normal_func_with_compare(a, b):
        return a is not b
    new_func = op_patcher.patch_func_or_module(normal_func_with_compare)
    assert normal_func_with_compare != new_func

    # case 2: bound function
    obj = object()
    bound_func = MethodType(normal_func, obj)
    new_func = op_patcher.patch_func_or_module(bound_func)
    assert bound_func == new_func

    obj = object()
    bound_func_with_compare = MethodType(normal_func_with_compare, obj)
    new_func = op_patcher.patch_func_or_module(bound_func)
    assert bound_func_with_compare != new_func

    # case 3: module
    class SimpleModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x):
            return x
    
    model = SimpleModel()
    orig_forward = model.forward
    model_with_orig_forward = op_patcher.patch_func_or_module(model)
    assert model == model_with_orig_forward and model_with_orig_forward.forward == orig_forward

    class SimpleModelWithCompare(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x, cp1, cp2):
            if cp1 is cp2:
                return x
            else:
                return x * 2

    model = SimpleModelWithCompare()
    orig_forward = model.forward
    model_with_new_forward = op_patcher.patch_func_or_module(model)
    assert model == model_with_new_forward and model_with_new_forward.forward != orig_forward

    # case 4: module.forward
    model = SimpleModel()
    orig_forward = model.forward
    new_forward = op_patcher.patch_func_or_module(model.forward)
    assert new_forward == orig_forward and not isinstance(new_forward, torch.nn.Module)

    model = SimpleModelWithCompare()
    orig_forward = model.forward
    new_forward = op_patcher.patch_func_or_module(model.forward)
    assert new_forward != orig_forward and not isinstance(new_forward, torch.nn.Module)
