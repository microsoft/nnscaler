#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import torch
from nnscaler import register_op


@torch.jit.script
def cus_add(a, b):
    return a + b

register_op('*, * -> *')(cus_add)


@torch.jit.script
def cus_sub(a, b):
    return a - b

register_op('*, * -> *')(cus_sub)
