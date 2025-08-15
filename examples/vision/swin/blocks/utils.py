#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import torch
import math


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