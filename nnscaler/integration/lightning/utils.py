#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import torch
from torch.optim.lbfgs import LBFGS

from lightning.pytorch.utilities.exceptions import MisconfigurationException


def inplace_optimizer_fn(optimizer, params):
    # hack to replace the optimizer's param_groups with the new params
    optimizer.param_groups[0]['params'] = list(params)
    # handle special cases. e.g. LBFGS
    if isinstance(optimizer, LBFGS):
        raise MisconfigurationException("LBFGS optimizer is not supported.")
    return optimizer
