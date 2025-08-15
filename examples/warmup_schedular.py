#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import math
from torch.optim.lr_scheduler import LRScheduler, Optimizer, _warn_get_lr_called_within_step


class WarmupCosineAnnealingLR(LRScheduler):
    r"""
    torch.optim.lr_scheduler.CosineAnnealingLR with warmup.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_steps (int): Number of warmup steps.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool | str): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

            .. deprecated:: 2.2
                ``verbose`` is deprecated. Please use ``get_last_lr()`` to access the
                learning rate.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        T_max: int,
        eta_min=0.0,
        last_epoch=-1,
        verbose="deprecated",
    ):  # noqa: D107
        self.warmup_steps = warmup_steps
        self.T_max = T_max - warmup_steps + 1
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        """Retrieve the learning rate of each parameter group."""
        _warn_get_lr_called_within_step(self)

        last_epoch_wo_warmup = self.last_epoch - self.warmup_steps + 1
        if last_epoch_wo_warmup < 0:
            return [base_lr * (self.last_epoch + 1) / self.warmup_steps for base_lr in self.base_lrs]
        elif last_epoch_wo_warmup == 0:
            return [base_lr  for base_lr in self.base_lrs]
        elif self._step_count == 1 and last_epoch_wo_warmup > 0:
            return [
                self.eta_min
                + (base_lr - self.eta_min)
                * (1 + math.cos((last_epoch_wo_warmup) * math.pi / self.T_max))
                / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        elif (last_epoch_wo_warmup - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [
                group["lr"]
                + (base_lr - self.eta_min) * (1 - math.cos(math.pi / self.T_max)) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        return [
            (1 + math.cos(math.pi * last_epoch_wo_warmup / self.T_max))
            / (1 + math.cos(math.pi * (last_epoch_wo_warmup - 1) / self.T_max))
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self):
        last_epoch_wo_warmup = self.last_epoch - self.warmup_steps + 1
        if last_epoch_wo_warmup < 0:
            return [base_lr * (self.last_epoch + 1) / self.warmup_steps for base_lr in self.base_lrs]
        else:
            return [
                self.eta_min
                + (base_lr - self.eta_min)
                * (1 + math.cos(math.pi * last_epoch_wo_warmup / self.T_max))
                / 2
                for base_lr in self.base_lrs
            ]
