#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

# Code modified from: https://github.com/Lightning-AI/pytorch-lightning/blob/master/src/lightning/pytorch/plugins/precision/fsdp.py

# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING, Any, Callable, ContextManager, Dict, Literal, Optional, Union

import torch
from torch import Tensor
import torch.distributed
from torch.optim import Optimizer
import torch.amp

import lightning.pytorch as pl
from lightning_utilities import apply_to_collection
from typing_extensions import get_args, override

from lightning.fabric.plugins.precision.amp import _optimizer_handles_unscaling
from lightning.fabric.plugins.precision.utils import _convert_fp_tensor, _DtypeContextManager
from lightning.pytorch.plugins.precision.precision import Precision
from lightning.fabric.utilities.types import Steppable
from lightning.pytorch.utilities import GradClipAlgorithmType


_PRECISION_INPUT = Literal["32-true", "16-true", "bf16-true", "16-mixed", "bf16-mixed"]


class NnScalerPrecision(Precision):
    """Precision plugin for training with nnscaler.

    .. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

    Args:
        precision: Full precision (32-true), half precision (16-true, bf16-true) or
            mixed precision (16-mixed, bf16-mixed).

    Raises:
        ValueError:
            If unsupported ``precision`` is provided.

    """

    def __init__(
        self,
        precision: _PRECISION_INPUT,
        scaler=None,
    ) -> None:
        """
        Args:
            scaler: a torch.amp.GradScaler-like object, supporting
                * ``scaler.scale(loss)`` multiplies a given loss by ``scaler``'s current scale factor.
                * ``scaler.step(optimizer)`` safely unscales gradients and calls ``optimizer.step()``.
                * ``scaler.update()`` updates ``scaler``'s scale factor.
        """
        supported_precision = get_args(_PRECISION_INPUT)
        if precision not in supported_precision:
            raise ValueError(
                f"`precision={precision!r})` is not supported in nnScaler."
                f" `precision` must be one of: {supported_precision}."
            )

        self.precision = precision
        self.scaler = scaler

        precision_to_type = {
            "bf16-mixed": torch.float32,
            "16-mixed": torch.float32,
            "bf16-true": torch.bfloat16,
            "16-true": torch.float16,
            "32-true": torch.float32,
        }
        self._desired_input_dtype = precision_to_type[self.precision]

    @override
    def convert_module(self, module: torch.nn.Module) -> torch.nn.Module:
        return module.to(dtype=self._desired_input_dtype)

    @override
    def convert_input(self, data: Any) -> Any:
        return apply_to_collection(data, function=_convert_fp_tensor, dtype=Tensor, dst_type=self._desired_input_dtype)

    @override
    def tensor_init_context(self) -> ContextManager:
        return _DtypeContextManager(self._desired_input_dtype)

    @override
    def module_init_context(self) -> ContextManager:
        return self.tensor_init_context()

    @override
    def forward_context(self) -> ContextManager:
        if "mixed" in self.precision:
            return torch.autocast("cuda", dtype=(torch.bfloat16 if self.precision == "bf16-mixed" else torch.float16))
        return self.tensor_init_context()

    @override
    def convert_input(self, data: Any) -> Any:
        return apply_to_collection(data, function=_convert_fp_tensor, dtype=Tensor, dst_type=self._desired_input_dtype)

    @override
    def convert_output(self, data: Any) -> Any:
        return apply_to_collection(data, function=_convert_fp_tensor, dtype=Tensor, dst_type=torch.get_default_dtype())

    @override
    def pre_backward(self, tensor: Tensor, module: "pl.LightningModule") -> Tensor:  # type: ignore[override]
        if self.scaler is not None:
            tensor = self.scaler.scale(tensor)
        return super().pre_backward(tensor, module)

    @override
    def _after_closure(self, model: "pl.LightningModule", optimizer: Steppable) -> None:
        if self.scaler is None:  # will be handled in optimizer_step instead of here when using scaler
            self._sync_grad(model, optimizer)
        super()._after_closure(model, optimizer)

    def _sync_grad(self, model: "pl.LightningModule", optimizer: Steppable):
        optimizer.sync_shard_grad()  # closure is used, so we have to sync gradients after closure
        cf = model._trainer.strategy.compute_config
        optimizer.scale_grads(cf.plan_ngpus / cf.runtime_ngpus)

    @override
    def optimizer_step(  # type: ignore[override]
        self,
        optimizer: Steppable,
        model: "pl.LightningModule",
        closure: Callable[[], Any],
        **kwargs: Any,
    ) -> Any:
        if self.scaler is None:
            return super().optimizer_step(optimizer, model=model, closure=closure, **kwargs)

        # TODO: test the following logic

        closure_result = closure()
        self._sync_grad(model, optimizer)
        # Note: `unscale` happens after the closure is executed, but before the `on_before_optimizer_step` hook.
        # Unscaling needs to be performed after grad sync but before gradient clipping
        self.scaler.unscale_(optimizer)

        self._after_closure(model, optimizer)

        if not model.automatic_optimization:
            raise ValueError("nnscaler does not support manual optimization.")
        if closure_result is None:
            # in manual optimization, the closure does not return a value
            raise ValueError("nnscaler does not support None as the return value of the closure.")

        # note: the scaler will skip the `optimizer.step` if nonfinite gradients are found
        step_output = self.scaler.step(optimizer, **kwargs)  # type: ignore[arg-type]
        self.scaler.update()
        return step_output

    def clip_gradients(
        self,
        optimizer: Optimizer,
        clip_val: Union[int, float] = 0.0,
        gradient_clip_algorithm: GradClipAlgorithmType = GradClipAlgorithmType.NORM,
    ) -> None:
        """Clips the gradients."""
        if clip_val <= 0:
            return
        if gradient_clip_algorithm == GradClipAlgorithmType.VALUE:
            raise ValueError('nnscaler does not support clipping gradients by value.')
        elif gradient_clip_algorithm == GradClipAlgorithmType.NORM:
            optimizer.clip_gnorm(clip_val)  # define in nnscaler

    @override
    def state_dict(self) -> Dict[str, Any]:
        if self.scaler is not None:
            return self.scaler.state_dict()
        return {}

    @override
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if self.scaler is not None:
            self.scaler.load_state_dict(state_dict)
