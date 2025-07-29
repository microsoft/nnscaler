from typing import TYPE_CHECKING, Any, Callable, ContextManager, Dict, Literal, Optional, Union

import torch
from torch import Tensor
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


_PRECISION_INPUT = Literal["32-true", "16-true", "bf16-true"]


class NnScalerPrecision(Precision):
    """Precision plugin for training with nnscaler.

    .. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

    Args:
        precision: Full precision (32-true), half precision (16-true, bf16-true)

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
        return _DtypeContextManager(self._desired_input_dtype)

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
    def optimizer_step(  # type: ignore[override]
        self,
        optimizer: Steppable,
        model: "pl.LightningModule",
        closure: Callable[[], Any],
        **kwargs: Any,
    ) -> Any:
        if self.scaler is None:
            return super().optimizer_step(optimizer, model=model, closure=closure, **kwargs)

        closure_result = closure()

        if not _optimizer_handles_unscaling(optimizer):
            # Unscaling needs to be performed here in case we are going to apply gradient clipping.
            # Optimizers that perform unscaling in their `.step()` method are not supported (e.g., fused Adam).
            # Note: `unscale` happens after the closure is executed, but before the `on_before_optimizer_step` hook.
            self.scaler.unscale_(optimizer)  # type: ignore[arg-type]

        self._after_closure(model, optimizer)
        skipped_backward = closure_result is None
        # in manual optimization, the closure does not return a value
        if not model.automatic_optimization or not skipped_backward:
            # note: the scaler will skip the `optimizer.step` if nonfinite gradients are found
            step_output = self.scaler.step(optimizer, **kwargs)  # type: ignore[arg-type]
            self.scaler.update()
            return step_output
        return closure_result

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
