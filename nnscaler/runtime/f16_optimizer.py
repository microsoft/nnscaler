#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

# CREDITS: This implementation is inspired by Fairseq https://github.com/facebookresearch/fairseq/blob/main/fairseq/optim/fp16_optimizer.py

import logging
from typing import Optional, TYPE_CHECKING

import torch

from nnscaler.cli.train_hook import TrainHook

if TYPE_CHECKING:
    from nnscaler.cli.trainer import Trainer

logger = logging.getLogger(__name__)


class MixedPrecisionF16OptimizerMixin(TrainHook):
    """
    A mixin class for mixed precision optimizer.
    Support both FP16 and BF16 parameters.

    1. It will create a copy of FP32 parameters and grads,
    and use the FP32 copy for optimization (via `build_fp32_params`).
    2. It will sync FP16 grads to FP32 grads before optimizer.step().
    3. It will sync FP32 params back to FP16 params after optimizer.step().
    4. It will zero FP16 grads and FP32 grads to zero in zero_grad().

    """
    def __init__(self, *args, **kwargs):
        # forward __init__ call to the next class in mro(method resolution order)
        super().__init__(*args, **kwargs)
        self._multiply_factor = 1.0

    def after_setup(self, trainer: 'Trainer') -> None:
        """
        Here we override the clip_gnorm and scale_grads methods in the optimizer.
        Reason:
        1. The original clip_gnorm and scale_grads methods apply to bf16 grads, which is not what we want.
           We need to apply them to fp32 grads.
        2. Combine the multiply_factors of clip_gnorm and scale_grads. So only one muliply is needed.
           This can mitigate the precision loss caused by multiple multiplications.
        Assumption:
        `clip_gnorm` is called immediately after `scale_grads` in training loop.
        """
        trainer.optimizer._clip_gnorm = trainer.optimizer.clip_gnorm
        trainer.optimizer.clip_gnorm = self.overrided_clip_gnorm
        trainer.optimizer._scale_grads = trainer.optimizer.scale_grads
        trainer.optimizer.scale_grads = self.overrided_scale_grads

    @classmethod
    def build_fp32_params(cls, params):
        # create FP32 copy of parameters and grads
        fp32_params = []
        for p in params:
            p32 = torch.nn.Parameter(p.data.float())
            p32.grad = torch.zeros_like(p32.data)
            fp32_params.append(p32)
        return fp32_params

    def step(self, closure=None):
        """Performs a single optimization step."""
        self._sync_f16_grads_to_fp32()
        super().step(closure)
        self._sync_fp32_params_to_f16()
        # No need to call gather_params here when zero is enabled,
        # as the gathered params are not in the optimizer

    def zero_grad(self, set_to_none: bool = True):
        """
        Clears the gradients of all optimized parameters.
        Will ignore `set_to_none` and always set fp16 grads to None, and fp32 grads to zero.
        """
        for p in self.f16_params:
            p.grad = None
        for p32 in self.fp32_params:
            if p32.grad is not None:
                p32.grad.zero_()

    def state_dict(self):
        """Return the optimizer's state dict."""
        state_dict = super().state_dict()

        # move fp32_params to the same level with 'exp_avg' and 'exp_avg_sq'
        # we do this to handle the merge of sharded checkpoint in nnscaler
        assert 'state' in state_dict, f'state not found in state_dict: {state_dict.keys()}'
        assert isinstance(state_dict['state'], dict), f'state is not a dict: {type(state_dict["state"])}'
        assert len(self.fp32_params) == len(state_dict['state']), \
                f'len(fp32_params) != len(state[state]): {len(self.fp32_params)} != {len(state_dict["state"])}'
        assert 'exp_avg' in state_dict['state'][0], f'currently only verified for adam-like optimizer'
        for key, value in state_dict['state'].items():
            assert self.fp32_params[key].shape == value['exp_avg'].shape, f'Shape mismatch: {value["exp_avg"].shape} vs {self.fp32_params[key].shape}'
            # .detach(): save tensor instead of Parameter.
            value['fp32_params'] = self.fp32_params[key].detach()

        return state_dict

    def load_state_dict(self, state_dict):
        """Load an optimizer state dict.
        This will also load the fp32_params from the state
        """
        if 'state' in state_dict and len(state_dict['state']) > 0 and 'fp32_params' in state_dict['state'][0]:
            logger.info('try to load fp32_params from state_dict in f16_optimizer')
            assert isinstance(self.fp32_params, list), f'fp32_params is not a list: {type(self.fp32_params)}'
            device = torch.cuda.current_device()
            for i, param in enumerate(self.fp32_params):
                ckpt_param = state_dict['state'][i]['fp32_params']
                assert param.shape == ckpt_param.shape, f'Shape mismatch: {param.shape} vs {ckpt_param.shape}'
                logger.info(f'param {i}, fp16 norm: {param.data.detach().norm().item()}, fp32 norm: {ckpt_param.data.detach().norm().item()}')
                param.data = state_dict['state'][i]['fp32_params'].data.to(device)
                # pop to avoid store a redundant copy in the wrapped optimizer
                state_dict['state'][i].pop('fp32_params')

            if len(self.param_groups) != 1:
                raise RuntimeError('only support one param group')
            self.param_groups[0]['params'] = self.fp32_params

        super().load_state_dict(state_dict)

    def _sync_f16_grads_to_fp32(self):
        # copy FP16 grads to FP32
        for p, p32 in zip(self.f16_params, self.fp32_params):
            if not p.requires_grad:
                continue
            if p.grad is not None:
                if p32.grad is None:
                    p32.grad = p.grad.data.float()
                else:
                    p32.grad.data.copy_(p.grad.data)
            else:
                p32.grad = torch.zeros_like(p.data, dtype=torch.float)
            if self._multiply_factor != 1.0:
                p32.grad.mul_(self._multiply_factor)
        self._multiply_factor = 1.0

    def _sync_fp32_params_to_f16(self):
        # copy FP32 params back into FP16 model
        for p, p32 in zip(self.f16_params, self.fp32_params):
            if not p.requires_grad:
                continue
            p.data.copy_(p32.data)

    def overrided_scale_grads(self, scale: float):
        """
        Scale the gradients by a factor.
        Will override the original scale_grads method in ParallelOptimizer.
        """
        self._multiply_factor *= scale

    def overrided_clip_gnorm(self, max_norm: Optional[float] = None) -> float:
        """
        Will override the original clip_gnorm method in ParallelOptimizer.
        """
        # self._clip_gnorm() is ParallelOptimizer.clip_gnorm
        grad_norm = self._multiply_factor * self._clip_gnorm()
        if max_norm is not None and max_norm > 0.0:
            clip_coef = (max_norm / (grad_norm + 1e-6)).clamp(max=1.0)
            self._multiply_factor *= clip_coef
        return grad_norm


class MixedPrecisionAdam(MixedPrecisionF16OptimizerMixin, torch.optim.Adam):
    def __init__(self, params, **kwargs):
        self.f16_params = list(params)
        self.fp32_params = self.build_fp32_params(self.f16_params)
        super().__init__(self.fp32_params, **kwargs)


class MixedPrecisionAdamW(MixedPrecisionF16OptimizerMixin, torch.optim.AdamW):
    def __init__(self, params, **kwargs):
        self.f16_params = list(params)
        self.fp32_params = self.build_fp32_params(self.f16_params)
        super().__init__(self.fp32_params, **kwargs)
