#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""
This file deals with some special nn modules which have control flows (if/else) in their forward function.
These control flows go different branches according to self.training.
So we rewrite these nn modules to update their forward function, the new forward function uses a registered
customized function to wrap the control flows. nnscaler treats the customized function as a black-box leaf node.

Currently, this file wraps the following nn modules:
    nn.BatchNorm2d
    nn.InstanceNorm2d

At last, we provide a utility function to replace the original nn modules with the wrapped nn modules.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Tuple, List, Dict
from typing import Tuple
import warnings
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.instancenorm import _InstanceNorm
from torch.nn.modules._functions import SyncBatchNorm as sync_batch_norm

from nnscaler.graph.function.function import _unwrap_value
from nnscaler.graph.parser.register import register_op
from nnscaler.ir.operator import IRFwOperation
from nnscaler.ir.cten import IRObject, IRTensor
from nnscaler.runtime.device import DeviceGroup


def wrap_batchnorm2d_func(
    input: Tensor,
    weight: Tensor,
    bias: Tensor,
    running_mean: Tensor,
    running_var: Tensor,
    num_batches_tracked: Tensor,
    momentum: float = 0.1,
    training: bool = True,
    track_running_stats: bool = True,
    eps: float = 1e-05,
    process_group: Tuple[int] = None,
) -> Tensor:
    """
    This function wraps the original batchnorm2d forward function, because it has both control flows and nccl communication.
    Most of the code is copied from https://pytorch.org/docs/stable/_modules/torch/nn/modules/batchnorm.html#SyncBatchNorm
    NOTE: the non-tensor inputs must be kwargs with default value.
    NOTE: the invocation of the function must use kw format to pass kwargs.
    NOTE: process_group and world_size is for the internal nccl communication, process_group specifies
    the group of devices that will perform the synchronization, and world_size specifies the number of devices.
    """
    if input.dim() != 4:
        raise ValueError(f"expected 4D input (got {input.dim()}D input)")

    # exponential_average_factor is set to self.momentum
    # (when it is available) only so that it gets updated
    # in ONNX graph when this node is exported to ONNX.
    if momentum is None:
        exponential_average_factor = 0.0
    else:
        exponential_average_factor = momentum

    if training and track_running_stats:
        # TODO: if statement only here to tell the jit to skip emitting this when it is None
        if num_batches_tracked is not None:  # type: ignore[has-type]
            num_batches_tracked.add_(1)  # type: ignore[has-type]
            if momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / float(num_batches_tracked)
            else:  # use exponential moving average
                exponential_average_factor = momentum

    r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """

    if training:
        bn_training = True
    else:
        bn_training = (running_mean is None) and (running_var is None)

    r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
    # If buffers are not to be tracked, ensure that they won't be updated
    running_mean = running_mean if not training or track_running_stats else None
    running_var = running_var if not training or track_running_stats else None
    # Don't sync batchnorm stats in inference mode (model.eval()).
    need_sync = bn_training and training and process_group is not None
    if need_sync:
        # currently only GPU/PrivateUse1 input is supported
        process_group = DeviceGroup().get_group(process_group)
        if process_group is None:
            process_group = torch.distributed.group.WORLD
        world_size = torch.distributed.get_world_size(process_group)
        need_sync = world_size > 1
    # fallback to framework BN when synchronization is not necessary
    if not need_sync:
        return F.batch_norm(
            input,
            running_mean,
            running_var,
            weight,
            bias,
            bn_training,
            exponential_average_factor,
            eps,
        )
    else:
        assert bn_training
        return sync_batch_norm.apply(
            input,
            weight,
            bias,
            running_mean,
            running_var,
            eps,
            exponential_average_factor,
            process_group,  # type: ignore[possibly-undefined]
            world_size,  # type: ignore[possibly-undefined]
        )


def batchnorm2d_annotation_fn(*inputs, **kwargs):
    assert (
        len(inputs) == 6
    ), f"Expected 6 inputs: input, weight, bias, running_mean, running_var, num_batches_tracked, but got {len(inputs)} {inputs}."
    input, weight, bias, running_mean, running_var, num_batches_tracked = inputs
    """
     Restrictions:
    1. If `weight` is None, then `bias` must also be None. This is because in the absence of `weight`,
       BatchNorm2d does not apply affine transformation, which means there is no need for `bias`.
    2. If `running_mean` is None, then `running_var` and `num_batches_tracked` must also be None.
       This is because `running_mean` and `running_var` are used for tracking the statistics of
       the batch normalization during training. If `running_mean` is not provided, it implies
       that the module should not track statistics, hence `running_var` and `num_batches_tracked`
       should also be absent.
       Reference: https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
    """
    weight = IRObject.try_unwrap(weight)
    bias = IRObject.try_unwrap(bias)
    running_mean = IRObject.try_unwrap(running_mean)
    running_var = IRObject.try_unwrap(running_var)
    num_batches_tracked = IRObject.try_unwrap(num_batches_tracked)

    if weight is None:
        assert bias is None
        wb_annos = "?, ?"
    else:
        assert isinstance(weight, IRTensor)
        assert isinstance(bias, IRTensor)
        wb_annos = "c, c"

    if running_mean is None:
        assert (
            running_var is None and num_batches_tracked is None
        ), "If running_mean is None, both running_var and num_batches_tracked must also be None"
        r_annos = "?, ?, ?"
    else:
        assert isinstance(running_mean, IRTensor)
        assert isinstance(running_var, IRTensor)
        assert isinstance(num_batches_tracked, IRTensor)
        r_annos = "c, c, 1"

    return "n c h^ w^, " + wb_annos + ", " + r_annos + " -> n c h^ w^"


class NnScalerBatchNorm2d(_BatchNorm):
    def forward(self, input: Tensor) -> Tensor:
        return wrap_batchnorm2d_func(
            input,
            self.weight,
            self.bias,
            self.running_mean,
            self.running_var,
            self.num_batches_tracked,
            momentum=self.momentum,
            training=self.training,
            track_running_stats=self.track_running_stats,
            eps=self.eps,
        )


def batchnorm2d_reinit(module: _BatchNorm) -> _BatchNorm:
    """Reinitialize the batchnorm2d module with the same parameters and arguments, but using
    the wrapped module NnScalerBatchNorm2d."""
    if not isinstance(module, _BatchNorm):
        raise TypeError(f"Expected module of type _BatchNorm, but got {type(module)}")
    new_module = NnScalerBatchNorm2d(
        module.num_features,
        module.eps,
        module.momentum,
        module.affine,
        module.track_running_stats,
    )
    if module.affine:
        with torch.no_grad():
            new_module.weight = module.weight
            new_module.bias = module.bias
    new_module.running_mean = module.running_mean
    new_module.running_var = module.running_var
    new_module.num_batches_tracked = module.num_batches_tracked
    return new_module


def emit_batchnorm2d(
    node: IRFwOperation,
    args: List[str],
    kwargs: Dict[str, str],
    runtime_devid: int,
    plan_ndevs: int,
    runtime_ndevs: int,
) -> str:
    """Special rule to generate batchnorm2d node"""

    signature = node.signature

    # Compute scale unit device ids
    offset = (runtime_devid // plan_ndevs) * plan_ndevs
    scale_unit_dev_ids = [local_rank + offset for local_rank in range(plan_ndevs)]

    kw_pairs = list()
    for key, val in kwargs.items():
        code = f"{key}={val}"
        kw_pairs.append(code)

    sub_input = node.inputs()[0]
    full_input = sub_input.parent
    partition_dims = [
        i for i, (s, f) in enumerate(zip(sub_input.shape, full_input.shape)) if s != f
    ]
    assert (
        len(partition_dims) <= 1
    ), f"only support one partition dim for now, got {partition_dims}"

    if len(partition_dims) == 1 and partition_dims[0] == 0:  # partition on batch dim
        # if batch dim is partitioned, it means batchnorm is partitioned in batch dim both
        # within scaleunit and across scaleunits
        kw_pairs.append(f"process_group={tuple(range(runtime_ndevs))}")
    else:
        # the synchronization should occur across scaleunits
        assert len(partition_dims) == 0 or partition_dims[0] != 0
        if runtime_ndevs == len(scale_unit_dev_ids):
            kw_pairs.append("process_group=None")
        else:
            start_id = runtime_devid % len(scale_unit_dev_ids)
            process_group = tuple(
                range(start_id, runtime_ndevs, len(scale_unit_dev_ids))
            )
            kw_pairs.append(f"process_group={process_group}")
            assert len(process_group) == runtime_ndevs // len(scale_unit_dev_ids)

    args_str = ", ".join(args)
    kwargs_str = ", ".join(kw_pairs)
    return f"{signature}({args_str}, {kwargs_str})"


register_op(batchnorm2d_annotation_fn, emit_fn=emit_batchnorm2d)(wrap_batchnorm2d_func)


"""
    This function wraps the original InstanceNorm2d forward function.

    The logic in this function is exactly the same as in the original PyTorch implementation.
    We copied the logic here to register it as a customized operation because nnscaler's
    `register_op` only supports functions, not nn.Module classes. Therefore, this function
    serves as a wrapper around the InstanceNorm2d forward logic, treating the entire function
    as a black-box leaf node in nnscaler.
"""


def wrap_instancenorm2d_func(
    input: Tensor,
    weight: Tensor,
    bias: Tensor,
    running_mean: Tensor,
    running_var: Tensor,
    momentum: float = 0.1,
    eps: float = 1e-05,
    training: bool = True,
    track_running_stats: bool = False,
    num_features: int = 0,
    affine: bool = False,
) -> Tensor:
    """
    This operation applies Instance Normalization over a 4D input (a mini-batch of 2D inputs with additional channel dimension)
    note: `InstanceNorm2d` is appliedon each channel of channeled data like RGB images,usually don't apply affine transform.
    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, H, W)` or :math:`(C, H, W)`
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        momentum: the value used for the running_mean and running_var computation. Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters, initialized the same way as done for batch normalization.
            Default: ``False``.
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics and always uses batch
            statistics in both training and eval modes. Default: ``False``
    Shape:
        - Input: :math:`(N, C, H, W)` or :math:`(C, H, W)`
        - Output: :math:`(N, C, H, W)` or :math:`(C, H, W)` (same shape as input)
    Reference:  https://pytorch.org/docs/stable/_modules/torch/nn/modules/instancenorm.html#InstanceNorm2d
    """

    def _get_no_batch_dim():
        """
        This function returns the dimension that indicates no batch dimension for InstanceNorm2d.
        For 2D data, typically we have the following dimensions:
        - 4D input: (N, C, H, W) where N is the batch size
        - 3D input: (C, H, W) without the batch dimension

        InstanceNorm2d can work with both 4D and 3D inputs.  When the input is 3D, we need to temporarily
        add a batch dimension to perform normalization, and then remove it afterwards.
        """
        return 3

    if input.dim() not in (3, 4):
        raise ValueError(f"expected 3D or 4D input (got {input.dim()}D input)")

    """
    Explanation:
    - For a 4D input (N, C, H, W), the channel dimension is the 2nd dimension (index 1).
    - For a 3D input (C, H, W), the channel dimension is the 1st dimension (index 0).
    This logic ensures that we correctly identify the channel dimension for both 3D and 4D inputs.
    """
    feature_dim = input.dim() - _get_no_batch_dim()
    if input.size(feature_dim) != num_features:
        if affine:
            raise ValueError(
                f"expected input's size at dim={feature_dim} to match num_features"
                f" ({num_features}), but got: {input.size(feature_dim)}."
            )
        else:
            warnings.warn(
                f"input's size at dim={feature_dim} does not match num_features. "
                "You can silence this warning by not passing in num_features, "
                "which is not used because affine=False"
            )

    if input.dim() == _get_no_batch_dim():
        return F.instance_norm(
            input.unsqueeze(0),
            running_mean,
            running_var,
            weight,
            bias,
            training or not track_running_stats,
            momentum,
            eps,
        ).squeeze(0)

    return F.instance_norm(
        input,
        running_mean,
        running_var,
        weight,
        bias,
        training or not track_running_stats,
        momentum,
        eps,
    )


def instancenorm2d_annotation_fn(*inputs, **kwargs):
    assert (
        len(inputs) == 5
    ), "Expected 5 inputs: input, weight, bias, running_mean, running_var"
    input, weight, bias, running_mean, running_var = inputs

    weight = IRObject.try_unwrap(weight)
    bias = IRObject.try_unwrap(bias)
    running_mean = IRObject.try_unwrap(running_mean)
    running_var = IRObject.try_unwrap(running_var)

    if weight is None:
        assert bias is None
        wb_annos = "?, ?"
    else:
        assert isinstance(weight, IRTensor)
        assert isinstance(bias, IRTensor)
        wb_annos = "c^, c^"

    if running_mean is None:
        assert (
            running_var is None
        ), "If running_mean is None, running_var must also be None"
        r_annos = "?, ?"
    else:
        assert isinstance(running_mean, IRTensor)
        assert isinstance(running_var, IRTensor)
        r_annos = "c^, c^"

    # FIXME: c cannot be partitioned, because the kwargs num_features cannot be updated for now
    return "n c^ h^ w^, " + wb_annos + ", " + r_annos + " -> n c^ h^ w^"


register_op(instancenorm2d_annotation_fn)(wrap_instancenorm2d_func)


class NnScalerInstanceNorm2d(_InstanceNorm):
    def forward(self, input: Tensor) -> Tensor:
        return wrap_instancenorm2d_func(
            input,
            self.weight,
            self.bias,
            self.running_mean,
            self.running_var,
            momentum=self.momentum,
            eps=self.eps,
            training=self.training,
            track_running_stats=self.track_running_stats,
            num_features=self.num_features,
            affine=self.affine,
        )


def instancenorm2d_reinit(module: _InstanceNorm) -> _InstanceNorm:
    """Reinitialize the instancenorm2d module with the same parameters and arguments, but using
    the wrapped module NnScalerInstanceNorm2d."""
    new_module = NnScalerInstanceNorm2d(
        module.num_features,
        module.eps,
        module.momentum,
        module.affine,
        module.track_running_stats,
    )
    if module.affine:
        with torch.no_grad():
            new_module.weight = module.weight
            new_module.bias = module.bias
    new_module.running_mean = module.running_mean
    new_module.running_var = module.running_var
    new_module.num_batches_tracked = module.num_batches_tracked
    return new_module


wrapped_modules = {
    torch.nn.BatchNorm2d: batchnorm2d_reinit,
    torch.nn.InstanceNorm2d: instancenorm2d_reinit,
}


_ORIGINAL_MODULE_ATTR = "__nnscaler_original_module__"


def convert_to_wrapnn(module: torch.nn.Module) -> torch.nn.Module:
    """Traverse the module and replace the original nn module with its wrapped version
    if it is in the `wrapped_modules`.
    Currently `wrapped_modules` contains `BatchNorm2d` and `InstanceNorm2d`.

    Please note the child modules of the input module will be replaced in-place.
    You can use `undo_convert_to_wrapnn` to revert the changes.

    It is necessary to call this function on user instantiated model before parallelizing
    the it, otherwise the modules in `wrapped_modules` cannot be partitioned, but be always
    replicated.

    Anyway, it is safe to call this function on the model, even if the model
    does not have the modules in `wrapped_modules`.
    """
    if type(module) in wrapped_modules:
        wrapped = wrapped_modules[type(module)](module)
        # module will be save to children module if we use setattr(wrapped,...)
        object.__setattr__(wrapped, _ORIGINAL_MODULE_ATTR, module)
        return wrapped

    for name, child in module.named_children():
        module.add_module(
            name, convert_to_wrapnn(child)
        )  # will inplace replace the module with the same name
    return module


def undo_convert_to_wrapnn(module: torch.nn.Module) -> torch.nn.Module:
    """
    Undo the effect of `convert_to_wrapnn` function.
    """
    if hasattr(module, _ORIGINAL_MODULE_ATTR):
        origin_module = getattr(module, _ORIGINAL_MODULE_ATTR)
        delattr(module, _ORIGINAL_MODULE_ATTR)
        return origin_module

    for name, child in module.named_children():
        module.add_module(
            name, undo_convert_to_wrapnn(child)
        )  # will inplace replace the module with the same name
    return module


@contextmanager
def wrapnn(module: torch.nn.Module, *, restore: bool = True):
    """
    wrap the nn module and undo the wrap after the context.
    Args:
        module: the nn module to wrap
        restore: whether to restore the original module after the context
    Returns:
        the wrapped module
    """
    try:
        yield convert_to_wrapnn(module)
    finally:
        # just restore the original module inplace
        # return value is discarded
        if restore:
            undo_convert_to_wrapnn(module)
