#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import types
import torch
from typing import Any, Optional
from dataclasses import asdict, replace
import inspect
import copy
import logging
from functools import partial

import nnscaler
from nnscaler.runtime.adapter.reducer import Reducer
from nnscaler.runtime.gnorm import ParamsInfo
from nnscaler.utils import fields

from .trainer_args import (
    TrainerArgs, PrecisionMixin, PolicyMixin, ModuleParallelizeConfig, ComputeConfig,
    load_type
)


logger = logging.getLogger(__name__)


def fork_rng():
    if torch.distributed.is_initialized():
        # only capture the random state of the current device
        # which is good enough for us
        device = torch.cuda.current_device()
        return torch.random.fork_rng([device])
    else:
        return torch.random.fork_rng()


class ModuleParallelizeConfigAdapter(PrecisionMixin, PolicyMixin):
    """
    Adapter for ModuleParallelizeConfig and TrainerArgs
    """
    def __init__(
            self, trainer_args: TrainerArgs,
            parallel_module: Optional[ModuleParallelizeConfig] = None,
            tracing_weights: Optional[dict[str, Any]] = None
    ):
        """
        Args:
            trainer_args: the trainer args
            parallelized_module: the parallelized module config.
                If None, the whole model will be parallelized
        """
        self.trainer_args = trainer_args
        self.parallel_module = parallel_module
        self.tracing_weights = tracing_weights

        # we don't want to load the tracing weights every time
        # It should be loaded only once outside, and passed to the adapter
        if self.parallel_module \
            and self.parallel_module.tracing_from_weights_prefix \
            and not self.tracing_weights:
            raise ValueError('tracing_weights should be provided when tracing_from_weights_prefix is set')

    @property
    def model_type(self):
        return (
            self.parallel_module.model_type
            if self.parallel_module
            else self.trainer_args.model_type
        )

    @property
    def compute_config(self):
        if self.parallel_module:
            if self.parallel_module.compute_config is not None:
                return self.parallel_module.compute_config.resolve(self.trainer_args.compute_config)
            else:
                return replace(self.trainer_args.compute_config, use_end2end=False)
        else:
            return self.trainer_args.compute_config

    @property
    def gen_savedir(self):
        return (
            self.parallel_module.gen_savedir
            if self.parallel_module and self.parallel_module.gen_savedir is not None
            else self.trainer_args.gen_savedir
        )

    @property
    def gen_reuse(self):
        return (
            self.parallel_module.gen_reuse
            if self.parallel_module and self.parallel_module.gen_reuse is not None
            else self.trainer_args.gen_reuse
        )

    @property
    def pas_policy(self):
        return (
            self.parallel_module.pas_policy
            if self.parallel_module and self.parallel_module.pas_policy is not None
            else self.trainer_args.pas_policy
        )

    @property
    def broadcast_strategy(self):
        return (
            self.parallel_module.broadcast_strategy
            if self.parallel_module and self.parallel_module.broadcast_strategy is not None
            else self.trainer_args.broadcast_strategy
        )

    @property
    def instance_name(self):
        return (
            self.parallel_module.instance_name
            if self.parallel_module and self.parallel_module.instance_name is not None
            else self.trainer_args.instance_name
        )

    @property
    def tracing_from_weights(self):
        return (
            self.parallel_module.tracing_from_weights
            if self.parallel_module
            else self.trainer_args.tracing_from_weights
        )

    def load_tracing_weights(self) -> Optional[dict[str, Any]]:
        tracing_weights = None
        if not self.parallel_module:
            # try to reuse the weights from the tracing weights
            tracing_weights = self.tracing_weights
            if self.tracing_from_weights and tracing_weights is None:
                tracing_weights = torch.load(self.tracing_from_weights)
        else:
            if self.tracing_from_weights:
                tracing_weights = torch.load(self.tracing_from_weights)
            elif self.parallel_module.tracing_from_weights_prefix:
                leading_key = self.parallel_module.tracing_from_weights_prefix + '.'
                tracing_weights = {}
                for key in self.tracing_weights:
                    if key.startswith(leading_key):
                        tracing_weights[key[len(leading_key):]] = self.tracing_weights[key]
        return tracing_weights

    @property
    def precision(self):
        return (
            self.parallel_module.precision
            if self.parallel_module and self.parallel_module.precision is not None
            else self.trainer_args.precision
        )

    def create_model(self, module_args: Optional[tuple[tuple, dict]]=None) -> torch.nn.Module:
        model = (
            self.parallel_module.create_model(self.trainer_args, module_args)
            if self.parallel_module
            else self.trainer_args.create_model()
        )
        model = self.to_precision(model)
        tracing_weights = self.load_tracing_weights()
        if tracing_weights:
            model.load_state_dict(tracing_weights)
        return model

    def create_dummy_forward_args(self, dummy_input) -> dict[str, Any]:
        if self.parallel_module:
            return self.fix_input(
                self.parallel_module.create_dummy_forward_args(self.trainer_args)
            )

        # forward args of whole model
        arg_names = list(
            inspect.signature(
                inspect.unwrap(getattr(self.model_type, 'forward'))
            ).parameters.keys()
        )
        return {arg_names[1]: self.fix_input(dummy_input)}  # arg_names[0] is self

    def resolve_compute_config(self):
        compute_config = copy.deepcopy(self.compute_config)
        compute_config.pas_config['__pas_name'] = self.pas_policy
        # autodist configs
        compute_config.pas_config['update_freq'] = self.trainer_args.update_freq
        compute_config.pas_config['use_bf16'] = self.param_dtype == torch.bfloat16
        compute_config.pas_config['use_fp16'] = self.param_dtype == torch.float16

        compute_config.user_config['__from_trainer_args'] = {
            'mbs': self.trainer_args.micro_batch_size,
            'gbs': self.trainer_args.global_batch_size,
            'precision': self.trainer_args.precision,
            'model_args': self.trainer_args.model.args,
        }
        return compute_config

    def parallelize(self,
        dummy_input: Optional[dict[str, Any]] = None, *,
        load_module: bool = True,
        module_args: Optional[tuple[tuple, dict]] = None
    ):
        pmodel_class = nnscaler.parallelize(
            self.model_type,
            self.create_dummy_forward_args(dummy_input),
            self.resolved_pas_policy,
            self.resolve_compute_config(),
            module_fn=partial(self.create_model, module_args=module_args),
            gen_savedir=self.gen_savedir,
            reuse=self.gen_reuse,
            instance_name=self.instance_name,
            broadcast_strategy=self.broadcast_strategy,
            load_module=load_module,
        )
        if load_module:
            return pmodel_class()
        return pmodel_class


def mixin_module(model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    if isinstance(model, nnscaler.ParallelModule):
        return model

    def train_step(self,
        samples: list[Any],
        is_dummy_batch: Optional[list[bool]] = None
    ) -> list[Any]:
        if is_dummy_batch is not None:
            if len(samples) != len(is_dummy_batch):
                raise ValueError('The length of samples and is_dummy_batch should be the same')
            samples = [
                sample
                for sample, is_dummy in zip(samples, is_dummy_batch)
                if not is_dummy
            ]
            if not samples:
                raise ValueError('No real samples in the batch')

            if not all(is_dummy_batch[len(samples):]):
                raise ValueError('Dummy samples should be at the end of the batch')

        forward_outputs = []
        for idx, sample in enumerate(samples):
            with nnscaler.sync_grad_when(idx == len(samples) - 1):
                output = model(sample)
                loss = output[0] if isinstance(output, tuple) else output
                loss.backward()
                forward_outputs.append(output)
        return forward_outputs

    def infer_step(self, samples: list[Any]) -> list[Any]:
        forward_outputs = []
        for sample in samples:
            output = model(sample)
            forward_outputs.append(output)
        return forward_outputs

    def parameters_for_calc_gnorm(self):
        parallel_modules = [m for m in model.modules() if isinstance(m, nnscaler.ParallelModule)]

        params_info = []
        for module in parallel_modules:
            params_info.extend(module.parameters_for_calc_gnorm())

        non_parallel_module_reducer: Reducer = optimizer._non_parallel_module_reducer
        if non_parallel_module_reducer:
            param_info = ParamsInfo(
                non_parallel_module_reducer.ranks,
                non_parallel_module_reducer.parameters_for_optimizer(),
                [],
                non_parallel_module_reducer.zero_ngroups
            )
            params_info.append(param_info)

        return params_info

    model.train_step = types.MethodType(train_step, model)
    model.infer_step = types.MethodType(infer_step, model)
    model.parameters_for_calc_gnorm = types.MethodType(parameters_for_calc_gnorm, model)
    return model


def parallelize_model(trainer_args: TrainerArgs, dummy_input: dict[str, Any], load_module: bool):
    tracing_weights = None
    if trainer_args.tracing_from_weights:
        tracing_weights = torch.load(trainer_args.tracing_from_weights)

    def _new_adapter(parallel_module=None):
        return ModuleParallelizeConfigAdapter(
            trainer_args, parallel_module,
            tracing_weights=tracing_weights
        )

    if not trainer_args.model.parallel_modules:
        # parallelize the whole model
        return _new_adapter().parallelize(dummy_input, load_module=load_module)

    if not load_module and all(pm.args is not None for pm in trainer_args.model.parallel_modules):
        for m in trainer_args.model.parallel_modules:
            _new_adapter(m).parallelize(dummy_input, load_module=False)
        return

    parallel_sub_modules = {
        load_type(m.type): m
        for m in trainer_args.model.parallel_modules
    }
    paralleled_sub_modules = set()

    def _default_new(cls, *args, **kwargs):
        return object.__new__(cls)

    # mock the __new__ method of sub modules to replace them with parallelized version
    # Please note mocking __new__ is very dangerous and error-prone
    # And once you set it, you can never restore it
    # Here we use _default_new to restore it,
    # Setting it to object.__new__ will be wrong
    # Deleting the __new__ method will also be wrong
    # See more https://github.com/python/cpython/issues/105888
    def _patch_new():
        for m in parallel_sub_modules:
            m.__new__ = __parallel__new__

    def _restore_new():
        for m in parallel_sub_modules:
            m.__new__ = _default_new

    # parallelize modules hook
    def __parallel__new__(cls, *args, **kwargs):
        try:
            _restore_new()
            # it can go here when a subclass module of a parallelized module is instantiated
            if cls not in parallel_sub_modules:
                # TODO: pass *args and **kwargs?
                return cls.__new__(cls)
            else:
                if cls in paralleled_sub_modules:
                    logger.warning(
                        f'Parallelized module {cls.__name__} is already created. Previously Parallelized version will be reused.'
                    )
                paralleled_sub_modules.add(cls)
                parallel_module_config = parallel_sub_modules[cls]
                adapter = _new_adapter(parallel_module_config)
                # fork the random state to
                # make sure the modules after parallelized module
                # are the same in all devices.
                # TODO: This will cause the random state to be different to non-parallel version.
                # This is a trade-off to make sure the parallelized module is consistent.
                # Maybe we can use torch.distributed.broadcast to sync the random state in all devices.
                with fork_rng():
                    return adapter.parallelize(dummy_input, load_module=load_module, module_args=(args, kwargs))
        finally:
            _patch_new()

    _patch_new()
    try:
        model = trainer_args.to_precision(trainer_args.create_model())
        missing_modules = set(parallel_sub_modules.keys()) - paralleled_sub_modules
        if missing_modules:
            logger.warning(
                f'The following modules are not parallelized because they are not used: {", ".join(m.__name__ for m in missing_modules)}'
            )
        if load_module:
            return model
    finally:
        _restore_new()
