#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from dataclasses import asdict, dataclass, field
import importlib
from typing import Any, Dict, List, Literal, Optional, TYPE_CHECKING, Union
from typing_extensions import get_args
from pathlib import Path
import logging
import copy
import os
import builtins

import torch
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
import yaml
import torch

from nnscaler.utils import transform_recursively
from nnscaler.parallel import ComputeConfig, build_optimizer, ReuseType, BroadcastGenFilesStrategy, _PREDEFINED_POLICIES
from nnscaler.runtime.module import ParallelModule

from .arg_parser import deserialize_dataclass, merge_args, parse_args, _TYPE_KEY, _VALUE_TYPE_KEY, _VALUE_KEY
from .loggers.logger_base import LoggerBase
from .train_hook import TrainHook


logger = logging.getLogger(__name__)


def load_type(type_name: str):
    """
    Load function/class from its full qualified name
    """
    if callable(type_name):  # a function or class
        return type_name

    parts = type_name.split('.')

    # s: the number of parts to be the namespace
    # s == 0: use builtins
    # so the range() part includes 0 (with stop=-1)
    for s in range(len(parts) - 1, -1, -1):
        if s == 0:
            nm = builtins
        else:
            namespace = '.'.join(parts[:s])
            try:
                nm = importlib.import_module(namespace)
                break
            except (ImportError, ModuleNotFoundError):
                pass

    try:
        for i in range(s, len(parts)):
            nm = getattr(nm, parts[i])
        return nm
    except AttributeError as e:
        raise RuntimeError(f"Failed to load type {type_name}") from e


@dataclass
class AggregatedOutputs:
    """
    Aggregated outputs from all micro-batches
    """
    # the aggregated loss as a sum
    loss_sum: float = None
    # number of mini batches
    num_batches: int = None
    # number of tokens (only used when grad_reduction is 'per-token-mean')
    num_tokens: Optional[int] = None
    # any other custom outputs
    aggregated_outputs: Any = None


@dataclass
class ModelConfig:
    type: str = None
    args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizerConfig:
    type: str = None
    args: Dict[str, Any] = field(default_factory=dict)
    clip_gnorm: float = 0.0

    # loss reduction method
    # mean: average the loss over all micro-batches
    # sum: sum the loss of all micro-batches
    # Please note in validation stage, this configuration is ignored
    # the loss is always averaged over all batches
    loss_reduction: str = 'mean'
    # different ways of calculating grad
    # sum: sum the gradients of all micro-batches
    # mean: average the gradients over all micro-batches
    # per-token-mean: average the gradients over all tokens
    #    you must specify `aggregate_outputs_fn` and return the number of tokens
    grad_reduction: str = 'mean'
    # the function to aggregate the outputs from all micro-batches
    # inputs: (list of local outputs, torch group)
    # output: AggregateOutputs
    # you can use `torch.distributed.*` functions to do the work
    aggregate_outputs_fn: str = None

    def __post_init__(self):
        if self.grad_reduction not in ('sum', 'mean', 'per-token-mean'):
            raise ValueError(f"Invalid gradient_accumulation {self.grad_reduction}")
        if self.grad_reduction == 'per-token-mean' and not self.aggregate_outputs_fn:
            raise ValueError("aggregate_outputs_fn is required when grad_reduction is 'per-token-mean'")
        if self.loss_reduction not in ('mean', 'sum'):
            raise ValueError(f"Invalid loss_reduction {self.loss_reduction}")

@dataclass
class DatasetConfig:
    type: str = None
    train_args: Dict[str, Any] = field(default_factory=dict)
    val_args: Dict[str, Any] = field(default_factory=dict)
    test_args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataloaderConfig:
    type: str = 'torch.utils.data.DataLoader'
    train_args: Dict[str, Any] = field(default_factory=dict)
    # default to train_args
    val_args: Dict[str, Any] = field(default_factory=dict)
    # default to train_args
    test_args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetSamplerConfig:
    type: str = 'torch.utils.data.DistributedSampler'
    train_args: Dict[str, Any] = field(default_factory=dict)
    val_args: Dict[str, Any] = field(default_factory=dict)
    test_args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LRSchedulerConfig:
    type: str = None
    args: Dict[str, Any] = field(default_factory=dict)
    interval: str = 'epoch'

    def __post_init__(self):
        if self.interval not in ('epoch', 'step'):
            raise ValueError(f"Invalid interval {self.interval}")


@dataclass
class CheckpointConfig:
    save_dir: str = './checkpoints'
    no_save: bool = False

    # `"sharded"`: Each rank saves its shard of weights and optimizer states to a file. The checkpoint is
    #   a folder with as many files as the world size.
    # `"deduped"`: Each rank saves its deduped shard of weights and optimizer states to a file. The checkpoint is
    #   a folder with as many files as the world size.
    # `"merged"`: everything has been merged into a single file.
    #   Used internally only when you merge the checkpoint files via `Trainer.merge_checkpoints`
    save_type: str = 'sharded'

    save_last: bool = True
    save_best: bool = True
    symlink_best_and_last: bool = True

    # save the checkpoint every n train steps
    # Please note we always run validation before saving the checkpoint
    every_n_train_steps: Optional[int] = None
    every_n_epochs: Optional[int] = None
    keep_last_n_checkpoints: Optional[int] = None

    # resume training from a checkpoint folder/file
    # can be 'last'/'best'/a specific folder/file
    # we will not resume if resume_from is last or best but the corresponding checkpoint does not exist
    resume_from: str = None

    def get_resume_checkpoint_dir(self) -> Optional[Path]:
        if not self.resume_from:
            return None
        if self.resume_from in ['last', 'best']:
            d = Path(self.save_dir) / self.resume_from
            if not d.exists():
                return None
            return d
        return Path(self.resume_from)

    def __post_init__(self):
        if self.resume_from:
            if self.resume_from in ['last', 'best']:
                if not self.save_dir:
                    raise ValueError("save_dir is required when resume_from is 'last'/'best'")
                if not (Path(self.save_dir) / self.resume_from).exists():
                    logger.warning(f"`{self.resume_from}` checkpoint does not exist. Will train from scratch.")
            elif not Path(self.resume_from).exists():
                raise ValueError(f"resume_from {self.resume_from} does not exist")
        if self.no_save:
            return

        if self.save_type not in ('sharded', 'deduped', 'merged'):
            raise ValueError(f"Invalid save_type {self.save_type}")
        if not self.save_dir:
            raise ValueError("save_dir is required")

        if self.every_n_epochs is not None and self.every_n_train_steps is not None:
            raise ValueError("Cannot specify both every_n_epochs and every_n_train_steps")
        if self.every_n_epochs is None and self.every_n_train_steps is None:
            self.every_n_epochs = 1  # default to 1 epoch

        if self.every_n_train_steps is not None and self.every_n_train_steps < 1:
            raise ValueError("every_n_train_steps must be positive")
        if self.every_n_epochs is not None and self.every_n_epochs < 1:
            raise ValueError("every_n_epochs must be positive")
        if self.keep_last_n_checkpoints is not None and self.keep_last_n_checkpoints < 1:
            raise ValueError("keep_last_n_checkpoints must be positive")


@dataclass
class LogConfig:
    type: str = None
    args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HookConfig:
    type: str = None
    args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HookMapConfig:
    after_setup: str = None

    on_train_start: str = None
    on_train_end: str = None
    on_val_start: str = None
    on_val_end: str = None

    on_epoch_start: str = None
    on_epoch_end: str = None

    on_train_step_start: str = None
    on_train_step_end: str = None
    on_val_step_start: str = None
    on_val_step_end: str = None

    after_aggregate_train_step_outputs: str = None
    after_aggregate_val_step_outputs: str = None

    before_zero_grad: str = None
    after_zero_grad: str = None

    before_sync_grad: str = None
    after_sync_grad: str = None

    before_gnorm_clip: str = None
    after_gnorm_clip: str = None

    before_optimizer_step: str = None
    after_optimizer_step: str = None

    on_load_checkpoint: str = None
    on_save_checkpoint: str = None


class ArgsTrainHook(TrainHook):
    def __init__(self, hook_config: HookMapConfig):
        self.config = hook_config
        for k, v in asdict(hook_config).items():
            if v:
                setattr(self, k, load_type(v))


_TENSOR_TYPE = Literal['param', 'buffer', 'input']
_PRECISION_TYPE = Literal['fp32', 'fp16', 'bf16', 'none']
_PRECISION_MAP = {
    'fp32': torch.float32,
    'fp16': torch.float16,
    'bf16': torch.bfloat16,
    'none': None  # as it is. no conversion will happen.
}


@dataclass
class TrainerArgs:
    compute_config: ComputeConfig = None

    gen_savedir: str = './.nnscaler'
    # the reuse strategy of the generated code
    # auto: automatically decide the reuse strategy (moo for compile, match for run)
    # Or one of match/override/moo/graph (see `nnscaler.ReuseType`)
    gen_reuse: str = 'auto'
    pas_policy: str = 'autodist'
    broadcast_strategy: str = 'all'
    instance_name: str = None
    # compile: compile the model but not training
    # run: compile and run the model
    run_mode: str = 'run'
    # the model state dict file for tracing.
    # It is only used in tracing to serve as the initial state dict of the model.
    tracing_from_weights: str = None

    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    dataloader: DataloaderConfig = field(default_factory=DataloaderConfig)
    dataset_sampler: DatasetSamplerConfig = field(default_factory=DatasetSamplerConfig)
    lr_scheduler: Optional[LRSchedulerConfig] = None
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    log: List[LogConfig] = field(default_factory=list)
    # It can be `HookConfig` or `HookMapConfig`
    hook: Union[HookConfig, HookMapConfig, None] = None

    # TODO: mixed precision support
    precision: Union[str, Dict[_TENSOR_TYPE, _PRECISION_TYPE], None] = None

    micro_batch_size: int = 1
    # You can set one of `global_batch_size` and `grad_accumulation_steps` option.
    # Please note if both are set, they must be consistent.
    # default is
    # global_batch_size = self.micro_batch_size*self.scaling_factor
    # grad_accumulation_steps = 1
    global_batch_size: Optional[int] = None
    grad_accumulation_steps: Optional[int] = None

    max_epochs: Optional[int] = None
    max_train_steps: Optional[int] = None
    max_val_steps: Optional[int] = None

    # validation frequency
    val_every_n_train_steps: Optional[int] = None
    val_every_n_epochs: Optional[int] = 1

    enable_progress_bar: bool = True
    # if progress_bar is disabled (enable_progress_bar is False),
    # the frequency to print the training progress
    # validation metrics will also be printed if it is not None.
    log_progress_every_n_train_steps: Optional[int] = 100

    seed: Optional[int] = None
    # environment initialization function
    # you can put your environment initialization code here
    init_env_fn: str = None

    def __post_init__(self):
        if not self.compute_config:
            raise ValueError("compute_config is required")
        if not self.compute_config.use_end2end:
            raise ValueError("use_end2end must be True")

        if not self.global_batch_size and not self.grad_accumulation_steps:
            self.global_batch_size = self.micro_batch_size*self.scaling_factor
            self.grad_accumulation_steps = 1
        elif not self.global_batch_size:
            self.global_batch_size = self.micro_batch_size*self.scaling_factor*self.grad_accumulation_steps
        elif not self.grad_accumulation_steps:
            self.grad_accumulation_steps = self.global_batch_size // (self.micro_batch_size*self.scaling_factor)

        if self.global_batch_size != self.micro_batch_size*self.scaling_factor*self.grad_accumulation_steps:
            raise ValueError(f"`global_batch_size` {self.global_batch_size} is not equal to `micro_batch_size*scaling_factor*grad_accumulation_steps` "
                             f"{self.micro_batch_size*self.scaling_factor*self.grad_accumulation_steps}")

        if self.run_mode not in ('compile', 'run'):
            raise ValueError(f"Invalid run_mode {self.run_mode}")

        if self.gen_reuse != 'auto':
            if self.gen_reuse not in [e.value for e in ReuseType]:
                raise ValueError(f"Invalid gen_reuse {self.gen_reuse}")
        else:
            self.gen_reuse = 'moo' if self.run_mode == 'compile' else 'match'

        if self.broadcast_strategy not in [e.value for e in BroadcastGenFilesStrategy]:
            raise ValueError(f"Invalid broadcast_strategy {self.broadcast_strategy}")

        supported_precision_type = get_args(_PRECISION_TYPE)
        supported_tensor_type = get_args(_TENSOR_TYPE)
        if not self.precision:
            self.precision = 'none'
        if isinstance(self.precision, str):
            self.precision = {k: self.precision for k in supported_tensor_type}
        for tensor_type in supported_tensor_type:
            if tensor_type not in self.precision:
                self.precision[tensor_type] = 'none'
            if self.precision[tensor_type] not in supported_precision_type:
                raise ValueError(f"Invalid precision {self.precision[tensor_type]} for {tensor_type}")
        if any(k not in supported_tensor_type for k in self.precision):
            raise ValueError(f"Invalid tensor type found in {self.precision.keys()}")

        if not self.max_epochs and not self.max_train_steps:
            raise ValueError("max_epochs or max_train_steps is required")
        if not self.model.type:
            raise ValueError("model type is required")
        if not self.optimizer.type:
            raise ValueError("optimizer type is required")
        if not self.dataset.type:
            raise ValueError("dataset type is required")
        if not self.dataloader.type:
            raise ValueError("dataloader type is required")
        if not self.dataset_sampler.type:
            raise ValueError("dataset_sampler type is required")
        if self.lr_scheduler and not self.lr_scheduler.type:
            raise ValueError("lr_scheduler type is required")

    @classmethod
    def from_cli(cls, argv: List[str]) -> 'TrainerArgs':
        d = {}
        if argv[0] == '-f':
            with open(argv[1], 'r') as f:
                d = yaml.safe_load(f)
            argv = argv[2:]

        merge_args(d, parse_args(argv))
        return cls.from_dict(d)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'TrainerArgs':
        ta = deserialize_dataclass(d, TrainerArgs)
        return ta

    def to_dict(self):
        # replace all callable with their full qualified name
        # please note it is not reversible if local functions are used
        return transform_recursively(
            asdict(self),
            lambda class_or_func: f'{class_or_func.__module__}.{class_or_func.__qualname__}',
            callable,
        )

    @classmethod
    def from_yaml(cls, path: str) -> 'TrainerArgs':
        with open(path, 'r') as f:
            return cls.from_dict(yaml.safe_load(f))

    @classmethod
    def create_kwarg(cls, value: Any):
        if isinstance(value, dict):
            value = {k: cls.create_kwarg(v) for k, v in value.items()}
            if _TYPE_KEY in value:
                value_type = load_type(value.pop(_TYPE_KEY))
                return value_type(**value)
            elif _VALUE_TYPE_KEY in value:
                if _VALUE_KEY not in value:
                    raise ValueError(f"`{_VALUE_KEY}` is required when `{_VALUE_TYPE_KEY}` is present")
                value_type = value.pop(_VALUE_TYPE_KEY)
                if value_type == 'function':  # when type is function, the value should be the full qualified name of the function
                    return load_type(value[_VALUE_KEY])
                else:
                    # call its __init__ function
                    value_type = load_type(value_type)
                    return value_type(value[_VALUE_KEY])
            else:
                return value
        elif isinstance(value, list):
            return [cls.create_kwarg(i) for i in value]
        elif isinstance(value, tuple):
            return tuple(cls.create_kwarg(i) for i in value)
        else:
            return value

    @property
    def model_type(self):
        return load_type(self.model.type)

    @property
    def resolved_aggregate_outputs_fn(self):
        if not self.optimizer.aggregate_outputs_fn:
            return None
        return load_type(self.optimizer.aggregate_outputs_fn)

    @property
    def resolved_pas_policy(self):
        if self.pas_policy in _PREDEFINED_POLICIES:
            return self.pas_policy
        return load_type(self.pas_policy)

    @property
    def scaling_factor(self):
        return self.compute_config.runtime_ngpus // self.compute_config.plan_ngpus

    @property
    def update_freq(self):
        return self.global_batch_size // self.micro_batch_size // self.scaling_factor

    @property
    def enable_log_progress(self):
        return not self.enable_progress_bar and self.log_progress_every_n_train_steps

    @property
    def compile_mode(self) -> bool:
        return self.run_mode == 'compile'

    @property
    def param_dtype(self) -> torch.dtype:
        return _PRECISION_MAP[self.precision['param']]

    @property
    def buffer_dtype(self) -> torch.dtype:
        return _PRECISION_MAP[self.precision['buffer']]

    @property
    def input_dtype(self) -> torch.dtype:
        return _PRECISION_MAP[self.precision['input']]

    def init_env(self):
        if self.seed is not None:
            import random
            import numpy as np
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)

        if self.init_env_fn is None:
            return
        init_env_fn = load_type(self.init_env_fn)
        init_env_fn(self)

    def create_model(self) -> torch.nn.Module:
        kwargs = self.create_kwarg(self.model.args)
        return self.model_type(**kwargs)

    def create_parallel_optimizer(self, parallel_model: ParallelModule):
        kwargs = self.create_kwarg(self.optimizer.args)
        optimizer_class = load_type(self.optimizer.type)
        return build_optimizer(parallel_model, optimizer_class, **kwargs)

    def create_dataset(self, stage='train'):
        dataset_args = getattr(self.dataset, f'{stage}_args')
        # Sometimes a user uses a parameterless dataset class/factory function.
        # To support this case, we will create train dataset even without any arguments.
        # but val/test dataset must have arguments.
        if not dataset_args and stage != 'train':
            logger.info(f"{stage} dataset will not be created because empty arguments are provided.")
            return None
        kwargs = self.create_kwarg(dataset_args)
        dataset_class = load_type(self.dataset.type)
        dataset = dataset_class(**kwargs)
        if isinstance(dataset_class, torch.utils.data.IterableDataset):
            raise ValueError("IterableDataset is not supported")
        return dataset

    def create_sampler(self, dataset, stage='train'):
        sampler_args = getattr(self.dataset_sampler, f'{stage}_args')
        sampler_args = sampler_args or self.dataset_sampler.train_args
        kwargs = self.create_kwarg(sampler_args)
        kwargs['dataset'] = dataset
        kwargs['num_replicas'] = self.compute_config.runtime_ngpus // self.compute_config.plan_ngpus
        # if not distributed, we use the rank 0 sampler
        kwargs['rank'] = int(os.environ.get('RANK', 0)) // self.compute_config.plan_ngpus
        sampler_class = load_type(self.dataset_sampler.type)
        return sampler_class(**kwargs)

    def create_dataloader(self, stage='train', dataset=None):
        dataloader_args = getattr(self.dataloader, f'{stage}_args')
        dataloader_args = dataloader_args or self.dataloader.train_args
        kwargs = self.create_kwarg(dataloader_args)
        if 'batch_size' in kwargs:
            raise ValueError("`batch_size` should not be specified in dataloader_args. "
                             "You should use `micro_batch_size` instead.")
        kwargs['dataset'] = dataset or self.create_dataset(stage)
        if kwargs['dataset'] is None:
            return None
        if 'collate_fn' in kwargs:
            # special handling for collate_fn as a function
            # here we don't use self.collate_fn to avoid its implementation hacking
            kwargs['collate_fn'] = load_type(kwargs['collate_fn'])
        kwargs['batch_size'] = self.micro_batch_size
        kwargs['sampler'] = self.create_sampler(kwargs['dataset'], stage)
        dataloader_class = load_type(self.dataloader.type)
        return dataloader_class(**kwargs)

    def create_lr_scheduler(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler:
        if not self.lr_scheduler:
            return None
        kwargs = self.create_kwarg(self.lr_scheduler.args)
        lr_scheduler_class = load_type(self.lr_scheduler.type)
        return lr_scheduler_class(optimizer, **kwargs)

    def create_loggers(self) -> List['LoggerBase']:
        loggers = []
        for log_config in self.log:
            kwargs = self.create_kwarg(log_config.args)
            logger_class = load_type(log_config.type)
            loggers.append(logger_class(**kwargs))
        return loggers

    def create_hook(self) -> TrainHook:
        if not self.hook:
            return TrainHook()  # empty hook

        if isinstance(self.hook, dict):
            if 'type' in self.hook:
                hook_config = HookConfig(**self.hook)
            else:
                hook_config = HookMapConfig(**self.hook)
        else:
            hook_config = self.hook

        if isinstance(hook_config, HookConfig):
            kwargs = self.create_kwarg(hook_config.args)
            return load_type(hook_config.type)(kwargs)
        elif isinstance(hook_config, HookMapConfig):
            return ArgsTrainHook(hook_config)
        else:
            raise ValueError(f"Invalid hook_config {hook_config}")
