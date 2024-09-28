#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from contextlib import contextmanager, nullcontext
from functools import partial
import logging
from pathlib import Path
import os
import types
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ContextManager,
    Dict,
    Generator,
    List,
    Literal,
    Mapping,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

import torch
from torch import Tensor
import torch.distributed
from torch.nn import Module
from torch.optim import Optimizer
from typing_extensions import TypeGuard, override
from torch.utils.data import DataLoader

import lightning.pytorch as pl
from lightning.pytorch.accelerators import Accelerator, CUDAAccelerator
from lightning.pytorch.plugins.precision import Precision
from lightning.pytorch.trainer.states import TrainerFn
from lightning.fabric.plugins import CheckpointIO, ClusterEnvironment
from lightning.pytorch.strategies.launchers.subprocess_script import _SubprocessScriptLauncher
from lightning.pytorch.strategies.parallel import ParallelStrategy
from lightning.fabric.strategies.registry import _StrategyRegistry
from lightning.fabric.strategies.strategy import (
    TBroadcast,
    _Sharded,
)
from lightning.fabric.utilities.distributed import (
    ReduceOp,
    _distributed_is_initialized,
    _sync_ddp_if_available,
)
from lightning.fabric.utilities.distributed import group as _group
from lightning.fabric.utilities.seed import reset_seed
from lightning.fabric.utilities.types import _PATH, _Stateful
from lightning.pytorch.core.optimizer import _init_optimizers_and_lr_schedulers
from lightning.pytorch.utilities.types import LRSchedulerConfig, STEP_OUTPUT
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.model_helpers import is_overridden
from lightning.pytorch.utilities.rank_zero import WarningCache, rank_zero_info, rank_zero_warn
from lightning.pytorch.utilities import GradClipAlgorithmType

import nnscaler
from nnscaler.integration.lightning.utils import inplace_optimizer_fn
from nnscaler.runtime.device import DeviceGroup
from nnscaler.utils import enforce_zero_num_worker
from .precision import NnScalerPrecision


logger = logging.getLogger(__name__)


class NnScalerStrategy(ParallelStrategy):
    r"""Strategy for nnscaler.

    .. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

    Arguments:
        state_dict_type: The format in which the state of the model and optimizers gets saved into the checkpoint.

            - ``"sharded"``: Each rank saves its shard of weights and optimizer states to a file. The checkpoint is
              a folder with as many files as the world size.
            - ``"deduped"``: Each rank saves its deduped shard of weights and optimizer states to a file. The checkpoint is
              a folder with as many files as the world size.
    """
    strategy_name = "nnscaler"
    _registered_strategies: List[str] = []
    _nnscaler_extra_state_key = 'nnscaler-extra-state'
    _state_dict_type_key = 'state-dict-type'
    _pl_module_name_key = 'pl_state_dict'  # save some extra pl module states
    _pmodule_attr_name = 'nnscaler_pmodule'
    _module_name_key = 'state_dict'
    _opt_name_key = 'optimizer_states'

    def __init__(
        self,
        accelerator: Optional[Accelerator] = None,
        parallel_devices: Optional[List[torch.device]] = None,
        cluster_environment: Optional[ClusterEnvironment] = None,
        precision_plugin: Optional[Precision] = None,
        compute_config: Optional[nnscaler.ComputeConfig] = None,
        state_dict_type: Literal["deduped", "sharded"] = "sharded",
        pas_policy: str = None,
        gen_savedir: Union[str, Path] = './.nnscaler',
        reuse: str = 'match',
        instance_name: Optional[str] = None,
        **kwargs
    ) -> None:
        super().__init__(
            accelerator=accelerator,
            parallel_devices=parallel_devices,
            cluster_environment=cluster_environment,
            precision_plugin=precision_plugin,
        )
        self._forward_redirection = None

        self._num_nodes = 1
        self.compute_config = compute_config
        self.pas_policy = pas_policy
        self.gen_savedir = gen_savedir
        self.reuse = reuse
        self.instance_name = instance_name
        if self.compute_config is None:
            raise ValueError("The `compute_config` must be provided to the `NnScalerStrategy`.")
        if self.pas_policy is None:
            raise ValueError("The `pas_policy` must be provided to the `NnScalerStrategy`.")

        self._state_dict_type = state_dict_type

    @override
    def setup_environment(self) -> None:
        if not isinstance(self.accelerator, CUDAAccelerator):
            raise RuntimeError(
                f"The nnscaler strategy is only supported on CUDA GPUs but `{self.accelerator.__class__.__name__}`"
                " is used."
            )
        super().setup_environment()
        self._setup_distributed()

    def _setup_distributed(self) -> None:
        assert self.parallel_devices is not None
        self._validate_device_index_selection()
        reset_seed()
        self.set_world_ranks()
        self._set_node_environment_variables()
        nnscaler.init()

    def set_world_ranks(self) -> None:
        if self.cluster_environment is not None:
            self.cluster_environment.set_global_rank(self.node_rank * self.num_processes + self.local_rank)
            self.cluster_environment.set_world_size(self.compute_config.runtime_ngpus)

    def _set_node_environment_variables(self) -> None:
        assert self.cluster_environment is not None
        os.environ["MASTER_ADDR"] = self.cluster_environment.main_address
        os.environ["MASTER_PORT"] = str(self.cluster_environment.main_port)
        os.environ["RANK"] = str(self.global_rank)
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["LOCAL_RANK"] = str(self.local_rank)

    def _validate_device_index_selection(self) -> None:
        selected_device_indices = [device.index for device in self.parallel_devices]
        expected_device_indices = list(range(len(self.parallel_devices)))
        if selected_device_indices != expected_device_indices:
            raise RuntimeError(
                f"The selected device indices {selected_device_indices!r} don't match the local rank values of processes."
                " If you need to select GPUs at a specific index, set the `CUDA_VISIBLE_DEVICES` environment variable"
                f" instead. For example: `CUDA_VISIBLE_DEVICES={','.join(str(i) for i in selected_device_indices)}`."
            )

    @property
    @override
    def restore_checkpoint_after_setup(self) -> bool:
        return True

    @property
    @override
    def lightning_restore_optimizer(self) -> bool:
        return False

    @property
    def is_distributed(self) -> bool:
        """
        Indicates we are running in distributed mode
        And `distributed_sampler_kwargs` will be used to configure the sampler
        """
        return True

    @override
    def setup(self, trainer: "pl.Trainer") -> None:
        super().setup(trainer)
        assert self._lightning_module is not None
        assert self._model is not None

        # nnscaler handles gradient clipping internally
        if is_overridden("configure_gradient_clipping", self.lightning_module, pl.LightningModule):
            rank_zero_warn(
                "Since nnscaler handles gradient clipping internally, the default"
                " `LightningModule.configure_gradient_clipping` implementation will not actually clip gradients."
                " The hook will still be called. Consider setting"
                " `Trainer(gradient_clip_val=..., gradient_clip_algorithm='norm')`"
                " which will use the internal mechanism."
            )

        if self.lightning_module.trainer.gradient_clip_algorithm == GradClipAlgorithmType.VALUE:
            raise MisconfigurationException("nnscaler does not support clipping gradients by value.")

    def _get_dummy_forward_args(self, model: pl.LightningModule) -> Dict[str, Any]:
        # two options to set the dummy forward arguments

        if hasattr(model, 'dummy_forward_args'):
            if not isinstance(model.dummy_forward_args, dict):
                raise ValueError("The `dummy_forward_args` must be a dictionary with forward arguments names as keys.")
            return model.dummy_forward_args

        if hasattr(model, 'dummy_forward_args_fn'):
            dummy_forward_args_fn = getattr(model, 'dummy_forward_args_fn')
            if not callable(dummy_forward_args_fn):
                raise ValueError("The `dummy_forward_args_fn` must be a callable function.")
            trainer = model._trainer
            data_source = trainer.fit_loop._data_source
            assert data_source is not None, "The `data_source` must be defined in the trainer."
            assert data_source.instance is not None, "The `instance` must be defined in the data source."
            with enforce_zero_num_worker(DataLoader):
                dataloader = data_source.dataloader()
                assert dataloader.num_workers == 0, "The dataloader must have `num_workers=0`."
                data = next(iter(dataloader))
                dummy_forward_args = dummy_forward_args_fn(data)
                if not isinstance(dummy_forward_args, dict):
                    raise ValueError("The return value of `dummy_forward_args_fn` must be a dictionary with forward arguments names as keys.")
                return dummy_forward_args

        raise ValueError("The `dummy_forward_args` or `dummy_forward_args_fn` must be defined in the module.")

    @override
    def _setup_model(self, model: Module) -> Module:
        """Set up a module for inference (no optimizers).
        """
        dummy_forward_args = self._get_dummy_forward_args(model)

        old_training_flag = model.training
        if not old_training_flag:
            logger.warning("The model is not in training mode. Setting it to training mode for parallelizing.")
        model.train()  # always use the model in training mode
        pmodule = nnscaler.parallelize(
            model,
            self.precision_plugin.convert_input(dummy_forward_args),
            self.pas_policy,
            self.compute_config,
            gen_savedir=self.gen_savedir,
            reuse=self.reuse,
            instance_name=self.instance_name,
            broadcast_strategy='all'
        )
        model.train(old_training_flag)
        pmodule.to(self.root_device)

        # update the device of the module
        model._device = self.root_device

        # set all module parameters of original model to None
        # to reduce the memory usage
        # In return, the original model will not be able to access the parameters anymore
        # but the forward will be redirected to the parallelized model
        # TODO: this doesn't work for pipeline because fullmap is not complete
        for attr in pmodule.fullmap.values():
            attr_name = attr.orig_name.split('.')[0]
            setattr(model, attr_name, None)

        # torch.nn.Module will add new attributes to the model automatically.
        setattr(model, self._pmodule_attr_name, pmodule)
        model.to(self.root_device)
        # rewrite model forward to parallelized model forward
        model.forward = pmodule.forward

        # patch log function to add sync_dist_group
        _, sync_group = self.compute_config.get_sync_group()

        _old_log = model.log
        def _new_log(self, *args, **kwargs) -> None:
            kwargs['sync_dist_group'] = sync_group
            _old_log(*args, **kwargs)
        model.log = types.MethodType(_new_log, model)
        model._old__log = _old_log

        return model

    @override
    def setup_optimizers(self, trainer: "pl.Trainer") -> None:
        assert self.lightning_module is not None

        # If we're setting up for evaluation after fitting, we need to discard the optimizers
        # since we're rewrapping the model, otherwise optimizer param references are no longer valid
        # and subsequent checkpoint saving can fail
        self._reset_optimizers_and_schedulers()

        optimizer, lr_scheduler = self._init_optimizers()
        if len(optimizer.param_groups) != 1:
            raise MisconfigurationException(
                "nnscaler currently only supports single optimizer with a single param group."
            )
        new_optimizer = nnscaler.build_optimizer(
            getattr(trainer.model, self._pmodule_attr_name),
            partial(inplace_optimizer_fn, optimizer)
        )
        # the lr_scheduler doesn't need to update when we change the optimizer's param_graups[0]['params']
        self.optimizers, self.lr_scheduler_configs = [new_optimizer], ([lr_scheduler] if lr_scheduler else [])

    def _init_optimizers(self) -> Tuple[Optimizer, Optional[LRSchedulerConfig]]:
        assert self.lightning_module is not None
        optimizers, lr_schedulers = _init_optimizers_and_lr_schedulers(self.lightning_module)
        if len(optimizers) > 1 or len(lr_schedulers) > 1:
            raise MisconfigurationException(
                "nnscaler currently only supports single optimizer, single optional scheduler."
            )
        return optimizers[0], lr_schedulers[0] if lr_schedulers else None

    @property
    @override
    def root_device(self) -> torch.device:
        assert self.parallel_devices is not None
        return self.parallel_devices[self.local_rank]

    @property
    def num_nodes(self) -> int:
        return self._num_nodes

    @num_nodes.setter
    def num_nodes(self, num_nodes: int) -> None:
        self._num_nodes = num_nodes

    @property
    def num_processes(self) -> int:
        return len(self.parallel_devices) if self.parallel_devices is not None else 0

    @property
    @override
    def distributed_sampler_kwargs(self) -> Dict[str, Any]:
        return {
            "num_replicas": self.compute_config.runtime_ngpus//self.compute_config.plan_ngpus,
            "rank": self.global_rank // self.compute_config.plan_ngpus
        }

    @property
    @override
    def precision_plugin(self) -> NnScalerPrecision:
        plugin = self._precision_plugin
        if plugin is not None:
            assert isinstance(plugin, NnScalerPrecision)
            return plugin
        return NnScalerPrecision("32-true")

    @precision_plugin.setter
    @override
    def precision_plugin(self, precision: Optional[NnScalerPrecision]) -> None:
        if precision is not None and not isinstance(precision, NnScalerPrecision):
           raise TypeError(f"The nnscaler strategy can only work with the `NnScalerPrecision` plugin, found {precision}")
        self._precision_plugin = precision

    @override
    def _configure_launcher(self) -> None:
        assert self.cluster_environment is not None
        if not self.cluster_environment.creates_processes_externally:
            self._launcher = _SubprocessScriptLauncher(self.cluster_environment, self.num_processes, self.num_nodes)

    @override
    def setup_optimizer(self, optimizer: Optimizer) -> Optimizer:
        """Optimizers can only be set up jointly with the model in this strategy.

        Please use :meth:`setup_module_and_optimizers` to set up both module and optimizer together.

        """
        raise NotImplementedError(self._err_msg_joint_setup_required())

    @override
    def model_to_device(self) -> None:
        assert self.model is not None
        self.model.to(self.root_device)

    @override
    def barrier(self, name: Optional[str] = None) -> None:
        if not _distributed_is_initialized():
            return
        assert torch.distributed.get_backend() == "nccl", "nnscaler only supports nccl backend"
        # https://github.com/pytorch/pytorch/issues/53658
        # It would be better to provide device_ids=[self.root_device.index]
        torch.distributed.barrier(device_ids=[self.root_device.index])

    @override
    def broadcast(self, obj: TBroadcast, src: int = 0) -> TBroadcast:
        assert torch.distributed.get_backend() == "nccl", "nnscaler only supports nccl backend"

        if not _distributed_is_initialized():
            return obj

        obj = [obj]
        torch.distributed.broadcast_object_list(obj, src, group=_group.WORLD)
        return obj[0]

    @override
    def reduce(
        self,
        tensor: Union[Tensor, Any],
        group: Optional[Any] = None,
        reduce_op: Optional[Union[ReduceOp, str]] = "mean",
    ) -> Tensor:
        """Reduces a tensor from several distributed processes to one aggregated tensor.

        Args:
            tensor: the tensor to sync and reduce
            group: the process group to gather results from. Defaults to all processes (world)
            reduce_op: the reduction operation. Defaults to 'mean'/'avg'.
                Can also be a string 'sum' to calculate the sum during reduction.

        Return:
            reduced value, except when the input was not a tensor the output remains is unchanged

        """
        assert torch.distributed.get_backend() == "nccl", "nnscaler only supports nccl backend"

        if not _distributed_is_initialized() or not isinstance(tensor, Tensor):
            return tensor

        op: Optional[ReduceOp]
        if isinstance(reduce_op, str):
            reduce_op = "avg" if reduce_op == "mean" else reduce_op
            op = getattr(ReduceOp, reduce_op.upper())
        else:
            op = reduce_op

        torch.distributed.all_reduce(tensor, op=op, group=group, async_op=False)
        return tensor

    @override
    def lightning_module_state_dict(self) -> Dict[str, Any]:
        assert self.model is not None
        # do it in `save_checkpoint`
        return {}

    @override
    def optimizer_state(self, optimizer: Optimizer) -> Dict[str, Tensor]:
        assert self.optimizers
        # do it in `save_checkpoint`
        return {}

    @override
    def load_model_state_dict(self, checkpoint: Mapping[str, Any], strict: bool = True) -> None:
        # Override to do nothing, already loaded the states in `load_checkpoint()`
        pass

    @override
    def load_optimizer_state_dict(self, checkpoint: Mapping[str, Any]) -> None:
        # Override to do nothing, already loaded the states in `load_checkpoint()`
        pass

    @override
    def save_checkpoint(
        self, checkpoint: Dict[str, Any], filepath: _PATH, storage_options: Optional[Any] = None
    ) -> None:
        """Save model, optimizer, and other state to a checkpoint on disk.
        """
        if storage_options is not None:
            raise TypeError(
                "`NnScalerStrategy.save_checkpoint(..., storage_options=...)` is not supported because"
                " `NnScalerStrategy` does not use the `CheckpointIO`."
            )

        # broadcast the path from rank 0 to ensure all the states are saved in a common path
        path = Path(self.broadcast(filepath))
        path.mkdir(parents=True, exist_ok=True)

        nnscaler_pmodule = getattr(self._lightning_module, self._pmodule_attr_name)
        pl_module_state_dict = self._lightning_module.state_dict()
        # remove the parallelized module state from it
        for key in list(pl_module_state_dict.keys()):
            if key.startswith(self._pmodule_attr_name + '.'):
                pl_module_state_dict.pop(key)

        nnscaler_extra_state = {
            self._state_dict_type_key: self._state_dict_type,
            self._pl_module_name_key: pl_module_state_dict
        }
        checkpoint[self._nnscaler_extra_state_key] = nnscaler_extra_state

        if self._state_dict_type == "deduped":
            module_state, opt_state = nnscaler.deduped_state_dict(
                nnscaler_pmodule,
                self.optimizers[0] if self.optimizers else None
            )
        else:
            module_state = nnscaler_pmodule.state_dict()
            if self.optimizers:
                opt_state = self.optimizers[0].state_dict()
            else:
                opt_state = None
        checkpoint[self._module_name_key] = module_state
        if opt_state:
            checkpoint[self._opt_name_key] = [opt_state]

        torch.save(checkpoint, path / f'{self.global_rank}.pt')

    @override
    def load_checkpoint(
        self, checkpoint_path: _PATH
    ) -> Dict[str, Any]:
        """
        Load the contents from a checkpoint and restore the state of the given objects.
        """
        # broadcast the path from rank 0 to ensure all the states are loaded from a common path
        path = Path(self.broadcast(checkpoint_path))
        assert self.model is not None
        assert self.lightning_module is not None

        if not path.exists():
            raise FileNotFoundError(f"Checkpoint file {path} not found.")

        if path.is_dir():
            state_dict: dict = torch.load(path / f'{self.global_rank}.pt')
        else:
            state_dict: dict = torch.load(path)
        nnscaler_extra_state = state_dict.pop(self._nnscaler_extra_state_key)
        # load the extra states of the pl module
        self._lightning_module.load_state_dict(nnscaler_extra_state[self._pl_module_name_key], strict=False)

        module_dict = state_dict[self._module_name_key]
        state_dict[self._module_name_key] = {}
        optimizer_dict = None
        if self._opt_name_key in state_dict:
            optimizer_dict = state_dict[self._opt_name_key][0]
            state_dict[self._opt_name_key] = [{}]

        state_dict_type = nnscaler_extra_state[self._state_dict_type_key]

        module = getattr(self._lightning_module, self._pmodule_attr_name)
        optimizer = self.optimizers[0] if self.optimizers else None

        if state_dict_type == 'merged':
            nnscaler.load_merged_state_dict(module, module_dict, optimizer, optimizer_dict)
        elif state_dict_type == "deduped":
            nnscaler.load_deduped_state_dict(module, module_dict, optimizer, optimizer_dict)
        else:  # sharded
            nnscaler.load_sharded_state_dict(module, module_dict, optimizer, optimizer_dict)

        return state_dict

    @classmethod
    def merge_checkpoint(cls, checkpoint_files: List[str], output_file: str) -> None:
        """
        Merge the checkpoint files into a single checkpoint file.

        Args:
            checkpoint_files: The list of checkpoint files to merge.
            output_file: The output file path.
        """
        state_dicts = [torch.load(f, map_location='cpu') for f in checkpoint_files]

        module_state_dicts = [s[cls._module_name_key] for s in state_dicts]
        opt_state_dicts = None
        if cls._opt_name_key in state_dicts[0]:
            opt_state_dicts = [s[cls._opt_name_key][0] for s in state_dicts]

        merged_module_state_dict, merged_opt_state_dict = nnscaler.merge_state_dicts(
            module_state_dicts,
            opt_state_dicts
        )
        merged_state_dict = state_dicts[0]
        # reuse all states from the first checkpoint except the module and optimizer states
        merged_state_dict[cls._module_name_key] = merged_module_state_dict
        merged_state_dict[cls._opt_name_key] = [merged_opt_state_dict]
        merged_state_dict[cls._nnscaler_extra_state_key][cls._state_dict_type_key] = 'merged'

        torch.save(merged_state_dict, output_file)

    @classmethod
    @override
    def register_strategies(cls, strategy_registry: _StrategyRegistry) -> None:
        if not torch.distributed.is_available():
            return

        strategy_registry.register(
            "nnscaler",
            cls,
            description="nnscaler training",
        )
        cls._registered_strategies.append("nnscaler")

    def _get_process_group_backend(self) -> str:
        return 'nccl'  # nnscaler only support nccl
