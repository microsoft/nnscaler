#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from enum import Enum
from functools import partial
import types
from typing import Callable, Any, Dict, Optional, Tuple, Type, Union, TypeVar, List, Set, Literal
from pathlib import Path
import inspect
import sys
import importlib
from dataclasses import dataclass, asdict, field
from contextlib import contextmanager
import logging
import copy
import os

import torch

from nnscaler.codegen import ModuleCodeGen
from nnscaler.codegen.schedule.schedule import ScheduleCodeGen

from nnscaler.execplan import ExecutionPlan
from nnscaler.execplan.planpass.fusion import DiffFusion
from nnscaler.execplan.planpass.grouping import Grouping

from nnscaler.graph import IRGraph
from nnscaler.graph import parser
from nnscaler.graph.function.anchor import IRGraphAnchor
from nnscaler.graph.function.pyfunc import IRPyFunc
from nnscaler.graph.function.wrapnn import convert_to_wrapnn, wrapnn
from nnscaler.graph.gener.gen import IRAdapterGener
from nnscaler.graph.parser import FxModuleParser
from nnscaler.graph.schedule.predefined import PredefinedSched
from nnscaler.graph.schedule.schedplan import SchedulePlan

from nnscaler.ir.cten import IRObject, IRTensor
from nnscaler.ir.operator import IRBpOperation, IRDataOperation
from nnscaler.ir.tensor import IRFullTensor
from nnscaler.ir.unique import IDGenerator

from nnscaler.runtime.adapter.reducer import Reducer
from nnscaler.runtime.device import DeviceGroup
from nnscaler.runtime.gnorm import calcuate_gnorm, clip_grads
from nnscaler.runtime.module import AttrMeta, CubeModule, ParallelModule, OriginModuleMetadata, ExtraState

from nnscaler.flags import CompileFlag, RuntimeFlag
import nnscaler.policies as policies
from nnscaler.program import disable_global_graph
from nnscaler.utils import get_member_by_name, setup_stride_broadcast_group, get_shared_params

logger = logging.getLogger(__name__)


_PREDEFINE_SCHEDS: Dict[str, Callable[[IRGraph, int, int], SchedulePlan]] = {}
_PREDEFINED_INFERENCE_SCHEDS = ['infer_pipe']
_PREDEFINE_SCHED_NAME_PREFIX = 'sched_'
for k, v in PredefinedSched.__dict__.items():
    if isinstance(v, staticmethod) and k.startswith(_PREDEFINE_SCHED_NAME_PREFIX):
        _PREDEFINE_SCHEDS[k[len(_PREDEFINE_SCHED_NAME_PREFIX):]] = getattr(PredefinedSched, k)  # be compatible with python 3.8

_PREDEFINED_POLICIES: Dict[str, Callable[[IRGraph, 'ComputeConfig'], IRGraph]] = {}
_PREDEFINED_POLICIES_NAME_PREFIX = 'pas_'
for k, v in policies.__dict__.items():
    if callable(v) and k.startswith(_PREDEFINED_POLICIES_NAME_PREFIX):
        _PREDEFINED_POLICIES[k[len(_PREDEFINED_POLICIES_NAME_PREFIX):]] = v


@dataclass(frozen=True)
class ComputeConfig:
    plan_ngpus: int
    runtime_ngpus: Optional[int] = None

    # whether to fold constant when generating code
    constant_folding: bool = False

    # how to execute the functions during trace
    trace_strategy: str = 'cuda_run_cpu_offload'

    use_zero: bool = False
    zero_ngroups: int = 1
    # whether to use reduce scatter for zero
    # Please note
    # 1. this only works when `use_zero` is True and `zero_ngroups` is 1.
    # 2. In some cases, it can introduce parity issue. So use it with caution.
    zero_use_reduce_scatter: bool = False

    # whether the generated code is for inference only
    inference_only: bool = False

    # end2end means,
    #  1. the first argument of `module.forward` must be the data sample
    #  2. the first return value of `module.forward` must be the loss
    #  which must be a scalar tensor
    use_end2end: bool = False

    # whether to use async reducer
    # if True, the gradient all-reduce will be async,
    # This only works when the `use_end2end` is `True` for now.
    use_async_reducer: bool = False
    # the maximal reducer weight bytes for one allreduce in megabytes
    # It is also effective for sync reducer.
    # None/0 means using the default value. (25MB for async, no limit for sync)
    reducer_bucket_cap_mb: Optional[float] = None

    # PAS policy settings
    # you can also put any other settings that can affect code generation here.
    # but please prefix the keys with `_` to avoid conflicts with predefined keys.
    pas_config: Dict[str, Any] = field(default_factory=dict)
    # the customized configs from user that can affect the graph and code generation.
    # you should put any configuration that may affect the traced graph here.
    # So we can track the changes and make sure the generated code is correct.
    # Example 1: save module configuration
    # ```python
    # class MyModule(torch.nn.Module):
    #   def __init__(self):
    #     super().__init__()
    #   def forward(self, x):
    #     ...
    #     if module_config.use_3d:
    #       ...
    # ```
    # here we can set `graph={'use_3d': module_config.use_3d}`,
    # and we can be sure different use_3d will never use the same generated code.
    # Example 2: save file stats
    # If you want to track all related file stats (just like traditional compilers do),
    # you can save the md5 of the files to save some bytes:
    # ```python
    # import hashlib
    # h = hashlib.md5()
    # for f in Path('./src').glob('**/*.py'):
    #   with open(f, 'rb') as f:
    #     h.update(f.read())
    # graph = {
    #   'files_md5': h.hexdigest()
    # }
    # ```
    user_config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.plan_ngpus <= 0:
            raise ValueError(f"plan_ngpus {self.plan_ngpus} must be > 0")
        if self.runtime_ngpus is None:
            super().__setattr__('runtime_ngpus', int(os.environ.get('WORLD_SIZE', 0)))
            if not self.runtime_ngpus:
                raise ValueError(f"runtime_ngpus is not set and WORLD_SIZE is not set.")
        if self.runtime_ngpus <= 0:
            raise ValueError(f"runtime_ngpus {self.runtime_ngpus} must be > 0")
        if self.runtime_ngpus % self.plan_ngpus != 0:
            raise ValueError(f"runtime_ngpus {self.runtime_ngpus} must be a multiple of plan_ngpus {self.plan_ngpus}")
        if self.use_zero and self.zero_ngroups <= 0:
            raise ValueError(f"zero_ngroups {self.zero_ngroups} must be > 0")
        if not self.use_zero and self.zero_ngroups != 1:
            logger.warning(f"use_zero is False, but zero_ngroups is {self.zero_ngroups}. Will set zero_ngroups to 1.")
            # have to use __setattr__ for frozen dataclass
            super().__setattr__('zero_ngroups', 1)

        if self.reducer_bucket_cap_mb and self.reducer_bucket_cap_mb < 0:
            raise ValueError(f"reducer_bucket_cap_mb {self.reducer_bucket_cap_mb} should not be negative.")

        # TODO: Please note in current implementation of Bucket,
        # zero_use_reduce_scatter still works when zero_ngroups > 1 in sync mode
        # Let's hide this feature for now for consistency.
        if self.use_zero and self.zero_use_reduce_scatter and self.zero_ngroups != 1:
            raise ValueError("zero_use_reduce_scatter is only supported when zero_ngroups is 1.")

    def apply_pipeline_scheduler(
            self,
            graph: IRGraph,
            pipeline_nstages: int,
            pipeline_nmicros: int,
            pipeline_scheduler: str,
    ) -> Optional[SchedulePlan]:
        """
        Apply the pipeline scheduler to the graph.
        """
        if not self.use_end2end:
            raise ValueError("pipeline is only supported in end2end mode")
        if pipeline_nmicros <= 0:
            raise ValueError(f"pipeline_nmicros {pipeline_nmicros} must be > 0.")
        if pipeline_nstages <= 0:
            raise ValueError(f"pipeline_nstages {pipeline_nstages} must be > 0.")
        if self.plan_ngpus % pipeline_nstages != 0:
            raise ValueError(f"pipeline_nstages {pipeline_nstages} must be a multiple of plan_ngpus {self.plan_ngpus}")
        if pipeline_scheduler not in _PREDEFINE_SCHEDS:
            raise ValueError(f"pipeline_scheduler {pipeline_scheduler} is not supported. "
                             f"Supported schedulers are {_PREDEFINE_SCHEDS.keys()}")
        if self.inference_only and pipeline_scheduler not in _PREDEFINED_INFERENCE_SCHEDS:
            raise ValueError(f"pipeline_scheduler {pipeline_scheduler} is not supported in inference mode. "
                             f"Supported schedulers are {_PREDEFINED_INFERENCE_SCHEDS}")
        if not self.inference_only and pipeline_scheduler in _PREDEFINED_INFERENCE_SCHEDS:
            raise ValueError(f"pipeline_scheduler {pipeline_scheduler} is not supported in training mode.")

        sched = _PREDEFINE_SCHEDS[pipeline_scheduler]
        return sched(graph, pipeline_nmicros, pipeline_nstages)

    @property
    def gpu_config(self) -> Dict[str, int]:
        return {
            'plan_ngpus': self.plan_ngpus,
            'runtime_ngpus': self.runtime_ngpus,
        }

    @property
    def graph_config(self) -> Dict[str, Any]:
      return {
            'constant_folding': self.constant_folding,
            'user_config': self.user_config,
            'inference_only': self.inference_only, # there will be no backward nodes in the graph in inference mode
            'end2end_mode': self.use_end2end,  # end2end_mode can affect the graph generation.
            'trace_strategy': self.trace_strategy,  # different strategy might lead to different graph
        }

    @property
    def module_dedup_group_size(self) -> int:
        """
        Get the size of the deduplication group of the model state dict, which is `plan_ngpus`.
        """
        return self.plan_ngpus

    @property
    def optimizer_dedup_group_size(self) -> int:
        """
        Get the size of the deduplication group of the optimizer state dict.

        Nonzero mode: the group size is the same with plan_ngpus
        Zero mode: the group size is `zero_group`, which equals `runtime_ngpus//zero_ngroups`
        """

        if self.use_zero:
            return self.runtime_ngpus // self.zero_ngroups
        else:
            return self.plan_ngpus

    def get_sync_group(self) -> Tuple[List[int], torch.distributed.ProcessGroup]:
        """
        Get sync group for the current rank.
        The sync group is a group of ranks that have exactly the same weights, but different inputs,
        so they should synchronize with each other to get the whole gradients/loss/etc.

        Please note if sync groups haven't been created, it will create them.
        So it will deadlock if only some of ranks call this function.

        Returns:
            Tuple[List[int], torch.distributed.ProcessGroup]: return the rank list of the group and its torch.distributed group
        """
        rank = torch.distributed.get_rank()
        # create all groups
        plan_ngpus = self.plan_ngpus
        runtime_ngpus = self.runtime_ngpus
        for i in range(plan_ngpus):
            DeviceGroup().get_group(
                list(range(i, runtime_ngpus, plan_ngpus))
            )
        rank_list = list(range(rank % plan_ngpus, runtime_ngpus, plan_ngpus))
        return rank_list, DeviceGroup().get_group(rank_list)

    @classmethod
    def safe_dump_to_file(cls, cfg: 'ComputeConfig', file: Union[str, Path]) -> None:
        """
        torch.save(cfg) is not safe when we change the fields of ComputeConfig.
        So we should use this method to save the config.
        """
        torch.save(asdict(cfg), file)

    @classmethod
    def safe_load_from_file(cls, file: Union[str, Path], return_none_on_error=True) -> Optional['ComputeConfig']:
        """
        Load the config from file.
        `return_none_on_error` controls the behaivor when the file not exists or failed to load.
        If `return_none_on_error` is True, will return None when failed to load.
        If `return_none_on_error` is False, will raise when failed to load.
        """
        if Path(file).exists():
            try:
                cfg = torch.load(file)
                if isinstance(cfg, dict): # in old version, we save the object directly (not save as dict)
                    # this can raise if cfg has extra keys.
                    # which means some fields of ComputeConfig has been removed(we should avoid this).
                    # in this case, we just return None.
                    return cls(**cfg)
                return cfg
            except Exception as e:
                if not return_none_on_error:
                    raise
                logger.warning(f"Failed to load ComputeConfig with error {str(e)}.")
        elif not return_none_on_error:
            raise FileNotFoundError(f"Failed to load compute config from {file}. File not found.")
        return None

    @classmethod
    def safe_equals(cls, a: Optional['ComputeConfig'], b: Optional['ComputeConfig']) -> bool:
        """
        Return False if a and b are from incompatible version of ComputeConfig
        This is only for backward compatibility, and will be removed in future
        and can use `==` when we save dict version of ComputeConfig to file.
        """
        try:
            return a == b
        except AttributeError:
            logger.warning("Failed to compare ComputeConfig. They are incompatible.")
            return False


@contextmanager
def _flags(flags, /, **kwargs):
    old_flags = {}
    for k, v in kwargs.items():
        old_flags[k] = getattr(flags, k)
        setattr(flags, k, v)
    try:
        yield
    finally:
        for k, v in old_flags.items():
            setattr(flags, k, v)


def _compile_flags(compute_config: ComputeConfig):
    return _flags(
        CompileFlag,
        async_reducer=compute_config.use_async_reducer, reducer_op='sum',
        max_reducer_bucket=int(compute_config.reducer_bucket_cap_mb * 1024 * 1024)
                if compute_config.reducer_bucket_cap_mb else None,
        async_comm=False,
        use_zero=compute_config.use_zero,
        zero_ngroups=compute_config.zero_ngroups,
        zero_use_reduce_scatter=compute_config.zero_use_reduce_scatter,
        trace_strategy=compute_config.trace_strategy,
    )


def _runtime_flags(**kwargs):
    return _flags(RuntimeFlag, **kwargs)


def _to_cpu(val: Any):
    """Complex to CPU"""
    if isinstance(val, tuple):
        return tuple(_to_cpu(t) for t in val)
    if isinstance(val, list):
        return list(_to_cpu(t) for t in val)
    if isinstance(val, dict):
        return {_to_cpu(key):_to_cpu(val) for key, val in val.items()}
    if isinstance(val, set):
        return {_to_cpu(t) for t in val}
    if isinstance(val, torch.Tensor):
        requires_grad = val.is_floating_point() or val.is_complex()
        return val.detach().clone().cpu().requires_grad_(requires_grad)
    return val


def _contains_uncommutable_data(ir_outputs: Any):
    """
    only IRObject (but not IRTensor) is not commutable between gpus.
    """
    if isinstance(ir_outputs, (tuple, list)):
        return any(_contains_uncommutable_data(t) for t in ir_outputs)
    elif isinstance(ir_outputs, dict):
        return any(_contains_uncommutable_data(k) or _contains_uncommutable_data(v) for k, v in ir_outputs.items())
    elif isinstance(ir_outputs, IRTensor):
        return False
    elif isinstance(ir_outputs, IRObject):
        return True
    return False


def _get_full_qualified_name(obj: Any) -> str:
    """Get full qualified name of an object"""
    if inspect.isclass(obj):
        return obj.__module__ + '.' + obj.__qualname__
    return obj.__module__ + '.' + obj.__class__.__qualname__


def _add_gen_savedir_to_syspath(gen_savedir: str) -> Path:
    gen_savedir = Path(gen_savedir).resolve()
    gen_savedir.mkdir(parents=True, exist_ok=True)
    if str(gen_savedir) not in sys.path:
        sys.path.append(str(gen_savedir))
    return gen_savedir


def _is_any_gencode_loaded(namespace: str) -> bool:
    """Check if a module is loaded"""
    for m in list(sys.modules.values()):  # list() to avoid mulitple thread confliction
        # m.__name__ doesn't always work as some module doesn't have __name__ attribute.
        if getattr(m, '__name__', '').startswith(namespace + '.' + _GENCODE_FILE_PREFIX):
            return True
    return False


def _get_arg_default_values(fn) -> Dict[str, Any]:
    args = inspect.signature(inspect.unwrap(fn))
    return {k: v.default for k, v in args.parameters.items()}


def _clean_files(_dir: Path, pattern = '*') -> None:
    """
    Clean files of a directory. No directories will be removed.
    """
    for f in _dir.glob(pattern):
        if f.is_file():
            f.unlink()


def _broadcast_single_value(src_rank, group, obj=None):
    sent_obj = [obj]
    torch.distributed.broadcast_object_list(
        sent_obj,
        src=src_rank,
        group=group,
    )
    return sent_obj[0]


_DEFAULT_INSTANCE_NAME = '_'
_GENCODE_FILE_PREFIX = 'gencode'
_GENCODE_FILE_TEMPLATE = _GENCODE_FILE_PREFIX + '{}.py'  # 'gencode{}.py'
_PARALLEL_MODULE_NAMESPACE = '_parallel_modules'
_GRAPH_DUMP_FILE = 'graph.ckp'
_FORWARD_ARGS_DUMP_FILE = 'forward_args.pkl'


class ReuseType(Enum):
    """The reuse type"""
    MATCH = 'match'        # reuse if present and match, error if present but not match, generate if not present.
    OVERRIDE = 'override'  # no reuse, everything will be regenerated.
    MOO = 'moo'            # (short for match or override)reuse if present and match, generate if not match or not present.
    GRAPH = 'graph'        # reuse graph only if present and match, generate otherwise.


class BroadcastGenFilesStrategy(Enum):
    """
    The broadcast strategy for generated files.
    Only new generated files can be broadcasted.
    The files includes:

    1. config file: compute config (compute_config.pt)
    2. trace files: graph dump (graph.ckp), forward args dump(forward_args.pkl),
       origin module metadata (origin_module_metadata.pt), init weights file(fullmodel.pt.*),
       param name mapping (dist_param_map.pt)
    3. code: generated code files (gencode*.py)

    Reused files will not be broadcasted with any of the following options.
    """

    # nothing will be broadcasted.
    # You need to do it by yourself or the generated files are saved in a shared directory (like azure blob).
    NONE = 'none'

    # broadcast all new generated files to all nodes.
    # This is useful when you want to run the same code on all nodes.
    # please note the init weight files can be huge.
    ALL = 'all'

    # broadcast all new generated files except init weights (fullmodel.pt.*).
    # Without weights,
    # you can only construct the parallel module with `init_params=False`.
    # You can then
    # 1. Load the weights from a checkpoint file with `module.load_state_dict` or `load_merged_state_dict`
    # 2. Or you can use `broadcast_weights` to get the weights from the workers in node0.
    #    (local world size should be bigger than plan_ngpus)
    NO_WEIGHTS = 'no_weights'

    # broadcast the new generated code (gencode*.py) and compute_config.pt only.
    # It's your responsibility to make sure other necessary files are available on all nodes.
    CODE = 'code'


class RegenStatus(Enum):
    NONE = 'none'   # nothing is regenerated.
    ALL = 'all'     # everything is regenerated, including graph and code
    CODE = 'code'   # only code is regenerated.


def _prepare_namespace(
        gen_savedir: str,
        module_or_module_class: Union[Type[torch.nn.Module], torch.nn.Module],
        instance_name: Optional[str] = None,
) -> Tuple[str, Path]:
    gen_savedir = _add_gen_savedir_to_syspath(gen_savedir)

    instance_name = instance_name or _DEFAULT_INSTANCE_NAME
    instance_name = instance_name.strip('.') if instance_name else ''
    instance_namespace = f'.{instance_name}' if instance_name else ''
    namespace = f'{_PARALLEL_MODULE_NAMESPACE}.{_get_full_qualified_name(module_or_module_class)}{instance_namespace}'

    outdir = gen_savedir / Path(namespace.replace('.', '/').strip('/'))
    outdir.mkdir(parents=True, exist_ok=True)

    return namespace, outdir


def _prepare_and_check_reusable(
        gen_savedir: str,
        module_or_module_class: Union[Type[torch.nn.Module], torch.nn.Module],
        compute_config: ComputeConfig,
        instance_name: Optional[str] = None,
        reuse: ReuseType = ReuseType.MATCH,
    ) -> Tuple[str, bool]:
    """
    Prepare the output directory for code generation, and also check if the existing code is reusable.

    Args:
        gen_savedir (str): the directory to save generated code
        module_or_module_class (Union[Type[torch.nn.Module], torch.nn.Module]): the original module or module class
        compute_config (ComputeConfig): the environment resource
        instance_name (Optional[str]): the instance name of the generated module. If it is None, will use the default name.
        reuse (ReuseType): specify which part can be reused.

    Returns:
        Tuple[str, bool]: the output directory and whether the existing code is reusable.

    Raises:
        RuntimeError: if the existing code is not reusable,
            will raise RuntimeError if the code is not reusable but the module is already loaded.
    """
    namespace, outdir = _prepare_namespace(gen_savedir, module_or_module_class, instance_name)

    # decision matrix for code generation
    # reuse flag | dir condition(imported, empty, match, unmatched) | action
    # ---------------------------------------------------------
    #   OVERRIDE   | empty           | generate
    #   OVERRIDE   | imported        | raise error
    #   OVERRIDE   | whatever match  | generate
    #   OVERRIDE   | unmatch         | generate
    #   GRAPH      | empty           | generate
    #   GRAPH      | imported        | raise error
    #   GRAPH      | graph match     | reuse graph, and regenerate code
    #   GRAPH      | all match       | reuse graph, and regenerate code
    #   GRAPH      | unmatch         | generate
    #   MATCH      | empty           | generate
    #   MATCH      | match           | reuse(do nothing)
    #   MATCH*     | whatever unmatch| raise error (except when there's no python source code, see below)
    #   MATCH      | imported        | doesn't matter
    #   MOO        | empty           | generate
    #   MOO        | match           | reuse(do nothing)
    #   MOO        | match graph     | reuse graph, and regenerate code
    #   MOO        | imported        | raise error if whatever unmatch
    #  *: The precondition for `except` part is the compute config should match.
    #     you can take it as a continous operation after a failed generation.
    reusable = False
    config_file = outdir / ParallelModule.COMPUTE_CONFIG_FILE
    old_config: Optional[ComputeConfig] = ComputeConfig.safe_load_from_file(config_file)
    is_config_match = ComputeConfig.safe_equals(old_config, compute_config)
    is_graph_config_match = old_config is not None and old_config.graph_config == compute_config.graph_config
    trace_meta_files = [
        outdir / FxModuleParser.ATTR_CONTENT_FILE_0,  # just check the first is good enough
        outdir / FxModuleParser.ATTR_MAP_FILE,
    ]

    if reuse == ReuseType.MATCH or reuse == ReuseType.MOO:
        # check if the module is already generated
        expected_output_files = [outdir / _GENCODE_FILE_TEMPLATE.format(rank) for rank in range(compute_config.runtime_ngpus)]
        expected_output_files.extend(trace_meta_files)
        expected_output_files.append(config_file)
        expected_output_files.append(outdir / _GRAPH_DUMP_FILE)
        expected_output_files.append(outdir / _FORWARD_ARGS_DUMP_FILE)
        expected_output_files.append(outdir / ParallelModule.ORIGIN_MODULE_METADATA_FILE)
        existing_output_files = [
            f for f in outdir.glob('*')
            if f.is_file() and (  # just take fullmodel.pt.0 to compare
                not f.name.startswith(FxModuleParser.ATTR_CONTENT_FILE_STEM)
                or f.name == FxModuleParser.ATTR_CONTENT_FILE_0
            )
        ]
        if existing_output_files:
            if is_config_match \
                and all([output_file.exists() for output_file in expected_output_files]) \
                and len(existing_output_files) == len(expected_output_files):
                reusable = True  # everything is matched.
            elif is_config_match \
                and all(f.suffix != '.py'  for f in existing_output_files):
                # No python source code is generated.
                # which means its last generation failed.
                # in this case, we can reuse the same directory safely.
                logger.info(f'Output directory {outdir} is not empty. '
                            f'But no python source code is present. '
                            f'Will reuse the directory and the graph dump if present.')
                # we have to trace the graph again if not all meta files are present.
                if not all([meta_file.exists() for meta_file in trace_meta_files]):
                    _clean_files(outdir)
            elif reuse == ReuseType.MATCH:
                raise RuntimeError(f'Output directory {outdir} is not empty. '
                                   f'And the existing files do not match with current config. '
                                   f'You can remove the directory and try again, '
                                   f'or set reuse to ReuseType.NONE/ReuseType.OVERRIDE to regenerate the code.')
            else:
                assert reuse == ReuseType.MOO
                if _is_any_gencode_loaded(namespace):
                    raise RuntimeError(f'Output directory {outdir} is already loaded. '
                                       f'You can not override a loaded module.')
                elif is_graph_config_match:
                    # reuse the graph dump
                    _clean_files(outdir, '*.py')
                else:
                    _clean_files(outdir)
    else:
        # check if the module is already loaded
        if _is_any_gencode_loaded(namespace):
            raise RuntimeError(f'Output directory {outdir} is already loaded. '
                               f'You can not override a loaded module.')
        # clear existing generated files
        if reuse == ReuseType.OVERRIDE \
            or not is_graph_config_match \
            or not all([meta_file.exists() for meta_file in trace_meta_files]):
            # we have to trace the graph again if not all meta files are present even when reuse=graph.
            glob_pattern = '*'
        else:
            glob_pattern = '*.py'  # so we can keep graph dumps.
        _clean_files(outdir, glob_pattern)

    return outdir, reusable


def _gen_graph(
    module: torch.nn.Module,
    dummy_forward_args: dict,
    outdir: Path,
    constant_folding: bool,
    end2end_mode: bool = False,
    inference_only: bool = False,
):
    # reset environment
    IDGenerator().clear()
    disable_global_graph()

    module.cpu()
    forward_args_default = _get_arg_default_values(module.forward)
    for v in forward_args_default.values():
        if v is not inspect.Parameter.empty and not isinstance(v, (int, str, float, bool, type(None))):
            raise ValueError(f"Default value type {type(v)} of forward args is not supported.")

    # generate fx graph
    dummy_forward_args = _to_cpu(dummy_forward_args)
    fx_graph = parser.to_fx_graph(module, dummy_forward_args)

    # generate ir logic graph
    graph = parser.to_ir_graph(
        fx_graph, dummy_forward_args, outdir, constant_folding
    )

    # generate dummy inputs for logic graph
    # that is, generate IRObject/IRFullTensor for fx graph dummy input
    fx_input_nodes = [node for node in fx_graph.graph.nodes if node.op == 'placeholder']
    # the inputs of graph is different with original forward args
    # so we get the real forward args from fx inputs
    forward_args = {
        node.target: forward_args_default.get(node.target, inspect.Parameter.empty)
        for node in fx_input_nodes
    }
    ir_dummy_inputs = []
    for node in fx_input_nodes:
        if node.target.startswith('*'):  # *args or **kwargs
            if node.target.strip('*') in dummy_forward_args:
                raise ValueError(f"Input {node.target}: *args or **kwargs is not suppported")
            ir_dummy_inputs.append(None)  # always set None to *args/**kwargs
        elif node.target in dummy_forward_args:
            ir_dummy_inputs.append(dummy_forward_args[node.target])
        elif forward_args[node.target] is not inspect.Parameter.empty:
            ir_dummy_inputs.append(forward_args[node.target])
        else:
            raise ValueError(f"Input {node.target} not in dummy forward args, nor has default value.")
    for i in range(len(ir_dummy_inputs)):
        # note: we will always set tensor to require gradient, which may
        # generate backward communications in adapter. However, as long as
        # the data doesn't require gradient in real runtime, the backward
        # communication will not be triggered.
        ir_dummy_inputs[i] = IRObject.from_complex(
            fx_input_nodes[i].target, ir_dummy_inputs[i],
            requires_grad=True,
            tosub=True,
            is_constant=False,
        )
        # if the input is a complex type, we should wrap it with IRObject
        if not isinstance(ir_dummy_inputs[i], IRObject):
            ir_dummy_inputs[i] = IRObject(fx_input_nodes[i].target, value=ir_dummy_inputs[i], is_constant=False)

    # generate complete ir graph
    ir_dummy_outputs = graph(*ir_dummy_inputs)
    if end2end_mode:
        # in end2end mode, we must use dataloader as the first argument of forward
        # we assume the first argument of forward is the data sample (which is a requirement in our doc)
        graph.use_dataloader_input()

        # we require the first output is the loss
        if isinstance(ir_dummy_outputs, (list, tuple)):
            ir_loss = ir_dummy_outputs[0]
        else:
            ir_loss = ir_dummy_outputs
        if not isinstance(ir_loss, IRTensor) or ir_loss.shape != (1,):
            # internally scalar tensor will be reshaped to (1,) in IRGraph
            raise RuntimeError(f"Loss can only be scalar tensor but got {ir_loss.shape if isinstance(ir_loss, IRTensor) else ir_loss}")
    else:
        ir_loss = None

    if not inference_only:
        graph.backward(ir_loss)
    else:
        graph.no_backward()

    return graph, forward_args


def _gencode(
        module_or_module_class: torch.nn.Module,
        dummy_forward_args: Dict[str, Any],
        pas_policy: Callable[[IRGraph, ComputeConfig], IRGraph],
        compute_config: ComputeConfig,
        outdir: Path,
        *,
        module_dtype:  Optional[torch.dtype] = None,
        module_fn: Optional[Callable[[], torch.nn.Module]] = None,
    ) -> RegenStatus:
    """
    Generate parallel module source code from a torch module, and save it to file.
    Generated module will be save according to its full qualified name.

    If you want to save multiple instances of the same module,
    you can specify the instance_name to distingish them.

    For example, if the module is `torchscale.x.y`, then the generated module will be save to
    `gen_savedir/_parallel_modules/torchscale/x/y/instance_name`.

    Args:
        module (torch.nn.Module): the module to be compiled
        dummy_forward_args (Dict[str, Any]): the dummy input for the module forward
        pas_policy (Callable[[IRGraph, ComputeConfig], IRGraph]): the pas policy
        compute_config (ComputeConfig): the environment resource
        outdir (Path): the directory to save generated code
        module_dtype (Optional[torch.dtype]): the dtype of the module. Keep as it is when it is None.
        module_fn (Optional[Callable[[], torch.nn.Module]]): the function to create the module. Will use __init__ if it is None.

    Returns:
        RegenStatus: which part is regenerated.
    """
    graph_ckp = outdir / _GRAPH_DUMP_FILE
    forward_args_ckp = outdir / _FORWARD_ARGS_DUMP_FILE
    origin_module_metadata_ckp = outdir / ParallelModule.ORIGIN_MODULE_METADATA_FILE
    ret = RegenStatus.NONE
    if not graph_ckp.exists() or not forward_args_ckp.exists() or not origin_module_metadata_ckp.exists():
        is_module_class = inspect.isclass(module_or_module_class)
        ret = RegenStatus.ALL
        if is_module_class:
            try:
                if module_fn is None:
                    # it should only have 1 `self` parameter
                    if len(inspect.signature(module_or_module_class.__init__).parameters) > 1:
                        raise ValueError("Module class __init__ should be parameter-free.")
                    module = module_or_module_class()
                else:
                    module = module_fn()
                    if type(module) != module_or_module_class:
                        raise ValueError(f"module_fn should return a {module_or_module_class} instance.")
            except Exception as e:
                raise RuntimeError(f"Error when creating module instance.") from e
        else:
            module = module_or_module_class

        if module_dtype is not None:
            module = module.to(dtype=module_dtype)

        if any(isinstance(m, CubeModule) for m in module.modules()):
            raise RuntimeError('Parallel modules can not be nested.')

        # save origin module metadata
        meta_info = OriginModuleMetadata(
            origin_param_names=[name for name, _ in module.named_parameters()],
            origin_state_dict_names=list(module.state_dict().keys()),
            origin_shared_param_names=get_shared_params(module),
        )
        torch.save(meta_info, origin_module_metadata_ckp)

        with wrapnn(module, restore=not is_module_class) as wrapped_module:
            graph, forward_args = _gen_graph(
                wrapped_module, dummy_forward_args, outdir,
                constant_folding=compute_config.constant_folding, end2end_mode=compute_config.use_end2end,
                inference_only=compute_config.inference_only,
            )

        graph.dump(graph_ckp)
        torch.save(forward_args, forward_args_ckp)

        if is_module_class:
            del module
    else:
        ret = RegenStatus.CODE
        logger.info(f"Reuse graph dump in {outdir}")
        graph = IRGraph.load(graph_ckp)
        forward_args = torch.load(forward_args_ckp)

    graph = pas_policy(graph, compute_config)
    if not isinstance(graph, IRGraph):
        raise RuntimeError("Expected policy return IRGraph")

    # currently graph.sched is only used for pipeline parallelism
    # so it is not none means we are in pipeline parallelism
    if graph.sched is not None and _contains_uncommutable_data(graph.outputs()):
        raise RuntimeError("Communication generation error: "
                           "some of outputs are not commutable between gpus, "
                           "which is not supported in pipeline parallelism.")

    # check assignment
    for node in graph.nodes(flatten=True):
        # skip graph anchor: will be removed
        # skip multiref and IRPyFunc: they will be managed by system
        if isinstance(node, IRGraphAnchor) or node.name == 'multiref':
            continue
        if isinstance(node, IRPyFunc):
            continue
        if isinstance(node, IRBpOperation) and node.mirror.name == 'multiref':
            continue
        if len(node.device) == 0:
            raise RuntimeError(f"Node {node} device is not set")
    # anchor node removed in gener
    graph = IRAdapterGener.gen(graph, cost_fn=None)
    if graph.sched is not None:
        graph.sched.apply()

    if isinstance(graph.sched, SchedulePlan):
        execplan = ExecutionPlan.from_schedplan(graph.sched)
    else:
        execplan = ExecutionPlan.from_graph(graph)

    execplan = DiffFusion.apply(execplan)
    # plan pass for computation grouping
    if not graph.sched:
        execplan = Grouping.apply(execplan)

    # code generation
    assert len(execplan.graph.device) == compute_config.plan_ngpus, f"{execplan.graph.device}"
    mgener = ModuleCodeGen(execplan, compute_config.runtime_ngpus)
    sgener = None
    if compute_config.use_end2end:
        sgener = ScheduleCodeGen(execplan, compute_config.runtime_ngpus)
    for rank in range(compute_config.runtime_ngpus):
        fname = outdir / _GENCODE_FILE_TEMPLATE.format(rank)
        mgener.gen(rank,
            forward_args=forward_args,
            outfile=fname,
            attach=False,
            as_parallel_module=True,
            end2end_mode=compute_config.use_end2end
        )
        # generate temporal schedule code only for end2end module
        # because the code generated is wrong for non-end2end module.
        if compute_config.use_end2end:
            sgener.gen(
                device=rank,
                outfile=fname,
                attach=True
            )

    return ret


def _load_parallel_module_class(
    module_class: Type[torch.nn.Module],
    *,
    gen_savedir: Union[str, Path] = './.nnscaler',
    instance_name: Optional[str] = None,
    rank: Optional[int] = None,
) -> Type[ParallelModule]:
    """
    Load the generated parallel module class, with train_step and infer_step assigned as member function..


    Please note that the parallel module class should be generated beforehand by _gencode().

    Args:
        module_class (Type[torch.nn.Module]): the original module class
        gen_savedir (Union[str, Path]): the directory to load generated code
        instance_name (Optional[str]): the instance name of the generated module. If it is None, will use the default name.
        rank (Optional[int]): the rank of the module. If it is None, will get the rank from torch.distributed.get_rank().
            This option is only useful for debugging or writing pre/post-processing tools.
            when you need to load the generated module in a non-torchrun environment.
    Returns:
        Type[ParallelModule]: the generated module class
    """
    rank = torch.distributed.get_rank() if rank is None else rank
    namespace, _ = _prepare_namespace(gen_savedir, module_class, instance_name)
    gen_imported = importlib.import_module(
        f'{namespace}.{Path(_GENCODE_FILE_TEMPLATE.format(rank)).stem}'
    )
    parallel_module_class = gen_imported.GenModel
    # rewrite class name and module name
    parallel_module_class.__name__ = module_class.__name__
    parallel_module_class.__qualname__ = module_class.__qualname__
    # parallel_module_class.__module__ = module_class.__module__
    parallel_module_class.__orig_module_class__ = module_class  # save the original module class
    # override train_step and infer_step only if they are defined in the generated module (end2end module only)
    parallel_module_class.runtime_version = getattr(gen_imported, 'runtime_version', None)
    parallel_module_class._train_step = getattr(gen_imported, '_train_step', parallel_module_class._train_step)
    parallel_module_class._infer_step = getattr(gen_imported, '_infer_step', parallel_module_class._infer_step)
    return parallel_module_class


def parallelize(
    module_or_module_class: Union[torch.nn.Module, Type[torch.nn.Module]],
    dummy_forward_args: Dict[str, Any],
    pas_policy: Union[str, Callable[[IRGraph, ComputeConfig], IRGraph]],
    compute_config: ComputeConfig,
    *,
    gen_savedir: Union[str, Path] = './.nnscaler',
    reuse: Union[ReuseType, str] = ReuseType.MATCH,
    instance_name: Optional[str] = None,
    load_module: bool = True,
    module_dtype:  Optional[torch.dtype] = None,
    module_fn: Optional[Callable[[], torch.nn.Module]] = None,
    init_module_params: bool = True,
    broadcast_strategy: Union[str, BroadcastGenFilesStrategy] = 'none',
) -> Union[None, ParallelModule, Type[ParallelModule]]:
    """
    Convert a torch.nn.Module object or class to ParallelModule object or class.

    If you want to save multiple instances of the same module,
    you can specify the instance_name to distinguish them.

    Currently you must use a shared file system to share the generated files (like mounted Azure Blob)
    Or you can unset load_module flag, and manually copy the generated files to other nodes.
    After all nodes have the generated files, you can call parallelize() again with load_module flag set.

    Note: if reuse is not set to ReuseType.MATCH,
    the generated code in outdir will be removed EVEN IF the code generation fails in this call.

    if the input is a module object.
    * The module object will be copied to cpu to handle possible insufficient gpu memory.
    * The training flag will be the same as the original module

    This function can be used to convert both module object and module class to parallel module or parallel module class.
    Among key-value arguments,
    module_fn and module_dtype control how to create the module object.
    whereas init_module_params controls how to load parallel module object after conversion is done.

    1. If the input is a module object, it will return a ParallelModule object if load_module is True.
       This is useful when the module is created by a factory function.

       a. module_fn is ignored.
       b. module_dtype is used to control the dtype of the input module.
       c. init_module_params is used to control whether to initialize the parallel module parameters when load it.

    2. If the input is a module class, it will return a ParallelModule sub class if load_module is True.

       a. module_fn is used to create the module object, or module's__init__ if not prent.
       b. module_dtype is used to control the dtype of the created module (by constructor or module_fn).
          Of course, it can be merged into module_fn.
       c. init_module_params is ignored.

    After the module is converted, you can use it to create module object by calling it like a module class.
    The module class is defined like:

    ::

        class GenModule(nnscaler.runtime.module.ParallelModule):
            def __init__(self, init_params=True):
                super().__init__()
                ...
            ...

    So you can use `init_params` in `__init__` to control whether to initialize the module parameters.
    For example, if you don't want to initialize module params:

    ::

        module = GenModule(init_params=False)

    Args:
        module_or_module_class (Union[torch.nn.Module, Type[torch.nn.Module]]): the module or module class to be compiled
        dummy_forward_args (Dict[str, Any]): the dummy input for the module forward
        pas_policy (Union[str, Callable[[IRGraph, ComputeConfig], IRGraph]]): the pas policy,
            it can be a name of builtin policies, or a custom policy function.
        compute_config (ComputeConfig): the environment resource
        reuse (ReuseType): specify which part can be reused.
        gen_savedir (Union[str, Path]): the directory to save generated code
        instance_name (Optional[str]): the instance name of the generated module. If it is None, will use the default name.
        load_module (bool): whether to load the generated module or module class after conversion is done.
        init_module_params (bool): If true, when we construct the module, all its parameters are initialized with the same value with when we traced.
            Otherwise, they will be empty tensor.
            This parameter will be passed to the module constructor,
            so it is only used when module_or_module_class is a module object, and load_module is true.
        module_dtype (Optional[torch.dtype]): the dtype of the module. Keep the module as it is if it is None.
        module_fn (Optional[Callable[[], torch.nn.Module]]): the function to create the module. Will use __init__ if it is None.
        broadcast_strategy (Union[str, BroadcastGenFilesStrategy]): the broadcast strategy for generated files.
            Please note that the broadcasting will only be done in torchrun environment,
            and will throw an error if torch.distributed is not initialized and broadcast_strategy is not NONE.
    Returns:
        Union[ParallelModule, Type[ParallelModule], None]:
            if load_module flag is set, return the converted ParallelModule object or class
            if load_module flag is not set, return None
    """
    if (
        isinstance(module_or_module_class, ParallelModule) or
        (inspect.isclass(module_or_module_class) and issubclass(module_or_module_class, ParallelModule))
    ):
        # already done
        return module_or_module_class if load_module else None

    if (
        isinstance(module_or_module_class, CubeModule) or
        (inspect.isclass(module_or_module_class) and issubclass(module_or_module_class, CubeModule))
    ):
        raise RuntimeError("Old style CubeModule is not supported")

    if isinstance(pas_policy, str):
        if not pas_policy in _PREDEFINED_POLICIES:
            raise ValueError(f"Invalid pas_policy: {pas_policy}")
        pas_policy = _PREDEFINED_POLICIES[pas_policy]

    is_module_class = inspect.isclass(module_or_module_class)
    module_class = module_or_module_class if is_module_class else module_or_module_class.__class__
    reuse = ReuseType(reuse) if isinstance(reuse, str) else reuse
    broadcast_strategy = BroadcastGenFilesStrategy(broadcast_strategy) if isinstance(broadcast_strategy, str) else broadcast_strategy

    # Call it here just to ensure the device group is initialized.
    # If the user initializes torch.distributed
    #     and doesn't call `nnscaler.init()` before calling this function, this is necessary.
    if torch.distributed.is_initialized():
        _ = DeviceGroup()

    # generate code only in node0
    # if it is not in a torchrun environment, just generate.
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        outdir, reusable = _prepare_and_check_reusable(gen_savedir, module_class, compute_config, instance_name, reuse)
        if not reusable:
            config_file = outdir / ParallelModule.COMPUTE_CONFIG_FILE
            ComputeConfig.safe_dump_to_file(compute_config, config_file)  # always refresh compute config
            with _compile_flags(compute_config):
                regen_status = _gencode(
                    module_or_module_class,
                    dummy_forward_args,
                    pas_policy,
                    compute_config,
                    outdir,
                    module_dtype=module_dtype,
                    module_fn=module_fn,
                )
        else:
            regen_status = RegenStatus.NONE
            logger.info(f"Reuse generated code in {outdir}")

    if torch.distributed.is_initialized():
        # code generation can take very long time (for example, over 1 hour)
        # It is not always OK to use torch.distributed.barrier() directly.
        # because the default timeout for nccl is 30 minutes
        # (we can't control the timeout setting if torch.distributed is not initialized by us)
        DeviceGroup().long_barrier()

    if broadcast_strategy != BroadcastGenFilesStrategy.NONE:
        if not torch.distributed.is_initialized(): # we only support loading in torchrun environment
            raise RuntimeError("Broadcast generated files failed: torch.distributed is not initialized.")
        torch.distributed.barrier()
        # sync regen_status
        curr_rank = torch.distributed.get_rank()
        if curr_rank == 0:
            sent_obj = [regen_status]
        else:
            sent_obj = [None]
        torch.distributed.broadcast_object_list(
            sent_obj,
            src=0,
        )
        if curr_rank != 0:
            regen_status = sent_obj[0]

        # narrow down broadcast_strategy according to regen_status
        if regen_status == RegenStatus.NONE:
            # we don't need to broadcast anything
            broadcast_strategy = BroadcastGenFilesStrategy.NONE
        elif regen_status == RegenStatus.CODE:
            # narrow ALL/NO_WEIGHTS down to code
            broadcast_strategy = BroadcastGenFilesStrategy.CODE
        else:
            # we don't need to narrow broadcast_strategy in this case
            # keep the original broadcast_strategy
            assert regen_status == RegenStatus.ALL

        # broadcast generated files according to regen_status
        if broadcast_strategy != BroadcastGenFilesStrategy.NONE:
            _broadcast_gen_files(
                module_class,
                gen_savedir=gen_savedir,
                instance_name=instance_name,
                broadcast_strategy=broadcast_strategy,
            )

    if load_module:
        if not torch.distributed.is_initialized(): # we only support loading in torchrun environment
            raise RuntimeError("Load ParallelModule failed: torch.distributed is not initialized.")
        torch.distributed.barrier()
        parallel_module_class = _load_parallel_module_class(
            module_class,
            gen_savedir=gen_savedir,
            instance_name=instance_name,
        )
        if is_module_class:
            return parallel_module_class
        else:
            parallel_module = parallel_module_class(init_module_params)
            parallel_module.train(module_or_module_class.training)  # set training state to the same as original module
            return parallel_module


@dataclass(unsafe_hash=True)
class ModuleParameterLocation:
    """
    the location of the parameters of a module in optimizer.param_groups[0]['params']
    [offset, offset + count) is the range of the parameters in optimizer.param_groups[0]['params']

    Args:
        offset: the first parameter's index in optimizer.state
        count: represents the number of parameters within this module.
    """
    offset: int
    count: int


@dataclass
class OptimizerExtraState:
    """
    Args:
        rank: the rank of the worker in torchrun
        name: the name of the optimizer type
        parallel_module_locs: the locations of the parameters of the parallelized module.
            the key is the module prefix of the parallel module.
            A module prefix is the same prefix used when you call `module.state_dict()` without the ending dot.
            For example, if you have a module

            ::

                module
                    submodule1_1
                        submodule2_1
                    submodule1_2

            then the prefix of `module` itself is `` (empty str).
            the prefix of `submodule1_1` is `submodule1_1`.
            the prefix of `submodule2_1` is `submodule1_1.submodule2_1`.
            etc.
    """
    rank: int
    name: str
    parallel_module_locs: Dict[str, ModuleParameterLocation]
    parallel_module_configs: Dict[str, ComputeConfig]

    def __post_init__(self):
        self.parallel_module_locs = {
            k: ModuleParameterLocation(**v) if isinstance(v, dict) else v
            for k, v in self.parallel_module_locs.items()
        }
        self.parallel_module_configs = {
            k: ComputeConfig(**v) if isinstance(v, dict) else v
            for k, v in self.parallel_module_configs.items()
        }


class ParallelOptimizer(torch.optim.Optimizer):
    """
    A optimizer stub to support parallelized module.
    The returned optimizer of build_optimizer() will have the same methods in this class.
    """

    # this is a reducer for non-parallel modules
    _non_parallel_module_reducer: Optional[Reducer] = None
    # the extra state that will be used when loading state dict.
    _extra_state: Optional[OptimizerExtraState] = None

    def sync_shard_grad(self):
        """
        Sync the shard gradients of the module from nodes with same shard to the optimizer.
        Please note this is called automatically in optimizer.step().
        But If you want to access the gradients before optimizer.step(),
        you need to call this function manually.
        """
        ...

    def clip_gnorm(self, max_norm: Optional[float] = None) -> torch.Tensor:
        """
        Clip the gradients with global norm, and return the global gnorm value.

        Args:
            max_norm (Optional[float]): the max global norm. If it is None, no clipping will be applied.

        Returns:
            torch.Tensor: the gradient norm.
        """
        ...

    def scale_grads(self, scale: float) -> None:
        """
        Scale the gradients of the module.

        Please note
        1. you can only call this function **after** `sync_shard_grad`,
        because the gradients are `None` until `sync_shard_grad` is called.
        2. Only the gradients of parameters in this optimizer be multiplied by this factor,
        (When ZERO is on, not all parameters of the module are added to the optimizer).

        Args:
            scale (float): the scale factor. Gradients will be multiplied by this factor.
        """
        ...

    def register_reducer_pre_hook(self, fn: Callable[[Reducer, torch.Tensor], None]):
        """
        Register pre hooks to reducers which will be applied before gradient synchronization.

        The pre-hooks will be applied one by one following the order of registration.

        Args:
            fn (Callable[[Reducer, torch.Tensor], None]): a callable function that takes a reducer and a gradient as input and optionally updates the gradient.
        """
        ...

    def register_reducer_post_hook(self, fn: Callable[[Reducer, torch.Tensor], None]):
        """
        Register post hooks to reducers which will be applied after gradient synchronization.

        The post-hooks will be applied one by one following the order of registration.

        Args:
            fn (Callable[[Reducer, torch.Tensor], None]): a callable function that takes a reducer and a gradient as input and optionally updates the gradient.
        """
        ...


OptimizerT = TypeVar('OptimizerT', bound=torch.optim.Optimizer)


def build_optimizer(
    module: torch.nn.Module,
    optimizer_fn: Union[Type[OptimizerT], Callable[..., OptimizerT]],
    *args,
    **kwargs,
) -> Union[OptimizerT, ParallelOptimizer]:
    """
    Build an optimizer for a module.

    To support parallelized module (ParallelModule), we hook 4 places in this function:

    1. optimizer constructor:
       the parameters of optimizer will not be the same with the parameters of the module if we use zero
       so we need to replace the parameters of optimizer with ParallelModule.parameters_for_optimizer
       It is impossible to make this change transparent to end users.
    2. optimizer.step():
       we need to call optimizer.sync_shard_grad() to sync the gradients of the module before optimizer.step().
       In zero mode, we have to call ParallelModule.gather_params() after optimizer.step()
    3. optimizer.zero_grad():
       We need to call ParallelModule.zero_grad() after optimizer.zero_grad()
    4. backward():
       you need to call optimizer.sync_shard_grad() manually if you want to read the gradients of the module before optimizer.step().

    Args:
        module (torch.nn.Module): the module to be optimized
        optimizer_fn (Union[Type[torch.optim.Optimizer], Callable[..., torch.optim.Optimizer]]):
            It can be the optimizer class or optimizer factory function.
            The first parameter of the optimizer_fn should be the parameters of the module.
        *args: other args for `optimizer_fn` besides parameters.
        **kwargs: the kwargs for optimizer constructor

    Returns:
        torch.optim.Optimizer: the optimizer you should use to train the module
        The optimizer is created by optimizer_fn,
        and will be patched with the methods in ParallelModule class to support parallelized module.
        Please note the type annotation of the returned optimizer (`Union[OptimizerT, ParallelOptimizer]`) is just for intellisense.
    """

    if isinstance(module, CubeModule) and not isinstance(module, ParallelModule):
        raise RuntimeError("Old style CubeModule is not supported")

    # only the root module can be end2end module.
    if any(m != module and isinstance(m, ParallelModule) and  m.compute_config.use_end2end for m in module.modules()):
        raise RuntimeError("End2End module cannot be nested in another module")

    RuntimeFlag.skip_reducer = True
    RuntimeFlag.skip_zero_grad = False

    non_parallel_module_reducer = None
    non_parallel_modules = [m for m in module.modules() if not isinstance(m, ParallelModule)]
    parallel_modules = [m for m in module.modules() if isinstance(m, ParallelModule)]
    if not parallel_modules:
        raise RuntimeError("No ParallelModule found in the module. Please make sure you have called parallelize() before build_optimizer().")

    # check if all ParallelModules have the same gpu_config
    compute_configs = [m.compute_config for m in parallel_modules]
    for i in range(1, len(compute_configs)):
        if compute_configs[i].gpu_config != compute_configs[0].gpu_config:
            raise RuntimeError("All ParallelModules should have the same gpu_config.")
    plan_ngpus, runtime_ngpus = compute_configs[0].plan_ngpus, compute_configs[0].runtime_ngpus

    # we need to add all parameters of non-parallel modules to a reducer to reduce grads
    # if there are non-parallel parameters
    if plan_ngpus != runtime_ngpus and non_parallel_modules and any(p.numel() for m in non_parallel_modules for p in m.parameters(False)):
        group, _ = compute_configs[0].get_sync_group()
        non_parallel_module_reducer = Reducer(group)
        for m in non_parallel_modules:
            for param in m.parameters(recurse=False): # only add leaf parameters to avoid duplicate
                non_parallel_module_reducer.add_param(param)
        non_parallel_module_reducer.build_buckets()

    opt_module_locs: Dict[str, ModuleParameterLocation] = {}
    def _local_parameters(module: torch.nn.Module):
        pm_suffix = "_PARALLEL_MODULE_PARAM_SUFFIX"
        gen = module._named_members(
            lambda m: [
                    (pm_suffix, p)  # (pm_suffix, p) to meet _named_members requirement
                    for p in (
                        m.parameters_for_optimizer() if m.compute_config.use_zero
                        else m.parameters() # `ParallelModule.merge_partial_states` supports parameters_for_optimizer() only in zero mode
                    )
                ]
                if isinstance(m, ParallelModule)
                else m._parameters.items()
        )
        for idx, (name, param) in enumerate(gen):
            if name.endswith(pm_suffix):  # is a parameter of ParallelModule
                # -1 for removing the dot
                # please note when the whole module is a ParallelModule,
                # the name will be empty after removing the suffix
                name = name[:-len(pm_suffix) - 1]
                if name not in opt_module_locs:
                    opt_module_locs[name] = ModuleParameterLocation(idx, 1)
                else:
                    opt_module_locs[name].count += 1
            yield param

    optimizer: torch.optim.Optimizer = optimizer_fn(_local_parameters(module), *args, **kwargs)
    optimizer._non_parallel_module_reducer = non_parallel_module_reducer
    optimizer._extra_state = OptimizerExtraState(
            rank=torch.distributed.get_rank(),
            name=type(optimizer).__name__,
            parallel_module_locs=opt_module_locs,
            parallel_module_configs={
                name: m.compute_config
                for name, m in module.named_modules()
                if isinstance(m, ParallelModule)
            }
    )

    def _step_pre_hook(opt, *args, **kwargs):
        opt.sync_shard_grad()

    def _step_post_hook(opt, *args, **kwargs):
        for m in parallel_modules:
            m.gather_params()

    # Please note:
    # register_step_pre_hook doesn't work expectly
    # when closure is used in optimizer.step()
    # in that case, you must call sync_shard_grad() manually
    optimizer.register_step_pre_hook(_step_pre_hook)
    optimizer.register_step_post_hook(_step_post_hook)

    orig_zero_grad = optimizer.zero_grad
    def _patched_zero_grad(self, set_to_none: bool = True):
        orig_zero_grad(set_to_none)
        for m in parallel_modules:
            m.zero_grad()
        if non_parallel_module_reducer:
            non_parallel_module_reducer.zero_grad()
    optimizer.zero_grad = types.MethodType(_patched_zero_grad, optimizer)

    orig_state_dict = optimizer.state_dict
    def _patched_state_dict(self):
        state_dict = orig_state_dict()
        state_dict[ParallelModule.EXTRA_STATE_KEY] = asdict(optimizer._extra_state)
        return state_dict
    optimizer.state_dict = types.MethodType(_patched_state_dict, optimizer)

    orig_load_state_dict = optimizer.load_state_dict
    def _patched_load_state_dict(self, state_dict):
        state_dict.pop(ParallelModule.EXTRA_STATE_KEY, None)
        orig_load_state_dict(state_dict)
    optimizer.load_state_dict = types.MethodType(_patched_load_state_dict, optimizer)

    def _sync_shard_grad(self):
        with _runtime_flags(skip_reducer=False):
            # HACK: we reuse the _sync_grad_required flag of the first parallel module
            # in order to support calling sync_shard_grad() multiple times.
            # _sync_grad_required will reset to `True` in forward() of ParallelModule.
            if parallel_modules[0]._sync_grad_required:
                for m in parallel_modules:
                    m.sync_grad()  # _sync_grad_required flag will reset inside sync_grad()

                if non_parallel_module_reducer:
                    non_parallel_module_reducer.sync_grads()

    optimizer.sync_shard_grad = types.MethodType(_sync_shard_grad, optimizer)

    @torch.no_grad()
    def _clip_gnorm(self, max_norm: Optional[float] = None):
        self.sync_shard_grad()
        total_norm_squared = 0.0
        grads: List[torch.Tensor] = []

        for m in parallel_modules:
            mnorm, mgrads = m.clip_gnorm(None)
            total_norm_squared += torch.square(mnorm)
            grads.extend(mgrads)

        if non_parallel_module_reducer:
            params = non_parallel_module_reducer.parameters_for_optimizer()
            mnorm, mgrads = calcuate_gnorm(params)
            total_norm_squared += torch.square(mnorm)
            grads.extend(mgrads)

        total_norm = torch.sqrt(total_norm_squared)
        if max_norm is not None and max_norm > 0:
            clip_grads(grads, total_norm, max_norm)

        return total_norm

    optimizer.clip_gnorm = types.MethodType(_clip_gnorm, optimizer)

    def _scale_grads(self, scale: float) -> None:
        if parallel_modules[0]._sync_grad_required:
            raise RuntimeError("You can only call scale_grads() after gradients are synchronized.")
        for pg in optimizer.param_groups:
            for p in pg['params']:
                if p.grad is not None:
                    p.grad.mul_(scale)

    optimizer.scale_grads = types.MethodType(_scale_grads, optimizer)

    def _register_reducer_pre_hook(self, fn: Callable[[Reducer, torch.Tensor], None]):
        for m in parallel_modules:
            for reducer in m.reducers:
                reducer.register_pre_hook(partial(fn, reducer))
        if non_parallel_module_reducer:
            non_parallel_module_reducer.register_pre_hook(partial(fn, non_parallel_module_reducer))

    def _register_reducer_post_hook(self, fn: Callable[[Reducer, torch.Tensor], None]):
        for m in parallel_modules:
            for reducer in m.reducers:
                reducer.register_post_hook(partial(fn, reducer))
        if non_parallel_module_reducer:
            non_parallel_module_reducer.register_post_hook(partial(fn, non_parallel_module_reducer))

    optimizer.register_reducer_pre_hook = types.MethodType(_register_reducer_pre_hook, optimizer)
    optimizer.register_reducer_post_hook = types.MethodType(_register_reducer_post_hook, optimizer)

    return optimizer


def _get_parallel_module_state_dict_info(
        model_state_dicts: List[Dict[str, Any]]
) -> Tuple[
    Dict[Tuple[str, ...], List[ExtraState]],    # parallel module extrastate for each rank
    Dict[Tuple[str,...], List[Dict[str, Any]]], # parallel module state dict for each rank
    Dict[str, Any]                              # non-parallel module state dict
]:
    # parted key model state dicts
    pk_model_state_dicts: List[Dict[Tuple[str,...], Any]] = []
    for model_state_dict in model_state_dicts:
        pk_model_state_dicts.append({tuple(k.split('.')): v for k, v in model_state_dict.items()})

    # find all parallel module state keys (whose key ends with ParallelModule.EXTRA_STATE_KEY)
    # key: the module prefix
    # value: the list of extra states from all ranks
    pm_extra_states: Dict[Tuple[str, ...], List[ExtraState]] = {}
    for pk_model_state_dict in pk_model_state_dicts:
        for k in pk_model_state_dict:
            if k[-1] == ParallelModule.EXTRA_STATE_KEY:
                module_prefix = k[:-1]
                if module_prefix not in pm_extra_states:
                    pm_extra_states[module_prefix] = [None] * len(pk_model_state_dicts)
                pm_extra_state = ExtraState(**pk_model_state_dict[k])
                pm_extra_states[module_prefix][pm_extra_state.rank] = pm_extra_state

    # collect ParallelModule state dicts
    # key is the module prefix of the parallel module in state dict
    # value is the list of state dicts of the parallel module from all ranks
    pm_state_dicts: Dict[Tuple[str,...], List[Dict[str, Any]]] = {}
    # non-parallel module state dict
    non_pm_state_dict: Dict[str, Any] = {}
    for pk_model_state_dict in pk_model_state_dicts:
        for k in pk_model_state_dict:
            if k[-1] == ParallelModule.EXTRA_STATE_KEY: # skip extra state, we already have them
                    continue
            module_prefix = k[:-1]
            if module_prefix in pm_extra_states:
                pm_extra_state = ExtraState(**pk_model_state_dict[module_prefix + (ParallelModule.EXTRA_STATE_KEY,)])
                module_dedup_group_size = pm_extra_state.compute_config.module_dedup_group_size
                if module_prefix not in pm_state_dicts:
                    pm_state_dicts[module_prefix] = [dict() for _ in range(module_dedup_group_size)]
                # only collect the state from the first module_dedup_group_size ranks
                if pm_extra_state.rank < module_dedup_group_size:
                    pm_state_dicts[module_prefix][pm_extra_state.rank][k[-1]] = pk_model_state_dict[k]
            else:
                # no further processing
                # here we assume values from all ranks are the same
                non_pm_state_dict['.'.join(k)] = pk_model_state_dict[k]

    return pm_extra_states, pm_state_dicts, non_pm_state_dict


def _get_optimizer_state_dict_info(
    optimizer_state_dicts: List[Dict[str, Any]]
) -> Tuple[
    List[OptimizerExtraState],
    Dict[str,                     # key: the module prefix
        List[Dict[                 # value: a list of dict from all ranks. The dict is
                str,               # key: the state key `state` (all other keys will be ignored.)
                Dict[              # value: a dict which is the same with opt_state_dict['state'], it is:
                    int,           # key: an integer representing the parameter index
                    Dict[str, Any] # value: a dict contains the parameter related info, the keys include 'step', 'exp_avg', 'exp_avg_sq'.
                ]
            ]
        ]
    ],
    Dict[str, Any]
]:
    """
    An example of optimizer state dict:
    {
        'state': {
            0: {'step': 10, 'exp_avg': ..., 'exp_avg_sq': ...},
            1: {'step': 10, 'exp_avg': ..., 'exp_avg_sq': ...},
            # no 2 here, because param 2 is not used
            3: {'step': 10, 'exp_avg': ..., 'exp_avg_sq': ...},
            4: {'step': 10, 'exp_avg': ..., 'exp_avg_sq': ...},
            5: {'step': 10, 'exp_avg': ..., 'exp_avg_sq': ...},
            6: {'step': 10, 'exp_avg': ..., 'exp_avg_sq': ...},
            # no 7 here, because param 7 is not used
        },
        'param_groups': [ {  # we only support the case when there is only one param_group
            'lr': ...,
            'betas': ...,
            'eps': ...,
            ...,
            'params': [0, 1, 2, 3, 4, 5, 6, 7]  # all params will be listed here, no matter it is used or not
        }]
    }
    """
    ret_opt_state_dict = {'state': {}}
    # collect optimizer state dicts
    # merge ParallelModule state dicts
    # here we only need to handle `state` key in the optimizer state dict
    # all other keys will be copied to the final state dict
    opt_extra_states: List[OptimizerExtraState] = [None] * len(optimizer_state_dicts)
    opt_state_dicts: Dict[str,     # key: the module prefix
        List[Dict[                 # value: a list of dict from all ranks. The dict is
                str,               # key: the state key `state` (all other keys will be ignored.)
                Dict[              # value: a dict which is the same with opt_state_dict['state'], it is:
                    int,           # key: an integer representing the parameter index
                    Dict[str, Any] # value: a dict contains the parameter related info, the keys include 'step', 'exp_avg', 'exp_avg_sq'.
                ]
            ]
        ]
    ] = {}
    for opt_state_dict in optimizer_state_dicts:
        opt_extra_state = OptimizerExtraState(**opt_state_dict[ParallelModule.EXTRA_STATE_KEY])
        if 'adam' not in opt_extra_state.name.lower():
            raise ValueError("Only Adam-like optimizers are supported.")
        opt_extra_states[opt_extra_state.rank] = opt_extra_state

        for module_prefix, loc in opt_extra_state.parallel_module_locs.items():
            opt_dedup_group_size = opt_extra_state.parallel_module_configs[module_prefix].optimizer_dedup_group_size
            if module_prefix not in opt_state_dicts:
                opt_state_dicts[module_prefix] = [dict(state={}, param_groups=[]) for _ in range(opt_dedup_group_size)]
            # only collect the state from the first optimizer_dedup_group_size ranks
            if opt_extra_state.rank < opt_dedup_group_size:
                for i in range(loc.offset, loc.offset + loc.count):
                    # if the parameter is not used or requires_grad is False, it will not be in the state dict
                    # for us, as we use a continous buffer, it will always have grad, so it will always be in the state dict
                    # the state for each parameters is inserted in Adam in a lazy way.
                    # see https://github.com/pytorch/pytorch/blob/dad1b765848c4f52501c4c60b1c3e6fbd3cc8837/torch/optim/adam.py#L103
                    assert i in opt_state_dict['state']
                    opt_state_dicts[module_prefix][opt_extra_state.rank]['state'][i - loc.offset] = opt_state_dict['state'][i]
                # TODO: inaccurate param_groups, for example, the 'params' in it is not right.
                # we have this to make `ParallelModule.merge_partial_states` happy.
                opt_state_dicts[module_prefix][opt_extra_state.rank]['param_groups'] = copy.deepcopy(opt_state_dict['param_groups'])

        for k, v in opt_state_dict.items():
            if k == ParallelModule.EXTRA_STATE_KEY or k == 'state':
                continue
            # no further processing
            # here we assume values from all ranks are the same
            # the value may change, so we deepcopy to make sure the input is not accidentally changed
            # for example, it will updated in `merge_state_dict` function.
            ret_opt_state_dict[k] = copy.deepcopy(v)

    return opt_extra_states, opt_state_dicts, ret_opt_state_dict


@torch.no_grad()
def merge_state_dicts(
    module_state_dicts: List[Dict[str, Any]],
    optimizer_state_dicts: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Dict[str, Any], Optional[List[Dict[str, Any]]]]:
    """
    Merge a list of shard state dicts (one for each rank) to a single full state dict
    Note: Only Adam-like optimizers are supported for merging

    Please Note:

    We don't garantee the devices of tensors are the same in the merged state dict.
    You can assume the device of the tensors in the merged state dict can be one of the following:

    1. the current device when running this function
    2. the current cuda device when running this function
    3. the device of the tensor in the original state dict

    When you load the state dict from file, you can just use `torch.load(..., map_location='...')` to unify the device of the tensors.

    Args:
        model_state_dicts (List[Dict[str, Any]]): the model state dicts from each rank
        optimizer_state_dicts (Optional[List[Dict[str, Any]]]): the optimizer state dicts from each rank

    Returns:
        Tuple[Dict[str, Any], Optional[List[Dict[str, Any]]]]: the merged model state dict and the merged optimizer state dict
    """
    if not module_state_dicts:
        raise ValueError("model_state_dicts should not be empty.")

    pm_extra_states, pm_state_dicts, ret_state_dict = _get_parallel_module_state_dict_info(module_state_dicts)
    if optimizer_state_dicts is not None:
        opt_extra_states, opt_state_dicts, ret_opt_state_dict = _get_optimizer_state_dict_info(optimizer_state_dicts)
        # the new optimizer state dict for ParallelModules
        # key: the parallel module location in the optimizer state
        # value: A tuple of
        #    0. the new state values for the parallel module
        #    (index is the parameter index in parallel module)
        #    1. the module prefix
        #    2. the original parameter names (OriginModuleMetadata.origin_param_names)
        opt_new_pm_states: Dict[ModuleParameterLocation, Tuple[Dict[int, Any], str, List[str]]] = {}
    else:
        opt_extra_states, opt_state_dicts, ret_opt_state_dict, opt_new_pm_states = None, None, None, None

    # merging parallel module state dicts,
    # non parallel module parts have been handled at _get_parallel_module_state_dict_info
    # and _get_optimizer_state_dict_info
    # every loop will merge one ParallelModule
    for k, state_dicts_for_merge in pm_state_dicts.items():
        extra_states = pm_extra_states[k]
        module_prefix = '.'.join(k)
        opt_state_dicts_for_merge = None if opt_state_dicts is None else opt_state_dicts[module_prefix]

        merge_partial_states_zero_idx_maps = [(e.model_idx2opt_idx, e.opt_idx2ranks) for e in extra_states]
        if not extra_states[0].compute_config.use_zero: # all ranks should have the same use_zero
            merge_partial_states_zero_idx_maps = None
        merged_state_dict, merged_opt_state_dict = ParallelModule.merge_state_dicts(
            [e.param_area_map for e in extra_states],
            state_dicts_for_merge,
            opt_state_dicts_for_merge,
            merge_partial_states_zero_idx_maps,
        )

        # merge back module state dict
        # all ranks have the same extra_states
        origin_state_dict_names = extra_states[0].origin_state_dict_names
        shared_param_names = extra_states[0].origin_shared_param_names
        for name in origin_state_dict_names:
            key = name if not module_prefix else f'{module_prefix}.{name}'
            if name in merged_state_dict:
                ret_state_dict[key] = merged_state_dict[name]
            else:
                name_in_merged = _get_valid_name_from_merged_model(name, shared_param_names, merged_state_dict)
                if name_in_merged is not None:
                    ret_state_dict[key] = merged_state_dict[name_in_merged]
                    key_in_merged = name_in_merged if not module_prefix else f'{module_prefix}.{name_in_merged}'
                    logger.warning(
                        f"Missing param/buffer {key} in merged_model_state_dict, "
                        f"safely using its shared param/buffer {key_in_merged} as {key}."
                    )
                else:
                    logger.warning(
                        f"Missing param/buffer {key} in merged_model_state_dict, "
                        f"high likely because {key} is created but not used in your model."
                    )

        # merge back opt state dict
        if opt_state_dicts is not None:
            opt_module_locs = [opt_extra_states[i].parallel_module_locs[module_prefix] for i in range(len(opt_extra_states))]

            # We can't assume all ranks have the same opt_module_locs (offset and count)
            # when we use pipeline parallelism, different ranks may have different opt_module_locs
            # fortunately, we can use the location information from any rank to do the merging in following
            # here we always use the location information from rank 0
            # for i in range(1, len(opt_module_locs)):
            #     assert opt_module_locs[i] == opt_module_locs[0]
            opt_new_pm_states[opt_module_locs[0]] = (merged_opt_state_dict['state'], module_prefix, extra_states[0].origin_param_names)

    if opt_new_pm_states:
        pm_orig_param_names: Dict[str, List[str]] = {}
        for k, extra_states in pm_extra_states.items():
            module_prefix = '.'.join(k)
            pm_orig_param_names[module_prefix] = ParallelModule.get_origin_parameter_names([e.param_area_map for e in extra_states])
        # now we can construct the merged state of optimizer from any rank
        # as said previously, the merge will be based on rank0's data
        orig_states: Dict[int, Any] = optimizer_state_dicts[0]['state']
        ret_states: Dict[int, Any] = {}  # see `_get_optimizer_state_dict_info` for the value structure.
        sorted_pm_locs = sorted(opt_new_pm_states.keys(), key=lambda x: x.offset)
        assert len(optimizer_state_dicts[0]['param_groups']) == 1
        orig_effective_state_len = len(optimizer_state_dicts[0]['param_groups'][0]['params'])
        orig_cur_index = 0        # index of orig_states
        ret_states_cur_index = 0  # index of ret_state_dict
        sorted_pm_locs_cur_index = 0 # index of sorted_pm_locs
        while orig_cur_index < orig_effective_state_len:
            if (
                sorted_pm_locs_cur_index >= len(sorted_pm_locs)  # after all parallel module parameters
                or orig_cur_index < sorted_pm_locs[sorted_pm_locs_cur_index].offset  # not in the range of current parallel module
            ):
                # non parallel module paramters
                if orig_cur_index in orig_states:
                    ret_states[ret_states_cur_index] = orig_states[orig_cur_index]
                orig_cur_index += 1
                ret_states_cur_index += 1
            else:
                # parallel module parameters
                pm_loc = sorted_pm_locs[sorted_pm_locs_cur_index]
                state, module_prefix, orignal_param_names = opt_new_pm_states[pm_loc]
                named_state = {}  #  the state dict with named keys
                for i, v in state.items():
                    named_state[pm_orig_param_names[module_prefix][i]] = v
                # reorder with the order of original param names
                for i, name in enumerate(orignal_param_names):
                    if name in named_state:
                        v = named_state[name]
                        ret_states[ret_states_cur_index + i] = v
                # always increase the index by the count of the original module parameters
                ret_states_cur_index += len(orignal_param_names)
                orig_cur_index += pm_loc.count
                sorted_pm_locs_cur_index += 1

        ret_opt_state_dict['state'] = ret_states
        ret_opt_state_dict['param_groups'][0]['params'] = list(range(ret_states_cur_index))

    return ret_state_dict, ret_opt_state_dict


@torch.no_grad()
def load_merged_state_dict(
    module: torch.nn.Module,
    module_state_dict: Dict[str, Any],
    optimizer: Optional[Union[torch.optim.Optimizer, ParallelOptimizer]] = None,
    optimizer_state_dict: Optional[Dict[str, Any]] = None,
    *,
    device: Union[str, torch.device] = None
):
    """
    Load the merged state dicts to the module, and optionally the optimizer to a specified device.

    Args:
        module (torch.nn.Module): the module to be loaded
        module_state_dict (Dict[str, Any]): the merged model state dict
        optimizer (Optional[torch.optim.Optimizer]): the optimizer to be loaded
        optimizer_state_dict (Optional[Dict[str, Any]]): the merged optimizer state dict
        device (Union[str, torch.device]): the device to put the module and optimizer state dicts.
            Use torch.cuda.current_device() if it is None.

    Returns:
        None
    """
    device = device or torch.cuda.current_device()

    # non ParallelModule parameters will be loaded here
    # there will be mismatched keys if the module is a ParallelModule or contains ParallelModule
    # so we need to ignore the mismatched keys
    module.load_state_dict(module_state_dict, strict=False)
    # load ParallelModule state dicts
    for name, child_module in module.named_modules():
        if isinstance(child_module, ParallelModule):
            prefix = name + '.' if name else ''
            child_module.load_merged_state_dict(module_state_dict, prefix=prefix)

    module.to(device)

    if optimizer is not None and optimizer_state_dict is not None:
        if 'adam' not in optimizer._extra_state.name.lower():
            raise ValueError("Only Adam-like optimizers are supported.")

        # handle non-paralleled module parameters
        # make sure the order of the parameters
        pm_name_locs: Dict[str, ModuleParameterLocation] = dict(sorted(optimizer._extra_state.parallel_module_locs.items(), key=lambda x: x[1].offset))
        pm_modules: List[torch.nn.Module] = []
        pm_locs = list(pm_name_locs.values())
        for name in pm_name_locs:
            m = get_member_by_name(module, name)
            if not isinstance(m, ParallelModule):
                raise ValueError(f"Module {name} is not a ParallelModule")
            pm_modules.append(m)

        merged_cur = 0  # the current index of the merged state dict
        pm_cur = 0      # the current index of the parallel module in pm_locs
        new_states: Dict[int, Dict[str, Any]] = {}
        new_cur = 0     # the current index of the new state dict
        assert len(optimizer_state_dict['param_groups']) == 1
        effective_state_len = len(optimizer_state_dict['param_groups'][0]['params'])
        while merged_cur < effective_state_len:
            # N: non-paralleled module parameters, P: paralleled module (will have multiple parameters)
            # The parameter list would look like: NNPNPPPN
            # []: the current processing parameter
            # <>: the current processing parallel module
            if (
                pm_cur >= len(pm_modules)  # NNPNPPP[N]:  the ending parameters, no current parallel module
                or new_cur < pm_locs[pm_cur].offset  # [N]N<P>NPPPN: other parameters
            ):
                # non-parallel module
                if merged_cur in optimizer_state_dict['state']:
                    new_states[new_cur] = optimizer_state_dict['state'][merged_cur]
                merged_cur += 1
                new_cur += 1
            else:
                # NNPN<[P]PP>N: the current parallel module
                # parallel module
                pm_param_count = len(pm_modules[pm_cur]._orign_module_metadata.origin_param_names)
                # will map `pm_param_count` parameters in merge state dict
                # to `pm_locs[pm_cur].count` in optimizer state.
                cur_states = {}
                for i in range(pm_param_count):
                    if merged_cur + i in optimizer_state_dict['state']:
                        cur_states[i] =optimizer_state_dict['state'][merged_cur + i]
                pm_new_states = _opt_load_merged_state_dict(pm_modules[pm_cur], cur_states)
                for idx, value in pm_new_states.items():
                    new_states[new_cur + idx] = value
                new_cur += pm_locs[pm_cur].count
                merged_cur += pm_param_count
                pm_cur += 1

        # move the new states to the device if needed
        for idx, state in new_states.items():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    new_states[idx][key] = value.to(device)

        new_optimizer_state_dict = {}
        new_optimizer_state_dict['state'] = new_states
        new_optimizer_state_dict['param_groups'] = copy.deepcopy(optimizer_state_dict['param_groups'])
        new_optimizer_state_dict['param_groups'][0]['params'] = list(range(new_cur))
        optimizer.load_state_dict(new_optimizer_state_dict)


def _opt_load_merged_state_dict(module: ParallelModule, states: Dict[int, Dict[str, Any]]):
    with torch.no_grad():
        # orig_name -> state
        orig_param_dict: Dict[str, Dict[str, Any]] = {}
        cnt = 0
        origin_param_names = module._orign_module_metadata.origin_param_names
        for name in origin_param_names:
            if cnt in states:  # some parameters may not in the sates when it is not used or requires_grad is False in training
                orig_param_dict[name] = states[cnt]
            cnt = cnt + 1

        if module.compute_config.use_zero:
            return _construct_optim_state_zero(module, orig_param_dict)
        else:
            return _construct_optim_state_nonzero(module, orig_param_dict)


def _construct_optim_state_zero(
        module: ParallelModule,
        orig_param_dict: Dict[str, Dict[str, Any]],
):
    dist_param_map = module.dist_param_map  # name in parallel module (without tid suffix) -> name in origin module
    param_area_map = module.fullmap         # str -> AttrMeta
    def _get_optimizer_state_of_param(param, param_ids, local_names):
        # find the parameter's optimizer state and pick the slices induced by tensor parallelism
        param_idx = param_ids.index(id(param))
        local_name = local_names[param_idx]
        return _extract_new_state(local_name, orig_param_dict, dist_param_map, param_area_map)

    # prepare param ids and corresponding local param names
    param_ids, local_names = [], []
    for local_name, param in module.named_parameters():
        param_ids.append(id(param))
        local_names.append(local_name)
    state_dict, opt_param_idx = {}, 0
    opt_param = module.parameters_for_optimizer()
    # first load the params' optimizer state for the reducers's flattened params
    for reducer in module.reducers:
        rank_idx, sub_ranks = module._get_zero_subranks(reducer)
        for bucket in reducer.buckets:
            # one bucket corresponds to one flattened param
            assert len(opt_param[opt_param_idx].shape) == 1
            assert bucket._contiguous_params.shape[0] % len(sub_ranks) == 0
            chunk_size = bucket._contiguous_params.shape[0] // len(sub_ranks)
            # the flattened param is in the range [bucket_chunk_start, bucket_chunk_end)
            bucket_chunk_start = rank_idx * chunk_size
            bucket_chunk_end = (rank_idx + 1) * chunk_size
            # NOTE: assume the traverse order of params is consistent
            # with them in contiguous buffer.
            # param_offset: the param's start offset in the contiguous buffer
            # chunk_offset: the current offset of the current rank corresponding chunk
            param_offset, chunk_offset = 0, 0
            step, opt_states, opt_state_keys = None, {}, None
            for param in bucket.params:
                sliced_new_val = _get_optimizer_state_of_param(param, param_ids, local_names)
                # there are padding in the chunk, so `param.numel()` doesn't work here
                param_numel = bucket.get_aligned_numel(param)
                # init the chunk's optimizer state
                if opt_state_keys is None:
                    opt_state_keys = [key for key in sliced_new_val]
                    if 'step' in sliced_new_val:
                        step = sliced_new_val['step']
                    if 'step' in sliced_new_val:
                        opt_state_keys.remove('step')
                    for key in opt_state_keys:
                        opt_states[key] = torch.zeros([chunk_size], dtype=sliced_new_val[key].dtype,
                                                        device=sliced_new_val[key].device, requires_grad=False)
                # copy the param's slices to the optimizer's chunk
                for key in opt_state_keys:
                    sliced_new_val[key] = sliced_new_val[key].view(-1)

                # parameter range: <>
                # bucket range: []
                # in the following branches, we check the range including paddings.
                # but in branch body, we only copy the valid range (without paddings) but update the chunk_offset with paddings.
                if param_offset < bucket_chunk_start \
                    and bucket_chunk_start < param_offset + param_numel < bucket_chunk_end:
                    # case: < [ > ]
                    copy_size = param_offset + param_numel - bucket_chunk_start
                    copy_size_without_padding = param_offset + param.numel() - bucket_chunk_start
                    if copy_size_without_padding > 0:
                        for key in opt_state_keys:
                            opt_states[key][chunk_offset:chunk_offset+copy_size_without_padding] = sliced_new_val[key][-copy_size_without_padding:]
                    chunk_offset += copy_size
                elif bucket_chunk_start <= param_offset < bucket_chunk_end \
                    and bucket_chunk_start <= param_offset + param_numel < bucket_chunk_end:
                    # case: [ <  > ]
                    for key in opt_state_keys:
                        opt_states[key][chunk_offset:chunk_offset+param.numel()] = sliced_new_val[key][:]
                    chunk_offset += param_numel
                elif bucket_chunk_start <= param_offset < bucket_chunk_end \
                    and param_offset + param_numel >= bucket_chunk_end:
                    # case: [ < ] >
                    copy_size = bucket_chunk_end - param_offset
                    copy_size_without_padding = min(copy_size, param.numel())
                    for key in opt_state_keys:
                        opt_states[key][chunk_offset:chunk_offset+copy_size_without_padding] = sliced_new_val[key][:copy_size_without_padding]
                    chunk_offset += copy_size
                elif param_offset < bucket_chunk_start \
                    and param_offset + param_numel >= bucket_chunk_end:
                    # case: < [ ] >
                    copy_size = bucket_chunk_end - bucket_chunk_start
                    copy_size_without_padding = min(copy_size, param_offset + param.numel() - bucket_chunk_start)
                    if copy_size_without_padding > 0:
                        for key in opt_state_keys:
                            opt_states[key][chunk_offset:chunk_offset + copy_size_without_padding] \
                                = sliced_new_val[key][bucket_chunk_start-param_offset:bucket_chunk_start-param_offset + copy_size_without_padding]
                    chunk_offset += copy_size
                else:
                    # case: [] <>, <> []
                    logger.debug(f'Skipped: parameter range({param_offset},{param_offset + param_numel}) vs. bucket range({bucket_chunk_start},{bucket_chunk_end})')
                param_offset += param_numel

            if step is not None:
                opt_states['step'] = step
            state_dict[opt_param_idx] = opt_states
            opt_param_idx += 1
    # load the params' optimizer state that are not in reducers
    # this part corresponds to nnscaler/runtime/module.py: parameters_for_optimizer
    reducer_pids = set()
    for reducer in module.reducers:
        reducer_pids.update(id(p) for p in reducer.params)
    for param in module.parameters():
        if id(param) not in reducer_pids:
            sliced_new_val = _get_optimizer_state_of_param(param, param_ids, local_names)
            state_dict[opt_param_idx] = sliced_new_val
            opt_param_idx += 1
    return state_dict


def _construct_optim_state_nonzero(
        module: ParallelModule,
        orig_param_dict: Dict[str, Dict[str, Any]]
):
    dist_param_map = module.dist_param_map  # name in parallel module (without tid suffix) -> name in origin module
    param_area_map = module.fullmap         # str -> AttrMeta

    new_states = {}
    for index, (local_name, _) in enumerate(module.named_parameters()):
        new_states[index] = _extract_new_state(local_name, orig_param_dict, dist_param_map, param_area_map)

    return new_states


def _extract_new_state(
        local_name: str,
        orig_param_dict: Dict[str, Dict[str, Any]],
        dist_param_map: Dict[str, str],
        param_area_map: Dict[str, AttrMeta],
):
    name = '_'.join(local_name.split('_')[:-1]) # remove the integer suffix
    assert name in dist_param_map
    attr_meta = param_area_map[local_name]
    new_val = orig_param_dict[dist_param_map[name]]
    sliced_new_val = {}
    for key in new_val:
        if key in ('step',):
            sliced_new_val[key] = new_val[key]
        else:
            sliced_new_val[key] = new_val[key][attr_meta.slicers] / attr_meta.val_chunks
    return sliced_new_val


def _get_valid_name_from_merged_model(
        target_name: str,
        shared_param_names: List[List[str]],
        merged_model_state_dict: Dict[str, Any]
) -> Optional[str]:
    """Find target_name in one set of shared_param_names, then find a name in merged_model_state_dict
    that is in the same set as target_name.
    """
    for shared_names in shared_param_names:
        if target_name in shared_names:
            for name in shared_names:
                if name in merged_model_state_dict:
                    return name
            break
    return None


def _broadcast_gen_files(
    module_class: Type[torch.nn.Module],
    *,
    gen_savedir: Union[str, Path] = './.nnscaler',
    instance_name: Optional[str] = None,
    broadcast_strategy: Union[str, BroadcastGenFilesStrategy],
):
    """
    Broadcast new generated files for a module to all nodes.

    Args:
        module_class (Type[torch.nn.Module]): the original torch module class
        gen_savedir (Union[str, Path]): the directory to save generated code
        instance_name (Optional[str]): the instance name of the generated module. If it is None, will use the default name.
        broadcast_strategy (Union[str, BroadcastGenFilesStrategy]): the broadcast strategy for generated files.

    Returns:
        None
    """

    broadcast_strategy = BroadcastGenFilesStrategy(broadcast_strategy) if isinstance(broadcast_strategy, str) else broadcast_strategy
    if broadcast_strategy == BroadcastGenFilesStrategy.NONE:
        return

    world_size = torch.distributed.get_world_size()
    local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', default=1))
    assert world_size % local_world_size == 0, "world_size should be a multiple of local_world_size"
    nnode = world_size // local_world_size

    if nnode == 1:
        # no need to broadcast generated files
        return

    curr_rank = torch.distributed.get_rank()
    ranks = list(range(0, world_size, local_world_size))
    group = DeviceGroup().get_group(ranks)

    # use the first rank of each node to broadcast
    if  curr_rank % local_world_size == 0:
        _, outdir = _prepare_namespace(gen_savedir, module_class, instance_name)
        files: List[str] = []
        # send file list
        if curr_rank == 0:
            for file in outdir.glob('*'):
                if file.is_file() and (
                    broadcast_strategy == BroadcastGenFilesStrategy.ALL or
                    (
                        broadcast_strategy == BroadcastGenFilesStrategy.NO_WEIGHTS
                        and not file.name.startswith(FxModuleParser.ATTR_CONTENT_FILE_STEM)
                    ) or
                    (
                        # broadcast code files and compute config file
                        # please note the compute config file can be updated
                        # even when the graph is reused.
                        broadcast_strategy == BroadcastGenFilesStrategy.CODE
                        and (file.suffix  == '.py' or file.name == ParallelModule.COMPUTE_CONFIG_FILE)
                    )
                ):
                    files.append(file.name)
            sent_obj = [files]
        else:
            sent_obj = [None]
        torch.distributed.broadcast_object_list(
            sent_obj,
            src=0,
            group=group,
        )
        # get file list
        if curr_rank != 0:
            files = sent_obj[0]

        logging.info(f'File list broadcasted ({len(files)} in total).')
        # send file content one by one
        for fname in files:
            if curr_rank == 0:
                with open(outdir / fname, 'rb') as f:
                    data = [f.read()]
            else:
                data = [None]
            torch.distributed.broadcast_object_list(data, src=0, group=group)
            if curr_rank != 0:
                with open(outdir / fname, 'wb') as f:
                    f.write(data[0])
            logging.info(f'File {fname} broadcasted.')

    # wait for all nodes to finish
    torch.distributed.barrier()


@torch.no_grad()
def deduped_state_dict(
    module: torch.nn.Module,
    optimizer: Optional[Union[torch.optim.Optimizer, ParallelOptimizer]] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Return the state dict only for the ranks that is necessary.
    For details, see `ComputeConfig.optimizer_dedup_group_size`
    and `ComputeConfig.module_dedup_group_size`.

    Args:
        module (torch.nn.Module): the module to get state dict
        optimizer (Optional[Union[torch.optim.Optimizer, ParallelOptimizer]]): the optimizer to get state dict

    Returns:
        Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]: the deduped state dict for the module and optimizer
    """

    cur_rank = torch.distributed.get_rank()
    module_state_dict, opt_state_dict = None, None
    parallel_modules = {prefix: m for prefix, m in module.named_modules() if isinstance(m, ParallelModule)}

    # The reason we use `Module.state_dict` on the whole to get the complete state dict
    # instead of call `Module.state_dict` on each submodule
    # is to make sure the hooks to state_dict are called.
    module_state_dict = module.state_dict()
    for key in list(module_state_dict.keys()):
        if key.endswith(ParallelModule.EXTRA_STATE_KEY): # never remove extra state
            continue
        prefix = '.'.join(key.split('.')[:-1]) # remove the last part of the key
        dedup_group_size = parallel_modules[prefix].module_dedup_group_size \
            if prefix in parallel_modules else 1
        # only keep the first `dedup_group_size` ranks' state
        if cur_rank >= dedup_group_size:
            module_state_dict.pop(key, None)

    if optimizer is not None:
        opt_state_dict = optimizer.state_dict()

        # get the locations of non-parallel module parameters
        # by removing the parallel module locations
        non_parallel_module_locs: Set[int] = set(opt_state_dict['param_groups'][0]['params'])
        for pm_loc in optimizer._extra_state.parallel_module_locs.values():
            non_parallel_module_locs.difference_update(range(pm_loc.offset, pm_loc.offset + pm_loc.count))

        # only keep non-parallel module parameters in rank 0
        if cur_rank > 0:
            for idx in non_parallel_module_locs:
                opt_state_dict['state'].pop(idx, None)

        for pm_prefix, pm_loc in optimizer._extra_state.parallel_module_locs.items():
            dedup_group_size = optimizer._extra_state.parallel_module_configs[pm_prefix].optimizer_dedup_group_size
            # only keep the first `dedup_group_size` ranks' state
            if cur_rank >= dedup_group_size:
                for idx in range(pm_loc.offset, pm_loc.offset + pm_loc.count):
                    opt_state_dict['state'].pop(idx, None)

    return module_state_dict, opt_state_dict


@torch.no_grad()
def load_deduped_state_dict(
    module: torch.nn.Module,
    module_state_dict: Dict[str, Any],
    optimizer: Optional[Union[torch.optim.Optimizer, ParallelOptimizer]] = None,
    optimizer_state_dict: Optional[Dict[str, Any]] = None,
    *,
    device: Union[str, torch.device] = None
) -> None:
    """
    Load the deduped state dicts to the module and optionally the optimizer to a specified device.

    Args:
        module (torch.nn.Module): the module to be loaded
        module_state_dict (Dict[str, Any]): the deduped model state dict
        optimizer (Optional[Union[torch.optim.Optimizer, ParallelOptimizer]]): the optimizer to be loaded
        optimizer_state_dict (Optional[Dict[str, Any]]): the deduped optimizer state dict
        device (Union[str, torch.device]): the device to put the module and optimizer state dicts.
            Use torch.cuda.current_device() if it is None.
    Returns:
        None
    """
    device = device or torch.cuda.current_device()

    # only load partial state for all ranks except rank 0
    module.load_state_dict(module_state_dict, strict=False)
    module.to(device)
    torch.distributed.barrier()

    # broadcast weights
    broadcast_weights(module)

    if optimizer is not None:
        if 'adam' not in optimizer._extra_state.name.lower():
            raise ValueError("Only Adam-like optimizers are supported.")
        if optimizer_state_dict is None:
            raise ValueError("optimizer_state_dict should be provided when optimizer is not None.")

        for idx, state in optimizer_state_dict['state'].items():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    optimizer_state_dict['state'][idx][key] = value.to(device)

        # get the locations of non-parallel module parameters
        # by removing the parallel module locations
        non_parallel_module_locs: Set[int] = set(optimizer_state_dict['param_groups'][0]['params'])
        # a list of tuple to track how to broadcast states
        # Tuple:
        #   0: a list of state idx
        #   1: the dedup group size for the state idx's
        opt_broadcast_groups: List[Tuple[List[int], int]] = []
        for prefix, pm_loc in optimizer._extra_state.parallel_module_locs.items():
            state_range = list(range(pm_loc.offset, pm_loc.offset + pm_loc.count))
            opt_broadcast_groups.append((state_range, optimizer._extra_state.parallel_module_configs[prefix].optimizer_dedup_group_size))
            non_parallel_module_locs.difference_update(state_range)
        # append also works
        # but insert to 0 feels better
        # the dedup size for non-parallel module is 1
        if non_parallel_module_locs:
            opt_broadcast_groups.insert(0, (list(non_parallel_module_locs), 1))

        for bg in opt_broadcast_groups:
            _broadcast_opt_state(optimizer_state_dict, *bg)
        optimizer.load_state_dict(optimizer_state_dict)

        torch.distributed.barrier()


def _broadcast_opt_state(optimizer_state_dict, state_indexes: List[int], dedup_group_size: int):
    rank = torch.distributed.get_rank()
    broadcast_group = setup_stride_broadcast_group(dedup_group_size)
    src_rank, curr_parallel_group, curr_parallel_group_ranks = broadcast_group.src_rank, broadcast_group.group, broadcast_group.ranks

    logging.info(f'Rank-{rank} is broadcasting states to ranks {curr_parallel_group_ranks}, broadcast root: {src_rank}...')

    # broadcast param groups and state keys/shapes/dtypes via broadcast_object_list
    if rank == src_rank:
        state_info = {}
        for idx in state_indexes:
            state_info[idx] = {key: (value.shape, value.dtype) for key, value in optimizer_state_dict['state'][idx].items()}
        sent = [state_info]
    else:
        sent = [None]
    torch.distributed.broadcast_object_list(
            sent,
            src=src_rank,
            group=curr_parallel_group,
    )
    if rank != src_rank:
        for k, v in sent[0].items():
            optimizer_state_dict['state'][k] = {
                key: torch.zeros(value[0], dtype=value[1], device=torch.cuda.current_device())
                for key, value in v.items()
            }

    # broadcast step
    # step is too small, so we can just broadcast all of them all together
    if rank == src_rank:
        step_stack = torch.stack(
            [optimizer_state_dict['state'][k]['step'] for k in state_indexes]
        )
    else:
        step_stack = torch.zeros(
            len(state_indexes),
            dtype=optimizer_state_dict['state'][0]['step'].dtype,
            device=torch.cuda.current_device()
        )
    torch.distributed.broadcast(step_stack, src=src_rank, group=curr_parallel_group)
    if rank != src_rank:
        for k, v in zip(state_indexes, step_stack):
            optimizer_state_dict['state'][k]['step'].copy_(v)

    # broadcast other states
    # TODO: can be slow?
    for k in state_indexes:
        keys = sorted(optimizer_state_dict['state'][k].keys())
        # for mixed precision f16 optimizer, we will add custom keys
        # assert set(keys) == {'step', 'exp_avg', 'exp_avg_sq'}
        keys.remove('step')  # we have done step in previous.
        for key in keys:
            value = optimizer_state_dict['state'][k][key]
            torch.distributed.broadcast(value.data, src=src_rank, group=curr_parallel_group)

    torch.distributed.barrier()


def broadcast_weights(module: torch.nn.Module, stride_size: Optional[int] = None):
    """
    Broadcast the weights of the module from the ranks in dedup group to all ranks.

    When you load the deduped state dict to broadcast the weights, you don't need to specify the `stride_size`.

    Args:
        module (torch.nn.Module): the module to be broadcasted
        stride_size (Optional[int]): the stride size for broadcast.
            If it is None, will use the dedup group size of each submodule.
    Returns:
        None
    """
    parallel_modules = {prefix: m for prefix, m in module.named_modules() if isinstance(m, ParallelModule)}

    for prefix, m in module.named_modules():
        if stride_size is not None:
            stride = stride_size
        elif prefix not in parallel_modules:
            stride = 1
        else:
            stride = parallel_modules[prefix].module_dedup_group_size
        _broadcast_weights(m, stride)


def _broadcast_weights(module: torch.nn.Module, stride_size: int):
    broadcast_group = setup_stride_broadcast_group(stride_size)
    rank = torch.distributed.get_rank()
    src_rank, curr_parallel_group, curr_parallel_group_ranks = broadcast_group.src_rank, broadcast_group.group, broadcast_group.ranks
    logging.info(f'Rank-{rank} is broadcasting weight to ranks {curr_parallel_group_ranks}, broadcast root: {src_rank}...')

    if isinstance(module, ParallelModule):
        if not _broadcast_single_value(src_rank, curr_parallel_group, module.non_presistent_buffers_inited):
            module._warn_uninitialized_non_persistent_buffers(raise_error=True)

    # we have a special optimization for ParallelModule
    params = module.parameters_for_broadcast() if isinstance(module, ParallelModule) else module._parameters.values()
    logging.info(f'Inplace broadcasting {len(params)} parameters...')
    for i, param in enumerate(params):
        torch.distributed.broadcast(param.data, src=src_rank, group=curr_parallel_group)
        logging.info(f'Inplace broadcasted {i+1}/{len(params)} parameters')

    # NOTE: may batch buffers for efficient broadcast,
    # current implementation is the most memory efficient way.
    logging.info(f'Inplace broadcasting {len(module._buffers)} buffers...')
    for _, buffer in module._buffers.items():
        torch.distributed.broadcast(buffer.data, src=src_rank, group=curr_parallel_group)

    if isinstance(module, ParallelModule):
        module.mark_non_persistent_buffers_inited()

    torch.distributed.barrier()


@torch.no_grad()
def load_sharded_state_dict(
    module: torch.nn.Module,
    module_state_dict: Dict[str, Any],
    optimizer: Optional[Union[torch.optim.Optimizer, ParallelOptimizer]] = None,
    optimizer_state_dict: Optional[Dict[str, Any]] = None,
    *,
    device: Union[str, torch.device] = None
):
    """
    Load the sharded state dicts to the module, and optionally the optimizer to a specified device.

    Args:
        module (torch.nn.Module): the module to be loaded
        module_state_dict (Dict[str, Any]): the sharded model state dict
        optimizer (Optional[torch.optim.Optimizer]): the optimizer to be loaded
        optimizer_state_dict (Optional[Dict[str, Any]]): the sharded optimizer state dict
        device (Union[str, torch.device]): the device to put the module and optimizer state dicts.
            Use torch.cuda.current_device() if it is None.

    Returns:
        None
    """

    device = device or torch.cuda.current_device()
    module.load_state_dict(module_state_dict)
    module.to(device)
    if optimizer:
        if optimizer_state_dict is None:
            raise ValueError("optimizer_state_dict should be provided when optimizer is not None.")
        optimizer.load_state_dict(optimizer_state_dict)


def sync_grad_when(cond: bool):
    """
    Context manager to enable/disable gradient synchronizations across workers.

    Within this context, gradients will be accumulated
    only when `cond` is True.

    This is needed when
    1. The mode is not end2end model.
        For end2end model, gradients are synchronized across workers automatically.
    2. async is enabled (`compute_config.use_async_reducer` is `True`).

    If both conditions are not satisfied, this function has no effect.

    Example:
        >>> model = parallelize(model, ...)
        >>> accum_steps = ...
        >>> for step in range(accum_steps)
        >>>     with sync_grad_when(step == accum_steps - 1):
        >>>         loss = ...
        >>>         loss.backward()
        >>> optimizer.step()
        >>> optimizer.zero_grad()

    Args:
        cond (bool): whether to synchronize gradients.
    """
    return _runtime_flags(skip_reducer=not cond)
