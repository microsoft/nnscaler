#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from pathlib import Path
import argparse
import logging
from .descs import MeshDesc
from .util import get_default_profile_path

_logger = logging.getLogger(__name__)


def _validate_file_path(path: str):
    if not Path(path).exists():
        raise ValueError(f'file path {path} does not exist')


def _validate_dir_path(path: str):
    if not Path(path).is_dir():
        raise ValueError(f'path {path} is not a directory')


class AutoDistConfig:
    r"""
    AutoDistConfig is the configuration for AutoDist. It contains the following fields:

    - task_name (`str`, *optional*, defaults to `'default'`):
        The name of the current task to distinguish runs.
    - consider_mem (`bool`, *optional*, defaults to `True`):
        Whether to consider memory when searching plans.
    - opt_resident_coef (`int`, *optional*, defaults to `2`):
        The coefficient of the optimizer resident state compare with the model weight size.
        For example: training a fp32 model with adam optimizer, movement1 and movement2 will be saved in the optimizer state,
        movement1 and movement2 are fp32 and have the same size with model weight,
        so the opt_resident_coef is (1 + 1) = 2.
        Common cases:
        - fp32 training w/ adam: (1 + 1) (fp32 movement1 + fp32 movement2)
        - fp16 & bf16 training w/ adam: (2 + 2 + 2) (fp32 movement1 + fp32 movement2 + fp32 weight)
        - fp16 & bf16 training w/ memory efficient adam: (2 + 2) (fp32 movement1 + fp32 movement2)
    - opt_transient_coef (`int`, *optional*, defaults to `0`):
        The coefficient of the optimizer transient state compare with the model weight size.
        For example: training a fp16 model with adam optimizer, fp16 gradient will transient convert to fp32,
        so the opt_transient_coef is 2.
        Common cases:
        - fp32 training w/ adam: 0
        - fp16 & bf16 training w/ adam w/o inkernal cast: (2) (fp32 gradient)
        - fp16 & bf16 training w/ memory efficient adam w/o inkernal cast: (2 + 2) (fp32 weight + fp32 gradient)
    - partition_constraints_path (`str`, *optional*, defaults to `''`):
        The path to the partition constraints file. Details can be found in docs/solver_interface/partition_constraints.md
    - profile_dir (`str`, *optional*, defaults to `~/.cache/nnscaler/autodist/1.0/get_node_arch()`):
        The directory to store the profiling results.
    - load_plan_path (`str`, *optional*, defaults to `''`):
        The path to the plan file to load. If specified, the plan will be loaded from the file instead of searching.
    - save_plan_path (`str`, *optional*, defaults to `'./{task_name}.json'`):
        The path to the plan file to save.
    - topk (`int`, *optional*, defaults to `20`):
        The number of plans to generate for robustness.
    - zero_stage (`int`, *optional*, defaults to `0`):
        The zero stage, see https://arxiv.org/abs/1910.02054 for details. Currently only support zero stage 0 and 1.
    - zero_ngroups (`int`, *optional*, defaults to `1`):
        The number of zero groups to balance memory usage and communication cost. The larger the number,
        more memory will be used and less communication cost will be incurred.
    - is_train (`bool`, *optional*, defaults to `True`):
        Whether the model is for training or inference.
    - mesh_row (`int`, *optional*, defaults to `1`):
        The number of available nodes.
    - mesh_col (`int`, *optional*, defaults to `1`):
        The number of available devices in each node.
    - recompute_modules (`str`, *optional*, defaults to `''`):
        The module names to recompute, separated by `,`. For example, `module1,module2`.
        Module name can be any suffix of the full module name, e.g., `module1` will match `x.module1`, `y.module1`,
        `x.module1` will match `x.module1` but not `y.module1`.
    - memory_constraint (`float`, *optional*, defaults to `32`):
        The memory constraint in each device in GB.
    - memory_granularity (`int`, *optional*, defaults to `1`):
        The memory granularity in Byte.
    - micro_batch_size (`int`, *optional*, defaults to `1`):
        The micro batch size.
    - update_freq (`int`, *optional*, defaults to `1`):
        The update frequency (micro batch size x update freq = real batch size).
    - world_size (`int`, *optional*, defaults to `1`):
        The total number of devices. (mesh_row x mesh_col x scale_factor = world_size)
    - nproc (`int`, *optional*, defaults to `1`):
        The number of processes in pipeline parallelism search.
    - ignore_small_tensor_threshold (`int`, *optional*, defaults to `1`):
        The tensor size threshold to ignore.
    - verbose (`bool`, *optional*, defaults to `False`):
        Whether to print verbose information.
    - re_profile (`bool`, *optional*, defaults to `False`):
        If set to `True`, the computation profiling results will be overridden.
    - pipeline_pivots (`str`, *optional*, defaults to `''`):
        The module names to pivot the pipeline, separated by `,`. For example, if `module1,module2`
        is specified, stages searched by pipeline solver only start from either `module1` or `module2`.
    - pipeline_nstages(`int | Literal['auto']`, *optional*, defaults to `'auto'`):
        When `pipeline_pivots` is not empty, this specify the number of stages in pipeline parallelism. `1` means not to use pipelines.
    - pipeline_scheduler (`str`, *optional*, defaults to `'1f1b'`):
        The pipeline scheduler to use. Currently only support `'1f1b'`.
    - max_pipeline_bubble_ratio (`float`, *optional*, defaults to `0.2`):
        The maximum bubble ratio in pipeline parallelism. The higher the ratio, the more bubbles will be allowed,
        the larger search space will be explored.
    - max_pipeline_unbalance_ratio (`float`, *optional*, defaults to `0.5`):
        The maximum unbalance ratio in pipeline parallelism. This is a metric control min_pipeline_stage_time / max_pipeline_stage_time.
        The higher the ratio, the more balance is required, the smaller search space will be explored.
    - solver (`str`, *optional*, defaults to `'dp'`):
        The solver to use in spmd parallelism. Currently only support
        `'dp'` (dynamic programming)
        `'ilp'` (integer linear programming).
    - parallel_profile (`bool`, *optional*, defaults to `True`):
        Whether to profile on multiple device in parallel. If set to `False`, the profiling will be done in a
        single device sequentially.
    - transient_mem_coef (`float`, *optional*, defaults to `2`):
        In autodist, a heuristic is used to estimate the transient memory size:
        `transient_mem_size = opt_transient_coef * (1st_largest_infer_mem + 2nd_largest_infer_mem)`. This formula
        is useful in many cases, but it may be too strict when some operators consume or generate a large tensor
        (>= 4GB). In this case, you can set `transient_mem_coef` to a smaller value to relax the constraint.
    """

    def __init__(self,
                 task_name='default',
                 consider_mem=True,
                 opt_resident_coef=2,
                 opt_transient_coef=0,
                 partition_constraints_path='',
                 profile_dir=get_default_profile_path(),
                 load_plan_path='',
                 save_plan_path='',
                 topk=20,
                 zero_stage=0,
                 zero_ngroups=1,
                 is_train=True,
                 mesh_row=1,
                 mesh_col=1,
                 recompute_modules='',
                 memory_constraint=32,
                 memory_granularity=1,
                 micro_batch_size=1,
                 update_freq=1,
                 world_size=1,
                 nproc=1,
                 ignore_small_tensor_threshold=1,
                 verbose=False,
                 re_profile=False,
                 pipeline_pivots='',
                 pipeline_nstages='auto',
                 pipeline_scheduler='1f1b',
                 max_pipeline_bubble_ratio=0.2,
                 max_pipeline_unbalance_ratio=0.5,
                 solver='dp',
                 parallel_profile=True,
                 transient_mem_coef=2,
                 **kwargs):
        self.pc_path = partition_constraints_path
        self.profile_dir = profile_dir
        self.topk = topk
        self.task_name = task_name
        self.load_plan_path = load_plan_path
        self.save_plan_path = save_plan_path

        self.consider_mem = consider_mem
        self.zero_stage = zero_stage
        self.zero_ngroups = zero_ngroups
        self.opt_resident_coef = opt_resident_coef
        self.opt_transient_coef = opt_transient_coef
        self.is_train = is_train
        self.mesh_desc = MeshDesc(mesh_row, mesh_col)
        self.recompute_modules = recompute_modules
        # from GB to Byte
        self.memory_constraint = int(memory_constraint * 1024 * 1024 * 1024)
        self.memory_granularity = memory_granularity
        self.micro_batch_size = micro_batch_size
        self.update_freq = update_freq
        self.world_size = world_size
        self.nproc = nproc

        self.ignore_small_tensor_threshold = ignore_small_tensor_threshold
        self.verbose = verbose
        self.re_profile = re_profile
        self.pipeline_pivots = pipeline_pivots
        self.pipeline_nstages = pipeline_nstages
        self.pipeline_scheduler = pipeline_scheduler
        if self.pipeline_scheduler != '1f1b':
            raise ValueError(f'pipeline scheduler {self.pipeline_scheduler} must be 1f1b')
        self.max_pipeline_bubble_ratio = max_pipeline_bubble_ratio
        self.max_pipeline_unbalance_ratio = max_pipeline_unbalance_ratio
        self.solver = solver
        if self.pipeline_enabled and solver != 'dp':
            _logger.warning(
                f'pipeline is enabled, but solver is not dp, set solver to dp'
            )
            self.solver = 'dp'
        self.parallel_profile = parallel_profile
        self.transient_mem_coef = transient_mem_coef

        ignored_keys = list(kwargs.keys())
        if ignored_keys:
            warning_msg = f'autodist config got unknown config keys: {ignored_keys}'
            _logger.warning(warning_msg)

        self._validate_config()

    def _validate_config(self):
        if self.pc_path:
            _validate_file_path(self.pc_path)

        if not Path(self.profile_dir).exists():
            _logger.info(f'create folder: {self.profile_dir}')
            Path(self.profile_dir).mkdir(parents=True, exist_ok=True)

        if self.pipeline_enabled:
            if self.max_pipeline_bubble_ratio <= 0 or self.max_pipeline_bubble_ratio >= 1:
                raise ValueError(
                    f'max pipeline bubble ratio {self.max_pipeline_bubble_ratio} must be in (0, 1)'
                )
            if self.max_pipeline_unbalance_ratio <= 0 or self.max_pipeline_unbalance_ratio >= 1:
                raise ValueError(
                    f'max pipeline unbalance ratio {self.max_pipeline_unbalance_ratio} must be in (0, 1)'
                )

        if self.topk < 1:
            raise ValueError(f'topk {self.topk} must be positive')

        if not self.task_name:
            raise RuntimeError('task name cannot be empty')

        if self.load_plan_path:
            _validate_file_path(self.load_plan_path)
            if self.save_plan_path:
                raise ValueError(
                    'cannot specify both load plan path and save plan path')

        if self.save_plan_path:
            Path(self.save_plan_path).parent.mkdir(parents=True, exist_ok=True)

        if self.zero_stage not in [0, 1]:
            raise ValueError(f'zero stage {self.zero_stage} must be 0 or 1')
        else:
            if self.zero_stage == 1:
                if self.world_size % self.zero_ngroups != 0:
                    raise ValueError(
                        f'world size {self.world_size} must be divisible by zero num groups {self.zero_ngroups}'
                    )
                scale_factor = self.world_size // self.mesh_desc.ngpus
                if scale_factor % self.zero_ngroups != 0:
                    raise ValueError(
                        f'world size {self.world_size} must be divisible by zero num groups {self.zero_ngroups}'
                    )

        if not self.solver in [
                'dp',
                'ilp',
        ]:
            raise ValueError(f'solver {self.solver} must be dp or ilp')

    def __repr__(self):
        return f'{self.__class__.__name__} {self.__dict__}'

    @property
    def ngpus(self):
        return self.mesh_desc.ngpus

    @property
    def pipeline_enabled(self) -> bool:
        # whether to explore pipeline
        # "auto" is considered as enabled although the exploration result might be not to use pipeline
        if not self.pipeline_pivots:
            return False
        return self.pipeline_nstages == 'auto' or self.pipeline_nstages > 1
