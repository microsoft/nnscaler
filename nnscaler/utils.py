#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import builtins
import importlib
from contextlib import contextmanager
from functools import wraps
from typing import (
    Generator, Optional, Tuple, Callable, Dict, List, Set, Any,
    Iterable, Type, Union, Protocol, ClassVar, cast, TypeVar
)
import logging
from pathlib import Path
import sys
from collections import defaultdict
from dataclasses import dataclass
import inspect
import os

import nnscaler
from nnscaler.flags import RuntimeFlag, CompileFlag

import torch

_logger = logging.getLogger(__name__)


def print_each_rank(msg: str, rank_only: Optional[int] = None, logger: Optional[logging.Logger] = None):
    """Logging the message.

    Args:
        msg (str): message to be logged.
        rank_only (int, optional):
            the rank to be logged. Defaults to None, which means all ranks.
        logger (logging.Logger, optional):
            the logger to use. Defaults to print.

    Returns:
        None
    """
    logger_fn = print if logger is None else logger.info
    if CompileFlag.dev_mode:
        logger_fn(msg)
        return

    myrank = torch.distributed.get_rank()
    for rank in range(torch.distributed.get_world_size()):
        if rank_only is None:
            if myrank == rank:
                logger_fn('rank [{}]: {}'.format(rank, msg))
        else:
            if myrank == rank_only and rank_only == rank:
                logger_fn('rank [{}]: {}'.format(rank, msg))
        torch.distributed.barrier()


def _load_module_attr(filename: str, name: str):
    # TODO: use `importlib.import_module` instead
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[name] = module  # so you can find the loaded module in sys.modules
    return module


def load_model(filename: Optional[str] = None, load_content: bool = True, fullmodel_filename: Optional[str] = None):
    filename = f'gencode{nnscaler.runtime.device.DeviceGroup().rank}.py' if filename is None else filename
    module = _load_module_attr(filename, Path(filename).stem)
    loaded_module: nnscaler.runtime.module.CubeModule = module.GenModel().cuda()
    non_persistent_buffers = loaded_module.get_non_persistent_buffers()
    if non_persistent_buffers:
        names = [name for name, _ in non_persistent_buffers.items()]
        _logger.warning(f'Detected non-persistent buffers: {names}, will load content, make sure fullmodel.pt.* are available and consistent.')
        if not load_content:
            load_content = True
    # load parameter content
    if load_content:
        _logger.info("loading parameter content...")
        if not fullmodel_filename:
            fullmodel_filename = str(Path(filename).with_name('fullmodel.pt'))
        loaded_module.load_attr_content(fullmodel_filename)
    # initialize reducer
    for reducer in loaded_module.reducers:
        reducer.build_buckets()
    return loaded_module


def load_default_schedule(filename: Optional[str] = None):
    filename = f'gencode{nnscaler.runtime.device.DeviceGroup().rank}.py' if filename is None else filename
    module = _load_module_attr(filename, Path(filename).stem)
    return module._train_step


def load_eval_schedule(filename: Optional[str] = None):
    filename = f'gencode{nnscaler.runtime.device.DeviceGroup().rank}.py' if filename is None else filename
    module = _load_module_attr(filename, Path(filename).stem)
    return module._infer_step


def get_member_by_name(model: torch.nn.Module, name: str) -> Any:
    """
    Get the member of the model by its full name.
    if name is empty, return the model itself.
    """
    if not name:
        return model
    sliced_names = name.split(".")
    model_attr = model
    for sliced_name in sliced_names:
        model_attr = getattr(model_attr, sliced_name)
    return model_attr


def get_shared_params(model: torch.nn.Module) -> List[List[str]]:
    paramid2name = defaultdict(set)
    for name in model.state_dict().keys():
        param = get_member_by_name(model, name)
        paramid = id(param)
        paramid2name[paramid].add(name)
    return [list(names) for _, names in paramid2name.items() if len(names) > 1]


@dataclass
class BroadcastGroup:
    src_rank: int      # the source rank in the group which the current rank belongs to
    ranks: List[int]   # the ranks in the group which the current rank belongs to
    group: torch.distributed.ProcessGroup


def setup_stride_broadcast_group(stride_size: int) -> BroadcastGroup:
    """
    Setup the broadcast group for the given stride size.

    For example, assume stride size is 4, then
    we will create 4 broadcasting groups:
        [0, 4, 8, ...],
        [1, 5, 9, ...],
        [2, 6, 10, ...],
        [3, 7, 11, ...]
    the broadcast will happen in above groups, the sending rank is the first rank in the group.

    Args:
        stride_size (int): the stride size.
    Returns:
        BroadcastGroup: the source rank and the broadcast group.
    """
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    for i in range(stride_size):
        ranks = list(range(i, world_size, stride_size))
        nnscaler.runtime.device.DeviceGroup().get_group(ranks)

    curr_parallel_group_ranks = list(range(rank % stride_size, world_size, stride_size))
    curr_parallel_group = nnscaler.runtime.device.DeviceGroup().get_group(curr_parallel_group_ranks)
    src_rank = min(curr_parallel_group_ranks)

    return BroadcastGroup(
        src_rank=src_rank,
        ranks=curr_parallel_group_ranks,
        group=curr_parallel_group
    )


def set_default_logger_level(level):
    """Set the logger level with predefined logging format.

    Args:
        level (int): the level of the logger.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


@contextmanager
def enforce_zero_num_worker(cls) -> Generator[None, None, None]:
    """Context manager to enforce the number of workers to be 0 in DataLoader."""
    _old__init__ = cls.__init__
    def _new__init__(self, *args, **kwargs) -> None:
        kwargs['num_workers'] = 0
        kwargs['prefetch_factor'] = None
        kwargs['persistent_workers'] = False
        _old__init__(self, *args, **kwargs)
    cls.__init__ = _new__init__
    yield
    cls.__init__ = _old__init__


def rank_zero_only(fn: Callable[..., None]) -> Callable[..., None]:
    """
    Wrap a function to call internal function only in rank zero.
    Function that can be used as a decorator to enable a function/method being called only on global rank 0.
    Please note
    1. that the fn should be no return values, and no side effect.
    So it is only recommend to use this decorator for logging or printing.
    2. `fn` will also be called if the distributed environment is not initialized.
    """

    @wraps(fn)
    def wrapped_fn(*args, **kwargs):
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else None
        if rank == 0 or rank is None:
            fn(*args, **kwargs)

    return wrapped_fn


_DICT_ITEMS_TYPE = type({}.items())
_DICT_KEYS_TYPE = type({}.keys())
_DICT_VALUES_TYPE = type({}.values())


def transform_recursively(data: Any, fn: Callable[[Any], Any],
    target_types: Union[Callable[[Any], bool], Type, Tuple[Type, ...]],
    collection_types = (tuple, list, dict), skip_dict_keys = True
) -> Any:
    """
    Transform the data with the given function, will recursively apply the function to the nested data.
    Args:
        data: the data to be transformed.
        fn: the function to apply.
        target_types: the target types to apply the function.
        collection_types: the collection types to apply the function to the nested data.
        skip_dict_keys: whether to skip the dict keys (for types dict, _DICT_ITEMS_TYPE).
            _DICT_KEYS_TYPE is not skipped, if you want to skip it, just remove it from the collection_types.
    """
    if isinstance(data, collection_types):
        if isinstance(data, tuple):
            return tuple(transform_recursively(t, fn, target_types, collection_types) for t in data)
        if isinstance(data, list):
            return list(transform_recursively(t, fn, target_types, collection_types) for t in data)
        if isinstance(data, set):
            return set(transform_recursively(t, fn, target_types, collection_types) for t in data)
        if isinstance(data, dict):
            return {
                k if skip_dict_keys else transform_recursively(k, fn, target_types, collection_types):
                transform_recursively(v, fn, target_types, collection_types)
                for k, v in data.items()
        }
        if isinstance(data, _DICT_ITEMS_TYPE):
            return {
                k if skip_dict_keys else transform_recursively(k, fn, target_types, collection_types):
                transform_recursively(v, fn, target_types, collection_types)
                for k, v in data
            }.items()
        if isinstance(data, _DICT_KEYS_TYPE):
            return {
                    transform_recursively(k, fn, target_types, collection_types): i
                    for i, k in enumerate(data)
            }.keys()
        if isinstance(data, _DICT_VALUES_TYPE):
            return {
                i: transform_recursively(v, fn, target_types, collection_types)
                for i, v in enumerate(data)
            }.values()
        if isinstance(data, slice):
            return slice(
                transform_recursively(data.start, fn, target_types, collection_types),
                transform_recursively(data.stop, fn, target_types, collection_types),
                transform_recursively(data.step, fn, target_types, collection_types)
            )
        raise ValueError(f"Unsupported collection type: {type(data)}")
    elif isinstance(target_types, (tuple, list)) or inspect.isclass(target_types):
        if isinstance(data, target_types):
            return fn(data)
    elif callable(target_types):  # not a class, but callable. treat as a check function.
        if target_types(data):
            return fn(data)
    return data


def is_running_distributed() -> bool:
    """Check if the current process is running under torchrun."""
    # TORCHELASTIC_RUN_ID is more unique than 'RANK'/'WORLD_SIZE'
    # so we use it to determine if the process is running under torchrun.
    # TODO: Is there a better way?
    return 'TORCHELASTIC_RUN_ID' in os.environ


def select_many(data: Iterable[Any], fn: Callable[[Any], Iterable[Any]]) -> Iterable[Any]:
    """Select many elements from the iterable with the given function."""
    for item in data:
        yield from fn(item)


# ref: https://stackoverflow.com/questions/128573/using-property-on-classmethods
class classproperty(property):
    """
    A simple class property decorator.
    """
    def __get__(self, obj, objtype=None):
        # obj will be None when accessed from the class like `MyClass.my_property`
        return super(classproperty, self).__get__(objtype)
    # This hack doesn't work for __set__ and __delete__.
    # so here __set__ and __delete__ are not implemented, and the property is read-only


# ref: https://stackoverflow.com/questions/54668000/type-hint-for-an-instance-of-a-non-specific-dataclass
class IsDataclass(Protocol):
    # as already noted in comments, checking for this attribute is currently
    # the most reliable way to ascertain that something is a dataclass
    __dataclass_fields__: ClassVar[Dict[str, Any]]


# ref: https://github.com/pydantic/pydantic/discussions/8600
@dataclass(frozen=True)
class _GetFields:
    _dataclass_type: Type[IsDataclass]

    def __getattr__(self, item: str) -> Any:
        if item in self._dataclass_type.__dataclass_fields__:
            return item
        raise AttributeError(f'"{item}" is not a valid field in type: {self._dataclass_type}')


TDataClass = TypeVar("TDataClass", bound=Type[IsDataclass])
def fields(model: TDataClass, /) -> TDataClass:
    """
    This function is used to get the field names(in str) of a dataclass.
    This is a workaround for the lack of `__name__` of dataclass field.
    """
    return cast(TDataClass, _GetFields(model))


def load_type(type_name: str):
    """
    Load function/class from its full qualified name
    """
    if callable(type_name):  # a function or class
        return type_name

    parts = type_name.split('.')

    last_ex = None
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
            except (ImportError, ModuleNotFoundError) as e:
                last_ex = e

    try:
        for i in range(s, len(parts)):
            nm = getattr(nm, parts[i])
        return nm
    except AttributeError as e:
        # give a hint of the import error
        # TODO: a better way?
        e.__context__ = last_ex
        raise RuntimeError(f"Failed to load type {type_name}") from e


class accum_mode:
    """Make cube execution in gradient accumulation mode.

    This is only required when `ASYNC_REDUCER=1`.

    A typical usage is:

    ```
    for _ in range(num_iters):
        for step in range(accum_steps):
            datas = next(dataloader)
            with nnscaler.accum_mode(begin=(step == 0), end=(step == accum_steps - 1)):
                train_iter(model, *datas)
        optimizer.step()
        optimizer.zero_grad()
    ```

    Or,

    ```
    for _ in range(num_iters):
        for step in nnscaler.accum_mode.steps(accum_steps):
            datas = next(dataloader)
            train_iter(model, *datas)
        optimizer.step()
        optimizer.zero_grad()
    ```
    """
    def __init__(self, begin: bool = True, end: bool = True):
        """Turn on/off accumulation mode.

        Args:
            begin (bool): Whether the iteration is the first accumulation step.
                If True, the `model.zero_grad()` will be enabled to zero out gradients
                of the parameters in the reducer.
            end (bool): Whether the iteration is the last accumulation step.
                If True, the `model.reduce_grad()` will be enabled to reduce gradients at
                the end of the iteration.
        """
        self.begin: bool = begin
        self.end: bool = end
        self.old: Tuple[bool, bool] = None

    def __enter__(self):
        """Enter the accumulation mode.

        Example usage:

        ```
        for _ in range(num_iters):
            for step in range(accum_steps):
                datas = next(dataloader)
                with nnscaler.accum_mode(begin=(step == 0), end=(step == accum_steps - 1)):
                    train_iter(model, *datas)
            optimizer.step()
            optimizer.zero_grad()
        ```

        """
        self.old = (RuntimeFlag.skip_zero_grad, RuntimeFlag.skip_reducer)
        RuntimeFlag.skip_zero_grad = (not self.begin)
        RuntimeFlag.skip_reducer = (not self.end)

    def __exit__(self, *args):
        RuntimeFlag.skip_zero_grad, RuntimeFlag.skip_reducer = self.old
        self.old = None

    @staticmethod
    def steps(nsteps: int):
        """Perform the accumulation in `nsteps` steps.

        This interface doesn't require to set the `begin` and `end` flags
        during the initilization of `accum_mode`.

        Example usage:

        ```
        for _ in range(num_iters):
            for step in nnscaler.accum_mode.steps(accum_steps):
                datas = next(dataloader)
                train_iter(model, *datas)
            optimizer.step()
            optimizer.zero_grad()
        ```

        Args:
            nsteps (int): The number of accumulation steps.

        Yield:
            int: The current step index.
        """
        old = (RuntimeFlag.skip_zero_grad, RuntimeFlag.skip_reducer)
        for step in range(nsteps):
            RuntimeFlag.skip_zero_grad = (not (step == 0))
            RuntimeFlag.skip_reducer = (not (step == nsteps - 1))
            yield step
        RuntimeFlag.skip_zero_grad, RuntimeFlag.skip_reducer = old
