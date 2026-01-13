#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import builtins
import importlib
from contextlib import contextmanager
from functools import wraps, cache
from typing import (
    Generator, Optional, Tuple, Callable, Dict, List, Set, Any,
    Iterable, Type, TypedDict, Union, Protocol, ClassVar, cast, TypeVar
)
import logging
from pathlib import Path
import sys
from collections import defaultdict
from dataclasses import dataclass, field
import inspect
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
import itertools
import ast

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


def set_member_by_name(model: Any, name: str, value: Any) -> None:
    """
    Set the member of the model by its full name.
    """
    if not name:
        raise ValueError("Name cannot be empty")
    class _ValueHolder:
        """
        A value holder.
        In python you can't call `setattr` on object, but you can call it on its subclasses.
        """
        pass
    sliced_names = name.split(".")
    model_attr = model
    for sliced_name in sliced_names[:-1]:
        if not hasattr(model_attr, sliced_name):
            setattr(model_attr, sliced_name, _ValueHolder())
        model_attr = getattr(model_attr, sliced_name)
    setattr(model_attr, sliced_names[-1], value)


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
TRANSFORM_SUPPORTED_COLLECTION_TYPES = (tuple, list, dict, set, slice, _DICT_ITEMS_TYPE, _DICT_KEYS_TYPE, _DICT_VALUES_TYPE)


def _transform_recursively(data: Any, fn: Callable[[Any], Any],
    target_types: Union[Callable[[Any], bool], Type, Tuple[Type, ...]],
    collection_types = (tuple, list, dict), skip_dict_keys = True
) -> tuple[bool, Any]:
    if collection_types is None:
        collection_types = TRANSFORM_SUPPORTED_COLLECTION_TYPES
    if isinstance(data, collection_types):
        if isinstance(data, tuple):
            result = tuple(_transform_recursively(t, fn, target_types, collection_types, skip_dict_keys) for t in data)
            changed = any(c for c, _ in result)
            if changed:
                return True, tuple(v for _, v in result)
            else:
                return False, data
        if isinstance(data, list):
            result = [_transform_recursively(t, fn, target_types, collection_types, skip_dict_keys) for t in data]
            changed = any(c for c, _ in result)
            if changed:
                return True, [v for _, v in result]
            else:
                return False, data
        if isinstance(data, set):
            result = [_transform_recursively(t, fn, target_types, collection_types, skip_dict_keys) for t in data]
            changed = any(c for c, _ in result)
            if changed:
                return True, {v for _, v in result}
            else:
                return False, data
        if isinstance(data, dict):
            if skip_dict_keys:
                keys = {k: (False, k) for k in data.keys()}
            else:
                keys = {
                    k: _transform_recursively(k, fn, target_types, collection_types, skip_dict_keys)
                    for k in data.keys()
                }
            changed = any(c for c, _ in keys.values())
            result = {
                k: _transform_recursively(v, fn, target_types, collection_types, skip_dict_keys)
                for k, v in data.items()
            }
            changed = changed or any(c for c, _ in result.values())
            if changed:
                return True, {
                    keys[k][1]: v for k, (_, v) in result.items()
                }
            else:
                return False, data
        if isinstance(data, _DICT_ITEMS_TYPE):
            if skip_dict_keys:
                keys = {k: (False, k) for k, _ in data}
            else:
                keys = {
                    k: _transform_recursively(k, fn, target_types, collection_types, skip_dict_keys)
                    for k, _ in data
                }

            changed = any(c for c, _ in keys.values())
            result = {
                k: _transform_recursively(v, fn, target_types, collection_types, skip_dict_keys)
                for k, v in data
            }
            changed = changed or any(c for c, _ in result.values())
            if changed:
                return True, {
                    keys[k][1]: v for k, (_, v) in result.items()
                }.items()
            else:
                return False, data
        if isinstance(data, _DICT_KEYS_TYPE):
            result = [
                _transform_recursively(k, fn, target_types, collection_types, skip_dict_keys)
                for k in data
            ]
            changed = any(c for c, _ in result)
            if changed:
                return True, {
                    v: i for i, (_, v) in enumerate(result)
                }.keys()
            else:
                return False, data
        if isinstance(data, _DICT_VALUES_TYPE):
            result = {
                i: _transform_recursively(v, fn, target_types, collection_types, skip_dict_keys)
                for i, v in enumerate(data)
            }
            changed = any(c for c, _ in result.values())
            if changed:
                return True, {
                    i: v for i, (_, v) in result.items()
                }.values()
            else:
                return False, data
        if isinstance(data, slice):
            result = (
                _transform_recursively(data.start, fn, target_types, collection_types, skip_dict_keys),
                _transform_recursively(data.stop, fn, target_types, collection_types, skip_dict_keys),
                _transform_recursively(data.step, fn, target_types, collection_types, skip_dict_keys),
            )
            if any(c for c, _ in result):
                return True, slice(
                    result[0][1],
                    result[1][1],
                    result[2][1]
                )
            else:
                return False, data
        raise ValueError(f"Unsupported collection type: {type(data)}")
    elif isinstance(target_types, (tuple, list)) or inspect.isclass(target_types):
        if isinstance(data, target_types):
            return True, fn(data)
    elif callable(target_types):  # not a class, but callable. treat as a check function.
        if target_types(data):
            return True, fn(data)
    return False, data


def transform_recursively(data: Any, fn: Callable[[Any], Any],
    target_types: Union[Callable[[Any], bool], Type, Tuple[Type, ...]],
    collection_types = (tuple, list, dict), skip_dict_keys = True
) -> Any:
    """
    Transform the data with the given function, will recursively apply the function to the nested data.
    Currently supported collection types is SUPPORTED_COLLECTION_TYPES.
    Args:
        data: the data to be transformed.
        fn: the function to apply.
        target_types: the target types to apply the function.
        collection_types: the collection types to apply the function to the nested data.
            Will handle all supported types if None.
        skip_dict_keys: whether to skip the dict keys (for types dict, _DICT_ITEMS_TYPE).
            _DICT_KEYS_TYPE is not skipped, if you want to skip it, just remove it from the collection_types.
    """
    _, result = _transform_recursively(data, fn, target_types, collection_types, skip_dict_keys)
    return result


def check_recursively(data, fn: Callable[[Any], bool],
    collection_types = (tuple, list, dict),
    skip_dict_keys = True
) -> bool:
    """
    Check the data with the given function, will recursively apply the function to the nested data.
    Args:
        data: the data to be checked.
        fn: the function to check.
        collection_types: the collection types to apply the function to the nested data.
        skip_dict_keys: whether to skip the dict keys (for types dict, _DICT_ITEMS_TYPE).
            _DICT_KEYS_TYPE is not skipped, if you want to skip it, just remove it from the collection_types.

    """
    if collection_types is None:
        collection_types = TRANSFORM_SUPPORTED_COLLECTION_TYPES

    if isinstance(data, collection_types):
        if isinstance(data, (list, tuple, set, _DICT_KEYS_TYPE, _DICT_VALUES_TYPE)):
            return any(check_recursively(t, fn, collection_types) for t in data)
        if isinstance(data, dict):
            if skip_dict_keys:
                return any(
                    check_recursively(v, fn, collection_types)
                    for v in data.values()
                )
            else:
                return any(
                    check_recursively(k, fn, collection_types) or check_recursively(v, fn, collection_types)
                    for k, v in data.items()
                )
        if isinstance(data, _DICT_ITEMS_TYPE):
            if skip_dict_keys:
                return any(
                    check_recursively(v, fn, collection_types)
                    for _, v in data
                )
            else:
                return any(
                    check_recursively(k, fn, collection_types) or check_recursively(v, fn, collection_types)
                    for k, v in data
                )
        if isinstance(data, slice):
            return any((
                check_recursively(data.start, fn, collection_types),
                check_recursively(data.stop, fn, collection_types),
                check_recursively(data.step, fn, collection_types)
            ))
        raise ValueError(f"Unsupported collection type: {type(data)}")

    return fn(data)


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


class _UncheckedFields:
    def __getattr__(self, item: str) -> Any:
        return item


TUncheckedClass = TypeVar("TAnyClass")
def unchecked_fields(_: TUncheckedClass, /) -> TUncheckedClass:
    """
    This function is used to get the field names(in str) of any object without checking
    This is a workaround for the lack of `__name__` of member.
    """
    return cast(TUncheckedClass, _UncheckedFields())


@cache
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


class AdamOptState(TypedDict):
    step: torch.Tensor
    exp_avg: torch.Tensor
    exp_avg_sq: torch.Tensor


class OptStateParamGroup(TypedDict):
    params: list[int]
    lr: int


class OptStateDict(TypedDict):
    state: dict[int, AdamOptState | dict[str, Any]]
    param_groups: list[OptStateParamGroup | dict[str, Any]]


def fn_field(**kwargs):
    metadata = kwargs.pop('metadata', {})
    metadata['deserialize'] = lambda t: None if t is None else load_type(t)
    return field(**kwargs, metadata=metadata)


TENSOR_DYNAMIC_DIMS_FIELD_NAME = '_nnscaler_dynamic_dims'
# for nnscaler custom class (TensorMetadata)
NNSCALER_DYNAMIC_DIMS_NAME = 'dynamic_dims'


def mark_dynamic(tensor: torch.Tensor, dims: int | list[int] | tuple[int]) -> torch.Tensor:
    """
    Mark the dim of a tensor as dynamic, which means it can be changed in the future.
    This is the same with `torch._dynamo.mark_dynamic`
    """
    dims = [dims] if isinstance(dims, int) else dims
    setattr(tensor, TENSOR_DYNAMIC_DIMS_FIELD_NAME, set(dims) if dims else set())
    return tensor


def copy_dynamic(src: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
    """
    Copy the dynamic dims from src to tensor, and return the tensor.
    """
    if hasattr(src, TENSOR_DYNAMIC_DIMS_FIELD_NAME):
        setattr(tensor, TENSOR_DYNAMIC_DIMS_FIELD_NAME, getattr(src, TENSOR_DYNAMIC_DIMS_FIELD_NAME))
    return tensor


def get_dynamic(tensor: Any) -> set[int]:
    """
    Get the dynamic dims of a tensor.
    It also works when tensor is not an instance of torch.Tensor
    """
    if isinstance(tensor, torch.Tensor):
        return getattr(tensor, TENSOR_DYNAMIC_DIMS_FIELD_NAME, set())
    else:
        return getattr(tensor, NNSCALER_DYNAMIC_DIMS_NAME, set())


@contextmanager
def suppress_warnings(message):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message=message)
        yield


def broadcast_files(
    file_groups: List[List[Union[str, Path]]],
    *,
    max_workers: int = 8,
):
    """Broadcast files from src to all other nodes. Files are grouped into file_groups,
    and each group of files are broadcasted together to get better performance.

    Args:
        files (List[List[str | Path]]): List of file groups to be broadcasted.
            Note that the file names should be the same across all ranks.
    """
    from nnscaler.runtime.device import DeviceGroup

    # filter out empty file groups
    file_groups = [
        fg for fg in file_groups if fg
    ]

    curr_rank = torch.distributed.get_rank()
    local_world_size = DeviceGroup().local_world_size
    world_size = torch.distributed.get_world_size()
    local_rank = curr_rank % local_world_size

    # create groups, make sure all groups are created correctly
    for i in range(local_world_size):
        group_ranks = list(range(i, world_size, local_world_size))
        DeviceGroup().get_group(group_ranks)

    # collect file sizes and broadcast
    if curr_rank == 0:
        file_group_sizes: List[List[int]] = [
            [os.path.getsize(file) for file in files] for files in file_groups
        ]
        exchange_objects = [file_group_sizes]
    else:
        exchange_objects = [None]

    torch.distributed.broadcast_object_list(exchange_objects, src=0)
    file_group_sizes = exchange_objects[0]

    # sort file_groups by size descending to improve overlapping
    file_groups_sizes_pairs = list(zip(file_groups, file_group_sizes))
    file_groups_sizes_pairs.sort(key=lambda x: sum(x[1]), reverse=True)
    file_groups = [pair[0] for pair in file_groups_sizes_pairs]
    file_group_sizes = [pair[1] for pair in file_groups_sizes_pairs]

    def _write_file(file: Union[str, Path], buffer, start, size):
        _logger.info(f'Rank {curr_rank}: Writing file {file} of size {size} bytes.')
        with open(file, 'wb') as f:
            f.write(buffer[start: start + size].numpy().tobytes())

    # this warning is caused by torch.frombuffer when the buffer is not writable
    # we can ignore it here
    @suppress_warnings('.*The given buffer is not writable.*')
    def _read_file(file, buffer, start, size):
        _logger.info(f'Rank {curr_rank}: Reading file {file} of size {size} bytes.')
        with open(file, 'rb') as f:
            buffer[start: start + size] = torch.frombuffer(f.read(), dtype=torch.uint8)

    def _write_files(buffer, files, file_sizes):
        buffer = buffer.cpu()
        file_starts = itertools.accumulate([0] + file_sizes[:-1])
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(
                lambda args: _write_file(args[0], buffer, args[1], args[2]),
                zip(files, file_starts, file_sizes)
            )

    def _send_file_group(src, files, file_sizes):
        total_size = sum(file_sizes)

        ranks = list(range(src, world_size, local_world_size))
        group = DeviceGroup().get_group(ranks)

        if curr_rank < local_world_size:
            file_starts = itertools.accumulate([0] + file_sizes[:-1])
            file_buffer = torch.empty(total_size, dtype=torch.uint8, device='cpu')
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                executor.map(
                    lambda args: _read_file(args[0], file_buffer, args[1], args[2]),
                    zip(files, file_starts, file_sizes)
                )
            file_buffer = file_buffer.cuda()
        else:
            file_buffer = torch.empty(total_size, dtype=torch.uint8, device='cuda')

        torch.distributed.broadcast(file_buffer, src=src, group=group)

        if curr_rank >= local_world_size:
            _write_files(file_buffer, files, file_sizes)

    for i in range(local_rank, len(file_groups), local_world_size):
        _send_file_group(local_rank, file_groups[i], file_group_sizes[i])
