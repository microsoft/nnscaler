#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import os
import re
import sys
from typing import Optional, Tuple, Type, Union, Pattern
from contextlib import contextmanager
from typing import Callable
import functools
import math
import random
from datetime import timedelta
from pathlib import Path
import shutil

import numpy as np

import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d

from nnscaler.parallel import ComputeConfig
import nnscaler
from nnscaler.runtime.module import ParallelModule
from nnscaler.runtime.device import DeviceGroup, CompileFlag


def init_parameter(model: torch.nn.Module, seed: int = 0):
    """
    Initialize a model's parameters with truncated normal distribution.
    """
    def trunc_normal_(tensor: torch.Tensor, mean: float = 0., std: float = 1., a: float = -2., b: float = 2.):
        with torch.no_grad():
            l = (1. + math.erf((a - mean) / std / math.sqrt(2.))) / 2.
            u = (1. + math.erf((b - mean) / std / math.sqrt(2.))) / 2.
            tensor.uniform_(2 * l - 1, 2 * u - 1)
            tensor.erfinv_()
            tensor.mul_(std * math.sqrt(2.))
            tensor.add_(mean)
            tensor.clamp_(min=a, max=b)
        return tensor

    torch.random.manual_seed(seed)
    random.seed(seed)

    for param in list(model.parameters()) + list(model.buffers()):
        if len(param.size()) > 1:
            trunc_normal_(param, std=.02)
        else:
            torch.nn.init.constant_(param, 0)


def init_random():
    np.random.seed(1)
    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1)


def assert_parity(baseline_fn: Callable, compile_fn: Callable, atol: float=1e-4) -> bool:
    """Compare the output of baseline_fn and compile_fn

    Error will raise if the output of two functions are not the same.

    Args:
        baseline_fn (Callable): a function that returns the output of baseline
        compile_fn (Callable): a function that returns the output of compile (cube)
        atol (Callable): absolute tolerance when comparing two torch tensors

    Returns:
        result (bool): True if the output of two functions are the same else raise Error
    """
    baseline_outputs = baseline_fn()
    compile_outputs = compile_fn()

    print(f'comparing\nGT:\t{baseline_outputs}\nOUT:\t{compile_outputs}')

    def assert_same_complex(gt, out):
        if isinstance(gt, tuple):
            assert isinstance(out, tuple)
            for ele_gt, ele_out in zip(gt, out):
                assert_same_complex(ele_gt, ele_out)
        elif isinstance(gt, list):
            assert isinstance(out, list)
            for ele_gt, ele_out in zip(gt, out):
                assert_same_complex(ele_gt, ele_out)
        elif isinstance(gt, dict):
            assert isinstance(out, dict)
            assert set(gt.keys()) == set(out.keys())
            for key in gt:
                assert_same_complex(gt[key], out[key])
        elif isinstance(gt, torch.Tensor):
            assert isinstance(out, torch.Tensor)
            assert torch.allclose(gt, out, atol=atol), f'mismatched: {gt} != {out}'
        elif isinstance(gt, float):
            assert isinstance(out, float)
            assert math.isclose(gt, out, abs_tol=atol), f'mismatched: {gt} != {out}'
        else:
            assert gt == out, f'mismatched: {gt} != {out}'
    assert_same_complex(baseline_outputs, compile_outputs)
    return None


@contextmanager
def replace_all_device_with(device='cpu', force=False):
    if not force and torch.cuda.is_available():
        # do not replace device if cuda is available
        yield
        return

    from nnscaler.graph.tracer import wrap_utils

    orig_to = torch.Tensor.to
    orig_cuda = torch.Tensor.cuda
    orig_cpu = torch.Tensor.cpu

    def patch_tensor_constructor(fn):
        orig_func = getattr(fn, '__cube_orig_func__', fn)  # to support nested patching
        def wrapper(*args, **kwargs):
            kwargs["device"] =device
            return orig_func(*args, **kwargs)
        wrapper.__name__ = orig_func.__name__
        wrapper.__qualname__ = orig_func.__qualname__
        if hasattr(orig_func, '__module__'):
            # torch.Tensor.new_empty, etc. don't have this attribute
            # TODO: FxModuleParser._find_module_of_method will fail if __module__ is not set
            wrapper.__module__ = orig_func.__module__
        wrapper.__cube_orig_func__ = orig_func
        return wrapper
    # these constructors are enough for most cases
    patched_tensor_constructor_names = [
        'empty', 'zeros', 'ones', 'full', 'eye',
        'linspace', 'logspace', 'arange',
        'rand', 'randn', 'randint', 'randperm',
        'randn_like', 'rand_like', 'randint_like',
        'tensor'
    ]
    old_tensor_constructors = {
        tf_name: getattr(torch, tf_name)
        for tf_name in patched_tensor_constructor_names
    }
    patched_tensor_constructors = {
        tf_name: patch_tensor_constructor(fn)
        for tf_name, fn in old_tensor_constructors.items()
    }

    patched_tensor_member_constructor_names = [
        'new_empty', 'new_zeros', 'new_ones', 'new_full', 'new_tensor'
    ]
    old_tensor_member_constructors = {
        tf_name: getattr(torch.Tensor, tf_name)
        for tf_name in patched_tensor_member_constructor_names
    }
    patched_tensor_member_constructors = {
        tf_name: patch_tensor_constructor(fn)
        for tf_name, fn in old_tensor_member_constructors.items()
    }

    def patched_to(self, *args, **kwargs):
        if len(args) > 0 and isinstance(args[0], (torch.device, str)):
            return orig_to(self, device, *args[1:], **kwargs)
        if 'device' in kwargs:
            kwargs['device'] = device
            return orig_to(self, *args, **kwargs)
        return orig_to(self, *args, **kwargs)

    def patched_cuda(self, *args, **kwargs):
        return orig_to(self, device)

    def patched_cpu(self, *args, **kwargs):
        return orig_to(self, device)

    try:
        torch.Tensor.to = patched_to
        torch.Tensor.cuda = patched_cuda
        torch.Tensor.cpu = patched_cpu
        # patch tensor constructors
        for tf_name, fn in old_tensor_constructors.items():
            setattr(torch, tf_name, patched_tensor_constructors[tf_name])

        # patch tensor member constructors
        for tf_name, fn in old_tensor_member_constructors.items():
            setattr(torch.Tensor, tf_name, patched_tensor_member_constructors[tf_name])

        # patch concrete tracer's autowrap leaf function
        for tf_name, fn in old_tensor_constructors.items():
            leaf_info = wrap_utils.default_autowrap_leaf_function.pop(fn, None)
            if leaf_info:
                wrap_utils.default_autowrap_leaf_function[
                    patched_tensor_constructors[tf_name]
                ] = leaf_info
        yield
    finally:
        for tf_name, fn in patched_tensor_constructors.items():
            leaf_info = wrap_utils.default_autowrap_leaf_function.pop(fn, None)
            if leaf_info:
                wrap_utils.default_autowrap_leaf_function[
                    old_tensor_constructors[tf_name]
                ] = leaf_info
        for tf_name, fn in old_tensor_member_constructors.items():
            setattr(torch.Tensor, tf_name, fn)
        for tf_name, fn in old_tensor_constructors.items():
            setattr(torch, tf_name, fn)
        torch.Tensor.to = orig_to
        torch.Tensor.cuda = orig_cuda
        torch.Tensor.cpu = orig_cpu


# mock process group is from pytorch testing code
# import torch.testing._internal.distributed.distributed_utils

class MockProcessGroup(dist.ProcessGroup):
    def __init__(self, rank, world):
        super().__init__(rank, world)

    def getBackendName(self):
        return "cube_mock_pg"


def create_mock_pg(prefix_store, rank, world_size, timeout):
    return MockProcessGroup(rank, world_size)


dist.Backend.register_backend('cube_mock_pg', create_mock_pg)


def mock_init_dist(rank, world_size):
    if dist.is_initialized():
        raise ValueError("dist is already initialized, cannot mock init")

    store = dist.HashStore()

    dist.init_process_group(
        backend="cube_mock_pg",
        rank=rank,
        world_size=world_size,
        store=store,
        group_name="cube_fake",
        timeout=timedelta(seconds=1))


@contextmanager
def mock_dist(rank, world_size):
    """
    Mock dist.init_process_group for testing
    """

    old_store_based_barrier = c10d._store_based_barrier
    old_new_group = dist.new_group
    try:
        c10d._store_based_barrier = lambda *args, **kwargs: None
        mock_init_dist(rank, world_size)
        dist.new_group = lambda *args, **kwargs: None
        yield
    finally:
        dist.destroy_process_group()
        c10d._store_based_barrier = old_store_based_barrier


@contextmanager
def mock_cube_env(rank, world_size):
    old_device_group = nnscaler.runtime.device._instance
    old_dev_mode = CompileFlag.dev_mode
    used_cuda_fns = ['set_device', 'current_device', 'default_stream']
    old_cuda_fns = {
        fname: getattr(torch.cuda, fname)
        for fname in used_cuda_fns
    }
    torchrun_envs = ['RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE', 'GROUP_RANK', 'TORCHELASTIC_RUN_ID']
    old_envs = {
        env: os.environ.get(env, None)
        for env in torchrun_envs
    }
    try:
        nnscaler.runtime.device._instance = None
        CompileFlag.dev_mode = False
        for fname, fn in old_cuda_fns.items():
            setattr(torch.cuda, fname, lambda *args, **kwargs: None)
        os.environ['RANK'] = os.environ['LOCAL_RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = os.environ['LOCAL_WORLD_SIZE'] = str(world_size)
        os.environ['GROUP_RANK'] = '0'
        os.environ['TORCHELASTIC_RUN_ID'] = '0' # fake torchrun env
        yield
    finally:
        for env, val in old_envs.items():
            if val is None:
                del os.environ[env]
            else:
                os.environ[env] = val
        for fname, fn in old_cuda_fns.items():
            setattr(torch.cuda, fname, fn)
        CompileFlag.dev_mode = old_dev_mode
        nnscaler.runtime.device._instance = old_device_group


@contextmanager
def mock_reducer_env(rank, runtime_ngpus, device='cpu'):
    with replace_all_device_with(device, True), mock_cube_env(rank, runtime_ngpus), mock_dist(rank, runtime_ngpus):
        yield


def new_empty(cube_module_cls: Type[ParallelModule], device='meta', init_params=False):
    """
    Create a new instance with empty weights.

    This is useful when you want to get model information (e.g. fullmap/zero) without allocating memory.
    """
    module_file = Path(sys.modules[cube_module_cls.__module__].__file__)
    compute_config = ComputeConfig.safe_load_from_file(module_file.with_name(f"{cube_module_cls.COMPUTE_CONFIG_FILE}"))
    with replace_all_device_with(device, True), mock_cube_env(cube_module_cls.rank, compute_config.runtime_ngpus), mock_dist(cube_module_cls.rank, compute_config.runtime_ngpus):
        return cube_module_cls(init_params=init_params)


def retry(*exceptions, max_tries=3, match: Optional[Union[str, Pattern[str]]] = None, delay=5):
    """
    Retry the function if an exception is raised.

    Args:
        max_tries (int): the maximum number of tries

    Example:
        @retry():
        def f(*args, **kwargs):
            ...
    """
    exceptions = exceptions or (Exception,)
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(max_tries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    matched = not match or re.search(match, str(e))
                    if i == max_tries - 1 or not matched:
                        raise
                    from time import sleep
                    print(f"retrying... {e} after {delay} seconds")
                    sleep(delay)
        return wrapper

    return decorator


@contextmanager
def catch_log(_logger, loglevel='DEBUG'):
    import logging
    from io import StringIO
    string_stream = StringIO()
    old = _logger.level
    _logger.setLevel(loglevel)
    handler = logging.StreamHandler(string_stream)
    handler.setLevel(loglevel)
    _logger.addHandler(handler)
    yield string_stream
    _logger.removeHandler(handler)
    _logger.setLevel(old)


@contextmanager
def catch_stdout():
    import sys
    from io import StringIO
    old = sys.stdout
    string_stream = StringIO()
    sys.stdout = string_stream
    yield string_stream
    sys.stdout = old


@contextmanager
def clear_dir_on_rank0(tempdir):
    if torch.distributed.get_rank() == 0 and tempdir.exists():
        shutil.rmtree(tempdir)
    yield tempdir
    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0 and tempdir.exists():
        shutil.rmtree(tempdir)
