#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from typing import Callable
import uuid
import torch

from torch.distributed.run import elastic_launch, LaunchConfig
from torch.distributed.elastic.multiprocessing.errors import ChildFailedError

from .utils import retry


@retry(ChildFailedError, delay=10, match='The server socket has failed to listen on any local network address.')
def launch_torchrun(nproc_per_node, worker_fn, *args, **kwargs):
    launch_config = LaunchConfig(
        min_nodes=1,
        max_nodes=1,
        nproc_per_node=nproc_per_node,
        rdzv_backend = "c10d",
        rdzv_endpoint = "localhost:29401",
        run_id = str(uuid.uuid4()),
        monitor_interval=0.1,
        max_restarts=0,
    )
    outputs = elastic_launch(launch_config, worker_fn)(*args, **kwargs)
    return outputs


def torchrun(nproc_per_node: int, test_fn: Callable, *args, **kwargs):
    """Test utility for torchrun

    Example usage:

    ```python
    from functools import partial
    test_function_name = partial(torchrun, 2, function_to_test)
    ```

    Args:
        nproc_per_node (int): number of gpus
        test_fn (function): test function, which should return None
        *args: args for worker_fn
        **kwargs: kwargs for worker_fn

    Returns:
        None
    """

    if not torch.cuda.is_available() or torch.cuda.device_count() < nproc_per_node:
        print(f"skip test on {nproc_per_node} gpus due to lack of cuda devices")
        return
    launch_torchrun(nproc_per_node, test_fn, *args, **kwargs)


def clone_to_cpu(tensor: torch.Tensor):
    # when you use launch_torchrun
    # you can't directly return a cuda tensor
    #   Error message: Producer process has been terminated before all shared CUDA tensors released. See Note [Sharing CUDA tensors]
    # So you can use this function to clone a tensor to cpu
    cloned_tensor = tensor.cpu().clone().detach().requires_grad_(tensor.requires_grad)
    if tensor.is_leaf and tensor.grad is not None:
        cloned_tensor.grad = tensor.grad.cpu().clone()
    return cloned_tensor


def clone_to_cpu_recursively(data):
    if isinstance(data, torch.Tensor):
        return clone_to_cpu(data)
    elif isinstance(data, dict):
        return {k: clone_to_cpu_recursively(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clone_to_cpu_recursively(v) for v in data]
    elif isinstance(data, tuple):
        return tuple(clone_to_cpu_recursively(v) for v in data)
    else:
        return data
