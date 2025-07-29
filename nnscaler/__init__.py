from .version import __version__
from .parallel import (
    ParallelModule,
    ComputeConfig,
    ReuseType,
    BroadcastGenFilesStrategy,
    parallelize,
    build_optimizer,
    merge_state_dicts,
    load_merged_state_dicts,
    deduped_state_dict,
    load_deduped_state_dict,
    broadcast_weights,
)
from nnscaler.graph.parser.register import register_op


def init():
    """
    Initialize the nnscaler library.

    It will initialize torch distributed nccl process_group
    and set the default cuda device according to the local rank of the process.

    It is recommended to call this function before any other nnscaler functions,
    although it is optional if you initialize the torch distributed nccl process_group
    and set the default cuda device by yourself.

    Please note that you should intialize torch distributed process_group with a large timeout (6 hours, for example),
    because the parallelization of modules may take a long time,
    and the default timeout (30 minutes) may be too short.

    Returns:
        None
    """
    from nnscaler import runtime
    _ = runtime.device.DeviceGroup()
    _ = runtime.resource.EnvResource()


def _check_torch_version():
    import torch
    import logging
    torch_version = str(torch.__version__).split('+')[0]
    torch_version = tuple(int(v) for v in torch_version.split('.')[:2])
    if torch_version < (2, 0):
        logging.warn(f"expected PyTorch version >= 2.0 but got {torch_version}")


_check_torch_version()
