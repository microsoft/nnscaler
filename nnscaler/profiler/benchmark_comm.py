#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from __future__ import annotations

from datetime import datetime
import json
import logging
import sys

import torch

import nnscaler
from nnscaler.autodist.util import get_node_arch, get_default_profile_path
from nnscaler.profiler import CudaTimer
from nnscaler.runtime.adapter.collectives import all_gather, all_reduce, all_to_all, reduce_scatter
from nnscaler.runtime.device import DeviceGroup
from nnscaler.utils import is_running_distributed


_logger = logging.getLogger('nnscaler.profiler')


# The profiling result of a primitive function, as two lists of the same length.
# The first list contains tensor sizes (in MB).
# The second list contains corresponding time consumption (in seconds).
PrimitiveProfile = tuple[list[float], list[float]]

# The profiling result of a GPU group.
# The key is a primitive function's name: "all gather", "all reduce", "reduce scatter", "all to all"
# The value is its profiling result.
# NOTE: the function names use spaces, not underscores
Profile = dict[str, PrimitiveProfile]


class CommProfiler:
    def __init__(self, warmup_times: int = 10, profile_times: int = 10):
        self.warmup_times = warmup_times
        self.profile_times = profile_times

    def profile_all(self) -> dict[str, Profile]:
        ret = {}

        # run on all nodes to simplify barrier
        ret.update(self.profile_single_node())

        if DeviceGroup().world_size > DeviceGroup().local_world_size:
            ret.update(self.profile_multi_nodes())

        return ret

    def profile_single_node(self) -> dict[str, Profile]:
        # The key is GPU numbers in string format: "2", "4", "8", ...
        ret = {}
        n_procs = DeviceGroup().local_world_size

        device_num = 2
        while device_num <= n_procs:
            key = str(device_num)
            if DeviceGroup().local_rank == 0:
                _logger.info(f'Profiling {key} GPUs...')
            ranks = tuple(range(device_num))

            # dist.new_group() must be invoked on all ranks,
            # but invoking primitives on all ranks will raise warning
            DeviceGroup().get_group(ranks)
            if DeviceGroup().rank in ranks:
                ret[key] = self.profile_ranks(ranks)
            DeviceGroup().long_barrier()

            device_num *= 2

        return ret

    def profile_multi_nodes(self) -> dict[str, Profile]:
        # The key is "{nnodes}x{ngpus}": "2x8", "4x8", "8x8", ...
        # Because 2x2 is likely to slower than 1x4, we only test N x local_world_size
        ret = {}

        # assuming all nodes have the same GPU numbers
        world_size = DeviceGroup().world_size
        local_world_size = DeviceGroup().local_world_size
        assert world_size % local_world_size == 0, 'The nodes are heterogeneous'

        n_nodes = world_size // local_world_size
        n_procs = local_world_size

        node_num = 2
        while node_num <= n_nodes:
            key = f'{node_num}x{n_procs}'
            if DeviceGroup().local_rank == 0:
                _logger.info(f'Profiling {key} GPUs...')
            ranks = list(range(n_procs * node_num))

            # dist.new_group() must be invoked on all ranks,
            # but invoking primitives on all ranks will raise warning
            DeviceGroup().get_group(ranks)
            if DeviceGroup().rank in ranks:
                ret[key] = self.profile_ranks(ranks)
            DeviceGroup().long_barrier()

            node_num *= 2

        return ret

    def profile_ranks(self, ranks: list[int]) -> Profile:
        profile_info = {}
        for primitive in ['all gather', 'all reduce', 'reduce scatter', 'all to all']:
            profile_info[primitive] = self.profile_primitive(primitive, ranks)
        return profile_info

    def profile_primitive(self, primitive: str, ranks: list[int]) -> PrimitiveProfile:
        b_size = 16
        sequence_len = 16
        quarter_mb_size_list = [
            1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048
        ]
        model_dim_list = [
            mem * 256 * 256 // b_size // sequence_len
            for mem in quarter_mb_size_list
        ]
        sizes_in_mb = [0.25 * val for val in quarter_mb_size_list]
        times_in_s = []
        for cur_sz, d_size in zip(sizes_in_mb, model_dim_list):
            assert d_size % len(ranks) == 0
            if primitive in ['all gather', 'all to all']:
                d_size = d_size // len(ranks)
            # Here dtype has little impact. Here we just use `float32`
            tensor = torch.rand([b_size, sequence_len, d_size],
                                dtype=torch.float32,
                                device=torch.cuda.current_device())
            # dim has no impact on transmission. In the following test, we use 0 for idim and 2 for odim.
            if primitive == 'all gather':
                func = lambda: all_gather(tensor=tensor, dim=2, ranks=ranks)
            elif primitive == 'all reduce':
                func = lambda: all_reduce(tensor=tensor, ranks=ranks)
            elif primitive == 'reduce scatter':
                func = lambda: reduce_scatter(tensor=tensor, dim=2, ranks=ranks)
            elif primitive == 'all to all':
                func = lambda: all_to_all(tensor=tensor, idim=0, odim=2, ranks=ranks)
            else:
                raise ValueError('Unknown primitive: {}'.format(primitive))
            for _ in range(self.warmup_times):
                func()
            CudaTimer().clear()
            for _ in range(self.profile_times):
                _otensor = func()
            cur_t = CudaTimer().instance.field_data['comm'] / self.profile_times
            times_in_s.append(cur_t)
        return sizes_in_mb, times_in_s


def main():
    if not is_running_distributed():
        print('Usage: torchrun {TORCHRUN_ARGS} -m nnscaler.profiler.benchmark_comm')
        sys.exit(1)

    nnscaler.init()

    if DeviceGroup().world_size == 1:
        _logger.warning('Single GPU profiling is not supported')
        return

    if DeviceGroup().local_rank == 0:
        nnscaler.utils.set_default_logger_level('INFO')
    else:
        nnscaler.utils.set_default_logger_level('DEBUG')

    CudaTimer(enable=True, predefined=True)

    comm_profiler = CommProfiler()
    profile_info = comm_profiler.profile_all()

    if DeviceGroup().rank == 0:
        comm_path = get_default_profile_path() / 'comm'
        if comm_path.exists():
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            backup_path = comm_path.with_name(f'comm-bak-{timestamp}')
            _logger.info('Profiling data already exists')
            _logger.info(f'Backup old data to {backup_path}')
            comm_path.rename(backup_path)

        comm_path.mkdir(parents=True, exist_ok=True)

        for key, profile in profile_info.items():
            if 'x' in key:
                # FIXME: saving inter-nodes results as intra
                x, y = key.split('x')
                key = str(int(x) * int(y))
            file_name = comm_path / f'intra_{key}.json'
            with open(file_name, 'w') as f:
                json.dump(profile, f, indent=2)

        _logger.info('Profiling done')

    elif DeviceGroup().local_rank == 0:
        _logger.info('Multi-nodes profiling done')


if __name__ == '__main__':
    main()
