#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import argparse
import json
import torch
from pathlib import Path
import os
from typing import Tuple, List, Dict

import nnscaler
from nnscaler.runtime.adapter.collectives import all_gather, all_reduce, all_to_all, reduce_scatter
from nnscaler.profiler import CudaTimer
from nnscaler.runtime.device import DeviceGroup
from nnscaler.autodist.util import get_node_arch, get_default_profile_path


class CommProfiler:

    def __init__(self,
                 nranks: int,
                 warmup_times: int = 10,
                 profile_times: int = 10) -> None:
        self.nranks = nranks
        self.warmup_times = warmup_times
        self.profile_times = profile_times
        self.ranks = tuple(range(self.nranks))

    def collect_profile_info(self,
                             primitive: str) -> Tuple[List[float], List[float]]:

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
            assert d_size % self.nranks == 0
            if primitive in ['all gather', 'all to all']:
                d_size = d_size // self.nranks
            tensor = torch.rand([b_size, sequence_len, d_size],
                                dtype=torch.float32,
                                device=torch.cuda.current_device())
            if primitive == 'all gather':
                func = all_gather
                kwargs = {'tensor': tensor, 'dim': 2, 'ranks': self.ranks}
            elif primitive == 'all reduce':
                func = all_reduce
                kwargs = {'tensor': tensor, 'ranks': self.ranks}
            elif primitive == 'reduce scatter':
                func = reduce_scatter
                kwargs = {'tensor': tensor, 'dim': 2, 'ranks': self.ranks}
            elif primitive == 'all to all':
                func = all_to_all
                kwargs = {
                    'tensor': tensor,
                    'idim': 0,
                    'odim': 2,
                    'ranks': self.ranks
                }
            else:
                raise ValueError('Unknown primitive: {}'.format(primitive))
            for _ in range(self.warmup_times):
                func(**kwargs)
            CudaTimer().clear()
            for _ in range(self.profile_times):
                otensor = func(**kwargs)
            cur_t = CudaTimer().instance.field_data['comm'] / self.profile_times
            times_in_s.append(cur_t)
        return sizes_in_mb, times_in_s

    def profile(self) -> Dict[str, Tuple[List[float], List[float]]]:
        profile_info = {}
        for primitive in [
                'all gather', 'all reduce', 'reduce scatter', 'all to all'
        ]:
            profile_info[primitive] = self.collect_profile_info(
                primitive=primitive)
        return profile_info

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Profile runtime communication cost')
    parser.add_argument('--comm_profile_dir',
                        type=str,
                        default=get_default_profile_path() / get_node_arch() / 'comm',
                        help='autodist comm profile folder')
    args = parser.parse_args()

    nnscaler.init()

    CudaTimer(enable=True, predefined=True)
    world_size = DeviceGroup().world_size
    comm_profiler = CommProfiler(nranks=world_size)

    profile_info = comm_profiler.profile()

    if torch.distributed.get_rank() == 0:
        dir_path = Path(args.comm_profile_dir)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
        file_name = dir_path / f'intra_{world_size}.json'
        with open(file_name, 'w') as f:
            json.dump(profile_info, f, indent=2)
