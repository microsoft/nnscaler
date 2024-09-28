#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""
Communication group settings among devices
"""
from typing import List, Dict, Optional
import numpy as np
import torch
import os
import logging
import datetime

from nnscaler.flags import CompileFlag
from nnscaler.utils import is_running_distributed

_logger = logging.getLogger(__name__)
_LARGE_TIMEOUT = datetime.timedelta(seconds=21600)


class _DeviceGroup:
    def __init__(self):
        if CompileFlag.dev_mode or not is_running_distributed():
            self.rank = 0
            self.world_size = 1
            self.local_world_size = 1
            self.local_rank = 0
            self.node_rank = 0
        else:
            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group(
                    backend='nccl', timeout=_LARGE_TIMEOUT
                )

            # disable it for now due to connection refused error when nnodes > 1
            # TODO: investigate the root cause
            # create a barrier group for synchronization
            # it is OK even the user has already created this gloo group
            # this new timeout will override the old one.
            # self.barrier_gloo_group = torch.distributed.new_group(
            #     backend='gloo', timeout=_LARGE_TIMEOUT
            # )

            self.rank = torch.distributed.get_rank()
            self.world_size = torch.distributed.get_world_size()
            # assume each node has the same device number
            self.local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE'))
            self.local_rank = int(os.environ.get('LOCAL_RANK'))
            self.node_rank = int(os.environ.get('GROUP_RANK'))

        torch.cuda.set_device(self.local_rank)
        self.groups: Dict = { '1'*self.world_size: None }
        self.streams: Dict[str, torch.cuda.Stream] = {
            'default': torch.cuda.default_stream()}

    def group_exists(self, ranks):
        """
        Check if group exists
        """
        rank_bits = self.bitmap(ranks)
        return rank_bits in self.groups

    def get_group(self, ranks):
        """
        Create and return rank groups on-demand

        None will be returned if length of ranks are equal to world size
        """
        if len(ranks) == self.world_size:
            return None
        rank_bits = self.bitmap(ranks)
        if rank_bits not in self.groups:
            self.groups[rank_bits] = torch.distributed.new_group(
                list(ranks), timeout=_LARGE_TIMEOUT)
        return self.groups[rank_bits]

    def long_barrier(self):
        """
        Barrier synchronization with very long timeout
        """
        # torch.distributed.barrier(group=self.barrier_gloo_group)
        torch.distributed.barrier()

    def get_stream(self, name: str) -> torch.cuda.Stream:
        """
        Get stream by name. If name doesn't exist,
        will create a new one.
        """
        return self.streams.setdefault(
            name, torch.cuda.Stream())

    def create_hybrid(self, group_num: List[int]) -> List[List[int]]:
        """
        Create hybrid (nested) groups given the each group number.

        The product of group_num should be same with total devices.
        """
        group_num = np.array(group_num)
        cnt = np.prod(group_num)
        if cnt != self.world_size:
            raise RuntimeError("product of group_num should be same with total device number")
        grid = np.arange(cnt).reshape(tuple(group_num))
        dims = list(range(len(group_num)))
        outputs = []
        for dim, num in enumerate(group_num):
            remain = np.prod(np.delete(group_num, dim))
            order = tuple(dims[:dim] + dims[dim+1:] + [dim])
            grid_dim = np.transpose(grid, order).reshape((remain,num))
            grid_dim = grid_dim.tolist()
            for ranks in grid_dim:
                # initialize group
                _ = self.get_group(ranks)
                if self.rank in ranks:
                    outputs.append(ranks)
        assert len(outputs) == len(group_num)
        return outputs

    def bitmap(self, ranks):
        """
        map the rank list to the bit map string
        """
        bits = '0' * self.world_size
        for rank in ranks:
            if rank >= len(bits):
                raise ValueError("rank {} out of range ({})".format(rank, len(bits)))
            bits = bits[0:rank] + '1' + bits[rank+1:]
        return bits

    def __repr__(self):
        msg = 'node rank: [{}] rank: [{}] local rank: [{}]\n'.format(self.node_rank, self.rank, self.local_rank)
        msg += 'communication groups (ranks):\n'
        for bitmap, group in self.groups.items():
            ranks = [rank for rank, bit in enumerate(bitmap) if bit == '1']
            if self.rank in ranks:
                msg += '\t group {}: my group rank: [{}]\n'.format(ranks, torch.distributed.get_rank(group))
        return msg


_instance: Optional[_DeviceGroup] = None

def DeviceGroup() -> _DeviceGroup:
    global _instance
    if _instance is None:
        _instance = _DeviceGroup()
    return _instance
