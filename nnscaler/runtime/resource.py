r"""
Runtime information
"""
from typing import Tuple

import torch
from nnscaler.flags import CompileFlag
from dataclasses import dataclass


@dataclass
class DeviceInfo:
    # memory in btypes
    memory: int = None


class EnvResource:

    class __EnvResource:

        def __init__(self):
            # number of gpus
            self.ngpus = 1 if CompileFlag.dev_mode else torch.distributed.get_world_size()
            # device topology
            self.topo = None
            self.gpus: Tuple[DeviceInfo] = self.get_device_capability()

        def get_device_capability(self) -> Tuple[DeviceInfo]:
            if CompileFlag.dev_mode:
                memory = [torch.cuda.get_device_properties(0).total_memory]
            else:
                rank = torch.distributed.get_rank()
                memory = torch.tensor(torch.cuda.get_device_properties(0).total_memory, 
                                      dtype=torch.int64, device=torch.cuda.current_device())
                all_device_mem = [torch.empty_like(memory) for _ in range(self.ngpus)]
                all_device_mem[rank] = memory.data
                torch.distributed.all_gather(all_device_mem, memory)
                torch.cuda.synchronize()
                memory = [t.item() for t in all_device_mem]
            return tuple(DeviceInfo(memory=mem) for mem in memory)

    instance = None

    def __init__(self):
        if not EnvResource.instance:
            EnvResource.instance = EnvResource.__EnvResource()

    def __getattr__(self, name):
        return getattr(self.instance, name)


    def __setattr__(self, name, val) -> None:
        setattr(EnvResource.instance, name, val)
