from typing import Optional
import time
import logging

import torch
from nnscaler.utils import print_each_rank

_logger = logging.getLogger(__name__)


class CudaTimer:
    r"""
    Singleton Cuda Timer

    Note that frequently using timer may decrease the performance.

    The runtime predefines the timer on each communication primitive.
    By default, the timer on communications are disabled for higher performance.
    For users who want to analyze communication overhead, turn on the timer
    by using `CudaTimer(enable=True, predefined=True)`.

    There are two switches to allow user to control the timer behaviour

    * enable:
        the overall controller to turn on/off the all profiling.
    * predefined:
        the controller to turn on/off the predefined timer (mostly are communications)
    """
    class __CudaTimer:

        def __init__(self, enable = True, predefined = False):
            self.start_t = None
            self.stop_t = None
            self.field = dict()
            self.field_data = dict()
            self.enabled = enable
            self.predefined = predefined
    
    instance = None

    def __init__(self, enable: Optional[bool] = None, predefined: Optional[bool] = None):
        # not have instance
        if not self.instance:
            enable = enable if enable is not None else True
            predefined = predefined if predefined is not None else False
            CudaTimer.instance = CudaTimer.__CudaTimer(enable, predefined)
        # have instance
        else:
            if enable is not None:
                self.instance.enabled = enable
            if predefined is not None:
                self.instance.predefined = predefined
    
    def start(self, field_name='default', predefined: bool = False, stream: Optional[torch.cuda.Stream] = None):
        """
        Start recording time on the the field

        Note `start` and `stop` on the same field can be called nestly

        @param field_name str
        @param is_predefined bool: whether the field is a predefined field
        @param stream Optional[torch.cuda.Stream]:
            if None (default), will synchronize all streams on the device before
            recording time. Otherwise, only synchronize the specified stream.

        @return None
        """
        if (not self.instance.enabled) or (predefined and not self.instance.predefined):
            return
        if stream is None:
            torch.cuda.synchronize()
        else:
            stream.synchronize()
        # torch.cuda.default_stream().synchronize()
        start_time = time.time()
        if field_name not in self.instance.field:
            self.instance.field[field_name] = list()
            self.instance.field_data[field_name] = 0
        self.instance.field[field_name].append(start_time)
    
    def stop(self, field_name='default', predefined: bool = False, stream: Optional[torch.cuda.Stream] = None) -> float:
        """
        Record the time span from last `start` on the same field_name to now

        @param field_name str
        @param is_predefined bool: whether the field is a predefined field
        @param stream Optional[torch.cuda.Stream]:
            if None (default), will synchronize all streams on the device before
            recording time. Otherwise, only synchronize the specified stream.

        @return None
        """
        if (not self.instance.enabled) or (predefined and not self.instance.predefined):
            return
        if field_name not in self.instance.field:
            raise RuntimeError("Missing start on the field")
        if stream is None:
            torch.cuda.synchronize()
        else:
            stream.synchronize()
        # torch.cuda.default_stream().synchronize()
        stop_time = time.time()
        start_time = self.instance.field[field_name].pop(-1)
        span = stop_time - start_time # in seconds
        self.instance.field_data[field_name] += span
        return span

    def duration(self, times: int, field_name: str = 'default') -> float:
        """
        Get dthe total span (wall clock) of a field name. The span is divided by times.

        @param times int: division factor
        @param filed_name str: the field name

        @return span float: wall clock in milliseconds.
        """
        if field_name not in self.instance.field:
            _logger.warning(f"CudaTimer: {field_name} doesn't record.")
            return 0.0
        if len(self.instance.field[field_name]) != 0:
            raise RuntimeError(f"timer for field {field_name} not stopped")
        return self.instance.field_data[field_name] / times * 1000  # in ms

    def __getattr__(self, name):
        return getattr(self.instance, name)

    def clear(self):
        CudaTimer.instance = CudaTimer.__CudaTimer(
            enable=self.enabled, predefined=self.predefined
        )

    def print_all(self, times: int, rank_only: Optional[int] = None):
        """
        Print the total span of each recorded field divided by `times`

        Note this should be called by each process

        @param times int: division factor
        @param rank_only Optional[int]: select only one rank for print

        @return None
        """
        msg = list()
        names = list(self.instance.field_data.keys())
        names.sort()
        for field_name in names:
            span = self.duration(times, field_name)
            msg.append('{} : {:.2f} ms'.format(field_name, span))
        msg = ' | '.join(msg)
        print_each_rank(msg, rank_only)

    def warmup(self, seconds=1.0):
        """
        Warm up GPU for `span` seconds.
        """
        print('> warming up for 1 second')
        data1 = torch.randn((4096, 4096), device=torch.cuda.current_device())
        data2 = torch.randn((4096, 4096), device=torch.cuda.current_device())
        # warm up 1s
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        torch.cuda.synchronize()
        start = time.time()
        while time.time() - start < seconds:
            _ = torch.matmul(data1, data2)
            # if torch.distributed.is_initialized():
            #     torch.distributed.all_reduce(out)
            torch.cuda.synchronize()
