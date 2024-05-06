# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Optional
import time
import logging

_logger = logging.getLogger(__name__)


class CpuTimer:
    r"""
    Singleton Cpu Timer

    Note that frequently using timer may decrease the performance.

    The runtime predefines the timer on each communication primitive.
    By default, the timer on communications are disabled for higher performance.
    For users who want to analyze communication overhead, turn on the timer
    by using `CpuTimer(enable=True, predefined=True)`.

    There are two switches to allow user to control the timer behaviour

    * enable:
        the overall controller to turn on/off the all profiling.
    * predefined:
        the controller to turn on/off the predefined timer (mostly are communications)
    """
    class __CpuTimer:

        def __init__(self, enable = True):
            self.field = dict()
            self.field_data = dict()
            self.enabled = enable
    
    instance = None

    def __init__(self, enable: Optional[bool] = None):
        # not have instance
        if not CpuTimer.instance:
            enable = enable if enable is not None else True
            CpuTimer.instance = CpuTimer.__CpuTimer(enable)
        # have instance
        else:
            if enable is not None:
                CpuTimer.instance.enabled = enable
    
    def start(self, field_name='default'):
        """
        Start recording time on the the field

        Note `start` and `stop` on the same field can be called nestly

        Args:
            field_name (str): the lable of the recorded time

        Returns:
            None
        """
        if not CpuTimer.instance.enabled:
            return
        # torch.cuda.default_stream().synchronize()
        start_time = time.time()
        if field_name not in CpuTimer.instance.field:
            CpuTimer.instance.field[field_name] = list()
            CpuTimer.instance.field_data[field_name] = 0
        CpuTimer.instance.field[field_name].append(start_time)
    
    def stop(self, field_name='default') -> float:
        """
        Record the time span from last `start` on the same field_name to now

        Args:
            field_name (str): the lable of the recorded time

        Returns:
            float: wall clock in milli-seconds
        """
        if not CpuTimer.instance.enabled:
            return
        if field_name not in CpuTimer.instance.field:
            raise RuntimeError("Missing start on the field")
        # torch.cuda.default_stream().synchronize()
        stop_time = time.time()
        start_time = CpuTimer.instance.field[field_name].pop(-1)
        span = stop_time - start_time # in seconds
        CpuTimer.instance.field_data[field_name] += span
        return span

    def duration(self, times: int, field_name: str = 'default') -> float:
        """
        Get the total span (wall clock) of a field name. The span is divided by times.

        Args:
            times (int): division factor to average the time
            filed_name (str): the field name

        Returns
            float : wall clock in milliseconds.
        """
        if field_name not in CpuTimer.instance.field:
            _logger.warning(f"CpuTimer: {field_name} doesn't record.")
            return 0.0
        if len(CpuTimer.instance.field[field_name]) != 0:
            raise RuntimeError(f"timer for field {field_name} not stopped")
        return CpuTimer.instance.field_data[field_name] / times * 1000  # in ms

    def __getattr__(self, name):
        return getattr(CpuTimer.instance, name)

    def clear(self):
        CpuTimer.instance = CpuTimer.__CpuTimer(
            enable=self.enabled, predefined=self.predefined
        )

    def print_all(self, times: int):
        """
        Print the total span of each recorded field divided by `times`

        Note this should be called by each process

        @param times int: division factor
        @param rank_only Optional[int]: select only one rank for print

        @return None
        """
        msg = list()
        names = list(CpuTimer.instance.field_data.keys())
        names.sort()
        for field_name in names:
            span = self.duration(times, field_name)
            msg.append('{} : {:.2f} ms'.format(field_name, span))
        msg = ' | '.join(msg)
        print(msg)

