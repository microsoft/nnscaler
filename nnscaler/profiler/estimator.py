#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from typing import Union, Tuple
import sys
import os
import logging

from nnscaler.ir.operator import IRFwOperation
from nnscaler.graph.segment import IRSegment
from nnscaler.graph.function import IRGraphAnchor
from nnscaler.profiler.database import ProfileDataBase

_logger = logging.getLogger(__name__)


class Estimator:
    """
    Estimator to measture the computation / memory cost of a subgraph
    """

    def __init__(self, cache='./profile_database.json'):

        self.cache_file = cache
        reload = cache if os.path.exists(cache) else None
        self.database = ProfileDataBase(reload)

    def __call__(self, nodes_or_segment: Union[Tuple[IRFwOperation], IRSegment], 
                 train: bool=False):
        """
        Profile the computation cost of a subgraph

        @param nodes_or_segment Tuple[IRFwOperation] | IRSegment

        @return latency float: latency in ms
        @return memory int: memory in bytes
        """
        nodes = nodes_or_segment.nodes() if isinstance(nodes_or_segment, IRSegment) else nodes_or_segment
        memory, latency = 0.0, 0.0
        for node in nodes:
            if node.name == 'multiref' or isinstance(node, IRGraphAnchor):
                continue
            # _, _, fw_span, bw_span, infer_mem, train_mem_info, _ = self.database.profile(node)
            try:
                _, _, fw_span, bw_span, infer_mem, train_mem_info, _ = self.database.profile(node)
            except Exception as e:
                color, default = '\033[31m', '\033[0m'
                error_msg = f'fail to run node: {node}\nerror: {e}'
                _logger.error(f'{color}{error_msg}{default}')
                fw_span, bw_span, infer_mem, train_mem_info = 0, 0, 0, [0]

            if train:
                memory += sum(train_mem_info)
                latency += fw_span + bw_span
            else:
                memory = max(memory, infer_mem)
                latency += fw_span
        return latency, memory

    def save(self):
        self.database.dump(self.cache_file, override=True)
