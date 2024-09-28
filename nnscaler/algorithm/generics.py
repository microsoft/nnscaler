#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from typing import List, Optional

from nnscaler.ir.cten import IRCell


class GenericDistAlgo:
    """!
    Generic distributed algorithm that partitions a node into sub-nodes.
    """

    def __init__(self, node: IRCell):
        if not isinstance(node, IRCell):
            raise TypeError("Expected node to be IRCell")
        self._node = node

    @property
    def node(self) -> IRCell:
        return self._node

    def satisfy(self, **config) -> bool:
        """!
        Check if the config satisfies instantiation conditions

        @param config Dict: configuration for the algorithm, like number of partitioned chunks.

        @return satisfy bool: True if the configuration can satisfy for this node
        """
        raise NotImplementedError

    def instantiate(self, **config) -> Optional[List[IRCell]]:
        """!
        Instantiate the algorithm given the config

        @param config Dict: configuration for the algorithm, like number of partitioned chunks.

        @return sub_nodes Optional[List[IRCell]]: if sucess, the partitioned sub nodes, else None
        """
        raise NotImplementedError
    
    def __repr__(self) -> str:
        return f'TransAlgo(node{self._node.cid})'
