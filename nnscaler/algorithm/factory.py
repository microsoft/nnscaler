#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from typing import Dict, Any, Union, List, Optional, Type, overload

from nnscaler.algorithm.generics import GenericDistAlgo


class _DistAlgorithmFactory:
    def __init__(self):
        self._algos: dict[type, dict[str, type[GenericDistAlgo]]] = {}
        self._load_predefined_algos()

    def exist(self, op: Type, tag: Optional[str] = None):
        """
        Check if the factory has op's algorithm recorded

        Returns:
            True if have, False if not
        """
        for op_class in op.mro():
            if op_class not in self._algos:
                continue
            if tag is None or tag in self._algos[op_class]:
                return True
        return False

    def register(self, op, algorithm: type[GenericDistAlgo], tag: str):
        """
        Register a holistic op (class) as one of the anchors
        """
        if op not in self._algos:
            self._algos[op] = dict()
        self._algos[op][tag] = algorithm

    def algorithms(self, op: Type) -> List[GenericDistAlgo]:
        """
        Get all transform algorithms for the op

        Args:
            op (IRFwOperation): the op to be transformed

        Returns:
            List[GenericDistAlgo]: the algorithms for the op
        """
        algos = [self._algos[op_class] for op_class in op.mro() if op_class in self._algos]
        # use dict to remove duplicates and keep order
        algos_all: dict[type[GenericDistAlgo], None] = {}
        for tag_algo_map in algos:
            for algo in tag_algo_map.values():
                algos_all[algo] = None
        return list(algos_all.keys())

    def algorithm(self, op: Type, tag: str) -> GenericDistAlgo:
        """
        Get best matched tranform algorithm for the op with tag

        Args:
            op (IRFwOperation): the op to be transformed
            tag (str): the tag of the algorithm

        Returns:
            GenericDistAlgo: the algorithm for the op

        Raises:
            ValueError: if the op + tag is not registered in the factory
        """
        for op_class in op.mro():
            if op_class not in self._algos:
                continue
            if tag in self._algos[op_class]:
                return self._algos[op_class][tag]
        raise ValueError("Op {op} + Tag {tag} is not registered in factory")

    def _load_predefined_algos(self):

        import nnscaler.algorithm.ops.dimops as dimops
        self.register(dimops.IRDimops, dimops.DimSplitEinops, tag='dim')

        import nnscaler.algorithm.ops.conv as conv
        self.register(conv.IRPad, conv.DimSplitPad, tag='dim')
        self.register(conv.IRConv2D, conv.DimSplitConv2D, tag='dim')
        self.register(conv.IRConv2D, conv.HaloSplitConv2D, tag='halo')
        self.register(conv.IRConv3D, conv.HaloSplitConv3D, tag='halo')


_instance: Optional[_DistAlgorithmFactory] = None
def DistAlgorithmFactory() -> _DistAlgorithmFactory:
    global _instance
    if _instance is None:
        _instance = _DistAlgorithmFactory()
    return _instance
