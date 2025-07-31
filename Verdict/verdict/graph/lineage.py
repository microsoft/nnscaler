#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from typing import Set, List, Tuple, Dict, Hashable

from .elements import Tensor
from .dfg import Node

ReduceTensors = List[Tensor]

SliceMap = List[Tuple[int, int]]  # [(start, end) for each dimention]


class Lineage:
    def __init__(self, Ts: Tensor, Tps: Set[Tensor]):
        self.Ts: Tensor = Ts
        self.Tps: Tuple[Tensor] = tuple(sorted(Tps))  # NOTE: Tps is a sorted tuple

        # Ts_slc -> [[Tp,...], [Tp,...]]
        self.slice_map: Dict[SliceMap, List[ReduceTensors]] = {}
        self.full_shape: List[int] = None
        
        # allow assign stage partition during building lineage (optimize cut_stage cost)
        self.is_stage_known: bool = False
        self.snodes: List[Node] = []
        self.pnodes: List[Node] = []
        self.dependency: List["Lineage"] = []
        

    @property
    def id(self) -> Hashable:
        return (self.Ts, self.Tps)

    def __repr__(self):
        return f"{self.Ts} == {self.Tps}"

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, value):
        return isinstance(value, Lineage) and self.id == value.id
