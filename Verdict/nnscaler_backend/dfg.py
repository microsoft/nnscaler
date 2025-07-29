from typing import Dict, List, final, Set, Tuple, Hashable
from collections import namedtuple
from pathlib import Path
from dataclasses import dataclass

from verdict.symbolics import Shape
from verdict.graph import DFG, World, DTag, SliceMap
from verdict.operators import OpName


ValueMap = Tuple[int, int]  # (value part id, total parts)


@dataclass
class LineageView:
    is_grad: bool
    is_loss: bool
    is_attr: bool

    ft_shape: Shape
    slcmap: SliceMap
    valmap: ValueMap


Node = namedtuple("Node", ["wtype", "rank", "mb", "cid", "irname"])
Tensor = namedtuple("Tensor", ["wtype", "rank", "mb", "tid", "v"])


class NNScalerDFG(DFG):
    def __init__(self, W: World):
        self._W = W
        self._path: str | Path = None
        self._nodes: List[Node] = []
        self._node2dtag: Dict[Node, DTag] = {}
        self._node2kwargs: Dict[Node, Dict] = {}
        self._node2opname: Dict[Node, OpName] = {}
        self._node2inputs: Dict[Node, List[Tensor]] = {}
        self._node2outputs: Dict[Node, List[Tensor]] = {}
        self._tensors: Set[Tensor] = set()
        self._tid2shape: Dict[int, List[int]] = {}
        self._initialized_tid: Set[int] = set()
        self._gid2wid: Dict[int, int] = {}
        self._ranktid2maxv: Dict[Tuple[int, int], int] = {}
        self._node2irstr: Dict[Node, str] = {}
        self._tid2lv: Dict[Tensor, LineageView] = {}
        self._shared_tensor_list: Dict[Hashable, List[Tensor]] = {}

        # from nnscaler.execplan.execplan import IRCell
        # from nnscaler.ir.adapter.prim import IRAdapterPrim
        # from nnscaler.ir.tensor import IRSubTensor

        # self._node2ir: Dict[Node, IRCell | IRAdapterPrim | None] = {}
        # self._tensor2ir: Dict[Tensor, IRSubTensor | None] = {}

    @property
    @final
    def W(self) -> World:
        return self._W

    @property
    @final
    def ID(self) -> str:
        return Path(self._path).stem

    @final
    def nodes(self) -> List[Node]:
        return self._nodes

    @final
    def node_dtag(self, node: Node) -> DTag:
        return self._node2dtag[node]

    @final
    def node_kwargs(self, node: Node) -> Dict:
        return self._node2kwargs.get(node, {})

    @final
    def node_inputs(self, node: Node) -> List[Tensor]:
        return self._node2inputs[node]

    @final
    def node_outputs(self, node: Node) -> List[Tensor]:
        return self._node2outputs[node]

    @final
    def node_opname(self, node: Node) -> OpName:
        return self._node2opname[node]

    @final
    def tensors(self) -> List[Tensor]:
        return self._tensors

    @final
    def tensor_shape(self, tensor: Tensor) -> List[int]:
        return tuple(int(x) for x in self._tid2shape[tensor.tid])

    @final
    def is_initialized(self, tensor: Tensor) -> bool:
        return tensor.tid in self._initialized_tid


def rank_to_dp(rank: int, W: World) -> int:
    return rank // W.plan_ndevs


def rank_to_tp(rank: int, W: World) -> int:
    return (rank % W.plan_ndevs) % W.num_tp


def rank_to_pp(rank: int, W: World) -> int:
    return (rank % W.plan_ndevs) // W.num_tp
