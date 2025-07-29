from typing import List, Tuple, Dict
from abc import ABC, abstractmethod

from verdict.graph import WType, DFG, Node, Tensor, Lineage, SliceMap
from verdict.symbolics import SymbTensor, SymbExpr, SymbolCtx


class GraphBackend(ABC):
    @staticmethod
    @abstractmethod
    def load_graph(G_path: str, W_path: str, wtype: WType | str) -> DFG:
        """Return an SSA graph representing a model."""

    @staticmethod
    @abstractmethod
    def get_ordered_lineages(Gs: DFG, Gp: DFG) -> List[Lineage]:
        """Return all lineages to guide stage partitioning and verification."""


class SymbolicBackend(ABC):
    @staticmethod
    @abstractmethod
    def create_ctx(*args, **kwargs) -> SymbolCtx:
        """Create a solver environment."""

    @staticmethod
    @abstractmethod
    def symbolize(T: Tensor, shape: Tuple[int], ctx: SymbolCtx) -> SymbTensor:
        """Create a symbolic tensor."""

    @staticmethod
    @abstractmethod
    def apply_op(
        node: Node, G: DFG, data: Dict[Tensor, SymbTensor], ctx: SymbolCtx
    ) -> Dict[Tensor, SymbTensor]:
        """Apply an operator."""

    @staticmethod
    @abstractmethod
    def express_lineage(
        lineage: Lineage, symb_tensors: Dict[Tensor, SymbTensor], ctx: SymbolCtx
    ) -> Dict[SliceMap, List[SymbExpr]]:
        """Express symbolic equivalence for a lineage."""

    @staticmethod
    @abstractmethod
    def check_always_hold(
        given: List[SymbExpr], grouped_always_hold: List[List[SymbExpr]], ctx: SymbolCtx
    ) -> bool:
        """Check whether a set of expressions always hold given known conditions."""
