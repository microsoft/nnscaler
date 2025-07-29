from typing import List

from verdict.graph import DFG, WType, Lineage

from nnscaler_backend.load_graph import load_graph
from nnscaler_backend.build_lineage import get_ordered_lineages


class nnScalerGraphBackend:
    @staticmethod
    def load_graph(G_path: str, W_path: str, wtype: WType | str) -> DFG:
        """Return an SSA graph representing a model."""
        return load_graph(G_path, W_path, wtype)

    @staticmethod
    def get_ordered_lineages(Gs: DFG, Gp: DFG) -> List[Lineage]:
        """Return all lineages to guide stage partitioning and verification."""
        return get_ordered_lineages(Gs, Gp)
