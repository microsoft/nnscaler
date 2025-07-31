#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from typing import List, Tuple, Dict

import z3

from verdict.graph import DFG, Node, Tensor, Lineage, SliceMap
from verdict.backend import SymbolicBackend
from verdict.operators import get_op
from verdict.symbolics import SymbTensor, SymbExpr, Shape, SymbolCtx, create_z3_tensor

from .core import express_lineage, check_always_hold


class z3Backend(SymbolicBackend):
    @staticmethod
    def create_ctx() -> SymbolCtx:
        """Create a solver environment."""
        return z3.Context()

    @staticmethod
    def symbolize(T: Tensor, shape: Shape, ctx: SymbolCtx) -> SymbTensor:
        """Create a symbolic tensor."""
        return create_z3_tensor(shape, str(T), z3.Real, ctx)

    @staticmethod
    def apply_op(
        node: Node,
        G: DFG,
        data: Dict[Tensor, SymbTensor],
        shapes: Dict[Tensor, Shape],
        ctx: SymbolCtx,
    ) -> Dict[Tensor, SymbTensor]:
        """Apply an operator."""
        APPLY_OP_BACKEND = "z3"
        op = get_op(G.node_opname(node))
        new_data = op.apply_op(node, G, data, shapes, APPLY_OP_BACKEND, ctx)
        return new_data

    @staticmethod
    def express_lineage(
        lineage: Lineage, symb_tensors: Dict[Tensor, SymbTensor], ctx: SymbolCtx
    ) -> Dict[SliceMap, List[SymbExpr]]:
        """Express symbolic equivalence for a lineage."""
        return express_lineage(lineage, symb_tensors, ctx)

    @staticmethod
    def check_always_hold(
        given: List[SymbExpr], grouped_always_hold: List[List[SymbExpr]], ctx: SymbolCtx
    ) -> bool:
        """Check whether a set of expressions always hold given known conditions."""
        return check_always_hold(given, grouped_always_hold, ctx)
