#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from typing import List, Dict, Tuple
import numpy as np

import z3
from functools import lru_cache

from verdict.graph import DFG, Node, Tensor, Lineage, WType
from verdict.config import Config
from verdict.operators import get_op, OpName
from verdict.log import logwarn
from verdict.symbolics import (
    SymbExpr,
    SymbShape,
    Shape,
    create_z3_tensor,
    concrete_z3,
    equalize_z3tensors,
)

from .stage import Stage


class ShapeReductionZ3:
    def __init__(self):
        self.ctx = z3.Context()

    def create_symbolic_shape(self, tensor_name: str, ndim: int) -> SymbShape:
        return create_z3_tensor(
            shape=(ndim,), prefix=f"{tensor_name}.D", dtype=z3.Int, ctx=self.ctx
        )

    def slice_sub_shape(
        self, full_symb: SymbShape, full_shape: Shape, sub_shape: Shape
    ) -> Tuple[SymbShape, List[SymbExpr]]:
        sub_symb: SymbShape = []
        constraints = []
        assert len(full_shape) == len(sub_shape)
        for d_full, d_sub, d in zip(full_shape, sub_shape, full_symb):
            assert d_full % d_sub == 0
            sub_symb.append(d * d_sub / d_full)
            constraints.append((d * d_sub) % d_full == 0)
        sub_symb = np.array(sub_symb)
        return sub_symb, constraints

    def enforce_shape_validity(self, symb_shape: SymbShape) -> List[SymbExpr]:
        return [d >= 1 for d in symb_shape]

    def init_shapes_from_lineage(
        self, l: Lineage, Gs: DFG, Gp: DFG
    ) -> Tuple[Dict[Tensor, SymbShape], List[SymbExpr]]:
        shapebook: Dict[Tensor, SymbShape] = {}
        constraints: List[SymbExpr] = []
        # init Ts symbolic shape
        Ts = l.Ts
        Ts_shape = Gs.tensor_shape(Ts)
        Ts_symb_shape = self.create_symbolic_shape(str(Ts), len(Ts_shape))
        shapebook[Ts] = Ts_symb_shape
        constraints.extend(self.enforce_shape_validity(Ts_symb_shape))
        
        @lru_cache()
        def _slice_sub_shape(_Tp_shape):
            return self.slice_sub_shape(
                Ts_symb_shape, Ts_shape, _Tp_shape
            )
            
        # init Tp symbolic shapes by slicing on Ts shapes
        for Tp in l.Tps:
            Tp_shape = Gp.tensor_shape(Tp)
            Tp_symb_shape, cons = _slice_sub_shape(Tp_shape)
            constraints.extend(cons)
            shapebook[Tp] = Tp_symb_shape
            constraints.extend(self.enforce_shape_validity(Tp_symb_shape))
        return shapebook, constraints

    def minimize(
        self, shapes: Dict[Tensor, SymbShape], constraints: List[SymbExpr]
    ) -> Dict[Tensor, Shape]:
        # deduplicate and simplify constraints
        def dedup_constraints(constraints: list[z3.ExprRef]) -> list[z3.ExprRef]:
            seen = {}
            for c in constraints:
                norm = z3.simplify(c)  # optional but improves equality
                if hash(norm) not in seen:
                    seen[hash(norm)] = norm
            return list(seen.values())
        constraints = dedup_constraints(constraints)
        goal = z3.Goal(ctx=self.ctx)
        goal.add(*constraints)
        tactic = z3.Then(
            z3.Tactic("ctx-simplify", ctx=self.ctx), z3.Tactic("lia2pb", ctx=self.ctx)
        )
        simplified_constraints = tactic(goal)
        # minimize goals
        opt = z3.Optimize(ctx=self.ctx)
        for subgoal in simplified_constraints:
            opt.add(subgoal)
        # NOTE: this objective is an approximation of solver complexity
        # we use sum(shape) instead of prod(shape) on purpose for solver efficiency
        objective = sum([sum(shape) for shape in shapes.values()])
        opt.minimize(objective)
        sat = opt.check()
        assert sat == z3.sat, sat

        # emit shapes
        rxshapes = {}
        model = opt.model()
        for t, symb_shape in shapes.items():
            rxshapes[t] = concrete_z3(symb_shape, model)
        return rxshapes


# NOTE: To adapt to a new rxshape engine, implement a class with methods
# the same as ShapeReductionZ3, and register here.
def _get_shape_reduction_engine(backend: str) -> ShapeReductionZ3:
    engines = {"z3": ShapeReductionZ3}
    return engines[backend]()


def shape_reduction(
    stage: Stage,
    Gs: DFG,
    Gp: DFG,
) -> Dict[Tensor, Shape]:
    """Shape reduction of a stage implemented with z3."""
    sse = _get_shape_reduction_engine(Config.rxshape_backend)
    shapebook: Dict[Tensor, SymbShape] = {}
    constraints: List[SymbExpr] = []
    
    # initialize input shapes enforcing input lineages
    for l in stage.input_lineages:
        shapes, shape_constraints = sse.init_shapes_from_lineage(l, Gs, Gp)
        shapebook.update(shapes)
        constraints.extend(shape_constraints)
    
    # An adhoc wred-specific shortcut for performance optimization
    is_wred_stage = len(stage.snodes) == 0 and all([Gp.node_opname(node) == OpName.CROSS_DP_WRED for node in stage.pnodes])
    if is_wred_stage:
        # TODO: currently bound to z3 as backend
        assert Config.rxshape_backend == "z3"
        input_tensors = set([t for node in stage.pnodes for t in Gp.node_inputs(node)])
        output_tensors = [t for node in stage.pnodes for t in Gp.node_outputs(node)]
        input_shapes = [shapebook[t] for t in input_tensors]
        shapebook.update({t:input_shapes[0] for t in output_tensors})
        constraints.extend(equalize_z3tensors(input_shapes))
    else:
        # apply constraints from ops
        for nodes, G in zip([stage.snodes, stage.pnodes], [Gs, Gp]):
            for node in nodes:
                opname = G.node_opname(node)
                op = get_op(opname)
                new_shapes, op_constraints = op.infer_rxshape(
                    node, G, shapebook, Config.rxshape_backend, sse.ctx
                )
                shapebook.update(new_shapes)
                constraints.extend(op_constraints)
        # apply output lineages
        for l in stage.output_lineages:
            shapes, shape_constraints = sse.init_shapes_from_lineage(l, Gs, Gp)
            constraints.extend(shape_constraints)
            for out_t in shapes:
                constraints.extend(equalize_z3tensors([shapes[out_t], shapebook[out_t]]))
            shapebook.update(shapes)
            
    result = sse.minimize(shapebook, constraints)
    
    return result
