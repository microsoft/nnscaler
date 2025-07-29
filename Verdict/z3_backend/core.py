from typing import List, Tuple, Dict, Any
from typing import Dict
import z3


from verdict.graph import Node, DFG, Tensor, Lineage, SliceMap
from verdict.log import logdebug
from verdict.symbolics import SymbTensor, SymbExpr, Shape, SymbolCtx, equalize_z3tensors
from verdict.report import report


def express_lineage(
    l: Lineage, symb_tensors: Dict[Tensor, SymbTensor], ctx: SymbolCtx
) -> Dict[SliceMap, List[SymbExpr]]:
    """Express symbolic equivalence for a lineage."""
    # NOTE: we currently assume full shape can be divided by reduced shapes
    Zs = symb_tensors[l.Ts]
    dim_scales = [l.full_shape[d] // Zs.shape[d] for d in range(len(l.full_shape))]
    slcmap_cons: Dict[SliceMap, List[SymbExpr]] = {}
    for org_slc, eq_copies in l.slice_map.items():
        rx_slcmap = tuple(
            (start // scale, end // scale)
            for (start, end), scale in zip(org_slc, dim_scales)
        )
        rx_slc = tuple(slice(start, end) for (start, end) in rx_slcmap)
        Zs_shard = Zs[rx_slc]
        for reduce_Tps in eq_copies:
            Zp = sum([symb_tensors[Tp] for Tp in reduce_Tps])
            cons = equalize_z3tensors([Zs_shard, Zp])
            slcmap_cons.setdefault(rx_slcmap, []).extend(cons)
    return slcmap_cons


def _tactic_check_unsat(constraints, ctx) -> bool:
    g = z3.Goal(ctx=ctx)
    g.add(*constraints)
    simplify = z3.Tactic("solve-eqs", ctx)
    simplified_goal = simplify(g)
    all_unsat = all(str(subgoal) == "[False]" for subgoal in simplified_goal)
    return all_unsat


def _solver_check_unsat(
    given: List[SymbExpr], grouped_always_hold: List[List[SymbExpr]], ctx: SymbolCtx
) -> bool:
    solver = z3.Solver(ctx=ctx)
    solver.add(*given)
    for group in grouped_always_hold:
        solver.push()
        neg_group = z3.simplify(z3.Not(z3.And(*group)))
        solver.add(neg_group)
        sat_result = solver.check()
        if sat_result != z3.unsat:
            report.sat = sat_result
            report.model = solver.model()
            return False
        solver.pop()
    return True


def check_always_hold(
    given: List[SymbExpr], grouped_always_hold: List[List[SymbExpr]], ctx: SymbolCtx
) -> bool:
    """Check whether a set of expressions always hold given known conditions."""
    always_hold = [expr for group in grouped_always_hold for expr in group]
    logdebug("... z3_backend.core.check_always_hold.neg_always_hold")
    neg_always_hold = z3.simplify(z3.Not(z3.And(*always_hold)))

    is_const_unsat = bool(neg_always_hold == False)
    if is_const_unsat:
        return True

    logdebug("... z3_backend.core.check_always_hold.tactic_check")
    tactic_unsat = _tactic_check_unsat(given + [neg_always_hold], ctx)
    if tactic_unsat:
        return True

    logdebug("... z3_backend.core.check_always_hold.solver_check")
    solver_unsat = _solver_check_unsat(given, grouped_always_hold, ctx)
    return solver_unsat
