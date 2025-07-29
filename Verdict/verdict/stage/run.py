from typing import List, Dict

import numpy as np

from verdict.graph import DFG, Node, Tensor, Lineage
from verdict.symbolics import SymbTensor, SymbExpr
from verdict.backend import SymbolicBackend
from verdict.operators import OpName
from verdict.debug_print import dump_stages
from verdict.log import logerr, logdebug
from verdict.report import report
from verdict.timer import Timer

from .stage import Stage
from .rxshape import Shape, shape_reduction


def run_stage(
    stage: Stage, Gs: DFG, Gp: DFG, symbolic_backend: SymbolicBackend
) -> bool:
    # check stage by rules instead of by symbolics
    # check stage by symbolic execution
    return _run_stage_symbolically(stage, Gs, Gp, symbolic_backend)


# TODO:
def _assert_check_stage_guard(stage: Stage, Gs: DFG, Gp: DFG) -> None:
    pass
    # Embedding: require input to be only sliced across dpmb, and embedw is replicated


def _run_stage_symbolically(
    stage: Stage, Gs: DFG, Gp: DFG, symbolic_backend: SymbolicBackend
) -> bool:
    T = Timer()
    T.start(f"Stage: {stage.id} TOTAL")
    
    logdebug("Stage start",
             Stage=stage.id, 
             n_snodes=len(stage.snodes),
             n_pnodes=len(stage.pnodes),
             head_snodes=[Gs.node_opname(n).value for n in stage.snodes[:1]],
             head_pnodes=[Gp.node_opname(n).value for n in stage.pnodes[:1]]
             )
    _assert_check_stage_guard(stage, Gs, Gp)

    # get reduced shapes
    T.start(f"Stage: {stage.id} rxshape")
    logdebug("Reducing shapes.", Stage=stage.id)
    rxshapes: Dict[Tensor, Shape] = shape_reduction(stage, Gs, Gp)
    T.end(f"Stage: {stage.id} rxshape")

    # initialize input tensors
    T.start(f"Stage: {stage.id} init inputs")
    logdebug("Initializing inputs.", Stage=stage.id)
    symb_ctx = symbolic_backend.create_ctx()
    data: Dict[Tensor, SymbTensor] = {}
    for l in stage.input_lineages:
        for t in [l.Ts, *l.Tps]:
            if t not in data:
                data[t] = symbolic_backend.symbolize(t, rxshapes[t], symb_ctx)
    T.end(f"Stage: {stage.id} init inputs")

    # apply symbolic operators, record everything in data
    T.start(f"Stage: {stage.id} apply op")
    logdebug("Applying symbolic operators.", Stage=stage.id)
    op_cons = []
    for nodes, G in zip([stage.snodes, stage.pnodes], [Gs, Gp]):
        for node in nodes:
            new_data, cons = symbolic_backend.apply_op(
                node, G, data, rxshapes, symb_ctx
            )
            # check shape consistency
            for out_t, out_symbt in new_data.items():
                if list(rxshapes[out_t]) == list(out_symbt.shape):
                    continue
                logerr(
                    "Tensor shape mismatch.",
                    Expect=list(rxshapes[out_t]),
                    Result=list(out_symbt.shape),
                    Node=node,
                )
                raise
            data.update(new_data)
            op_cons.extend(cons)
    T.end(f"Stage: {stage.id} apply op")

    # input lineage
    T.start(f"Stage: {stage.id} input lng")
    logdebug("Enforcing input lineages.", Stage=stage.id)
    given: List[SymbExpr] = []
    for l in stage.input_lineages:
        slcmap_cons = symbolic_backend.express_lineage(l, data, symb_ctx)
        given.extend([expr for cons in slcmap_cons.values() for expr in cons])
    T.end(f"Stage: {stage.id} input lng")

    # output lineage
    T.start(f"Stage: {stage.id} out lng")
    logdebug("Enforcing output lineages.", Stage=stage.id)
    grouped_always_hold: List[List[SymbExpr]] = []
    for l in stage.output_lineages:
        slcmap_cons = symbolic_backend.express_lineage(l, data, symb_ctx)
        grouped_always_hold.extend(list(slcmap_cons.values()))
    T.end(f"Stage: {stage.id} out lng")

    # check equivalence
    T.start(f"Stage: {stage.id} check eq")
    logdebug("Checking stage equivalence.", Stage=stage.id)
    always_hold = symbolic_backend.check_always_hold(
        given + op_cons, grouped_always_hold, symb_ctx
    )
    logdebug("Stage result.", Stage=stage.id, result=always_hold)
    T.end(f"Stage: {stage.id} check eq")

    T.end(f"Stage: {stage.id} TOTAL")
    T.display()
    # emit report if fails
    if not always_hold:
        report.stage = stage
        report.Gs = Gs
        report.Gp = Gp
        report.data = data
        logerr(report.dump_z3())
        raise RuntimeError(f"Stage {stage.id} equivalence fails.")
    return always_hold
