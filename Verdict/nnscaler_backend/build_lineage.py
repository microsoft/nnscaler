from typing import List, Dict, Tuple, Hashable, Set
from collections import defaultdict

from verdict.graph import Lineage, SliceMap, ReduceTensors
from verdict.operators import OpName
from verdict.utils import unique, idempotent_update

from .dfg import (
    NNScalerDFG,
    Node,
    Tensor,
    rank_to_dp,
    rank_to_tp,
    rank_to_pp,
)


def _is_original_op(opname: OpName) -> bool:
    # communication operator
    if opname.value[0].endswith("Prim"):
        return False
    # grdient finalization
    if opname in [OpName.LOCAL_GRAD_ACCUM, OpName.CROSS_DP_WRED]:
        return False
    # identity & multiref
    if opname.value[0] in ["identity", "multiref"]:
        return False
    return True


def _is_wred(opname: OpName) -> bool:
    return opname == OpName.CROSS_DP_WRED


# Reorganize parallelized graph as the following view
# DP0 TP0 MB0: FW..., BW...
#         MB1: FW..., BW...
#     TP1 MB0: FW..., BW...
#         MB1: FW..., BW...
# DP1 TP0 MB0: FW..., BW...
#         MB1: FW..., BW...
#     TP1 MB0: FW..., BW...
#         MB1: FW..., BW...
def _reorganize_Gp_nodes(nodes: List[Node], dfg: NNScalerDFG) -> List[List[List[Node]]]:
    W = dfg.W

    # a helper function to init multi-level dict
    def nested_grid():
        return defaultdict(lambda: nested_grid())

    # index nodes by dp, tp, pp, mb, fw
    indexed_nodes: Dict[Tuple[int, int, int, int, bool], List[Node]] = {}
    for node in nodes:
        dp = rank_to_dp(node.rank, W)
        tp = rank_to_tp(node.rank, W)
        pp = rank_to_pp(node.rank, W)
        mb = node.mb
        fw = dfg.node_opname(node).value[1]
        assert isinstance(fw, bool)
        key = (dp, tp, pp, mb, fw)
        indexed_nodes.setdefault(key, []).append(node)

    # explicitly init grid
    grid: List[List[List[List[Node]]]] = [
        [[[] for _ in range(W.num_mb)] for _ in range(W.num_tp)]
        for _ in range(W.num_dp)
    ]

    # feed indexed nodes into the grid
    for dp in range(W.num_dp):
        for tp in range(W.num_tp):
            for mb in range(W.num_mb):
                target = grid[dp][tp][mb]
                # add in sequential order:
                # fw.pp0, ..., fw.pp-1, bw.pp-1, ..., bw.pp0
                for pp in range(W.num_pp):
                    target.extend(indexed_nodes[(dp, tp, pp, mb, True)])
                for pp in range(W.num_pp - 1, -1, -1):
                    target.extend(indexed_nodes[(dp, tp, pp, mb, False)])
    return grid


class AlignedNodesWithLineages:
    def __init__(self):
        self.snode: Node = None
        self.pnodes: List[Node] = []
        self.input_lineages: List[Lineage] = []
        self.output_lineages: List[Lineage] = []


def _infer_lineages_from_alignable_ops(
    Gs_alignable_ops: List[Node],
    Gp_alignable_ops: List[Node],
    Gs: NNScalerDFG,
    Gp: NNScalerDFG,
) -> List[Lineage]:
    Gp_aglinable_op_grid = _reorganize_Gp_nodes(Gp_alignable_ops, Gp)
    # align nodes into bundles
    aligned_ops: List[AlignedNodesWithLineages] = []
    for node_ptr, snode in enumerate(Gs_alignable_ops):
        bundle = AlignedNodesWithLineages()
        bundle.snode = snode
        # node alignment
        for dp in range(Gp.W.num_dp):
            for tp in range(Gp.W.num_tp):
                for mb in range(Gp.W.num_mb):
                    bundle.pnodes.append(Gp_aglinable_op_grid[dp][tp][mb][node_ptr])
        # tensor alignment of inputs
        for input_ptr, Ts in enumerate(Gs.node_inputs(snode)):
            Tps = [Gp.node_inputs(pnode)[input_ptr] for pnode in bundle.pnodes]
            lineage = Lineage(Ts, Tps)
            bundle.input_lineages.append(lineage)
        # tensor alignment of outputs
        for output_ptr, Ts in enumerate(Gs.node_outputs(snode)):
            Tps = [Gp.node_outputs(pnode)[output_ptr] for pnode in bundle.pnodes]
            lineage = Lineage(Ts, Tps)
            bundle.output_lineages.append(lineage)
            # stage is determined by aligned ops
            lineage.is_stage_known = True
            lineage.snodes = [snode]
            lineage.pnodes = bundle.pnodes
            lineage.dependency = bundle.input_lineages
        aligned_ops.append(bundle)

    # emit lineage for each bundle's inputs and outputs
    lineages: List[Lineage] = []
    for bundle in aligned_ops:
        for l in bundle.input_lineages + bundle.output_lineages:
            lineages.append(l)
    return lineages


def _infer_lineages_from_wreds(
    wreds: List[Node],
    Gp_gid_to_Gs_grad: Dict[int, Tensor],
    Gs: NNScalerDFG,
    Gp: NNScalerDFG,
) -> List[Lineage]:
    # NOTE: we want to consolidate wreds that finalize the same gradient counterparts,
    # which helps enforces `ordered` lineages; else the cut_stage may behave unexpectedly
    Gs_grad_to_unreduced_inputs: Dict[Tensor, Set[Tensor]] = {}
    Gs_grad_to_reduced_outputs: Dict[Tensor, Set[Tensor]] = {}
    Gs_grad_to_wred: Dict[Tensor, List[Node]] = {}
    for wred in wreds:
        wred_inputs = Gp.node_inputs(wred)
        wred_outputs = Gp.node_outputs(wred)
        # wred should match a single weight grad in Gs
        Gs_grad = unique([Gp_gid_to_Gs_grad[t.tid] for t in wred_inputs])
        # emit lineage for wreds
        Gs_grad_to_unreduced_inputs.setdefault(Gs_grad, set()).update(set(wred_inputs))
        Gs_grad_to_reduced_outputs.setdefault(Gs_grad, set()).update(set(wred_outputs))
        Gs_grad_to_wred.setdefault(Gs_grad, []).append(wred)

    lineages: List[Lineage] = []
    for Gs_grad in Gs_grad_to_unreduced_inputs:
        input_lineage = Lineage(Gs_grad, Gs_grad_to_unreduced_inputs[Gs_grad])
        lineages.append(input_lineage)
        output_lineage = Lineage(Gs_grad, Gs_grad_to_reduced_outputs[Gs_grad])
        lineages.append(output_lineage)
        output_lineage.is_stage_known = True
        output_lineage.snodes = []
        output_lineage.pnodes = Gs_grad_to_wred[Gs_grad]
        output_lineage.dependency = [input_lineage]
    return lineages


def _explore_final_lineages(
    Gs_grad_to_lineage: Dict[Tensor, Lineage], Gs: NNScalerDFG, Gp: NNScalerDFG
) -> List[Lineage]:
    """This is to discover the final lineages for weight grads that is produced
    by gradient accumulation, but don't need the cross-dp wreducers."""
    # enforce all the weight gradients are finalized,
    # and respect parallelization semantics
    explored_final_lineages: List[Lineage] = []
    for Gs_grad in Gs_grad_to_lineage:
        # if the lineage is not a final lineage, craft the final lineage
        # by advancing the tensor versions for Tps
        l = Gs_grad_to_lineage[Gs_grad]
        Tps: List[Tensor] = l.Tps
        is_final_lineage = all(
            Tp.v == Gp._ranktid2maxv[(Tp.rank, Tp.tid)] for Tp in Tps
        )
        if not is_final_lineage:
            final_Tps = [
                Tensor(Tp.rank, -1, Tp.tid, Gp._ranktid2maxv[(Tp.rank, Tp.tid)])
                for Tp in Tps
            ]
            explored_final_lineages.append(Lineage(l.Ts, final_Tps))
    return explored_final_lineages


def get_ordered_lineages(Gs: NNScalerDFG, Gp: NNScalerDFG) -> List[Lineage]:
    """Get the list of lineage in visting order of node execution."""
    # prepare nodes
    Gs_alignable_ops: List[Node] = []
    Gp_alignable_ops: List[Node] = []
    reused_Ts: Set[Tensor] = set()
    wreds: List[Node] = []
    for node in Gs.nodes():
        reused_Ts.update(set(Gs.node_inputs(node)))
        reused_Ts.update(set(t for t in Gs.node_outputs(node) if t.mb == -1))
        if _is_original_op(Gs.node_opname(node)):
            Gs_alignable_ops.append(node)
    for node in Gp.nodes():
        opname = Gp.node_opname(node)
        if _is_original_op(opname):
            Gp_alignable_ops.append(node)
        elif _is_wred(opname):
            wreds.append(node)

    # infer lineages from alignable ops
    lineages_from_aligned_ops = _infer_lineages_from_alignable_ops(
        Gs_alignable_ops, Gp_alignable_ops, Gs, Gp
    )

    # prepare Gs gradient mappings
    Gs_grad_to_lineage: Dict[Tensor, Lineage] = {}
    Gp_gid_to_Gs_grad: Dict[int, Tensor] = {}
    for l in lineages_from_aligned_ops:
        Gs_grad: Tensor = l.Ts
        is_Ts_weight_grad = Gs_grad.tid in Gs._gid2wid
        if is_Ts_weight_grad:
            Gs_grad_to_lineage[Gs_grad] = l
            for Tp in l.Tps:
                Tp: Tensor = Tp
                Gp_gid_to_Gs_grad[Tp.tid] = Gs_grad

    # emit lineages for weight reducers; refresh finalized Gs grads
    lineages_from_wreds = _infer_lineages_from_wreds(wreds, Gp_gid_to_Gs_grad, Gs, Gp)
    Gs_grad_to_lineage.update({l.Ts: l for l in lineages_from_wreds})

    # emit lineages for non-final lineages; refresh finalized Gs grads
    lineages_final = _explore_final_lineages(Gs_grad_to_lineage, Gs, Gp)
    Gs_grad_to_lineage.update({l.Ts: l for l in lineages_final})

    # prune lngs if the lineage is not used and is not weight' gradients
    pruned_lineages_from_aligned_ops = [
        l for l in lineages_from_aligned_ops if l.Ts in reused_Ts or l.Ts.mb == -1
    ]

    for lngs, src in zip(
        [pruned_lineages_from_aligned_ops, lineages_from_wreds, lineages_final],
        # [False, True, True],
        ["OP", "WRED", "FINAL"],
    ):
        for l in lngs:
            l.slice_map = _get_lng_slcmap(l, Gs, Gp, src)
            l.full_shape = Gs.tensor_shape(l.Ts)
            l.src = src

    # check semantics for final lineage to respect parallelization consistency
    for Gs_grad in Gs_grad_to_lineage:
        Ts_shape = Gs.tensor_shape(l.Ts)
        Ts_slices = list(l.slice_map.keys())
        assert _dim_cover_check(Ts_slices, Ts_shape)
    lineages = pruned_lineages_from_aligned_ops + lineages_from_wreds + lineages_final
    
    dedup_lngs = []
    visited = set()
    for l in lineages:
        if l not in visited:
            visited.add(l)
            dedup_lngs.append(l)
    return dedup_lngs


def _dim_cover_check(slices_list: List[List[slice]], shape: List[int]) -> bool:
    for dim, dim_size in enumerate(shape):
        covered = set()
        for sl in slices_list:
            s = sl[dim]
            start = s[0] or 0
            stop = s[1] if s[1] is not None else dim_size
            step = 1
            covered.update(range(start, stop, step))
        if set(range(dim_size)) != covered:
            return False
    return True


def _get_lng_slcmap(
    l: Lineage, Gs: NNScalerDFG, Gp: NNScalerDFG, src: str
) -> Dict[SliceMap, List[ReduceTensors]]:
    """Inplace set lng slice map."""
    # variable facility
    total_dp, total_mb = Gp.W.num_dp, Gp.W.num_mb
    Ts: Tensor = l.Ts
    Tps: List[Tensor] = l.Tps
    Ts_lv = Gs._tid2lv[Ts.tid]
    Tp_lvs = [Gp._tid2lv[Tp.tid] for Tp in Tps]
    s_ft_shape = Ts_lv.ft_shape
    p_ft_shape = unique([lv.ft_shape for lv in Tp_lvs])

    # flag_semantics
    is_parent_shape_eq = s_ft_shape == p_ft_shape

    # map each Tp to the Ts slice
    slc2Tps: Dict[SliceMap, List[Tensor]] = {}
    require_dl_shift = False
    dl_shidf_dim = None
    if not is_parent_shape_eq:
        require_dl_shift = True
        dl_shidf_dim = unique(
            [i for i in range(len(s_ft_shape)) if s_ft_shape[i] != p_ft_shape[i]]
        )
        assert (
            s_ft_shape[dl_shidf_dim] == p_ft_shape[dl_shidf_dim] * total_dp * total_mb
        ), f"shapes violate assumption that dp&nm should have/have-not the same SLC effect {l}"
    for Tp, lv in zip(Tps, Tp_lvs):
        slcmap = lv.slcmap
        if require_dl_shift:
            assert Tp.mb != -1, f"Unexpected dpmb slice on weight {Tp}."
            dp = rank_to_dp(Tp.rank, Gp.W)
            batch_id = dp * total_mb + Tp.mb
            start, end = slcmap[dl_shidf_dim]
            offset = batch_id * lv.ft_shape[dl_shidf_dim]
            slcmap = tuple(
                [
                    slc if d != dl_shidf_dim else (start + offset, end + offset)
                    for d, slc in enumerate(slcmap)
                ]
            )
        slc2Tps.setdefault(slcmap, []).append(Tp)

    # for each slice's Tps, group them into equal copies
    slc2rts: Dict[SliceMap, List[ReduceTensors]] = {}
    is_fw = not Ts_lv.is_grad
    # NOTE: adhoc patch for created mask as attr
    is_mask = Gs.tensor_shape(l.Ts)[:2] == (1, 1)
    is_attr = Ts_lv.is_attr or is_mask
    is_loss = Ts_lv.is_loss
    is_actv = not is_loss and not is_attr
    is_wred_lng = src == "WRED"

    # for each sharding
    for slc, slc_Tps in slc2Tps.items():
        # (dp,mb) -> [*tp]
        dpmb2Tps: Dict[Tuple[int, int], List[Tensor]] = _group_by_dpmb(slc_Tps, Gp)
        """
        FW & BW ACTIVATIONS / FW LOSS
        NOTE:
            - mb can only serve as slice, not reduction,
            so dpmb- and dp- grouping is effectively the same,
            - across dp/dpmb: must remain replicated
            - within dp/dpmb: reduction indicated by valmap
        """
        if (is_fw and is_loss) or is_actv:
            unique([Tp.mb for Tp in slc_Tps])  # mb shall only serve as slice
            eq_copies = []
            for (dp, mb), dpmb_Tps in dpmb2Tps.items():
                n_vparts = unique([Gp._tid2lv[Tp.tid].valmap[1] for Tp in dpmb_Tps])
                if n_vparts == 1:
                    eq_copies.extend([[Tp] for Tp in dpmb_Tps])
                else:
                    assert n_vparts == Gp.W.num_tp
                    eq_copies.extend([dpmb_Tps])
            slc2rts[slc] = eq_copies

        """
        FW WEIGHTS / BW LOSS
        NOTE: 
            - fw weights and bw losses are initialized, thus 
            replicated in each sharding no matter dp or mb
        """
        if (is_fw and is_attr) or (not is_fw and is_loss):
            eq_copies = [[Tp] for Tp in slc_Tps]
            slc2rts[slc] = eq_copies

        """
        BW ATTRS
        NOTE: 
            - bw weight grads are all mb=-1, but differentiated with version
            - they should first reduce locally, then reduce cross-dp
        """
        if is_attr and not is_fw:
            assert unique([Tp.mb for Tp in slc_Tps]) == -1
            # identify cross-dp wred group, and merge on condition
            # the comm_group key is (rank, tid)
            comm_group_map: Dict[Tuple[int, int], Hashable] = (
                _get_wred_inverse_index_comm_group(Gp)
            )
            if is_wred_lng:
                is_wred_output = unique(
                    [Tp.v == Gp._ranktid2maxv[(Tp.rank, Tp.tid)] for Tp in slc_Tps]
                )
                if is_wred_output:
                    eq_copies = [[Tp] for Tp in slc_Tps]
                else:
                    eq_copies = list(
                        _group_by(
                            slc_Tps, lambda Tp: comm_group_map[(Tp.rank, Tp.tid)]
                        ).values()
                    )
            else:
                # merge mbs (grad accumulation)
                rank2Tps: Dict[int, List[Tensor]] = _group_by(
                    slc_Tps, lambda Tp: Tp.rank
                )
                # each item in comm2Tps is an equal copy
                comm2Tps = {}
                for rank, rank_Tps in rank2Tps.items():
                    tid = unique([Tp.tid for Tp in rank_Tps])
                    comm_group = comm_group_map[(rank, tid)]
                    comm2Tps.setdefault(comm_group, []).extend(rank_Tps)
                eq_copies = list(comm2Tps.values())
            slc2rts[slc] = eq_copies

            # print(">>")
            # for arr in eq_copies:
            #     print(arr)
    return slc2rts


def _group_by(arr, key_func) -> Dict:
    ret = {}
    for x in arr:
        key = key_func(x)
        ret.setdefault(key, []).append(x)
    return ret


def _group_by_dpmb(
    Tps: List[Tensor], Gp: NNScalerDFG
) -> Dict[Tuple[int, int], List[Tensor]]:
    key_func = lambda Tp: (rank_to_dp(Tp.rank, Gp.W), Tp.mb)
    return _group_by(Tps, key_func)


def _get_wred_inverse_index_comm_group(
    Gp: NNScalerDFG,
) -> Dict[Tuple[int, int], Hashable]:
    if not hasattr(Gp, "_cached_wred_inv_index"):
        shared_tensor_list = Gp._shared_tensor_list
        ret = {}
        for key, tensors in shared_tensor_list.items():
            if len(key) != 2:
                continue
            for t in tensors:
                idempotent_update(ret, {(t.rank, t.tid): key})
        Gp._cached_wred_inv_index = ret
    return Gp._cached_wred_inv_index
