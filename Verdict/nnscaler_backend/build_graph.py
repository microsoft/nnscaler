#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from typing import List, Dict, Tuple, Set, Hashable, Any
from collections import defaultdict
from copy import copy
from multiprocessing import Pool
from tqdm import tqdm
from pathlib import Path
import pickle

import nnscaler.ir.adapter.prim as PTypes
from nnscaler.codegen.module.module import ModuleCodeGen
from nnscaler.ir.adapter import IRWeightReducer, IRAdapter
from nnscaler.ir.operator import IRDataOperation, IRBpOperation, IRFwOperation
from nnscaler.ir.adapter.prim import ChunkPrim, IRAdapterPrim
from nnscaler.graph.graph import IRSegment
from nnscaler.graph.function.dimops import IRDimops
from nnscaler.codegen.module.module import ModuleCodeGen
from nnscaler.execplan.execplan import ExeReuseCell, IRCell
from nnscaler.ir.tensor import IRSubTensor, IRFullTensor

from verdict.operators import OpName, KW_CC_IDX, KW_CONSTS_KEY
from verdict.graph import World, WType, DTag
from verdict.utils import idempotent_update, unique
from verdict.config import Config
from verdict.log import loginfo

from .dfg import (
    NNScalerDFG,
    Node,
    Tensor,
    rank_to_pp,
    rank_to_dp,
    rank_to_tp,
    LineageView,
)


class Cell:
    def __init__(self, ir: IRCell, rank: int, wtype: WType):
        # Transient attributes
        self.ir: IRCell | None = ir
        self.adapter: IRAdapter | None = None
        self._input_irs: List[IRSubTensor] = []
        self._output_irs: List[IRSubTensor] = []
        self.wtype: WType = wtype
        self._wred_wid: int = None
        self._grad_accum_transition: List[Tuple[Tensor, Tensor, Tensor]] = []
        self._input_consts: List = []
        
        # Persistent attributes
        self.irstr: str = str(ir)
        self.rank: int = rank
        self.mb: int = 0

        self.node: Node = None
        self.opname: OpName = None
        self.inputs: List[Tensor] = []
        self.outputs: List[Tensor] = []
        self.tid_shapes: Dict[int, List[int]] = {}
        self.initialized_tid: Set[int] = set()
        self.kwargs: Dict = {}

        self._collective_group_id: Hashable = None
        self._collective_indmap: Dict[Tensor, Any] = {}
        self._gid2wid: Dict[Hashable] = {}
        self._tid2lv: Dict[Tensor, LineageView] = {}
    
    __slots__ = ("ir", "irstr", "rank", "wtype", "mb", "adapter",
                 "node", "opname", "inputs", "outputs", "tid_shapes",
                 "initialized_tid", "kwargs", "_input_irs", "_output_irs",
                 "_input_consts", "_grad_accum_transition", "_wred_wid",
                 "_collective_group_id", "_collective_indmap", "_gid2wid",
                 "_tid2lv")
    
    def __getstate__(self):
        return {
            "irstr": self.irstr,
            "rank": self.rank,
            "mb": self.mb,
            "node": self.node,
            "opname": self.opname,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "tid_shapes": self.tid_shapes,
            "initialized_tid": self.initialized_tid,
            "kwargs": self.kwargs,
            "_collective_group_id": self._collective_group_id,
            "_collective_indmap": self._collective_indmap,
            "_gid2wid": self._gid2wid,
            "_tid2lv": self._tid2lv,
        }
    
    def __setstate__(self, state):
        # Transient attributes
        self.ir = None
        self.adapter = None
        self._input_irs = []
        self._output_irs = []
        self.wtype = None
        self._wred_wid = None
        self._grad_accum_transition = []
        self._input_consts = []
        # Persistent attributes
        self.irstr = state["irstr"]
        self.rank = state["rank"]
        self.mb = state["mb"]
        self.node = state["node"]
        self.opname = state["opname"]
        self.inputs = state["inputs"]
        self.outputs = state["outputs"]
        self.tid_shapes = state["tid_shapes"]
        self.initialized_tid = state["initialized_tid"]
        self.kwargs = state["kwargs"]
        self._collective_group_id = state["_collective_group_id"]
        self._collective_indmap = state["_collective_indmap"]
        self._gid2wid = state["_gid2wid"]
        self._tid2lv = state["_tid2lv"]


DiffFusedAdapterPrims = (
    PTypes.AllReduceIdentityPrim,
    PTypes.IdentityAllreducePrim,
    PTypes.AllReduceAllReducePrim,
    PTypes.ReduceScatterAllGatherPrim,
    PTypes.AllGatherReduceScatterPrim,
    PTypes.AllGatherSplitPrim,
    PTypes.SplitAllGatherPrim,
    PTypes.AllToAllAllToAllPrim,
    PTypes.ReduceBroadcastPrim,
    PTypes.BroadcastRedducePrim,
)


def _flatten_exereuse_then_scale(
    cells: List[IRCell], mg: ModuleCodeGen, rank: int
) -> List[IRCell]:
    ret = []
    for ir in cells:
        if isinstance(ir, ExeReuseCell):
            ir = ir.cell
        ir = mg.scale(ir, rank)
        ret.append(ir)
    return ret


def _set_mb(cells: List[Cell]) -> List[Cell]:
    mb_counter: Dict[int, int] = defaultdict(int)  # cid -> mb
    for cell in cells:
        cid = cell.ir.cid
        cell.mb = mb_counter[cid]
        mb_counter[cid] += 1
    return cells


def _flatten_segment(cells: List[Cell]) -> List[Cell]:
    ret: List[Cell] = []
    for cell in cells:
        if isinstance(cell.ir, IRSegment):
            for node in cell.ir.nodes():
                new_cell = Cell(node, cell.rank, cell.wtype)
                new_cell.mb = cell.mb
                ret.append(new_cell)
        else:
            ret.append(cell)
    return ret


def _flatten_adapter(cells: List[Cell]) -> List[Cell]:
    ret: List[Cell] = []
    for cell in cells:
        if isinstance(cell.ir, IRAdapter):
            for prim in cell.ir.prims:
                new_cell = Cell(prim, cell.rank, cell.wtype)
                new_cell.mb = cell.mb
                new_cell.adapter = cell.ir
                ret.append(new_cell)
        else:
            ret.append(cell)
    return ret


def _remove_dummy_dataloader_redundant_identity(
    cells: List[Cell], W: World
) -> List[Cell]:
    ret: List[Cell] = []
    for cell in cells:
        rank = cell.rank
        pp = rank_to_pp(rank, W)
        if cell.opname is OpName.DATALOADER and pp > 0:
            continue
        if cell.opname in [OpName.IdentityPrim, OpName.FW_identity, OpName.BW_identity]:
            assert len(cell.inputs) == len(cell.outputs) == 1
            if cell.inputs[0] == cell.outputs[0]:
                continue
        ret.append(cell)
    return ret


def _assert_check_ir_types(cells: List[Cell]) -> List[Cell]:
    for cell in cells:
        assert isinstance(
            cell.ir,
            (IRDataOperation, IRDimops, IRBpOperation, IRAdapterPrim, IRWeightReducer),
        ), f"Unexpected IR type: {type(cell.ir)} of {cell.ir}"
    return cells


def _set_node_SSA(cells: List[Cell]) -> List[Cell]:
    for cell in cells:
        is_prim = isinstance(cell.ir, IRAdapterPrim)
        cid = cell.adapter.cid if is_prim else cell.ir.cid
        if isinstance(cell.ir, IRDimops):
            name = cell.ir.name
        elif isinstance(cell.ir, IRBpOperation):
            name = f"BW.{cell.ir.mirror.name}"
        else:
            name = cell.ir.__class__.__name__
        cell.node = Node(cell.wtype.value, cell.rank, cell.mb, cid, name)
    return cells


def _set_node_opname(cells: List[Cell]) -> List[Cell]:
    for cell in cells:
        is_prim = isinstance(cell.ir, IRAdapterPrim)
        is_bwop = isinstance(cell.ir, IRBpOperation)
        is_wred = isinstance(cell.ir, IRWeightReducer)
        if is_prim:
            fw = cell.adapter.isfw()
            name_dict = {
                (PTypes.AllToAllAllToAllPrim, True): OpName.AllToAllPrim,
                (PTypes.AllToAllAllToAllPrim, False): OpName.AllToAllPrim,
                (PTypes.AllReduceIdentityPrim, True): OpName.AllReducePrim,
                (PTypes.AllReduceIdentityPrim, False): OpName.IdentityPrim,
                (PTypes.SplitAllGatherPrim, True): OpName.ChunkPrim,
                (PTypes.SplitAllGatherPrim, False): OpName.AllGatherPrim,
                (PTypes.AllGatherSplitPrim, True): OpName.AllGatherPrim,
                (PTypes.AllGatherSplitPrim, False): OpName.ChunkPrim,
                (PTypes.ChunkPrim, True): OpName.ChunkPrim,
                (PTypes.ChunkPrim, False): OpName.ChunkPrim,
                (PTypes.MovePrim, True): OpName.MovePrim,
                (PTypes.MovePrim, False): OpName.MovePrim,
                (PTypes.AllGatherPrim, True): OpName.AllGatherPrim,
                (PTypes.AllGatherPrim, False): OpName.AllGatherPrim,
                (PTypes.AllReduceAllReducePrim, True): OpName.AllReducePrim,
                (PTypes.AllReduceAllReducePrim, False): OpName.AllReducePrim,
                (PTypes.AllReducePrim, True): OpName.AllReducePrim,
                (PTypes.AllReducePrim, False): OpName.AllReducePrim,
                (PTypes.IdentityAllreducePrim, True): OpName.IdentityPrim,
                (PTypes.IdentityAllreducePrim, False): OpName.AllReducePrim,
                (PTypes.BroadcastPrim, True): OpName.BroadcastPrim,
                (PTypes.BroadcastPrim, False): OpName.BroadcastPrim,
            }
            cell.opname = name_dict[(type(cell.ir), fw)]
        else:
            if is_bwop:
                name = cell.ir.mirror.name
                fw = cell.ir.isfw()
            elif is_wred:
                name = cell.ir.name
                fw = None
            else:
                name = cell.ir.name
                fw = cell.ir.isfw()
            cell.opname = OpName((name, fw))
    return cells


def _extract_dataflow_irs(cells: List[Cell]) -> List[Cell]:
    for cell in cells:
        is_dataloader = isinstance(cell.ir, IRDataOperation)
        is_prim = isinstance(cell.ir, IRAdapterPrim)
        is_bwop = isinstance(cell.ir, IRBpOperation)
        is_identity_multiref = cell.opname.value[0] in ["identity", "multiref"]
        if is_prim:
            prim = cell.ir
            if isinstance(prim, DiffFusedAdapterPrims):
                # for fused prims, their data flow is in adapter cell
                cell._input_irs = cell.adapter.inputs()
                cell._output_irs = cell.adapter.outputs()
            else:
                # for regular prim, build data flow from prim
                cell._input_irs = prim.inputs()
                cell._output_irs = prim.outputs()
        else:
            for t in cell.ir.inputs():
                if isinstance(t, IRSubTensor):
                    cell._input_irs.append(t)
                else:
                    cell._input_consts.append(t)
            cell._output_irs = cell.ir.outputs()
            if is_bwop and not is_identity_multiref:
                # bw ops need its fw inputs for gradient computation
                for t in cell.ir.mirror.inputs():
                    if isinstance(t, IRSubTensor):
                        cell._input_irs.append(t)
                    else:
                        cell._input_consts.append(t)
            elif is_dataloader:
                cell._input_irs = []
        # format to lists
        cell._input_irs = list(cell._input_irs)
        cell._input_consts = list(cell._input_consts)
        cell._output_irs = list(cell._output_irs)

        for ir in cell._input_irs + cell._output_irs:
            idempotent_update(cell.tid_shapes, {ir.tid: ir.shape})
            if isinstance(cell.ir, IRDataOperation):
                cell.initialized_tid.add(ir.tid)
            elif ir.is_attr() and not ir.is_grad():
                cell.initialized_tid.add(ir.tid)
            elif ir.is_loss() and ir.is_grad():
                cell.initialized_tid.add(ir.tid)
    return cells


def _set_node_kwargs(cells: List[Cell]) -> List[Cell]:

    for cell in cells:
        is_bwop = isinstance(cell.ir, IRBpOperation)
        is_identity_multiref = cell.opname.value[0] in ["identity", "multiref"]
        cell.kwargs = cell.ir.kwargs
        # bw kwargs inherits fw kwargs
        if is_bwop and not is_identity_multiref:
            idempotent_update(cell.kwargs, cell.ir.mirror.kwargs)
        # include consts
        cell.kwargs[KW_CONSTS_KEY] = cell._input_consts.copy()
        # resolve chunk idx
        if cell.opname == OpName.ChunkPrim:
            out_ir: IRSubTensor = unique(cell._output_irs)
            dim = cell.ir.kwargs["dim"]
            indmap_start, indmap_end = out_ir.indmap[dim]
            chunk_size = indmap_end - indmap_start
            chunk_idx = indmap_start // chunk_size
            cell.kwargs[KW_CC_IDX] = chunk_idx
    return cells


def _set_dataflow_partial_SSA_wo_version(cells: List[Cell]) -> List[Cell]:
    for cell in cells:
        cell.inputs = []
        # NOTE: all tensors should inherit rank from cell instead from its ir
        # as prims can have local-view device
        for t in cell._input_irs:
            # for fw weight, mb=-1 means shared across mbs
            # for bw weight, mb=-1 means they will be automatically accumulated
            mb = -1 if t.is_attr() else cell.mb
            cell.inputs.append(Tensor(cell.wtype.value, cell.rank, mb, t.tid, None))
        for t in cell._output_irs:
            # for fw weight, mb=-1 means shared across mbs
            # for bw weight, mb=-1 means they will be automatically accumulated
            mb = -1 if t.is_attr() else cell.mb
            cell.outputs.append(Tensor(cell.wtype.value, cell.rank, mb, t.tid, None))
        # if is send recv prim
        if isinstance(cell.ir, (PTypes.MovePrim, PTypes.BroadcastPrim)):
            # if is the sender
            sender_rank = cell.ir.kwargs["src"]
            fly_rank = -1 - sender_rank  # use sender's inverted rank
            if cell.rank == sender_rank:
                # original ir has missing outputs for sender prim
                cell.outputs = [
                    Tensor(t.wtype, fly_rank, t.mb, t.tid, None) for t in cell.inputs
                ]
                cell._output_irs = cell._input_irs
            else:
                # original ir has missing inputs for receiver prim
                if isinstance(cell.ir, PTypes.BroadcastPrim):
                    assert cell.rank in cell.ir.kwargs["ranks"]
                elif isinstance(cell.ir, PTypes.MovePrim):
                    assert cell.rank == cell.ir.kwargs["dst"]
                else:
                    raise NotImplementedError
                cell.inputs = [
                    Tensor(t.wtype, fly_rank, t.mb, t.tid, None) for t in cell.outputs
                ]
                cell._input_irs = cell._output_irs
    return cells


def _set_tensor_version_to_enforce_SSA(cells: List[Cell]) -> List[Cell]:
    # keys are Tensor with version as None
    # consuming a tensor directly consult the dict to extract latest version
    # producing a tensor will produce a tensor with incremented version
    # so produced version must >= 1
    #   if v == 1, means it is produced first time
    #   else, i.e. v > 1, means an earlie partial value is generated
    #       then its subsequent nodes should consume v+1,
    #       where +1 indicates an injected grad accumulation
    cnt_tensor_as_produced: Dict[Tensor, int] = defaultdict(int)
    ret = []
    for cell in cells:
        discard = False
        inputs_wo_v = cell.inputs
        inputs_ssa = [
            Tensor(t.wtype, t.rank, t.mb, t.tid, cnt_tensor_as_produced[t])
            for t in inputs_wo_v
        ]
        outputs_wo_v = cell.outputs
        output_ssa = []
        is_sender = (
            isinstance(cell.ir, (PTypes.MovePrim, PTypes.BroadcastPrim))
            and cell.rank == cell.ir.kwargs["src"]
        )
        isChunk = cell.opname == OpName.ChunkPrim
        for t in outputs_wo_v:
            # assert t not in inputs_wo_v, f"{cell.ir}, {cell.inputs}, {cell.outputs}, {cell.node}"
            if is_sender:
                curr_v = 0
            elif isChunk and cnt_tensor_as_produced[t]:
                # PATCH: confirmed by Yi, needs an adhoc patch to 
                # bypass ChunkPrim that re-outputs an exisiting tensor
                # NOTE: this patch may be error prone; need revisit
                discard = True
                break
            else:
                # produce a tensor and subsequent cells just use this
                curr_v = cnt_tensor_as_produced[t] + 1
                next_v = cnt_tensor_as_produced[t] + 1
                # detect grad accumulation when curr_v > 1, meaning the tensor
                # is already produced at least once
                if curr_v > 1:
                    prev_v = cnt_tensor_as_produced[t]
                    next_v = curr_v + 1
                    prev_tensor = Tensor(t.wtype, t.rank, t.mb, t.tid, prev_v)
                    curr_tensor = Tensor(t.wtype, t.rank, t.mb, t.tid, curr_v)
                    next_tensor = Tensor(t.wtype, t.rank, t.mb, t.tid, next_v)
                    # record tensor transition
                    cell._grad_accum_transition.append(
                        (prev_tensor, curr_tensor, next_tensor)
                    )
                cnt_tensor_as_produced[t] = next_v
            # NOTE: current cell still produces curr_v instead of next_v
            # as we will inject the grad accumuluation later
            output_ssa.append(Tensor(t.wtype, t.rank, t.mb, t.tid, curr_v))
        if discard:
            continue
        cell.inputs = inputs_ssa
        cell.outputs = output_ssa
        ret.append(cell)
    return ret


def _inject_grad_accum(cells: List[Cell]) -> List[Cell]:
    ret: List[Cell] = []
    cnt_injected_grad_accum: int = 0
    for cell in cells:
        # always append the original cell
        ret.append(cell)
        # sequentially emit grad accumulations
        for in_t1, in_t2, out_t in cell._grad_accum_transition:
            tid = unique([in_t1.tid, in_t2.tid, out_t.tid])
            # inject new node
            ga_cell = Cell(None, cell.rank, cell.wtype)
            ga_cell.node = Node(
                cell.wtype.value,
                cell.rank,
                cell.mb,
                cnt_injected_grad_accum,
                OpName.LOCAL_GRAD_ACCUM.value[0],
            )
            cnt_injected_grad_accum += 1
            ga_cell.opname = OpName.LOCAL_GRAD_ACCUM
            ga_cell.inputs = [in_t1, in_t2]
            ga_cell.outputs = [out_t]
            ga_cell.tid_shapes[tid] = cell.tid_shapes[tid]
            ret.append(ga_cell)
    return ret


def _set_wred_local_grads(cells: List[Cell]) -> List[Cell]:
    # the following dicts checks the 1-1-maping between a weight and
    # its grads subtensors, which are potentially
    # value-partitioned and automatically accumulated, have the same
    # IRSubTensor.tid, such that later tensor version can preserve
    # correct semantics for IRWeightReducer, which only specify the
    # fw weights, while actually consumes their gradient tensors
    # essentially, it want to avoid grad accum for irs with different tids
    wid2gid: Dict[int, int] = {}
    gid2wid: Dict[int, int] = {}
    tid2maxv: Dict[int, int] = {}
    ret: List[Cell] = []
    for cell in cells:
        for t in cell.inputs + cell.outputs:
            tid2maxv[t.tid] = max(tid2maxv.get(t.tid, t.v), t.v)
        if isinstance(cell.ir, IRDimops):
            # record w to grad mappings
            for input_ir in cell._input_irs:
                if input_ir.is_attr() and input_ir.grad is not None:
                    grad: IRSubTensor = input_ir.grad
                    weight_tid = input_ir.tid
                    grad_tid = grad.tid
                    # register the 1-1-mapping
                    idempotent_update(wid2gid, {weight_tid: grad_tid})
                    idempotent_update(gid2wid, {grad_tid: weight_tid})
                    cell._gid2wid[grad_tid] = weight_tid
        elif isinstance(cell.ir, IRBpOperation):
            fw_node = cell.ir.mirror
            assert fw_node is not None
            for output_ir in cell._output_irs:
                if output_ir.is_attr():
                    grad_tid = output_ir.tid
                    # a weight's gradient must be registered by a fw op
                    assert grad_tid in gid2wid
        if isinstance(cell.ir, IRWeightReducer):
            # by default, IRWeightReducer has weights as inputs instead of
            # gradients, so we need to modify its dataflow to be represented
            # by collective grads, which also include counterpart tensors
            # from the reducer-group ranks
            # we also decompose WRED into per-weight
            assert cell.outputs == []
            for w, w_ir in zip(cell.inputs, cell._input_irs):
                new_cell = copy(cell)
                grad_tid = wid2gid[w.tid]
                grad_v = tid2maxv[grad_tid]
                local_grad = Tensor(w.wtype, new_cell.rank, -1, grad_tid, grad_v)
                # output_tensor
                finalized_grad = Tensor(
                    w.wtype, new_cell.rank, -1, grad_tid, grad_v + 1
                )
                new_cell.inputs = [local_grad]
                new_cell._input_irs = [w_ir]
                new_cell.outputs = [finalized_grad]
                new_cell.node = new_cell.node._replace(
                    irname=new_cell.node.irname + f"-w{w.tid}"
                )
                new_cell._wred_wid = w.tid
                ret.append(new_cell)
        else:
            ret.append(cell)

    return ret


def _set_collective_group_id(cells: List[Cell], W: World) -> List[Cell]:
    for cell in cells:
        if cell.opname is OpName.CROSS_DP_WRED:
            cell._collective_group_id = (cell.node.cid, cell._wred_wid)
            assert len(cell._input_irs) == len(cell.inputs)
            cell._collective_indmap.update(
                {t: ir.indmap for t, ir in zip(cell.inputs, cell._input_irs)}
            )
        elif cell.opname in [
            OpName.AllGatherPrim,
            OpName.AllReducePrim,
            OpName.AllToAllPrim,
        ]:
            cell._collective_group_id = (
                cell.node.cid,
                cell.opname.value[0],
                rank_to_dp(cell.rank, W),
                cell.mb,
            )
            assert len(cell._input_irs) == len(cell.inputs)
            cell._collective_indmap.update(
                {t: ir.indmap for t, ir in zip(cell.inputs, cell._input_irs)}
            )

    return cells


def _extract_lv(
    cells: List[Cell],
) -> List[Cell]:
    for cell in cells:
        for t, ir in zip(
            [*cell.inputs, *cell.outputs], [*cell._input_irs, *cell._output_irs]
        ):
            if cell.opname in [OpName.LOCAL_GRAD_ACCUM, OpName.CROSS_DP_WRED]:
                continue
            idempotent_update(
                cell._tid2lv,
                {
                    t.tid: LineageView(
                        ir.is_grad(),
                        ir.is_loss(),
                        ir.is_attr(),
                        ir.parent.shape,
                        ir.indmap,
                        ir.valmap,
                    )
                },
            )
    return cells

def _bypass_known_bugs(
    cells: List[Cell],
) -> List[Cell]:
    for cell in cells:
        if cell.opname == OpName.BW_multiref:
            inputs = []
            for t in cell.inputs:
                if t.v  > 0:
                    inputs.append(t)
            cell.inputs = inputs
    return cells        

def _prepare_rank_cells(W: World, mg: ModuleCodeGen, rank: int) -> List[Cell]:
    """Build the rank nodes, ensuring SSA for nodes and tensors."""
    # NOTE: We adopt procedure/pass-style design pattern to maintain
    # better readability as logics are highly coupled with
    # NNScaler implementations. It may slightly compromise
    # performance, but the trade-off is expect small.

    # get raw irs of from NNScaler (with dp-local ranks)
    cells = mg.execplan.seq(rank % W.plan_ndevs)
    # scale dp-local ranks to global ranks
    cells = _flatten_exereuse_then_scale(cells, mg, rank)
    # wrap ir into Cell to facilitate data flow analysis
    cells = [Cell(ir, rank, W.wtype) for ir in cells]
    # distinguish cells with micro-batch id
    cells = _set_mb(cells)
    # flatten the segment wrapper into operators
    cells = _flatten_segment(cells)
    # flatten the adapter into prims
    cells = _flatten_adapter(cells)

    # assert check types of cell.ir as safe guard
    cells = _assert_check_ir_types(cells)

    # determine SSA id for each node
    cells = _set_node_SSA(cells)
    # determine the semantic operator of the node
    cells = _set_node_opname(cells)

    # unify the access of input/output tensor irs
    cells = _extract_dataflow_irs(cells)
    # unify the access of input/output tensor irs
    cells = _set_node_kwargs(cells)
    # set the dataflow using Tensor, with missing version
    cells = _set_dataflow_partial_SSA_wo_version(cells)
    # remove useless dataloader
    cells = _remove_dummy_dataloader_redundant_identity(cells, W)
    # prepare tid with version to allow later tensor SSA
    cells = _set_tensor_version_to_enforce_SSA(cells)

    # scan tids' versions and explicitly inject gradient accumulation
    cells = _inject_grad_accum(cells)
    # extend weight reducer's input dataflow
    cells = _set_wred_local_grads(cells)
    # set collective communication group id for input fusion
    cells = _set_collective_group_id(cells, W)

    # set lineage view
    cells = _extract_lv(cells)
    cells = _bypass_known_bugs(cells)

    return cells


def _fuse_collective_inputs(cells: List[Cell]):
    shared_tensor_list: Dict[Hashable, List[Tensor]] = {}
    tensor_indmap: Dict[Tensor, Any] = {}
    for cell in cells:
        key = cell._collective_group_id
        if key is not None:
            shared_tensor_list.setdefault(key, [])
            shared_tensor_list[key].extend(cell.inputs)
            tensor_indmap.update({t: cell._collective_indmap[t] for t in cell.inputs})
            cell.inputs = shared_tensor_list[key]
    for key, tensor_list in shared_tensor_list.items():
        # inplace sort
        tensor_list.sort(key=lambda t: tensor_indmap[t])
    return cells, shared_tensor_list


def _inverse_prop_multiref_valmap(cells: List[Cell], tid2lv: Dict[int, LineageView]):
    # we first store the updates out-of-place to ensure consistency
    tid2lv_updates: Dict[int, LineageView] = {}
    for cell in cells:
        if cell.opname == OpName.BW_multiref:
            output: Tensor = unique(cell.outputs)
            out_valmap = tid2lv[output.tid].valmap
            for t in cell.inputs:
                lv = tid2lv[t.tid]
                new_lv = copy(lv)
                new_lv.valmap = out_valmap
                idempotent_update(tid2lv_updates, {t.tid: new_lv})
    return tid2lv_updates


def _emit_graph(dfg: NNScalerDFG, cells: List[Cell]) -> None:
    for cell in cells:
        dfg._nodes.append(cell.node)
        idempotent_update(
            dfg._node2kwargs,
            {cell.node: cell.kwargs},
        )
        idempotent_update(dfg._node2opname, {cell.node: cell.opname})
        idempotent_update(dfg._node2inputs, {cell.node: cell.inputs})
        idempotent_update(dfg._node2outputs, {cell.node: cell.outputs})
        tensors = {*cell.inputs, *cell.outputs}
        dfg._tensors.update(tensors)
        for t in tensors:
            key = (t.rank, t.tid)
            dfg._ranktid2maxv[key] = max(dfg._ranktid2maxv.get(key, t.v), t.v)
        idempotent_update(dfg._tid2shape, cell.tid_shapes)
        dfg._initialized_tid.update(cell.initialized_tid)
        idempotent_update(dfg._gid2wid, cell._gid2wid)
        dfg._node2irstr[cell.node] = f"{cell.irstr} {cell.kwargs}"
        idempotent_update(
            dfg._node2dtag,
            {
                cell.node: DTag(
                    cell.rank,
                    rank_to_dp(cell.rank, dfg.W),
                    rank_to_tp(cell.rank, dfg.W),
                    rank_to_pp(cell.rank, dfg.W),
                    cell.mb,
                ),
            },
        )
        idempotent_update(dfg._tid2lv, cell._tid2lv)

    assert len(dfg.nodes()) == len(set(dfg.nodes()))


_GLOBAL_MG = None
_GLOBAL_W = None

def _init_pool(mg_obj, w_obj):
    global _GLOBAL_MG, _GLOBAL_W
    _GLOBAL_MG = mg_obj
    _GLOBAL_W = w_obj
    

def _prepare_rank_cell_worker(args):
    rank, path = args
    cells: List[Cell] = _prepare_rank_cells(_GLOBAL_W, _GLOBAL_MG, rank)
    # rm transient irs to enable multiprocessing / better performance
    for cell in cells:
        del cell.ir
        del cell.adapter
        del cell._input_irs
        del cell._output_irs
        del cell.wtype
        del cell._wred_wid
        del cell._grad_accum_transition
        del cell._input_consts
    with open(path, "wb") as fp:
        pickle.dump(cells, fp, protocol=pickle.HIGHEST_PROTOCOL)

def build_graph(W: World, mg: ModuleCodeGen, G_path: str) -> NNScalerDFG:
    dfg = NNScalerDFG(W)
    dfg._path = G_path
    ranks = list(range(W.runtime_ndevs))
    loginfo("Loading rank cells.", wtype=W.wtype.value)
    G_stem = Path(G_path).stem
    cells_dir: Path = Config.cache_dir / G_stem / "cells" 
    cells_dir.mkdir(parents=True, exist_ok=True)
    rank_paths = {
        r: cells_dir / f"r{r}.pkl" for r in ranks
    }
    args_list = [(r, rank_paths[r]) for r in ranks]
    with Pool(
        processes=Config.max_ser_proc, initializer=_init_pool, initargs=(mg,W)
    ) as pool:
        pool.map(_prepare_rank_cell_worker, args_list)
    cells: List[Cell] = []
    for path in tqdm(rank_paths.values(), desc="Collecting ranks."):
        with open(path, "rb") as fp:
            rank_cells = pickle.load(fp)
            cells.extend(rank_cells)
    cells, shared_tensor_list = _fuse_collective_inputs(cells)
    _emit_graph(dfg, cells)
    tid2lv_updates = _inverse_prop_multiref_valmap(cells, dfg._tid2lv)
    dfg._tid2lv.update(tid2lv_updates)
    dfg._shared_tensor_list = shared_tensor_list
    return dfg
