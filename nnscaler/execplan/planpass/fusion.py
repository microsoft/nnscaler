from typing import List, Union, Set
import logging
from nnscaler.graph.graph import IRSegment

from nnscaler.ir.adapter import IRAdapter

from nnscaler.execplan import ExecutionPlan
from nnscaler.execplan.execplan import ExeReuseCell
from nnscaler.execplan.planpass.planpass import PlanPass

from nnscaler.ir.adapter.prim import IRAdapterPrim
from nnscaler.ir.adapter.prim import AllReducePrim, AllGatherPrim, ReduceScatterPrim, AllToAllPrim
from nnscaler.ir.adapter.prim import IdentityPrim, ChunkPrim
from nnscaler.ir.adapter.prim import IdentityAllreducePrim, AllReduceIdentityPrim, AllReduceAllReducePrim
from nnscaler.ir.adapter.prim import AllGatherReduceScatterPrim, ReduceScatterAllGatherPrim
from nnscaler.ir.adapter.prim import SplitAllGatherPrim, AllGatherSplitPrim
from nnscaler.ir.adapter.prim import AllToAllAllToAllPrim


_logger = logging.getLogger(__name__)


class DiffFusion(PlanPass):

    @staticmethod
    def apply(execplan: ExecutionPlan) -> ExecutionPlan:
        """
        Fuse the non-differentiable adapters into differentiable adapters.
        """
        cnt = 0
        for devid in execplan.devices():
            # fadapters: Set[IRAdapter] = set()
            visited = set()
            for node in execplan.seq(devid):
                if isinstance(node, ExeReuseCell):
                    node = node.cell
                if node in visited:
                    continue
                if isinstance(node, IRAdapter) and node.isfw():
                    ret = DiffFusion.nnfuse(node)
                    cnt = cnt+1 if ret else cnt
                elif isinstance(node, IRSegment) and node.isfw():
                    for fadapter in node.select(ntype=IRAdapter):
                        ret = DiffFusion.nnfuse(fadapter)
                        cnt = cnt+1 if ret else cnt
                visited.add(node)
        _logger.info(f'adapter fusion: successfully fuse {cnt} differentiable adapters')
        return execplan

    @staticmethod
    def _apply(cell: IRSegment) -> int:
        cnt = 0
        for node in cell.nodes():
            if isinstance(node, IRAdapter) and node.isfw():
                ret = DiffFusion.nnfuse(node)
                # if not ret and not node.differentiable:
                #     raise NotImplementedError(
                #         f"Adapter within IRSegment cannot fuse to differientiable adapter"
                #         f"\nforward: {node.extra_repr()}"
                #         f"\nbackward: {node.mirror.extra_repr()}"
                #     )
                cnt = cnt + 1 if ret else cnt
            elif isinstance(node, IRSegment) and node.isfw():
                cnt += DiffFusion._apply(node)
        return cnt

    @staticmethod
    def nnfuse(fadapter: IRAdapter) -> bool:
        """
        Fuse the forward adapter with its backward adapter into differentiable
        communications. Note this is an inplacement update

        Return:
            success: boolean
        """
        if not isinstance(fadapter.mirror, IRAdapter):
            return False
        badapter: IRAdapter = fadapter.mirror
        fprims, bprims = fadapter.prims, badapter.prims

        def is_allreduce(prims: List[IRAdapterPrim]) -> bool:
            return len(prims) == 1 and all(isinstance(prim, AllReducePrim) for prim in prims)

        def is_identity(prims: List[IRAdapterPrim]) -> bool:
            return len(prims) == 1 and all(isinstance(prim, IdentityPrim) for prim in prims)

        def is_redsca(prims: List[IRAdapterPrim]) -> bool:  # reduce-scatter
            return len(prims) == 1 and all(isinstance(prim, ReduceScatterPrim) for prim in prims)

        def is_allgather(prims: List[IRAdapterPrim]) -> bool:
            return len(prims) == 1 and all(isinstance(prim, AllGatherPrim) for prim in prims)

        def is_chunk(prims: List[IRAdapterPrim]) -> bool:
            return len(prims) == 1 and all(isinstance(prim, ChunkPrim) for prim in prims)

        def is_alltoall(prims: List[IRAdapterPrim]) -> bool:
            return len(prims) == 1 and all(isinstance(prim, AllToAllPrim) for prim in prims)

        prims = None
        # allreduce-identity
        if is_allreduce(fprims) and is_identity(bprims):
            prims = [AllReduceIdentityPrim(p.inputs(), p.outputs(), **p.kwargs) for p in fprims]
        # identity-allreduce
        elif is_identity(fprims) and is_allreduce(bprims):
            prims = [IdentityAllreducePrim(p.inputs(), p.outputs(), **bprims[0].kwargs) for p in fprims]
        # allreduce-allreduce
        elif is_allreduce(fprims) and is_allreduce(bprims):
            prims = [AllReduceAllReducePrim(p.inputs(), p.outputs(), **p.kwargs) for p in fprims]
        # allgather-reducescatter
        elif is_allgather(fprims) and is_redsca(bprims):
            prims = [AllGatherReduceScatterPrim(p.inputs(), p.outputs(), **p.kwargs) for p in fprims]
        # reducescatter-allgather
        elif is_redsca(fprims) and is_allgather(bprims):
            prims = [ReduceScatterAllGatherPrim(p.inputs(), p.outputs(), **p.kwargs) for p in fprims]
        # allgather-chunk
        elif is_allgather(fprims) and is_chunk(bprims):
            prims = [AllGatherSplitPrim(p.inputs(), p.outputs(), **p.kwargs) for p in fprims]
        # chunk-allgather
        elif is_chunk(fprims) and is_allgather(bprims):
            prims = [SplitAllGatherPrim(p.inputs(), p.outputs(), **p.kwargs) for p in fprims]
        # all-to-all
        elif is_alltoall(fprims) and is_alltoall(bprims):
            prims = [AllToAllAllToAllPrim(p.inputs(), p.outputs(), **p.kwargs) for p in fprims]
        
        if prims is not None:
            fadapter.prims = prims
            badapter.prims = prims
            fadapter.custom = False
            fadapter.differentiable = True
            badapter.custom = False
            badapter.differentiable = True
            return True
        return False
