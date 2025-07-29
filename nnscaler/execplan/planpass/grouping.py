"""
Operation grouping
"""
from typing import List, Dict, Tuple

from nnscaler.execplan import ExecutionPlan
from nnscaler.execplan.planpass.planpass import PlanPass
from nnscaler.ir.adapter import IRAdapter
from nnscaler.ir.adapter.prim import IdentityPrim
from nnscaler.ir.operator import IRFwOperation
from nnscaler.graph.function.pyfunc import IRPyFunc
from nnscaler.ir.cten import IRCell

from nnscaler.flags import CompileFlag
        

class Grouping(PlanPass):

    @staticmethod
    def apply(execplan: ExecutionPlan) -> ExecutionPlan:
        """
        Group contiguous differentiable operators segments

        Note non-differentiable IRAdapter with all identity operators will be
        removed from execution plan.
        """
        graph = execplan.graph
        fgroups, bgroups = Grouping.group(execplan)
        for devid in execplan.devices():
            for fpieces, bpieces in zip(fgroups[devid], bgroups[devid]):
                fsubgraph = graph.create_segment(fpieces)
                if len(bpieces) > 0:
                    bsubgraph = graph.create_segment(bpieces)
                    IRCell.make_pair(fsubgraph, bsubgraph)
                subgraphs = [fsubgraph] if len(bpieces) == 0 else [fsubgraph, fsubgraph.mirror]
                for subgraph in subgraphs:
                    # update execution plan: replace the nodes with the subgraph
                    pieces = subgraph.nodes()
                    idx = execplan.seq(devid).index(pieces[0])
                    execplan.at(devid).insert(idx, subgraph)
                    for node in pieces:
                        execplan.at(devid).remove(node)
            # remove identity adapter
            for adapter in execplan.seq(devid):
                if isinstance(adapter, IRAdapter):
                    if all(isinstance(prim, IdentityPrim) for prim in adapter.prims):
                        execplan.at(devid).remove(adapter)
        return execplan

    @staticmethod
    def group(execplan) -> Tuple[Dict[int, List[List[IRCell]]],]:
        """
        Return forward groups and corresponding
        backward groups for each device.

        Each group can be indexed by device id.
        Each device id contains a list of forward / backward operations

        Returns:
            Tuple: (fgroups, bgroups)
        """
        def differentiable(fnode):
            # nnfusion special handle: break IRAdapter and IRPyFunc
            if CompileFlag.use_nnfusion:
                if isinstance(fnode, IRAdapter): return False
            if isinstance(fnode, IRFwOperation):
                return True
            if isinstance(fnode, IRAdapter) and fnode.isfw():
                if fnode.differentiable: return True
                if fnode.mirror is None: return True  # not require backward
            return False

        fgroups, bgroups = dict(), dict()
        for devid in execplan.devices():
            fgroups[devid], bgroups[devid] = list(), list()
            nodes = execplan.seq(devid)
            break_idx = [idx for idx, node in enumerate(nodes) if not differentiable(node)]
            for start, end in zip([-1] + break_idx, break_idx + [len(nodes)]):
                if start+1 == end: continue
                fpieces = nodes[start+1:end]
                bpieces = [node.mirror for node in fpieces[::-1] if node.mirror is not None]
                fgroups[devid].append(nodes[start+1:end])
                bgroups[devid].append(bpieces)

        return fgroups, bgroups
