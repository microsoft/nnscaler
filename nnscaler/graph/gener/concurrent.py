#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""
Concurrent producer / consumer Adapter Generator
"""
from typing import List, Optional, Dict, Tuple, Callable
import copy
import numpy as np
import logging
from contextlib import contextmanager

from nnscaler.ir.tensor import IRFullTensor, IRSubTensor, IndexMap, ValueMap
from nnscaler.ir.adapter.prim import IRAdapterPrim, ReduceScatterPrim, AllToAllPrim
from nnscaler.ir.adapter import IRAdapter
from nnscaler.ir.adapter.prim import SelectPrim, MovePrim, SumPrim, MergeDimPrim
from nnscaler.ir.adapter.prim import BroadcastPrim

from nnscaler.graph.gener.rvd.layout import RVDLayout
from nnscaler.graph.gener.rvd.intra import IntraPathFinder
from nnscaler.graph.gener.rvd.inter import InterPathFinder
from nnscaler.flags import CompileFlag


_logger = logging.getLogger(__name__)

if CompileFlag.disable_intra_rvd:
    _logger.warning('Detected disabling intra-RVD collective generation, which may have big impact on performance.')
if CompileFlag.disable_inter_rvd:
    _logger.warning('Detected disabling inter-RVD collective generation, which may have big impact on performance.')
if CompileFlag.disable_comm_fusion:
    _logger.warning('Detected disabling general communication fusion, which may have big impact on performance in certain cases.')


@contextmanager
def _temp_disable_reduce_scatter_adapter():
    assert not CompileFlag.disable_reduce_scatter_adapter, "Already disabled"
    CompileFlag.disable_reduce_scatter_adapter = True
    yield
    CompileFlag.disable_reduce_scatter_adapter = False


class ConcurrentGener:

    @staticmethod
    def gen(fptensors: List[IRSubTensor], fctensors: List[IRSubTensor],
            bptensors: List[IRSubTensor], bctensors: List[IRSubTensor],
            cost_fn: Optional[Callable] = None) -> Optional[IRAdapter]:
        """
        Generate forward adapter and backward adapter

        @param fptensors List[IRSubTensor]: forward producer tensors
        @param fctensors List[IRSubTensor]: forward consumer tensors
        @param bptensors List[IRSubTensor]: backward producer tensors
        @param bctensors List[IRSubTensor]: backward consumer tensors
        @param cost_fn Optional[Callable]: takes in an IRAdapterPrim and outputs a cost in float

        @return fadapter IRAdapter: forward adapter
        """
        pdevs = tuple(t.device[0] for t in fptensors)
        cdevs = tuple(t.device[0] for t in fctensors)

        fadapter: IRAdapter = None

        # case 1: sharing device (intra-rvd)
        inshard = (set(pdevs) == set(cdevs)) and (len(fptensors) == len(fctensors)) and (len(pdevs) == len(fptensors))
        if (not CompileFlag.disable_intra_rvd) and inshard and len(pdevs) > 1:
            try:
                fadapter = ConcurrentGener.gen_intra_rvd(fptensors, fctensors, bptensors, bctensors, cost_fn)
            except Exception as e:
                fadapter = None
                color, default = '\033[33m' , '\033[0m'
                msg = (
                    f"{color}========== Fail to use intra-RVD ==========\n"
                    f"full tensor: {fptensors[0].parent} | is grad: {fptensors[0].parent.is_grad()}\n"
                    f"Reason: {str(e)}\n"
                    f"Switch to general P2P communication.\n"
                    f"===========================================\n{default}"
                )
                _logger.warning(f'intra-RVD:\n{msg}')

        # Case 2: sperating device (inter-rvd)
        if (not CompileFlag.disable_inter_rvd) and len(set(pdevs).intersection(cdevs)) == 0:
            try:
                fadapter = ConcurrentGener.gen_inter_rvd(fptensors, fctensors, bptensors, bctensors, cost_fn)
            except Exception as e:
                fadapter = None
                color, default = '\033[33m' , '\033[0m'
                msg = (
                    f"{color}========== Fail to use inter-RVD ==========\n"
                    f"full tensor: {fptensors[0].parent}\n"
                    f"Reason: {str(e)}\n"
                    f"Switch to general P2P communication.\n"
                    f"===========================================\n{default}"
                )
                _logger.warning(f'inter-RVD:\n{msg}')

        # Case 3: General cases
        # warnings.warn('The adapter is generated using P2P communication')
        if fadapter is None:
            fadapter = ConcurrentGener.gen_general(fptensors, fctensors, bptensors, bctensors)

        if set(pdevs) == set(cdevs) and fadapter.mirror is not None:
            fadapter.differentiable = True
            fadapter.mirror.differentiable = True

        return fadapter

    @staticmethod
    def _path(
        path_fn: Callable,
        ilayout: RVDLayout, olayout: RVDLayout,
        cost_fn: Optional[Callable] = None
    ) -> List[IRAdapterPrim]:
        prims = path_fn(ilayout, olayout, cost_fn)
        if any(isinstance(prim, AllToAllPrim) and not prim.is_valid() for prim in prims):
            if not CompileFlag.disable_reduce_scatter_adapter \
                and any(isinstance(prim, ReduceScatterPrim) for prim in prims):
                _logger.warning(
                    'Detected invalid AllToAllPrim, retrying with reduce-scatter disabled.'
                    'Please report this issue to the developers.'
                )
                # the problem may be caused by the ReduceScatterPrim
                # let's retry without it.
                with _temp_disable_reduce_scatter_adapter():
                    prims = path_fn(ilayout, olayout, cost_fn)

        if any(not prim.is_valid() for prim in prims):
            # will use `ConcurrentGener.gen_general` to generate adapter
            raise RuntimeError('Invalid primitives detected. Please report this issue to the developers.')

        return prims

    @staticmethod
    def _intra_path(
        ilayout: RVDLayout, olayout: RVDLayout,
        cost_fn: Optional[Callable] = None
    ) -> List[IRAdapterPrim]:
        return ConcurrentGener._path(IntraPathFinder.path, ilayout, olayout, cost_fn)

    @staticmethod
    def _inter_path(
        ilayout: RVDLayout, olayout: RVDLayout,
        cost_fn: Optional[Callable] = None
    ) -> List[IRAdapterPrim]:
        return ConcurrentGener._path(InterPathFinder.path, ilayout, olayout, cost_fn)

    @staticmethod
    def gen_intra_rvd(fptensors: List[IRSubTensor], fctensors: List[IRSubTensor],
                      bptensors: List[IRSubTensor], bctensors: List[IRSubTensor],
                      cost_fn: Optional[Callable] = None) -> IRAdapter:
        """
        Generate forward and backward adapter for concurrent produced tensors and consumed tensors.

        @param fptensors List[IRSubTensor]: forward produced tensors
        @param fctensors List[IRSubTensor]: forward consumed tensors
        @param bptensors List[IRSubTensor]: backward produced tensors
        @param bctensors List[IRSubTensor]: backward consumed tensors
        @param cost_fn Optional[Callable]: takes in an IRAdapterPrim and outputs a cost in float

        @return adapter IRAdapter: forward IRAdapter with backward (if has) in its .mirror attribute.
        """
        ftensor = fptensors[0].parent
        # producer grid layout
        ilayout = RVDLayout.togrid(ftensor, fptensors)
        devs = [ptensor.device for ptensor in ilayout.mat.flatten()]
        # re-order ctensors to match with placement of ptensors
        ctensors = [None] * len(devs)
        for ctensor in fctensors:
            idx = devs.index(ctensor.device)
            ctensors[idx] = ctensor
        assert all(t is not None for t in ctensors), f"empty device slot {ctensors}"
        olayout = RVDLayout.togrid(ftensor, ctensors)
        # get forward primitives
        fprims = ConcurrentGener._intra_path(ilayout, olayout, cost_fn)

        fadapter = IRAdapter(fptensors, fctensors)
        fadapter.prims = fprims

        # generate backward
        grad: IRFullTensor = ftensor.grad
        bprims = []
        if len(bptensors) > 0 and len(bctensors) > 0:
            # reorder ptensors to match with forward
            ptensors = [None] * len(devs)
            for bptensor in bptensors:
                idx = devs.index(bptensor.device)
                assert ptensors[idx] is None, "same device of different tensors"
                ptensors[idx] = bptensor
            assert all(t is not None for t in ptensors), f"empty device slot from {bptensors}"
            ilayout = RVDLayout.togrid(grad, ptensors)
            olayout = RVDLayout.togrid(grad, bctensors)
            # paths, bprims = ilayout.path(olayout)
            bprims = ConcurrentGener._intra_path(ilayout, olayout, cost_fn)
            # generate backward adapter
            badapter = IRAdapter(bptensors, bctensors)
            badapter.prims = bprims
            IRAdapter.make_pair(fadapter, badapter)

        return fadapter

    @staticmethod
    def gen_inter_rvd(fptensors: List[IRSubTensor], fctensors: List[IRSubTensor],
                      bptensors: List[IRSubTensor], bctensors: List[IRSubTensor],
                      cost_fn: Optional[Callable] = None) -> IRAdapter:
        """
        Generate communication adapters for inter-RVD scenarios.
        This assumes ptensors and ctensors can be represented by RVD layout.

        @param fptensors List[IRSubTensor]: produced tensors
        @param fctensors List[IRSubTensor]: consumed tensors
        @param bptensors List[IRSubTensor]: produced tensors
        @param bctensors List[IRSubTensor]: consumed tensors
        @param cost_fn Optional[Callable]: takes in an IRAdapterPrim and outputs a cost in float

        @return fadapter IRAdapter
        """
        ftensor = fptensors[0].parent
        ilayout = RVDLayout.togrid(ftensor, fptensors)
        olayout = RVDLayout.togrid(ftensor, fctensors)
        fprims = ConcurrentGener._inter_path(ilayout, olayout, cost_fn)
        fadapter = IRAdapter(fptensors, fctensors)
        fadapter.prims = fprims

        grad: IRFullTensor = ftensor.grad
        if len(bptensors) > 0 or len(bctensors) > 0:
            ilayout = RVDLayout.togrid(grad, bptensors)
            olayout = RVDLayout.togrid(grad, bctensors)
            bprims = ConcurrentGener._inter_path(ilayout, olayout, cost_fn)
            badapter = IRAdapter(bptensors, bctensors)
            badapter.prims = bprims
            IRAdapter.make_pair(fadapter, badapter)
        return fadapter

    @staticmethod
    def gen_general(fptensors: List[IRSubTensor], fctensors: List[IRSubTensor],
                    bptensors: List[IRSubTensor], bctensors: List[IRSubTensor]) -> IRAdapter:
        """
        A general way to generate adapter.

        @param ftensor IRFullTensor
        @return adapter IRAdapter
        """
        fprims = []
        fpdevs = set(t.device[0] for t in fptensors)
        fcomm_workload = {t.device[0]: 0 for t in fptensors}
        # first try collectives
        ret = False
        if not CompileFlag.disable_comm_fusion:
            ret, prims = ConcurrentGener.gen_subtensor_coll(fctensors, fptensors, fcomm_workload)
        if ret:
            fprims += prims
        # otherwise use general p2p send recv
        else:
            for ctensor in fctensors:
                fprims += ConcurrentGener.gen_subtensor(ctensor, fptensors, fcomm_workload)
        fadapter = IRAdapter(fptensors,fctensors)
        fadapter.prims = fprims
        # backward
        if len(bptensors) > 0 and len(bctensors) > 0:
            bprims = []
            bcomm_workload = {t.device[0]: 0 for t in bptensors}
            # first try collectives
            if not CompileFlag.disable_comm_fusion:
                ret, prims = ConcurrentGener.gen_subtensor_coll(bctensors, bptensors, bcomm_workload)
            if ret:
                bprims += prims
            # otherwise use general p2p send recv
            else:
                for cgrad in bctensors:
                    bprims += ConcurrentGener.gen_subtensor(cgrad, bptensors, bcomm_workload)
            badapter = IRAdapter(bptensors, bctensors)
            badapter.prims = bprims
            IRAdapter.make_pair(fadapter, badapter)
        return fadapter

    @staticmethod
    def gen_subtensor_coll(ctensors: List[IRSubTensor], ptensors: List[IRSubTensor], workload: Dict[int, int]) -> Tuple[bool, List[IRAdapterPrim]]:
        """
        Generate communication primitives for a tensor using collectives of
        broadcast, [reduce, gather and scatter]. => [...] Not supported yet.

        @param ctensors List[IRSubTensor]: the consumed tensors as destination.
        @param ptensors List[IRSubTensor]: the produced tensors as source

        @return success bool: whether succeed in generate collective
        @return prims List[IRAdapterPrim]: the primitives for adapter
        """
        ret = False
        prims = []
        fuse_broadcast = True
        # check broadcast
        if len(ptensors) >= len(ctensors) or len(ptensors) == 0:
            fuse_broadcast = False
        else:
            for ptensor in ptensors:
                if not all(ptensor == ctensor for ctensor in ctensors):
                    fuse_broadcast = False
                    break
        # fuse to broadcast
        if fuse_broadcast:
            cdev_tensors, pdev_tensors = dict(), dict()
            for ptensor in ptensors:
                pdev_tensors.setdefault(ptensor.device[0], []).append(ptensor)
            for ctensor in ctensors:
                # not consider self-transmission
                if ctensor.device[0] in pdev_tensors: continue
                cdev_tensors.setdefault(ctensor.device[0], []).append(ctensor)
            if len(cdev_tensors) // len(pdev_tensors) <= 1: # can simply use send recv
                return False, []
            pdevs = list(pdev_tensors.keys())
            cdevs = list(cdev_tensors.keys())
            broadcast_ndevs = len(cdevs) // len(pdevs)
            start = 0
            for idx, pdev in enumerate(pdevs):
                addone = 1 if idx < (len(cdevs) % len(pdevs)) else 0
                end = start + broadcast_ndevs + addone
                pdev_ctensors = [cdev_tensors[devid][0] for devid in cdevs[start:end]]
                pdev_ctensors += [pdev_tensors[pdev][0]]
                prims.append(BroadcastPrim([pdev_tensors[pdev][0]], pdev_ctensors))
                start = end
            ret = True
        return ret, prims

    @staticmethod
    def gen_subtensor(ctensor: IRSubTensor, ptensors: List[IRSubTensor], workload: Dict[int, int]) -> List[IRAdapterPrim]:
        """
        Generate communiction primitives for ctensor

        @param ctensor IRSubTensor: the consumed tensor as destination
        @param ptensors List[IRSubTensor]: the produced tensors as source

        @return prims List[IRAdapterPrim]: the primitives for adapter
        """
        # category to local tensor and remote tensor
        local = [t for t in ptensors if t.device == ctensor.device]
        # reorder remote devices: higher priority to use tensor with lower communication workload
        devices = np.array([devid for devid in workload.keys()], dtype=int)
        volume = np.array([workload[devid] for devid in workload.keys()])
        indices = np.argsort(volume)
        sorted_devices = devices[list(indices)]
        remote: List[IRSubTensor] = []
        for devid in sorted_devices:
            if devid == ctensor.device[0]: continue
            remote += [t for t in ptensors if t.device[0] == devid]

        prims = []

        # ==== select ==== #
        intersections: List[IRSubTensor] = []
        # check local
        for itensor in local+remote:
            if itensor.device == ctensor.device and itensor == ctensor:
                return []
            common: Optional[IRSubTensor] = itensor.common(ctensor)
            if common is None:
                continue
            common.cell = itensor.cell
            intersections.append(common)
            # create select primitive
            if common != itensor:
                indmap = []
                for dim in range(itensor.ndims):
                    (s1, e1), (s2, e2) = itensor.indmap[dim], common.indmap[dim]
                    start = s2 - s1
                    end = start + e2 - s2
                    indmap.append((start, end))
                indmap = IndexMap(tuple(indmap))
                if itensor.valmap == common.valmap:
                    valmap = ValueMap((0, 1))
                else:
                    assert itensor.valmap == (0, 1)
                    valmap = common.valmap
                select_prim = SelectPrim(itensor, indmap, valmap, common)
                prims.append(select_prim)
            if itensor.device == ctensor.device and common == ctensor:
                return [select_prim]
            # TODO: check union == subtensor
            if common == ctensor:
                break

        # print(intersections)
        # ====== move ===== #
        tmoved = []
        for tensor in intersections:
            assert len(tensor.device) == 1 and len(ctensor.device) == 1, "Expected only one device."
            mtensor = tensor
            if tensor.device != ctensor.device:
                mtensor = copy.copy(tensor)
                mtensor.cell = ctensor.cell
                prims.append(MovePrim([tensor], [mtensor]))
                workload[tensor.device[0]] += tensor.nelement()
            tmoved.append(mtensor)

        # ===== merge ===== #
        remain_tensors: List[IRSubTensor] = copy.copy(tmoved)
        if ctensor in remain_tensors:
            return prims
        out = None
        while out != ctensor:
            out, merged = None, False
            for idx1 in range(len(remain_tensors) - 1):
                for idx2 in range(idx1+1, len(remain_tensors)):
                    t1, t2 = remain_tensors[idx1], remain_tensors[idx2]
                    catdim = t1.catdim(t2)
                    if catdim is not None:
                        tensors = [t1, t2] if t1.indmap[catdim][0] < t2.indmap[catdim][0] else [t2, t1]
                        out = tensors[0].concat(tensors[1], dim=catdim)
                        out.cell = ctensor.cell
                        prims.append(MergeDimPrim(tensors, out, catdim))
                        merged = True
                        break
                    # reduction
                    if t1.accumable(t2):
                        out = t1.accum(t2)
                        out.cell = ctensor.cell
                        prims.append(SumPrim([t1, t2], out))
                        merged = True
                        break
                if merged:
                    remain_tensors.remove(t1)
                    remain_tensors.remove(t2)
                    remain_tensors.append(out)
                    break
            if out is None:
                ptensors = '\n\t'.join(t.extra_repr() for t in ptensors)
                remain = '\n\t'.join(t.extra_repr() for t in remain_tensors)
                raise RuntimeError(
                    f"Fail to build adapter.\n"
                    f"FullTensor:{ctensor.parent}\n"
                    f"Produced Tensors:\n\t{ptensors}\n"
                    f"Consumed Tensors:\n\t{ctensor.extra_repr()}\n"
                    f"Consumer:\n\t{ctensor.cell}\n"
                    f"Remain Tensor:\n\t{remain}"
                )
        return prims
