#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from typing import Callable, Dict, List, Tuple, Optional, Set
from functools import partial
import numpy as np
import copy
import logging
import torch

from nnscaler.ir.cten import IRCell
from nnscaler.ir.tensor import IRFullTensor, IRSubTensor

from nnscaler.ir.adapter.prim import IRAdapterPrim
from nnscaler.ir.adapter.prim import AllGatherPrim      # d2r
from nnscaler.ir.adapter.prim import AllToAllPrim       # d2d
from nnscaler.ir.adapter.prim import AllReducePrim      # v2r
from nnscaler.ir.adapter.prim import ReduceScatterPrim  # v2d
from nnscaler.ir.adapter.prim import ChunkPrim          # r2d
from nnscaler.ir.adapter.prim import VChunkPrim         # r2v

from nnscaler.graph import IRGraph
from nnscaler.graph.segment import IRSegment
from nnscaler.graph.gener.rvd.layout import RVDLayout

from nnscaler.graph.gener.utils import tensor_vd_repr

from nnscaler.utils import classproperty
from nnscaler.flags import CompileFlag


_logger = logging.getLogger(__name__)
TShape = Tuple[int, ...]
TRVD = Tuple[int, ...]


class IntraTransition:
    """
    Intra-RVD transition primitives
    """

    @staticmethod
    def d2r(rvd: TRVD, dim: int, chunks: int) -> Tuple[TRVD, Callable]:
        """
        intra-RVD primitive D->R: allgather

        @param rvd Tuple[int]: input RVD
        @param dim int: tensor dimension
        @param chunks int: the number of chunks to transfer

        @return rvd Tuple[int]: output RVD
        @return prim Callable: IRAdapter primitive
        """
        assert rvd[2+dim] % chunks == 0, f"not dividable dim: {rvd[2+dim]} // {chunks}"
        rvd = list(rvd)
        rvd[0], rvd[2+dim] = rvd[0] * chunks, rvd[2+dim] // chunks
        return rvd, partial(AllGatherPrim, dim=dim)

    @staticmethod
    def d2d(rvd: TRVD, from_dim: int, to_dim: int, chunks: int) -> Tuple[TRVD, Callable]:
        """
        intra-RVD primitive D(...,i,..)->D(..,j,...): alltoall

        @param rvd Tuple[int]: input RVD
        @param from_dim int: source tensor axis
        @param to_dim int: destination tensor axis
        @param chunks int: the number of chunks to transfer

        @return rvd Tuple[int]: output RVD
        @return prim Callable: IRAdapter primitive
        """
        assert rvd[2+from_dim] % chunks == 0, f"not dividable dim: {rvd[2+from_dim]} // {chunks}"
        rvd = list(rvd)
        rvd[2+from_dim], rvd[2+to_dim] = rvd[2+from_dim] // chunks, rvd[2+to_dim] * chunks
        return rvd, partial(AllToAllPrim, idim=from_dim, odim=to_dim)

    @staticmethod
    def v2r(rvd: TRVD, chunks: int) -> Tuple[TRVD, Callable]:
        """
        intra-RVD primitive V->R: allreduce

        @param dim int: tensor dimension
        @param chunks int: the number of chunks to transfer

        @return rvd Tuple[int]: output RVD
        @return prim Callable: IRAdapter primitive
        """
        assert rvd[1] % chunks == 0, f"not dividable value chunks: {rvd[1]} // {chunks}"
        rvd = list(rvd)
        rvd[1], rvd[0] = rvd[1] // chunks, rvd[0] * chunks
        return rvd, AllReducePrim

    @staticmethod
    def v2d(rvd: TRVD, dim: int, chunks: int) -> Tuple[TRVD, Callable]:
        """
        intra-RVD primitive V->D: reduce-scatter

        @param dim int: tensor dimension
        @param chunks int: the number of chunks to transfer

        @return rvd Tuple[int]: output RVD
        """
        assert rvd[1] % chunks == 0, f"not dividable value chunks: {rvd[1]} // {chunks}"
        rvd = list(rvd)
        rvd[1], rvd[2+dim] = rvd[1] // chunks, rvd[2+dim] * chunks
        return rvd, partial(ReduceScatterPrim, dim=dim)

    @staticmethod
    def r2d(rvd: TRVD, dim: int, chunks: int) -> Tuple:
        """
        intra-RVD primitive R->D: schunk

        @param dim int: tensor axis
        @param chunks int: the number of chunks to transfer

        @return rvd Tuple[int]: output RVD
        @return prim Callable: IRAdapter primitive
        """
        assert rvd[0] % chunks == 0, f"not dividable replica: {rvd[0]} // {chunks}"
        rvd = list(rvd)
        rvd[0], rvd[2+dim] = rvd[0] // chunks, rvd[2+dim] * chunks
        return rvd, partial(ChunkPrim, dim=dim)

    @staticmethod
    def r2v(rvd: TRVD, chunks: int) -> Tuple:
        """
        intra-RVD primitive R->V: vchunk

        @param chunks int: the number of chunks to transfer

        @return rvd Tuple[int]: output RVD
        @return prim Callable: IRAdapter primitive
        """
        assert rvd[0] % chunks == 0, f"not dividable replica: {rvd[0]} // {chunks}"
        rvd = list(rvd)
        rvd[0], rvd[1] = rvd[0] // chunks, rvd[1] * chunks
        return rvd, VChunkPrim

    @staticmethod
    def transitionable(src_rvd: TRVD, dst_rvd: TRVD) -> Optional[Callable]:
        """
        Check wheter a primitive exists to transform src_rvd to dst_rvd

        @param src_rvd TRVD: source RVD
        @param dst_rvd TRVD: destination RVD

        @return trans_fn Optional[Callable]: None indicates no primitive found.
        """
        trans_fn = None
        incd = [dim for dim, (d1, d2) in enumerate(zip(src_rvd, dst_rvd)) if d1 < d2]
        decd = [dim for dim, (d1, d2) in enumerate(zip(src_rvd, dst_rvd)) if d1 > d2]
        if len(incd) != 1 or len(decd) != 1: return None
        incd, decd = incd[0], decd[0]
        # TODO: optimize: enable following may miss best solution
        # ========= prune graph to avoid device mis-alignment ======== #
        # for d in range(min(incd, decd) + 1, max(incd, decd)):
        #     if src_rvd[d] != 1 or dst_rvd[d] != 1: return None
        # ============================================================ #
        # if incd == 1: return None
        if decd >= 2 and incd == 0:   # d2r
            trans_fn = partial(IntraTransition.d2r, dim=decd-2)
        elif decd >= 2 and incd >= 2: # d2d
            trans_fn = partial(IntraTransition.d2d, from_dim=decd-2, to_dim=incd-2)
        elif decd == 1 and incd == 0: # v2r
            trans_fn = IntraTransition.v2r
        elif decd == 1 and incd >= 2: # v2d
            trans_fn = partial(IntraTransition.v2d, dim=incd-2)
        elif decd == 0 and incd >= 2: # r2d
            trans_fn = partial(IntraTransition.r2d, dim=incd-2)
        elif decd == 0 and incd == 1: # r2v
            trans_fn = IntraTransition.r2v
        return trans_fn

    @staticmethod
    def transition(src_layout: RVDLayout, dst_rvd: TRVD) -> List[Tuple[RVDLayout, List[IRAdapterPrim]]]:
        """
        Transfer from source RVD to destination RVD.
        Get all possible device-placement choices for RVD
        (for returned RVDLayout, only device placement are different.)
        given the fixed device placement of RVD.

        Args:
            src_layout (RVDLayout): source ilayout
            dst_rvd (Tuple[int, ...]): destination RVD

        Returns:
            List[Tuple[GridLayout, List[IRAdapterPrim]], ...]:
                tuple of pairs of <layout, [prims]> with each has a different device mapping.
        """
        src_rvd = src_layout.vec
        if src_rvd == dst_rvd: return [(src_layout, [])]
        trans_fn = IntraTransition.transitionable(src_rvd, dst_rvd)
        assert trans_fn is not None, f"Cannot find primitive: {src_rvd} -> {dst_rvd}"
        # get primitive
        incd = [dim for dim, (d1, d2) in enumerate(zip(src_rvd, dst_rvd)) if d1 < d2][0]
        decd = [dim for dim, (d1, d2) in enumerate(zip(src_rvd, dst_rvd)) if d1 > d2][0]
        chunks = src_rvd[decd] // dst_rvd[decd]
        _, primitive = trans_fn(src_rvd, chunks=chunks)

        # get device spaces
        optional_dims = {0, 1}
        devices = tuple(t.device[0] for t in src_layout.mat.flatten())

        ilayouts: List[RVDLayout] = [src_layout]
        olayouts: List[RVDLayout] = [RVDLayout.grid(src_layout.ftensor, r=dst_rvd[0], v=dst_rvd[1], dims=dst_rvd[2:])]
        # setup ilayout choices
        # add alternative choices for device placement with inner-transpose
        if decd in optional_dims:
            ftensor = src_layout.ftensor
            for k in range(2, src_rvd[decd]):
                if src_rvd[decd] % k != 0: continue
                ilayout = RVDLayout.grid(
                    ftensor, r=src_rvd[0], v=src_rvd[1], dims=src_rvd[2:], devices=devices)
                ilayout.inner_transpose(decd, k)
                ilayouts.append(ilayout)

        # get olayouts with device placement
        rets = []
        for ilayout in ilayouts:
            for olayout in olayouts:
                if len(ilayouts) > 1: olayout = copy.copy(olayout)
                if len(olayouts) > 1: ilayout = copy.copy(ilayout)
                imat = RVDLayout.dim2last(ilayout.mat, decd, chunks)
                omat = RVDLayout.dim2last(olayout.mat, incd, chunks)
                for itensor, otensor in zip(imat.flatten(), omat.flatten()):
                    otensor.cell = itensor.cell
                prims = []
                for itensors, otensors in zip(imat.reshape(-1, chunks), omat.reshape(-1, chunks)):
                    prims.append(primitive(itensors.tolist(), otensors.tolist()))
                rets.append((olayout, prims))
        return rets


class IntraPathFinder:
    """
    intra-RVD Path finder for generating communication plans for RVDLayout
    """
    # Key is configuration.
    # Currently only CompileFlag.disable_reduce_scatter_adapter is considered
    # intra-shard: cached nodes. paths[shape][i][j] = List[int] of indices from (src -> dst]
    _config_cached_intra_nodes: Dict[Tuple, Dict[Tuple[TShape, int], Tuple[TRVD]]] = {}
    _config_cached_intra_edges: Dict[Tuple, Dict[Tuple[TShape, int], np.ndarray]] = {}
    _config_cached_intra_paths: Dict[Tuple, Dict[Tuple[TShape, int], Dict[TRVD, List[List[int]]]]] = {}

    @classproperty
    def _cached_intra_nodes(cls):
        return cls._config_cached_intra_nodes.setdefault((CompileFlag.disable_reduce_scatter_adapter,), {})

    @classproperty
    def _cached_intra_edges(cls):
        return cls._config_cached_intra_edges.setdefault((CompileFlag.disable_reduce_scatter_adapter,), {})

    @classproperty
    def _cached_intra_paths(cls):
        return cls._config_cached_intra_paths.setdefault((CompileFlag.disable_reduce_scatter_adapter,), {})

    # type annotation because type cannot be inferred from `classproperty`
    _cached_intra_nodes: Dict[Tuple[TShape, int], Tuple[TRVD]]
    _cached_intra_edges: Dict[Tuple[TShape, int], np.ndarray]
    _cached_intra_paths: Dict[Tuple[TShape, int], Dict[TRVD, List[List[int]]]]

    @staticmethod
    def path(ilayout: RVDLayout, olayout: RVDLayout,
             cost_fn: Optional[Callable] = None) -> List[IRAdapterPrim]:
        """
        Get primitive path of transforming ilayout into olayout.
        ilayout must have same device set with olayout

        @param ilayout RVDLayout: input tensor layout
        @param olayout RVDLayout: output tensor layout
        @param cost_fn Optional[Callable]: cost function of each primitive.
            Default (None) will use communication volume as metrics

        @return all_primitives List[IRAdapterPrims]: all primitives for communication path
        """
        assert ilayout.ftensor == olayout.ftensor, f"ilayout and olayout should have a same full tensor"
        ftensor = ilayout.ftensor
        src, dst = tuple(ilayout.vec), tuple(olayout.vec)
        rvds = IntraPathFinder.get_optimal_path(ftensor, src, dst, cost_fn)

        # search for correct device mapping
        align, all_prims = IntraPathFinder.device_align(ilayout, olayout, rvds)

        if not align:
            warn_msg = (
                f"Fail to align intra-RVD devices. {ftensor}\n"
                f"Path: {' -> '.join(str(rvd) for rvd in rvds)}\n"
                f"ptensors:\n\t" + "\n\t".join(tensor_vd_repr(ptensor) for ptensor in ilayout.mat.flatten()) + "\n"
                f"ctensors:\n\t" + "\n\t".join(tensor_vd_repr(ctensor) for ctensor in olayout.mat.flatten()) + "\n"
                f"Switch to a fixed plan: ilayout -> FullReplica -> olayout"
            )
            color, default = '\033[33m' , '\033[0m'
            _logger.warning(f'intra-RVD:\n{color+warn_msg+default}')
            all_prims = IntraPathFinder.backup_path(ilayout, olayout, cost_fn)

        return all_prims

    @staticmethod
    def backup_path(ilayout: RVDLayout, olayout: RVDLayout,
                    cost_fn: Optional[Callable] = None) -> List[IRAdapterPrim]:
        """
        Get primitive path of transforming ilayout into olayout.
        ilayout has the same device set with olayout.

        The path generation searches for a default communication plan
        by ilayout -> FullReplica -> olayout.

        @param ilayout RVDLayout: input tensor layout
        @param olayout RVDLayout: output tensor layout
        @param cost_fn Optional[Callable]: cost function of each primitive.
            Default (None) will use transmission volume as metrics

        @return all_primitives List[IRAdapterPrims]: all primitives for communication path
        """
        assert ilayout.ftensor == olayout.ftensor, f"ilayout and olayout should have a same full tensor"
        ftensor = ilayout.ftensor
        src, dst = tuple(ilayout.vec), tuple(olayout.vec)
        # create all-replicate rvd
        rlayout = RVDLayout.grid(ftensor, r=ilayout.ndevs, v=1, dims=tuple(1 for _ in range(ilayout.ndims-2)))
        for rt, ot in zip(rlayout.mat.flatten(), olayout.mat.flatten()):
            rt.cell = ot.cell
        rep = tuple(rlayout.vec)
        # search for left primitives
        left: List[TRVD] = IntraPathFinder.get_optimal_path(ftensor, src, rep, cost_fn)
        align, lprims = IntraPathFinder.device_align(ilayout, rlayout, left)
        assert align, f"Fail to align devices of backup plan at left side: {src} -> {rep}"
        # search
        right: List[TRVD] = IntraPathFinder.get_optimal_path(ftensor, rep, dst, cost_fn)
        align, rprims = IntraPathFinder.device_align(rlayout, olayout, right)
        assert align, f"Fail to align devices of backup plan at right side: {rep} -> {dst}"
        return lprims + rprims

    @staticmethod
    def device_align(ilayout: RVDLayout, olayout: RVDLayout,
                     rvd_path: Tuple[TRVD, ...], _all_prims: Optional[None] = None) -> Tuple[bool, List[IRAdapterPrim]]:
        """
        Align devices for intra-RVD
        We recursively search for the correct device mapping from `ilayout` to `olayout`
        The success of the search is determined by the device placement of `ilayout` and `olayout`.

        `rvd_path` is the transition path from ilayout to olayout.
        The first item can be assumed ilayout (R/V/D are same but device placement may be different in recursive calls),
        and the last item is olayout.

        The exit condition is when the length of `rvd_path` is 1,
        which means ilayout and olayout have the same R/V/D,
        and we just check the device placement are compatible (via `RVDLayout.align`).

        Args:
            ilayouts (RVDLayout): source layout
            olayout (RVDLayout): target layout with correct device mapping
            rvd_hops (Tuple[TRVD, ...]): the hops from ilayout to olayout, which
                contains ilayout and olayout at beginning and last, respectively.
            _all_prims (List[IRAdapterPrim]): the previous primitives, only for recursive calls
        Returns:
            Tuple[bool, List[IRAdapterPrim]]:
                - success bool: True if found device, else False.
                - primitives List[IRAdapterPrim]: the correspoinding primitives
        """
        _all_prims = [] if _all_prims is None else _all_prims
        assert ilayout.vec == rvd_path[0] and olayout.vec == rvd_path[-1]
        if len(rvd_path) == 1:
            if not ilayout.align(olayout):
                return False, []
            return True, _all_prims
        else:
            layout_prims = IntraTransition.transition(ilayout, rvd_path[1])
            for (hop_layout, hop_prims) in layout_prims:
                ret, ret_prims = IntraPathFinder.device_align(
                    hop_layout, olayout, rvd_path[1:], _all_prims + hop_prims)
                if ret:
                    return True, ret_prims
            return False, []

    @staticmethod
    def get_optimal_path(ftensor, src_rvd: TRVD, dst_rvd: TRVD,
                         cost_fn: Optional[Callable] = None) -> Tuple[TRVD]:
        """
        Get optimal RVD path from source RVD to destination RVD

        @param src_rvd Tuple[int]: source RVD
        @param dst_rvd Tuple[int]: destination RVD

        @return path Tuple[Tuple[int]]:
            The first one is src_rvd. The last one is dst_rvd.
            Otherwise they are intermediate RVD status
        """
        # Please note the following int can be either python int or np.int*

        src_rvd, dst_rvd = tuple(src_rvd), tuple(dst_rvd)
        if src_rvd == dst_rvd: return [src_rvd, dst_rvd]

        cost_fn = IntraPathFinder.default_cost_fn if cost_fn is None else cost_fn
        shape = tuple(ftensor.shape)
        ndevs = np.prod(np.array(src_rvd, dtype=int))
        key = (shape, ndevs)

        # get paths using dijkstra algorithm or cached
        if key in IntraPathFinder._cached_intra_paths and src_rvd in IntraPathFinder._cached_intra_paths[key]:
            paths = IntraPathFinder._cached_intra_paths[key][src_rvd]
        else:
            # initialize the graph if not cached
            if key not in IntraPathFinder._cached_intra_nodes:
                nodes, edges = IntraPathFinder.init_graph(ftensor, ndevs, cost_fn)
                IntraPathFinder._cached_intra_nodes[key] = nodes
                IntraPathFinder._cached_intra_edges[key] = edges
                IntraPathFinder._cached_intra_paths[key] = {}
            nodes = IntraPathFinder._cached_intra_nodes[key]
            edges = IntraPathFinder._cached_intra_edges[key]
            # build and initialize cost table
            cost = np.full((len(nodes),), np.inf)
            cost[nodes.index(src_rvd)] = 0
            # setup unvisited and visited set
            unvisited = set(range(len(nodes)))
            visited = set()
            paths = [[] for _ in range(len(nodes))]
            paths[nodes.index(src_rvd)] = [nodes.index(src_rvd)]
            # dijkstra body
            while len(unvisited) > 0:
                min_cost, visit = np.inf, None
                for idx in unvisited:
                    if cost[idx] < min_cost:
                        min_cost = idx
                        visit = idx
                if visit is None: break  # for remaining states that cannot reach
                for neighbor in np.where(edges[visit] != np.inf)[0]:
                    new_cost = cost[visit] + edges[visit, neighbor]
                    if cost[neighbor] == np.inf or new_cost < cost[neighbor]:
                        cost[neighbor] = new_cost
                        paths[neighbor] = paths[visit] + [neighbor]
                unvisited.remove(visit)
                visited.add(visit)
            IntraPathFinder._cached_intra_paths[key][src_rvd] = paths

        # for idx, path in enumerate(paths):
        #     print(f"{src} -> {nodes[idx]}: {' -> '.join([str(nodes[i]) for i in path])} | cost: {cost[idx]}")

        # get layout
        nodes = IntraPathFinder._cached_intra_nodes[key]
        path: List[int] = paths[nodes.index(dst_rvd)]
        rvds: List[Tuple[int]] = [nodes[idx] for idx in path]
        rvds = tuple(
            tuple(int(x) for x in rvd)  # make sure all int (not np.int*) for rvds
            for rvd in rvds
        )
        assert len(path) > 0, f"Un-reachable src RVD ({src_rvd}) -> dst RVD ({dst_rvd})"
        # print(f'get optimal path from {src_rvd} -> {dst_rvd}: {rvds}')
        return rvds

    @staticmethod
    def get_backup_path(ftensor: IRFullTensor, src_rvd: TRVD, dst_rvd: TRVD,
                        cost_fn: Optional[Callable] = None) -> Tuple[TRVD]:
        """
        Get backup path
        """
        rep = (np.prod(np.array(src_rvd)), 1) + (1,) * len(ftensor.shape)
        # search for left primitives
        left: List[TRVD] = IntraPathFinder.get_optimal_path(ftensor, src_rvd, rep, cost_fn)
        # search
        right: List[TRVD] = IntraPathFinder.get_optimal_path(ftensor, rep, dst_rvd, cost_fn)
        # omit right[0] as same with left[-1]
        return left + right[1:]

    @staticmethod
    def get_device_space(ftensor: IRFullTensor, rvd_paths: List[TRVD], placement: Tuple[int, ...]) -> Set[Tuple[int, ...]]:
        """
        Get all possible device placement of the destination RVD given the rvd transition paths.

        Args:
            ftensor (IRFullTensor): the full tensor
            rvd_paths (List[TRVDS]): transition RVD paths from source to destination
            placement (Tuple[int, ...]): device placement of the first RVD in rvd_paths
        Returns:
            Set[Tuple[int, ...]]: all possible device placement of the destination RVD
        """
        init, hops = rvd_paths[0], rvd_paths[1:]
        rvds: List[RVDLayout] = [RVDLayout.grid(ftensor, r=init[0], v=init[1], dims=init[2:], devices=placement)]
        for hop in hops:
            for _ in range(len(rvds)):
                layout = rvds.pop(0)
                rets = IntraTransition.transition(layout, hop)
                for (olayout, _) in rets:
                    rvds.append(olayout)
        devices: Set[Tuple[int]] = set()
        for rvd in rvds:
            assert rvd.vec == tuple(hops[-1])
            devices.add(tuple(t.device[0] for t in rvd.mat.flatten()))
        return devices

    @staticmethod
    def init_graph(ftensor: IRFullTensor, ndevs: int, cost_fn: Optional[Callable] = None) -> Tuple[List[TRVD], np.ndarray]:
        """
        Initialize the graph of RVD status graph.

        @param ftensor IRFullTensor: the full tensor
        @param ndevs int: total device number

        @return nodes Tuple[TRVD]
        @return edges np.ndarray: edges among nodes
        """
        cost_fn = IntraPathFinder.default_cost_fn if cost_fn is None else cost_fn
        nodes = tuple(IntraPathFinder.get_rvd_space(ftensor, ndevs))
        edges = np.full((len(nodes), len(nodes)), np.inf)
        # initialize the cost
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                if i == j: continue
                src, dst = nodes[i], nodes[j]
                if IntraTransition.transitionable(src, dst) is None: continue
                cost = IntraPathFinder.estimate_cost(ftensor, [src, dst], cost_fn)
                edges[i, j] = cost
        return nodes, edges

    @staticmethod
    def get_rvd_space(ftensor: IRFullTensor, ndevs: int) -> List[Tuple[int, ...]]:
        """
        Get all possible RVD representations given ftensor and device number.

        Args:
            ftensor (IRFullTensor): the full tensor
            ndevs (int): the number of devices
        Returns:
            List[Tuple[int, ...]]: all possible RVD representations
        """
        all_layouts: List[int] = []

        def factors(ndevs: int, length: int):
            if length == 1: yield [ndevs]
            else:
                for i in range(1, ndevs + 1):
                    if ndevs % i == 0:
                        for res in factors(ndevs // i, length - 1):
                            yield [i] + res

        for rvd in factors(ndevs, 2+len(ftensor.shape)):
            skip = False
            for dimlen, pnum in zip(ftensor.shape, rvd[2:]):
                if dimlen % pnum != 0:
                    skip = True
                    break
            if not skip:
                all_layouts.append(tuple(rvd))
        return all_layouts

    @staticmethod
    def estimate_cost(ftensor: IRFullTensor, rvd_paths: List[Tuple[TRVD]], cost_fn: Optional[Callable] = None) -> float:
        """
        Estimate transition cost
        """
        cost_fn = IntraPathFinder.default_cost_fn if cost_fn is None else cost_fn
        cost = 0.0
        if len(rvd_paths) == 0: return cost
        if len(rvd_paths) == 2 and (rvd_paths[0] == rvd_paths[1]): return cost
        src, hops = rvd_paths[0], rvd_paths[1:]
        for hop in hops:
            trans_fn = IntraTransition.transitionable(src, hop)
            assert trans_fn is not None, "Fails to find primitive for estimating cost"
            incd = [dim for dim, (d1, d2) in enumerate(zip(src, hop)) if d1 < d2][0]
            decd = [dim for dim, (d1, d2) in enumerate(zip(src, hop)) if d1 > d2][0]
            chunks = src[decd] // hop[decd]
            _, primitive = trans_fn(src, chunks=chunks)
            ilayout: RVDLayout = RVDLayout.grid(ftensor, r=src[0], v=src[1], dims=src[2:])
            olayout: RVDLayout = RVDLayout.grid(ftensor, r=hop[0], v=hop[1], dims=hop[2:])
            imat = RVDLayout.dim2last(ilayout.mat, decd, chunks)
            omat = RVDLayout.dim2last(olayout.mat, incd, chunks)
            prim = primitive(imat.reshape(-1, chunks)[0].tolist(), omat.reshape(-1, chunks)[0].tolist())
            cost += cost_fn(prim)
            src = hop
        return cost

    @staticmethod
    def default_cost_fn(prim: IRAdapterPrim) -> int:
        return prim.volume() + 1 # 1 is hop penalty


class IntraAutoPlacer:

    @staticmethod
    def auto_place(graph: IRSegment, ftensor: IRFullTensor,
                   producers: List[IRCell], consumers: List[IRCell],
                   cost_fn: Optional[Callable] = None) -> List[int]:
        """
        Automatically find good device placement for consumers given the producer placement
        The backward will also be considered.

        @param graph IRSegment
        @param ftensor IRFullTensor
        @param producers List[IRCell]: producers that must be assigned to devices
        @param consumers List[IRCell]: consumers that are about to be assigned

        @return placement List[int]: the adviced placement
            corresponding to each consumer in consumers.
        """
        assert not ftensor.is_param(), f"Cannot automatically assign device given weight tensor"
        assert all(len(p.device) > 0 for p in producers), f"Expect all producers have been assigned to a device"

        devices = [p.device[0] for p in producers]
        assert len(set(devices)) == len(producers),f"Expect each producer is on a different device"

        assert len(producers) == len(consumers), \
            f"Expect same number of producer and consumer, but got {len(producers)} producers and {len(consumers)} consumers"

        if len(producers) == 1:
            return [producers[0].device[0]]

        # reorder producer to match with device order
        producers = sorted(producers, key=lambda n: n.device[0])
        # get forward produced tensors
        fptensors: List[IRSubTensor] = []
        fctensors: List[IRSubTensor] = []
        for producer in producers:
            assert producer in graph.producers(ftensor), f"Producer {producer} doesn't generate ftensor: {ftensor}"
            pidx = graph.producers(ftensor).index(producer)
            fptensors.append(graph.ptensors(ftensor)[pidx])
        for consumer in consumers:
            assert consumer in graph.consumers(ftensor), f"Consumer {producer} doesn't take ftensor: {ftensor}"
            cidx = graph.consumers(ftensor).index(consumer)
            fctensors.append(graph.ctensors(ftensor)[cidx])

        # get backward producer and consumer tensors
        bptensors, bctensors = None, None
        if ftensor.grad is not None:
            bptensors = [t.grad for t in fctensors]
            bctensors = [t.grad for t in fptensors]
        # get RVD representation
        fw_src = RVDLayout.togrid(ftensor, fptensors)
        fw_src_rvd = fw_src.vec
        fw_dst = RVDLayout.togrid(ftensor, fctensors)
        fw_dst_rvd = fw_dst.vec
        bw_src_rvd, bw_dst_rvd = None, None
        if ftensor.grad is not None:
            bw_src_rvd = RVDLayout.togrid(ftensor.grad, bptensors).vec
            bw_dst_rvd = RVDLayout.togrid(ftensor.grad, bctensors).vec

        # get placement advice
        devices = [t.device[0] for t in fw_src.mat.flatten()]
        placement, _ = IntraAutoPlacer.advice(
            ftensor.shape,
            fw_src_rvd, fw_dst_rvd, bw_src_rvd, bw_dst_rvd,
            devices, cost_fn)

        # assign to device
        ordered_placement = [None] * len(consumers)
        for devid, t in zip(placement, fw_dst.mat.flatten()):
            ordered_placement[consumers.index(t.cell)] = devid
        assert all(devid is not None for devid in ordered_placement), f"Internal Error"

        return ordered_placement

    @staticmethod
    def advice(shape: TShape,
               fw_src_rvd: TRVD, fw_dst_rvd: TRVD,
               bw_src_rvd: Optional[TRVD], bw_dst_rvd: Optional[TRVD],
               src_placement: List[int],
               cost_fn: Optional[Callable] = None) -> Tuple[Tuple[int, ...], float]:
        """
        Search for a good device placement for destination RVD partition (fw_dst_rvd and bw_src_rvd)

        Args:
            shape (Tuple[int]): full tensor shape
            fw_src_rvd (TRVD): forward producer RVD layout vector
            fw_dst_rvd (TRVD): forward consumer RVD layout vector
            bw_src_rvd (Optional[TRVD]): backward producer RVD layout vector
            bw_dst_rvd (Optional[TRVD]): backward consumer RVD layout vector
            src_placement (List[int]): device placement of source RVD
            cost_fn (Optional[Callable]): cost function of each primitive.
                Default (None) will use communication volume as metrics
        Returns:
            Tuple[int, ...]: best device placement for RVD tensors
            float: Cost of communication plan
        """
        src_placement = tuple(src_placement)
        ftensor = IRFullTensor(shape, dtype=torch.float16)
        cost_fn = IntraPathFinder.default_cost_fn if cost_fn is None else cost_fn

        # forward pass
        fw_rvd_hops = IntraPathFinder.get_optimal_path(
            ftensor, fw_src_rvd, fw_dst_rvd, cost_fn=cost_fn)
        fw_consumer_devices: Set[Tuple[int]] = IntraPathFinder.get_device_space(
            ftensor, fw_rvd_hops, src_placement)

        # backward pass
        if (bw_src_rvd is not None) and (bw_dst_rvd is not None):
            bw_rvd_hops = IntraPathFinder.get_optimal_path(
                ftensor, bw_src_rvd, bw_dst_rvd, cost_fn=cost_fn)
            devices = set()
            for bw_producer_devs in fw_consumer_devices:
                bw_consumer_devices = IntraPathFinder.get_device_space(
                    ftensor, bw_rvd_hops, bw_producer_devs
                )
                # FIXME: this comparison on tuples some misses possible placement
                # that can be actually aligned by using layout.align (false possitive).
                if src_placement in bw_consumer_devices:
                    devices.add(bw_producer_devs)
                    break
        else:
            devices = fw_consumer_devices

        placement = None
        # - if find, choose one
        # FIXME: looks the above code (`devices = fw_consumer_devices` as a fallback) should be removed.
        # so here we check whether we have found a valid placement.
        if len(devices) > 0:
            placement = list(devices)[0]
        # - if not find, keep forward one as optimal and adopt backup plan for backward one
        else:
            placement = list(fw_consumer_devices)[0]
            msg = (f"================ forward-backward mis-aligned! ============== \n"
                  f"fw device choices: {fw_consumer_devices} | hops: {'->'.join(str(rvd) for rvd in fw_rvd_hops)}\n"
                  f"bw hops: {'->'.join(str(rvd) for rvd in bw_rvd_hops)}\n"
                  f"using placement: {placement}\n"
                  f"=============================================================")
            _logger.warning(f'intra-RVD:\n{msg}')
            bw_rvd_hops = IntraPathFinder.get_backup_path(ftensor, bw_src_rvd, bw_dst_rvd, cost_fn)

        # estimate cost
        cost = IntraPathFinder.estimate_cost(ftensor, fw_rvd_hops, cost_fn)
        if (bw_src_rvd is not None) and (bw_dst_rvd is not None):
            cost += IntraPathFinder.estimate_cost(ftensor, bw_rvd_hops, cost_fn)
        return placement, cost
