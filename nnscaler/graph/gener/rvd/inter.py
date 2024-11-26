#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from typing import Callable, Dict, List, Tuple, Optional, Set, Union
from functools import partial
import numpy as np

from nnscaler.ir.tensor import IRFullTensor

from nnscaler.ir.adapter.prim import IRAdapterPrim
from nnscaler.ir.adapter.prim import MovePrim           # p2p
from nnscaler.ir.adapter.prim import BroadcastPrim
from nnscaler.ir.adapter.prim import RDScatterPrim, RVScatterPrim
from nnscaler.ir.adapter.prim import RDGatherPrim, RVGatherPrim

from nnscaler.graph.gener.rvd.layout import RVDLayout
from nnscaler.graph.gener.rvd.intra import IntraPathFinder

from nnscaler.utils import classproperty
from nnscaler.flags import CompileFlag


TShape = Tuple[int, ...]
TRVD = Tuple[int, ...]
InterRVD = Tuple[str, int,] # ('p', 2, 1, 1, ...) or ('c', 2, 1, 1, ...)


class InterTransition:
    """
    Inter-RVD transition primitives
    """

    @staticmethod
    def incr(rvd: TRVD, chunks: int) -> Tuple[TRVD, Callable]:
        """
        Inter-RVD extended primitive: increase replica number

        @param rvd Tuple[int]: source rvd
        @param chunks int: the number to multiply

        @return rvd Tuple[int]: transformed RVD
        @return prim Callable: primitive class
        """
        rvd = list(rvd)
        rvd[0] = rvd[0] * chunks
        return rvd, MovePrim if chunks == 1 else BroadcastPrim

    @staticmethod
    def decr(rvd: TRVD, chunks: int) -> Tuple[TRVD, Callable]:
        """
        Inter-RVD extended primitive: decrease replica number

        @param rvd Tuple[int]: source rvd
        @param chunks int: the number to divide

        @return rvd Tuple[int]: transformed RVD
        @return prim Callable: primitive class
        """
        assert rvd[0] % chunks == 0, f"not divisible replica {rvd[0]} // {chunks}"
        rvd = list(rvd)
        rvd[0] = rvd[0] // chunks
        return rvd, MovePrim

    @staticmethod
    def incd(rvd: TRVD, dim: int, chunks: int) -> Tuple[TRVD, Callable]:
        """
        Inter-RVD extended primitive: increase tensor dimension partition

        @param rvd Tuple[int]: source rvd
        @param dim int: the tensor axes to increase
        @param chunks int: the number to multiply

        @return rvd Tuple[int]: transformed RVD
        @return prim Callable: primitive class
        """
        rvd = list(rvd)
        rvd[2+dim] = rvd[2+dim] * chunks
        return rvd, partial(RDScatterPrim, dim=dim)

    @staticmethod
    def decd(rvd: TRVD, dim: int, chunks: int) -> Tuple[TRVD, Callable]:
        """
        Inter-RVD extended primitive: decrease tensor dimension partition

        @param rvd Tuple[int]: source rvd
        @param dim int: the tensor axes to decrease
        @param chunks int: the number to divide

        @return rvd Tuple[int]: transformed RVD
        @return prim Callable: primitive class
        """
        assert rvd[2+dim] % chunks == 0, f"not divisible dim: {rvd[2+dim]} % {chunks} != 0"
        rvd = list(rvd)
        rvd[2+dim] = rvd[2+dim] // chunks
        return rvd, partial(RDGatherPrim, dim=dim)

    @staticmethod
    def incv(rvd: TRVD, chunks: int) -> Tuple[TRVD, Callable]:
        """
        Inter-RVD extended primitive: increase value partition

        @param rvd Tuple[int]: source rvd
        @param chunks int: the number to multiply

        @return rvd Tuple[int]: transformed RVD
        @return prim Callable: primitive class
        """
        rvd = list(rvd)
        rvd[1] *= chunks
        return rvd, RVScatterPrim

    @staticmethod
    def decv(rvd: TRVD, chunks: int) -> Tuple[TRVD, Callable]:
        """
        Inter-RVD extended primitive: decrease value partition

        @param rvd Tuple[int]: source rvd
        @param chunks int: the number to divide

        @return rvd Tuple[int]: transformed RVD
        @return prim Callable: primitive class
        """
        assert rvd[1] % chunks == 0, f"not divisable value split: {rvd[1]} % {chunks} != 0"
        rvd = list(rvd)
        rvd[1] = rvd[1] // chunks
        return rvd, RVGatherPrim

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
        if len(incd) == 0 and len(decd) == 0:
            decd = [0]
        # only support one dimension change
        # this only happens when the device number changes
        if len(incd) + len(decd) != 1: return trans_fn
        if len(incd) == 1:
            incd = incd[0]
            if incd == 0: # incr
                return InterTransition.incr
            elif incd == 1:
                return InterTransition.incv
            else:
                return partial(InterTransition.incd, dim=incd-2)
        else:
            decd = decd[0]
            if decd == 0: # decr
                return InterTransition.decr
            elif decd == 1:
                return InterTransition.decv
            else:
                return partial(InterTransition.decd, dim=decd-2)

    @staticmethod
    def transition(src_layout: RVDLayout, dst_rvd: TRVD, placement: Optional[Tuple[int]] = None) -> Tuple[RVDLayout, List[IRAdapterPrim]]:
        """
        Transfer from source RVD to destination RVD.
        Get all possible device-placement choices for RVD
        given the fixed device placement of RVD.

        @param src_layout RVDLayout: source ilayout
        @param dst_rvd Tuple[int]: destination RVD
        @param placement Tuple[int]: output layout device placement

        @return rets Tuple[GridLayout, List[IRAdapterPrim]]:
           pairs of <layout, [prims]> of output
        """

        src_rvd = src_layout.vec
        ftensor = src_layout.ftensor
        dst_layout: RVDLayout = RVDLayout.grid(ftensor, r=dst_rvd[0], v=dst_rvd[1], dims=dst_rvd[2:], devices=placement)
        trans_fn = InterTransition.transitionable(src_rvd, dst_rvd)
        assert trans_fn is not None, f"Cannot find primitive: {src_rvd} -> {dst_rvd}"
        # get primitive
        incd = [dim for dim, (d1, d2) in enumerate(zip(src_rvd, dst_rvd)) if d1 < d2]
        decd = [dim for dim, (d1, d2) in enumerate(zip(src_rvd, dst_rvd)) if d1 > d2]
        if len(incd) == 0 and len(decd) == 0:
            decd = [0]

        if len(incd) == 1:
            change_dim = incd[0]
            chunks = dst_rvd[change_dim] // src_rvd[change_dim]
        else:
            change_dim = decd[0]
            chunks = src_rvd[change_dim] // dst_rvd[change_dim]
        _, primitive = trans_fn(src_rvd, chunks=chunks)

        imat = RVDLayout.dim2last(src_layout.mat, change_dim, src_rvd[change_dim])
        omat = RVDLayout.dim2last(dst_layout.mat, change_dim, dst_rvd[change_dim])

        prims = []
        if len(incd) == 1:
            for src, dsts in zip(imat.flatten(), omat.reshape(-1, chunks)):
                dsts = dsts.tolist()
                if primitive is BroadcastPrim:
                    dsts = [src] + dsts
                prims.append(primitive([src], dsts))
        else:
            for srcs, dst in zip(imat.reshape(-1, chunks), omat.flatten()):
                srcs = srcs.tolist()
                if primitive is MovePrim:
                    srcs = [srcs[0]]
                prims.append(primitive(srcs, [dst]))
        return dst_layout, prims


class InterPathFinder:
    """
    inter-RVD Path finder for generating communication plans for RVDLayout
    """
    # Key is configuration.
    # Currently only CompileFlag.disable_reduce_scatter_adapter is considered
    _config_cached_inter_nodes: Dict[Tuple, Dict[Tuple[TShape, int, int], Tuple[Tuple[InterRVD]]]] = {}
    _config_cached_inter_edges: Dict[Tuple, Dict[Tuple[TShape, int, int], Tuple[np.ndarray]]] = {}
    _config_cached_inter_paths: Dict[Tuple, Dict[Tuple[TShape, int, int], Dict[TRVD, List[List[int]]]]] = {}

    @classproperty
    def _cached_inter_nodes(cls):
        return cls._config_cached_inter_nodes.setdefault((CompileFlag.disable_reduce_scatter_adapter,), {})

    @classproperty
    def _cached_inter_edges(cls):
        return cls._config_cached_inter_edges.setdefault((CompileFlag.disable_reduce_scatter_adapter,), {})

    @classproperty
    def _cached_inter_paths(cls):
        return cls._config_cached_inter_paths.setdefault((CompileFlag.disable_reduce_scatter_adapter,), {})

    # type annotation because type cannot be inferred from `classproperty`
    _cached_inter_nodes: Dict[Tuple[TShape, int, int], Tuple[Tuple[InterRVD]]]
    _cached_inter_edges: Dict[Tuple[TShape, int, int], Tuple[np.ndarray]]
    _cached_inter_paths: Dict[Tuple[TShape, int, int], Dict[TRVD, List[List[int]]]]

    @staticmethod
    def path(ilayout: RVDLayout, olayout: RVDLayout, cost_fn: Optional[Callable] = None) -> List[IRAdapterPrim]:
        """
        Get primitive path of transforming ilayout into olayout.
        ilayout must locate on different device set of olayout

        @param ilayout RVDLayout: input tensor layout
        @param olayout RVDLayout: output tensor layout
        @param cost_fn Optional[Callable]: cost function of each primitive.
            Default (None) will use transmission volume as metrics

        @return all_primitives List[IRAdapterPrims]: all primitives for communication path
        """
        ftensor: IRFullTensor = ilayout.ftensor
        cost_fn = InterPathFinder.default_cost_fn if cost_fn is None else cost_fn

        inter_rvds: List[InterRVD] = InterPathFinder.get_optimal_path(
            ftensor, ilayout.vec, olayout.vec, cost_fn)

        all_prims = InterPathFinder.device_align(ilayout, olayout, inter_rvds)
        return all_prims

    @staticmethod
    def device_align(ilayout: RVDLayout, olayout: RVDLayout,
                     rvd_paths: Tuple[InterRVD]) -> Tuple[IRAdapterPrim]:
        """
        Align devices for inter-RVD

        @param ilayouts List[RVDLayout]: searched layouts
        @param olayout RVDLayout: target layout with correct device mapping
        @param rvd_hops: Tuple[TRVD]: the hops from ilayout to olayout, which
            contains ilayout and olayout at beginning and last, respectively.

        @return primitives List[IRAdapterPrim]: the correspoinding primitives
        """
        # decode producer and consumer part
        prvds, crvds = InterPathFinder.decode(rvd_paths)

        # get possible consumer deivce space: try with reversed path
        cdev_space = IntraPathFinder.get_device_space(
            olayout.ftensor, crvds[::-1],
            tuple(t.device[0] for t in olayout.mat.flatten()))

        # setup producer primitives
        producer_out_devs = None
        pdev_space = IntraPathFinder.get_device_space(
            ilayout.ftensor, prvds,
            tuple(t.device[0] for t in ilayout.mat.flatten())
        )
        for pdevs in pdev_space:
            producer_out_devs = pdevs
            playout = RVDLayout.grid(
                ilayout.ftensor, r=prvds[-1][0], v=prvds[-1][1],
                dims=prvds[-1][2:], devices=pdevs
            )
            align, pprims = IntraPathFinder.device_align(ilayout, playout, prvds)
            assert align, "Internal Error: inter-rvd producer side device fails to align"
            break # we only take the first one
        assert producer_out_devs is not None, f"Can't find inter-rvd producer out device placement"

        # setup consumer primitives and entry device placement
        consumer_entry_devs = None
        for cdevs in cdev_space:
            clayout = RVDLayout.grid(
                olayout.ftensor, r=crvds[0][0], v=crvds[0][1],
                dims=crvds[0][2:], devices=cdevs)
            align, cprims = IntraPathFinder.device_align(clayout, olayout, crvds)
            if align:
                consumer_entry_devs = cdevs
                break
        assert consumer_entry_devs is not None, f"Can't find inter-rvd consumer entry device placement."

        # setup inter-primitive
        _, iprims = InterTransition.transition(playout, crvds[0], consumer_entry_devs)

        # merge together
        return pprims + iprims + cprims

    @staticmethod
    def get_optimal_path(ftensor, src_rvd: TRVD, dst_rvd: TRVD, cost_fn: Optional[Callable] = None) -> List[InterRVD]:
        """
        Get optimal RVD path from source RVD to destination RVD

        @param src_rvd Tuple[int]: source RVD
        @param dst_rvd Tuple[int]: destination RVD

        @return path Tuple[InterRVD]:
            The first one is src_rvd. The last one is dst_rvd.
            Otherwise they are intermediate RVD status
        """
        # Please note the following int can be either python int or np.int*

        src_ndevs = np.prod(src_rvd)
        src = ('p',) + src_rvd
        dst_ndevs = np.prod(dst_rvd)
        dst = ('c',) + dst_rvd

        key = (tuple(ftensor.shape), np.prod(src_rvd), np.prod(dst_rvd))

        if key in InterPathFinder._cached_inter_nodes and src in InterPathFinder._cached_inter_paths[key]:
            nodes = InterPathFinder._cached_inter_nodes[key]
            paths = InterPathFinder._cached_inter_paths[key][src]
        else:
            if key in InterPathFinder._cached_inter_nodes:
                nodes = InterPathFinder._cached_inter_nodes[key]
                edges = InterPathFinder._cached_inter_edges[key]
            else:
                nodes, edges = InterPathFinder.init_graph(ftensor, src_ndevs, dst_ndevs, cost_fn)
                InterPathFinder._cached_inter_nodes[key] = nodes
                InterPathFinder._cached_inter_edges[key] = edges
                InterPathFinder._cached_inter_paths[key] = {}
            # build cost
            cost = np.full((len(nodes),), np.inf)
            cost[nodes.index(src)] = 0
            # setup unvisited and visited set
            unvisited = set(range(len(nodes)))
            visited = set()
            paths = [[] for _ in range(len(nodes))]
            paths[nodes.index(src)] = [nodes.index(src)]
            # dijkstra body
            while len(unvisited) > 0:
                min_cost, visit = np.inf, None
                for idx in unvisited:
                    if cost[idx] < min_cost:
                        min_cost = idx
                        visit = idx
                if visit is None: break
                for neighbor in np.where(edges[visit] != np.inf)[0]:
                    new_cost = cost[visit] + edges[visit, neighbor]
                    if cost[neighbor] == np.inf or new_cost < cost[neighbor]:
                        cost[neighbor] = new_cost
                        paths[neighbor] = paths[visit] + [neighbor]
                unvisited.remove(visit)
                visited.add(visit)
            InterPathFinder._cached_inter_paths[key][src] = paths

        # print for debug
        # for idx, path in enumerate(paths):
        #     print(f"{src} -> {nodes[idx]}: {' -> '.join([str(nodes[i]) for i in path])} | cost: {cost[idx]}")

        path = paths[nodes.index(dst)]
        assert len(path) > 0, f"Un-reachable src RVD {src} -> dst RVD {dst}"
        inter_rvds = tuple(nodes[idx] for idx in path)
        inter_rvds = tuple(
            (rvd[0],) + tuple(int(x) for x in rvd[1:])  # make sure all int (not np.int*) for rvd[1:]
            for rvd in inter_rvds
        )
        return inter_rvds

    @staticmethod
    def init_graph(ftensor: IRFullTensor, src_ndevs: int, dst_ndevs: int, cost_fn: Callable) -> Tuple[List[TRVD], np.ndarray]:
        """
        Initialize the graph of RVD status graph.

        An additional positition tage is append to at the first element of each node, i.e.,
            For source (producer) layout: ('p', 2,1,1,2) means <R(2), V(1), D(1,2)>
            For dest (consumer) layout: ('c', 2,1,1,2) means <R(2), V(1), D(1,2)>

        @param ftensor IRFullTensor: the full tensor
        @param idevs int: total device number of source tensor

        @return nodes Tuple[TRVD]
        @return edges np.ndarray: edges among nodes
        """
        shape = tuple(ftensor.shape)

        if (shape, src_ndevs) in IntraPathFinder._cached_intra_nodes:
            src_nodes = IntraPathFinder._cached_intra_nodes[(shape, src_ndevs)]
            src_edges = IntraPathFinder._cached_intra_edges[(shape, src_ndevs)]
        else:
            src_nodes, src_edges = IntraPathFinder.init_graph(ftensor, src_ndevs, cost_fn)
            IntraPathFinder._cached_intra_nodes[(shape, src_ndevs)] = src_nodes
            IntraPathFinder._cached_intra_edges[(shape, src_ndevs)] = src_edges
            IntraPathFinder._cached_intra_paths[(shape, src_ndevs)] = {}

        if (shape, dst_ndevs) in IntraPathFinder._cached_intra_nodes:
            dst_nodes = IntraPathFinder._cached_intra_nodes[(shape, dst_ndevs)]
            dst_edges = IntraPathFinder._cached_intra_edges[(shape, dst_ndevs)]
        else:
            dst_nodes, dst_edges = IntraPathFinder.init_graph(ftensor, dst_ndevs, cost_fn)
            IntraPathFinder._cached_intra_nodes[(shape, dst_ndevs)] = dst_nodes
            IntraPathFinder._cached_intra_edges[(shape, dst_ndevs)] = dst_edges
            IntraPathFinder._cached_intra_paths[(shape, dst_ndevs)] = {}
        nodes = tuple(('p',) + n for n in src_nodes ) + tuple(('c',) + n for n in dst_nodes)
        edges = np.full((len(nodes), len(nodes)), np.inf)
        edges[:len(src_nodes), :len(src_nodes)] = src_edges
        edges[len(src_nodes):,len(src_nodes):] = dst_edges
        # NVLink: 300GBps Inter-node: 100Gbps
        for i in range(len(src_nodes)):
            for j in range(len(dst_nodes)):
                src, dst = src_nodes[i], dst_nodes[j]
                if InterTransition.transitionable(src, dst) is None: continue
                cost = InterPathFinder.estimate_cost(
                    ftensor, (('p',) + src, ('c',) + dst), cost_fn)
                # set for [i, len(src_nodes) + j]
                edges[i, len(src_nodes) + j] = cost
                # set for [len(src_nodes) + j, i]
                edges[len(src_nodes) + j, i] = cost
        return nodes, edges

    @staticmethod
    def decode(inter_rvds: Tuple[InterRVD]) -> Tuple[Tuple[TRVD], Tuple[TRVD]]:
        """
        Decode searched inter-rvd paths into intra-rvd representations (TRVD)
        for producer and consumer side.

        @param inter_rvds Tuple[InterRVD]

        @return prvds Tuple[TRVD]: rvd paths of producer side
        @return crvds Tuple[TRVD]: rvd paths of consumer side
        """
        bps = [idx for idx in range(len(inter_rvds) - 1) if inter_rvds[idx][0] != inter_rvds[idx+1][0]]
        assert len(bps) == 1, \
            f"Expect path to be producer intra-rvd -> inter -> consumer intra-rvd: {inter_rvds}"
        bp = bps[0]

        prvds = tuple(rvd for rvd in inter_rvds[:bp+1])
        assert all(rvd[0] == 'p' for rvd in prvds)
        prvds = tuple(rvd[1:] for rvd in prvds)
        if len(prvds) == 1:
            prvds = prvds * 2

        crvds = tuple(rvd for rvd in inter_rvds if rvd[0] == 'c')
        assert all(rvd[0] == 'c' for rvd in crvds)
        crvds = tuple(rvd[1:] for rvd in crvds)
        if len(crvds) == 1:
            crvds = crvds * 2

        return prvds, crvds

    @staticmethod
    def estimate_cost(ftensor: IRFullTensor, rvd_paths: Tuple[InterRVD],
                      cost_fn: Optional[Callable] = None) -> float:
        """
        Estimate transition cost

        @return cost float
        """
        cost_fn = InterPathFinder.default_cost_fn if cost_fn is None else cost_fn
        # decode producer and consumer part
        prvds, crvds = InterPathFinder.decode(rvd_paths)
        # producer cost
        pcost = IntraPathFinder.estimate_cost(ftensor, prvds, cost_fn)
        # consumer cost
        ccost = IntraPathFinder.estimate_cost(ftensor, crvds, cost_fn)
        # inter-cost
        pndevs = np.prod(prvds[-1])
        cndevs = np.prod(crvds[0])
        playout = RVDLayout.grid(
            ftensor, r=prvds[-1][0], v=prvds[-1][1],
            dims=prvds[-1][2:], devices=list(range(pndevs)))
        _, prims = InterTransition.transition(playout, crvds[0], list(range(pndevs, pndevs + cndevs)))
        icost = cost_fn(prims[0])
        # gather all
        # consider differnt linkbandwidth intra NVLink 300GB/s vs. inter-node 100Gbps
        comm_factor = 24
        return pcost + ccost + icost * comm_factor

    @staticmethod
    def default_cost_fn(prim: IRAdapterPrim) -> int:
        return prim.volume() + 1 # 1 is hop penalty
