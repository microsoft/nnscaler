import networkx as nx
from tqdm import tqdm
from typing import List, Dict, Set, Hashable

from verdict.graph import DFG, Node, Tensor, Lineage
from verdict.log import logwarn, logerr, loginfo, logdebug
from verdict.operators import OpName
from verdict.utils import idempotent_update

from .stage import Stage

nxTensor = "tensor"
nxOp = "op"

        
def _DFG_to_nxG(dfg: DFG) -> nx.DiGraph:
    nodes = dfg.nodes()
    nxG = nx.DiGraph()
    added = set()
    for node in nodes:
        nxG.add_node(node, type=nxOp)
        for t in dfg.node_inputs(node):
            if t not in added:
                nxG.add_node(t, type=nxTensor)
                added.add(t)
            nxG.add_edge(t, node)
        for t in dfg.node_outputs(node):
            if t not in added:
                nxG.add_node(t, type=nxTensor)
                added.add(t)
            nxG.add_edge(node, t)
    return nxG


def _bw_slice_with_terminable(
    graph: nx.DiGraph,
    target_nodes: List[Hashable],
    terminable_nodes: Set[Hashable],
) -> Set[Hashable]:
    visited = set()
    queue = list(target_nodes)
    frontier = set()
    while queue:
        node = queue.pop()
        if node in visited:
            continue
        visited.add(node)
        # if a node is specified to terminate its backward slicing
        if node in terminable_nodes:
            frontier.add(node)
            continue
        for pred in graph.predecessors(node):
            if pred not in visited:
                queue.append(pred)
    return frontier


def _slice_subgraph_w_barrier(
    graph: nx.DiGraph,
    reversed_graph: nx.DiGraph,
    sources: list[Hashable],  # verified tensors
    targets: list[Hashable],  # tensors to verify
) -> nx.DiGraph:
    """Only allow paths from target to source that do NOT traverse through other sources"""
    barrier = set(sources)
    result_nodes = set()
    for target in targets:
        if target not in reversed_graph:
            continue
        stack = [target]
        visited = set()
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            if node in barrier:
                continue  # stop here
            result_nodes.add(node)
            stack.extend(
                reversed_graph.successors(node)
            )  # successors in reversed graph = predecessors in original
    return graph.subgraph(result_nodes | set(targets))


def _is_lng_initialized(l: Lineage, Gs: DFG, Gp: DFG) -> bool:
    return Gs.is_initialized(l.Ts) and all(Gp.is_initialized(Tp) for Tp in l.Tps)


def cut_stages(Gs: DFG, Gp: DFG, ordered_lngs: List[Lineage]) -> List[Stage]:
    """Lineage-guided stage cutting."""
    nxGs = _DFG_to_nxG(Gs)
    nxGp = _DFG_to_nxG(Gp)
    nxGs_rev = nxGs.reverse(copy=False)
    nxGp_rev = nxGp.reverse(copy=False)
    
    verified_Ts_lngs: Dict[Tensor, List[Lineage]] = {}
    stages: List[Stage] = []
    stage_i: int = 0
    visited: Set[Lineage] = set()
    Tp2lngs: Dict[Tensor, List[Lineage]] = {}
    
    for i, target_lng in tqdm(enumerate(ordered_lngs), total=len(ordered_lngs), desc="scheduling from lineages"):
        # 0.1 if a lng is initialized, no need to form an empty stage
        if _is_lng_initialized(target_lng, Gs, Gp):
            verified_Ts_lngs.setdefault(target_lng.Ts, []).append(target_lng)
            for Tp in target_lng.Tps:
                Tp2lngs.setdefault(Tp, []).append(target_lng)
            visited.add(target_lng)
            continue

        # 0.2 if a lng has already formed a stage as its output, no need to reform
        if target_lng in visited:
            continue
        visited.add(target_lng)
        
        if target_lng.is_stage_known:
            # 1. if a lng is hinted a formed stage, directly form it
            stage = Stage(
                id=stage_i,
                snodes=target_lng.snodes,
                pnodes=target_lng.pnodes,
                input_lineages=target_lng.dependency,
                output_lineages=[target_lng],
            )
            stage_i += 1
            stages.append(stage)
        else:
            logdebug("cut stage by graph slicing", target_lng=i)
            # 1. start from the Ts, get the src tensors in Gs
            Gs_target: Tensor = target_lng.Ts
            Gs_srcs: Set[Tensor] = _bw_slice_with_terminable(
                nxGs, [Gs_target], verified_Ts_lngs
            )
            # 2. determine the src boundary for Gp based on lineages
            Gp_targets: List[Tensor] = target_lng.Tps
            Gp_srcs: Set[Tensor] = set()
            src_lngs = []

            def find_nearest_lng_for_each_Gs_src():
                # NOTE: for simplilicity and efficiency, we assume target_sample should work
                # if not, redesign this function later
                target_sample = Gp_targets[0]
                for Ts in Gs_srcs:
                    src_lng_candidates = set(verified_Ts_lngs[Ts])
                    closest: Lineage | None = None
                    for parent, children in nx.bfs_successors(nxGp_rev, target_sample):
                        # target_sample is a Tp, which can also be already verified
                        # as Gs contain multiref
                        if parent == target_sample:
                            children.append(parent)
                        for element in children:
                            if element not in Tp2lngs:
                                continue
                            for lng in set(Tp2lngs[element]):
                                if lng in src_lng_candidates:
                                    closest = lng
                                    break
                            if closest is not None:
                                break
                        if closest is not None:
                            break
                    if closest is None:
                        logerr(
                            "Target Tp is not reachable from any source Tp bound to Ts's lineage.",
                            target_Tp=target_sample,
                            Ts=Ts,
                            Tp_heads=[l.Tps[0] for l in src_lng_candidates],
                        )
                        raise RuntimeError
                    src_lngs.append(closest)
                    Gp_srcs.update(closest.Tps)

            find_nearest_lng_for_each_Gs_src()

            # 3. cut the subgraph
            sub_nxGs = _slice_subgraph_w_barrier(nxGs, nxGs_rev, Gs_srcs, [Gs_target])
            sub_nxGp = _slice_subgraph_w_barrier(nxGp, nxGp_rev, Gp_srcs, Gp_targets)

            # 4. form a stage
            is_op = lambda n, G: G.nodes[n]["type"] == nxOp
            sub_snodes = [n for n in nx.topological_sort(sub_nxGs) if is_op(n, sub_nxGs)]
            sub_pnodes = [n for n in nx.topological_sort(sub_nxGp) if is_op(n, sub_nxGp)]
            stage = Stage(
                id=stage_i,
                snodes=sub_snodes,
                pnodes=sub_pnodes,
                input_lineages=src_lngs,
                output_lineages=[target_lng],
            )
            stage_i += 1
            stages.append(stage)
            
        # set the target_lng as verified; also record inverse mapping
        verified_Ts_lngs.setdefault(target_lng.Ts, []).append(target_lng)
        for Tp in target_lng.Tps:
            Tp2lngs.setdefault(Tp, []).append(target_lng)
            
    stages = _prune_and_consolidate(stages)
    
    _empirical_check_stage_sizes(stages, Gs, Gp)
    _check_input_equivs_validity(stages, Gs, Gp)

    return stages


def _prune_and_consolidate(stages: List[Stage]) -> List[Stage]:
    """Prune and consolidate stages."""
    # 1. remove a stage if nodes are empty
    # 2. hash and consolidate stages that have exactly the same nodes
    ret: List[Stage] = []
    stage_i: int = 0
    nodes_to_stage: Dict[Node, Stage] = {}
    for stage in stages:
        if len(stage.snodes) == len(stage.pnodes) == 0:
            continue
        nodes_as_key = tuple(sorted(set(stage.snodes + stage.pnodes)))
        if nodes_as_key in nodes_to_stage:
            # if there is a prior stage already doing the same work
            # assert check input lineages are the same,
            # then merge the output lineages
            assert nodes_to_stage[nodes_as_key].input_lineages == stage.input_lineages
            nodes_to_stage[nodes_as_key].output_lineages.extend(stage.output_lineages)
        else:
            stage_i += 1
            stage.id = stage_i
            nodes_to_stage[nodes_as_key] = stage
            ret.append(stage)
    return ret


def _is_original_op(opname: OpName) -> bool:
    return not (
        opname.value[1] is None
        or opname.value[0]
        in [
            "identity",
            "multiref",
        ]
    )


def _empirical_check_stage_sizes(stages: List[Stage], Gs: DFG, Gp: DFG) -> None:
    # empirical warning
    expected_stage_size = Gp.W.num_dp * Gp.W.num_tp * Gp.W.num_mb
    for stage in stages:
        stage_i = stage.id
        snodes = stage.snodes
        pnodes = stage.pnodes
        # count original ops
        num_original_ops_Gs = sum(
            [1 for node in snodes if _is_original_op(Gs.node_opname(node))]
        )
        num_original_ops_Gp = sum(
            [1 for node in pnodes if _is_original_op(Gp.node_opname(node))]
        )
        # warn the general large stage
        if len(pnodes) > 10 * expected_stage_size:
            logwarn(
                "Too many nodes for a stage.",
                stage=stage_i,
                num_snodes=len(snodes),
                num_pnodes=len(pnodes),
                num_original_ops_Gs=num_original_ops_Gs,
                num_original_ops_Gp=num_original_ops_Gp,
            )
        # warn the large adapter stage
        if len(snodes) == 0:
            if num_original_ops_Gp > 0:
                logwarn(
                    "Large adapter stage.",
                    stage=stage_i,
                    num_snodes=len(snodes),
                    num_pnodes=len(pnodes),
                    num_original_ops_Gs=num_original_ops_Gs,
                    num_original_ops_Gp=num_original_ops_Gp,
                )
        # warn the large comp stage
        if len(snodes) > 0:
            if num_original_ops_Gp > expected_stage_size:
                logwarn(
                    "Large computation stage",
                    stage=stage_i,
                    num_snodes=len(snodes),
                    num_pnodes=len(pnodes),
                    num_original_ops_Gs=num_original_ops_Gs,
                    num_original_ops_Gp=num_original_ops_Gp,
                )


def _check_input_equivs_validity(stages: List[Stage], Gs: DFG, Gp: DFG) -> None:
    """Check tensor consistency between stages; ensuring all input lineages are initialized or produced by prior stages."""
    verified_lineages: Set[Lineage] = set()
    for stage in stages:
        for l in stage.input_lineages:
            if l not in verified_lineages:
                assert Gs.is_initialized(l.Ts)
                assert all(Gp.is_initialized(Tp) for Tp in l.Tps)
            verified_lineages.add(l)
        for l in stage.output_lineages:
            verified_lineages.add(l)
    return
