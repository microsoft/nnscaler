#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from typing import List, Dict
from pathlib import Path

from verdict.graph import Node, DFG, Lineage
from verdict.log import logdebug


def dump_nodes(nodes: List[Node], G: DFG, path: Path | str | None = None) -> str:
    lines: List[str] = []
    for node in nodes:
        lines.append(f"{node} {G.node_opname(node)}")
        if G._node2irstr:
            lines.append(G._node2irstr[node])
        lines.append(f"{G.node_inputs(node)}")
        lines.append(f"{G.node_outputs(node)}")
        lines.append("")
    result = "\n".join(lines)
    if path is not None:
        write(path, result)
    return result


def dump_stages(
    stages: List["Stage"],
    Gs: DFG,
    Gp: DFG,
    path: Path | str | None = None,
    stats: bool = False,
) -> str:
    lines: List[str] = []
    sid_to_num_pnodes: Dict[int, int] = {}
    for stage in stages:
        sid_to_num_pnodes[stage.id] = len(stage.pnodes)
        lines.append(
            f"🎯 Stage {stage.id}, snodes: {len(stage.snodes)}, pnodes: {len(stage.pnodes)}"
        )
        lines.append("S.NODES:")
        for node in stage.snodes:
            lines.append(f"  {Gs.node_opname(node)} {node}")
            for t in Gs.node_inputs(node):
                lines.append(f"    IN {t} {Gs.tensor_shape(t)}")
            for t in Gs.node_outputs(node):
                lines.append(f"    OUT {t} {Gs.tensor_shape(t)}")
        lines.append("P.NODES:")
        rkmb_nodes = {}
        for node in stage.pnodes:
            rkmb_nodes.setdefault((node.rank, node.mb), []).append(node)
        for rk, mb in sorted(rkmb_nodes.keys()):
            nodes = rkmb_nodes[(rk, mb)]
            for node in nodes:
                lines.append(f"  {Gp.node_opname(node)} {node}")
                for t in Gp.node_inputs(node):
                    lines.append(f"    IN {t} {Gp.tensor_shape(t)}")
                for t in Gp.node_outputs(node):
                    lines.append(f"    OUT {t} {Gp.tensor_shape(t)}")
        lines.append("LNG.IN:")
        for l in stage.input_lineages:
            lines.append(f"=== {l.Ts}")
            for Tp in l.Tps:
                lines.append(f"    {Tp}")
            # for slcmap, eq_copies in l.slice_map.items():
            #     lines.append(f"{slcmap}")
            #     for eqcopy in eq_copies:
            #         lines.append(f"    {eqcopy}")
        lines.append("LNG.OUT:")
        for l in stage.output_lineages:
            lines.append(f"=== {l.Ts}")
            for Tp in l.Tps:
                lines.append(f"    {Tp}")
            # for slcmap, eq_copies in l.slice_map.items():
            #     lines.append(f"{slcmap}")
            #     for eqcopy in eq_copies:
            #         lines.append(f"    {eqcopy}")

        lines.append("")

    # stats
    if stats:
        lines.append("Statistics:")
        lines.append(f"Total stages: {len(stages)}")
        top_items = sorted(
            sid_to_num_pnodes.items(), key=lambda item: item[1], reverse=True
        )[:100]
        for key, value in top_items:
            lines.append(f"Stage {key}: {value} pnodes.")

    result = "\n".join(lines)
    if path is not None:
        write(path, result)
    return result


def write(path: Path | str, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    logdebug(f"✏️  Writing file.", path=path)
    Path(path).write_text(text)


def dump_lineages(
    ls: List[Lineage], Gs: DFG, Gp: DFG, path: Path | str | None = None
) -> str:
    lines: List[str] = []
    for l in ls:
        lines.append(f"{l.Ts}")
        for Tp in l.Tps:
            lines.append(f"{Tp}")
        lines.append("")
    result = "\n".join(lines)
    if path is not None:
        write(path, result)
    return result
