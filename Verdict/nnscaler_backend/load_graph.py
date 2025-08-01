#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import re
import json
import pickle
from typing import Dict
from pathlib import Path

from nnscaler.codegen.module.module import ModuleCodeGen

from verdict.graph import World, WType, DFG
from verdict.utils import unique

from nnscaler_backend.build_graph import build_graph


def load_graph(G_path: str, W_path: str, wtype: WType | str) -> DFG:
    if isinstance(wtype, str):
        wtype = WType(wtype)

    with open(G_path, "rb") as fp:
        mg: ModuleCodeGen = pickle.load(fp)
    mg_spec = {
        "wtype": wtype,
        "plan_ndevs": len(mg.devices),
        "runtime_ndevs": mg.runtime_ndevs,
    }

    # W_path Backward Compatibility:
    # Ideally, world spec is specified in a json file.
    # In case of missing file, parse the spec from G_path

    W_path: str | Path = W_path or Path(G_path).with_suffix(".json")
    if W_path.exists():
        with open(W_path, "r") as fp:
            world_spec: Dict = json.load(fp)
            world: World = World(**mg_spec, **world_spec)
    else:
        world: World = _load_world_from_gpath(G_path, mg_spec)

    _sanity_check_world(world)
    return build_graph(world, mg, G_path)


def _load_world_from_gpath(G_path: str, mg_spec: Dict) -> World:
    p = Path(G_path)
    fname = p.stem
    model_name = fname.split("_")[0]
    is_moe = "moe" in model_name.lower()

    def search(var):
        matches = re.findall(rf"_{var}(\d+)", fname)
        return int(unique(matches))

    return World(
        **mg_spec,
        model_name=model_name,
        num_dp=search("dp"),
        num_tp=search("tp"),
        num_pp=search("pp"),
        num_mb=search("nm"),
        gbs=search("gbs"),
        num_layers=search("ly"),
        num_heads=search("h"),
        hidden_size=search("hi"),
        seqlen=search("sq"),
        n_activated_experts=search("a") if is_moe else None,
        n_routed_experts=search("r") if is_moe else None,
    )


def _sanity_check_world(w: World):
    assert (
        w.num_tp * w.num_pp == w.plan_ndevs
    ), f"{w.num_tp} * {w.num_pp} != {w.plan_ndevs}"
    assert (
        w.num_dp * w.plan_ndevs == w.runtime_ndevs
    ), f"{w.num_dp} * {w.plan_ndevs} != {w.runtime_ndevs}"
