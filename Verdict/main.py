import sys

sys.path.append("..")


import z3
import csv
import argparse
from typing import List
from dataclasses import dataclass, asdict
from pathlib import Path
from pprint import pprint
import traceback

from verdict.config import Config
from verdict.log import setup_logger, logerr, loginfo
from verdict.verifier import StageParallelVerifier
from verdict.timer import timer

from nnscaler_backend import nnScalerGraphBackend
from z3_backend import z3Backend
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


def prepare(cfg: Config):
    setup_logger(cfg.loglevel)
    z3.set_param("smt.random_seed", cfg.seed)
    z3.set_param("memory_max_size", 0)
    sys.setrecursionlimit(10000)


def dump_stats(v: StageParallelVerifier, err: Exception | None, sm: str, pm: str):
    @dataclass
    class Stats:
        # job meta
        success: bool = None
        num_layers: int = None
        num_dp: int = None
        num_pp: int = None
        num_tp: int = None
        num_mb: int = None
        gbs: int = None
        num_heads: int = None
        hidden_size: int = None
        seqlen: int = None
        n_activated_experts: int = None
        n_routed_experts: int = None

        # time profile
        t_graph: float = None
        t_lineage: float = None
        t_schedule: float = None
        t_vrf: float = None
        t_total: float = None

        # exception
        sm: str = None
        pm: str = None
        error: List[str] = None

    # init Stats
    stats = Stats(
        success=err is None, sm=sm, pm=pm, error=[] if err is None else [str(err)]
    )

    # set Stats
    try:
        for k, v_ in asdict(v.Wp).items():
            if k in Stats.__annotations__:
                setattr(stats, k, v_)
        if Config.time:
            stats.t_graph = timer.get("load Gs") + timer.get("load Gp")
            stats.t_lineage = timer.get("align lineages")
            stats.t_schedule = timer.get("cut stages")
            stats.t_vrf = timer.get("run stages")
            stats.t_total = timer.get("main")
    except Exception as e:
        if err is None:
            logerr(e, traceback=traceback.format_exc())
            stats.error.append(str(e))

    # print Stats
    pprint(stats)

    # dump Stats
    path = Path(Config.stats_dir / "stats.csv")
    file_exists = path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=Stats.__annotations__.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(asdict(stats))


def main(sm: str, pm: str) -> StageParallelVerifier:
    loginfo(f"🚀 Start verifying {pm}.")
    timer.start("main")
    v = StageParallelVerifier(
        Gs_path=sm,
        Ws_path=None,
        Gp_path=pm,
        Wp_path=None,
        graph_backend=nnScalerGraphBackend,
        symbolic_backend=z3Backend,
    )
    v.launch()
    timer.end("main")
    timer.display()
    return v


def main_w_stats(sm: str, pm: str):
    v = None
    err = None
    try:
        v = main(sm, pm)
    except Exception as e:
        err = e
        logerr(e, traceback=traceback.format_exc())
    dump_stats(v, err, sm, pm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sm", type=str, default="")
    parser.add_argument("--pm", type=str, default="")
    args, config_args = parser.parse_known_args()
    Config.update_from_args(config_args)
    Config.display()
    prepare(Config)

    sm = "../genmodel/mgeners/llama3adpt_mgener_dp1_pp1_tp1_nm1_gbs64_ly2_h32_hi4096_sq128.pkl"
    # pm = "../genmodel/mgeners/llama3adptMeg_mgener_dp2_pp2_tp2_nm1_gbs64_ly2_h32_hi4096_sq128.pkl"

    sm = "../genmodel/mgeners/llama3adptMegE_mgener_dp1_pp1_tp1_nm1_gbs512_ly2_h32_hi4096_sq128.pkl"
    pm = "../genmodel/mgeners/llama3adptMegE_mgener_dp2_pp2_tp2_nm2_gbs512_ly2_h32_hi4096_sq128.pkl"

    main_w_stats(args.sm or sm, args.pm or pm)
