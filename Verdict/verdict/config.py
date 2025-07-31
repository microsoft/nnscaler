#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import json
import argparse
from dataclasses import dataclass, replace, asdict
from pathlib import Path
from functools import lru_cache
from typing import List

WORKSPACE = Path(__file__).parent.parent / "data"


# @dataclass
class Config:
    # workspace to store reusable data, e.g. serialized graphs
    cache_dir: Path = WORKSPACE / "cache"
    # directory to store logs
    log_dir: Path = WORKSPACE / "logs"
    # directory to store stats
    stats_dir: Path = WORKSPACE / "stats"
    # max graph-serialization parallel workers
    max_ser_proc: int = 30
    # max stage-parallel verification workers
    max_vrf_proc: int = 30
    # whether to use cached serialized graphs
    use_cache_nodes: bool = False
    # whether to use cached serialized graphs
    use_cache_stages: bool = False
    # default log level
    loglevel: str = "DEBUG"
    # z3 random seed
    seed: int = 0
    # time
    time: bool = False
    # mem
    mem: bool = True

    # debug print
    dump_nodes: bool = False
    dump_lineages: bool = False
    dump_stages: bool = False

    # backends
    rxshape_backend: str = "z3"

    @classmethod
    def update_from_args(cls, list_args: List[str] | None = None):
        parser = argparse.ArgumentParser()
        parser.add_argument("--cache_dir", type=str)
        parser.add_argument("--log_dir", type=str)
        parser.add_argument("--stats_dir", type=str)
        parser.add_argument("--max_ser_proc", type=int)
        parser.add_argument("--max_vrf_proc", type=int)
        parser.add_argument(
            "--use_cache_nodes", dest="use_cache_nodes", action="store_true"
        )
        parser.add_argument(
            "--no_cache_nodes", dest="use_cache_nodes", action="store_false"
        )
        parser.set_defaults(use_cache_nodes=cls.use_cache_nodes)
        parser.add_argument(
            "--use_cache_stages", dest="use_cache_stages", action="store_true"
        )
        parser.add_argument(
            "--no_cache_stages", dest="use_cache_stages", action="store_false"
        )
        parser.set_defaults(use_cache_stages=cls.use_cache_stages)
        parser.add_argument("--loglevel", type=str)
        parser.add_argument("--seed", type=int)
        parser.add_argument("--time", dest="time", action="store_true")
        parser.add_argument("--dump_nodes", dest="dump_nodes", action="store_true")
        parser.add_argument("--dump_lineages", dest="dump_lineages", action="store_true")
        parser.add_argument("--dump_stages", dest="dump_stages", action="store_true")

        args = parser.parse_args(list_args)
        overrides = {k: v for k, v in vars(args).items() if v is not None}

        # Convert strings to Paths where applicable
        for k, v in overrides.items():
            if k in {"cache_dir", "log_dir", "stats_dir"} and not isinstance(v, Path):
                v = Path(v)
            setattr(cls, k, v)

    @classmethod
    def display(cls):
        config_dict = {}
        for k, v in cls.__dict__.items():
            if k.startswith("_") or isinstance(v, (classmethod, staticmethod)):
                continue
            if isinstance(v, Path):
                config_dict[k] = str(v)
            elif not callable(v):
                config_dict[k] = v
        print(json.dumps(config_dict, indent=2))
