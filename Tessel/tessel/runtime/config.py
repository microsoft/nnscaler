# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Optional
from dataclasses import dataclass
import argparse


@dataclass
class TesselConfig:

    # =============== staged spmd solver config ===============

    # maximal number of data parallelism size
    max_dp_size: Optional[int] = None
    # maximal number of tensor parallelism size
    max_tp_size: Optional[int] = None
    # maximal number of pipeline parallelism size
    max_pp_size: Optional[int] = None
    # minimal number of pipeline parallelism size
    min_pp_size: Optional[int] = 1
    # maximal number of graph chunks
    max_layer_num: int = 12
    # maximal parameter limit (GB) at first stage
    param_limit_gb: Optional[int] = None
    # total memory limit fraction, the memory limit will be device_mem * mem_frac
    memory_fraction: float = 1.0
    # db cache file
    db_cache: str = 'profile.json'
    # enable recompute for every operators
    recompute: bool = False


def build_parser() -> argparse.ArgumentParser:
    """
    The parser can be registered in user traing script using:

    .. code-block:: python

        tparser = tessel.build_parser()
        parser = argparser.ArgumentParser(parents=[tparser], ...)
        parser.add_argument(...)  # user arguments

    Returns:
        argparse.ArgumentParser: the parser
    ```
    """
    parser = argparse.ArgumentParser(description='tessel policy searching configuration',
                                     add_help=False)
    parser.add_argument('--max-pp', type=int, default=32,
                        help='max number of pipeline stages')
    parser.add_argument('--min-pp', type=int, default=1,
                        help='min size of pipeline parallelism')
    parser.add_argument('--max-tp', type=int, default=32,
                        help='max size of tensor paralllelism')
    parser.add_argument('--max-dp', type=int, default=None,
                        help='max size of data paralllelism')
    parser.add_argument('--max-layer-num', type=int, default=12,
                        help='max number of graph chunks')
    parser.add_argument('--recompute', action='store_true', default=False,
                        help='enable recompute for every operators')
    parser.add_argument('--mem-frac', type=float, default=1.0,
                        help='memory limit fraction, ranging from 0.0 to 1.0')
    parser.add_argument('--param-limit', type=int, default=None,
                        help='max parameter limit (GB) at first stage')
    parser.add_argument('--db-cache', type=str, default='profile.json',
                        help='profiled database cache file')
    return parser


def build_config(parser: argparse.ArgumentParser) -> TesselConfig:
    """
    Build the solver config from argument parser

    Args:
        parser (argparse.ArgumentParser): the argument parser

    Returns:
        TesselConfig: the solver config
    """
    args = parser.parse_args()
    config = TesselConfig(
        max_dp_size=args.max_dp,
        max_tp_size=args.max_tp,
        max_pp_size=args.max_pp,
        min_pp_size=args.min_pp,
        max_layer_num=args.max_layer_num,
        param_limit_gb=args.param_limit,
        memory_fraction=args.mem_frac,
        db_cache=args.db_cache,
        recompute=args.recompute,
    )
    return config