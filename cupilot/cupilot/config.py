# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Optional
from dataclasses import dataclass
import argparse


@dataclass
class CupilotConfig:

    # =============== staged spmd solver config ===============

    # maximal number of data parallelism size
    max_dp_size: Optional[int] = None
    # maximal number of tensor parallelism size
    max_tp_size: Optional[int] = None
    # maximal number of pipeline parallelism size
    max_pp_size: Optional[int] = None
    # maximal number of graph chunks
    max_block_num: int = 12
    # maximal memory limit (GB) at the first device
    dev0_mem_limit_gb: Optional[int] = None
    # enable memory saving mode by partitioning every operator if possible.
    memory_saving: bool = False
    # total memory limit fraction, the memory limit will be device_mem * mem_frac
    memory_fraction: float = 1.0
    # db cache file
    db_cache: str = 'profile.json'
    # enable recompute for every operators
    recompute: bool = False
    # number of zero groups, default (0) not to use zero
    zero_size: int = 0
    # order plan json file
    order_plan: Optional[str] = None


def build_parser() -> argparse.ArgumentParser:
    """
    The parser can be registered in user traing script using:

    .. code-block:: python

        cuparser = cupilot.build_parser()
        parser = argparser.ArgumentParser(parents=[cuparser], ...)
        parser.add_argument(...)  # user arguments

    Returns:
        argparse.ArgumentParser: the parser
    ```
    """
    parser = argparse.ArgumentParser(description='cupilot policy searching configuration',
                                     add_help=False)
    parser.add_argument('--max-pp', type=int, default=32,
                        help='max number of pipeline stages')
    parser.add_argument('--max-tp', type=int, default=32,
                        help='max size of tensor paralllelism')
    parser.add_argument('--max-dp', type=int, default=None,
                        help='max size of data paralllelism')
    parser.add_argument('--max-block-num', type=int, default=12,
                        help='max number of graph chunks')
    # memory optimizations
    parser.add_argument('--recompute', action='store_true', default=False,
                        help='enable recompute for every operators')
    parser.add_argument('--zero-size', type=int, default=0,
                        help='number of zero groups on optimizer states')
    # search flags
    parser.add_argument('--mem-saving', action='store_true', default=False,
                        help='enable memory saving mode by partitioning every operator if possible.')
    parser.add_argument('--mem-frac', type=float, default=1.0,
                        help='memory limit fraction, ranging from 0.0 to 1.0')
    parser.add_argument('--dev0-mem-limit', type=int, default=None,
                        help='max memory limit (GB) of the first device')
    parser.add_argument('--db-cache', type=str, default='profile.json',
                        help='profiled database cache file')
    parser.add_argument('--order-plan', type=str, default=None,
                        help='order plan json file')
    return parser


def build_config(parser: argparse.ArgumentParser) -> CupilotConfig:
    """
    Build the solver config from argument parser

    Args:
        parser (argparse.ArgumentParser): the argument parser

    Returns:
        CupilotConfig: the solver config
    """
    args = parser.parse_args()
    config = CupilotConfig(
        max_dp_size=args.max_dp,
        max_tp_size=args.max_tp,
        max_pp_size=args.max_pp,
        max_block_num=args.max_block_num,
        dev0_mem_limit_gb=args.dev0_mem_limit,
        memory_saving=args.mem_saving,
        memory_fraction=args.mem_frac,
        db_cache=args.db_cache,
        recompute=args.recompute,
        zero_size=args.zero_size,
        order_plan=args.order_plan,
    )
    return config
