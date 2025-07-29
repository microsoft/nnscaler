from enum import Enum
from dataclasses import dataclass


@dataclass(frozen=True)
class DTag:
    rank: int
    dp: int
    tp: int
    pp: int
    mb: int


class WType(Enum):
    S = "s"
    P = "p"


@dataclass(frozen=True)
class World:
    wtype: WType = None
    plan_ndevs: int = None
    runtime_ndevs: int = None
    model_name: str = None
    num_dp: int = None
    num_tp: int = None
    num_pp: int = None
    num_mb: int = None
    gbs: int = None
    num_layers: int = None
    num_heads: int = None
    hidden_size: int = None
    seqlen: int = None
    n_activated_experts: int = None
    n_routed_experts: int = None
