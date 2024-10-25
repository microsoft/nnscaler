#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

# CREDITS: This implementation is inspired by Fairseq https://github.com/facebookresearch/fairseq/blob/main/fairseq/utils.py

from typing import List, Dict, Tuple, Optional, TYPE_CHECKING
from dataclasses import dataclass
from collections import defaultdict

import torch
import torch.distributed as dist

try:
    from amp_C import multi_tensor_l2norm
    multi_tensor_l2norm_available = True
except ImportError:
    multi_tensor_l2norm_available = False


if TYPE_CHECKING:
    from nnscaler.runtime.module import CubeModule


@dataclass
class ParamsInfo:
    # An instance of ParamsInfo corresponds to a group of parameters in cube reducer,
    # or a single parameter without cube reducer.
    ranks: Tuple[int]
    params: List[torch.nn.Parameter]
    param_names: List[str]
    zero_ngroups: int


@dataclass
class TidReplicaInfo:
    # the number of the replicas of the (partitioned) parameter with tid
    nreplicated: int
    # the number of all the involved ranks for this parameter with tid
    nranks: int


def _calc_grad_shape(slicers_list):
    # caculate the shape of each full parameters/grads
    tid2shape = {}
    for rank_slicers in slicers_list:
        for tid, slicers in rank_slicers.items():
            if tid not in tid2shape:
                tid2shape[tid] = [0 for _ in slicers]
            for i, slicer in enumerate(slicers):
                # slicer: (start, end, step)
                if slicer.stop > tid2shape[tid][i]:
                    tid2shape[tid][i] = slicer.stop
    # caculate the number of replicas of each model parameter
    tid2nreplicas = {}
    for rank_slicers in slicers_list:
        for tid, slicers in rank_slicers.items():
            if tid not in tid2nreplicas:
                tid2nreplicas[tid] = 0
            factor = 1
            for i, slicer in enumerate(slicers):
                factor *= (slicer.stop - slicer.start) / tid2shape[tid][i]
            tid2nreplicas[tid] += factor
    return tid2nreplicas


def prepare_for_grad_clip_legacy(cube_model: 'CubeModule', curr_rank: int) -> Dict[int, List[torch.nn.Parameter]]:
    assert curr_rank == dist.get_rank()
    tid2param, tid2slicers = {}, {}
    for name, param in cube_model.named_parameters():
        assert name in cube_model.fullmap
        if param.requires_grad:
            tid = cube_model.tid_of_param_name(name)
            slicers = cube_model.fullmap[name].slicers
            tid2param[tid] = param
            tid2slicers[tid] = slicers
    # gather all parameters' slicers
    tid2ranks_list = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(tid2ranks_list, tid2slicers)
    tid2nreplicas = _calc_grad_shape(tid2ranks_list)
    nreplicas2localparams = defaultdict(list)
    for tid, param in tid2param.items():
        nreplicas = tid2nreplicas[tid]
        nreplicas2localparams[nreplicas].append(param)
    return nreplicas2localparams


def _check_is_ordered(ranks: Tuple[int]) -> bool:
    for i in range(len(ranks)-1):
        if ranks[i] >= ranks[i+1]:
            return False
    return True


def _check_no_intersection(ranks_set):
    # ranks_set: set of tuple
    # check intersection between any two tuples
    ranks = set()
    for r in ranks_set:
        old_len = len(ranks)
        ranks.update(r)
        if len(ranks)  - old_len != len(r):
            return False
    return True


def _calc_grad_replicas(tid2ranks_list: List[Dict[int, Tuple[int]]]) -> Dict[int, TidReplicaInfo]:
    """This function is used to calculate the number of replicas of each model parameter.
    Each parameter has a tuple of `len(ranksset)` (we call it nreplicated) and `nranks`,
    because a parameter may be replicated (not data parallelism) which is supported by nnscaler.
    It affects the calculation of gnorm. So nreplicated represents the number of
    non-data-parallelism replicas for this parameter, and nranks represents the number of
    all the involved ranks for this parameter.

    Args:
        tid2ranks_list: list of dict, each dict is tid2ranks

    Returns:
        tid2nreplicas: dict, tid -> TidReplicaInfo
    """
    # caculate the number of replicas of each model parameter
    tid2nreplicas = {}
    tid2ranksset = defaultdict(set)
    for tid2ranks in tid2ranks_list:
        for tid, ranks in tid2ranks.items():
            assert _check_is_ordered(ranks)
            assert isinstance(ranks, tuple), f'ranks {ranks} should be tuple'
            tid2ranksset[tid].add(ranks)
    # the ranks have been deduplicated using set.
    # so the number of ranks represents the number of replicas (pure replicate not data parallelism),
    # where each ranks is the unit of ZeRO (or reducer).
    for tid, ranksset in tid2ranksset.items():
        assert _check_no_intersection(ranksset)
        nranks = sum([len(ranks) for ranks in ranksset])
        tid2nreplicas[tid] = TidReplicaInfo(len(ranksset), nranks)
    return tid2nreplicas


def prepare_for_grad_clip(cube_model: 'CubeModule', is_zero: bool) -> Dict[int, List[torch.nn.Parameter]]:
    params_info_for_gnorm = cube_model.parameters_for_calc_gnorm()
    tid2ranks = {}
    tid2info_list_seq = {}
    for seq, params_info in enumerate(params_info_for_gnorm):
        # params_info is ParamsInfo, which is defined in this file
        assert isinstance(params_info.ranks, tuple), f'ranks {params_info.ranks} should be tuple'
        for param in params_info.params:
            assert param.requires_grad
        for name in params_info.param_names:
            tid = cube_model.tid_of_param_name(name)
            tid2ranks[tid] = params_info.ranks
            tid2info_list_seq[tid] = seq
    tid2ranks_list = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(tid2ranks_list, tid2ranks)
    tid2nreplicas = _calc_grad_replicas(tid2ranks_list)
    # populate nreplicas2localparams
    nreplicas2localparams = defaultdict(list)
    processed_seqs = {}
    for tid, replicated_info in tid2nreplicas.items():
        if tid not in tid2info_list_seq:
            # because tid2nreplicas is from all the ranks,
            # if this parameter (tid) does not belong to this rank,
            # it is safe to skip it.
            continue
        seq = tid2info_list_seq[tid]
        params_info = params_info_for_gnorm[seq]
        if seq in processed_seqs:
            assert processed_seqs[seq] == replicated_info, \
                'the params belonging to the same seq should have the same nreplicated and nranks'
            continue
        # If ZeRO is not used, the number of replicas of a parameter (partition) is its involved ranks,
        # no matter it is pure replicated or data-parallelism replicated. For calculating gnorm, these
        # two kinds of replicas are the same, because in data-parallelism, gradients are also allreduced
        # before gnorm calculation.
        # If ZeRO is used, the number of replicas of a parameter (partition) is the number of pure replicated
        # multiplied by the number of ZeRO groups. Multiplying the number of pure replicated is easy
        # to understand. Multiplying the number of ZeRO groups is because the gradients of each ZeRO group
        # are full model gradients, so the number of ZeRO groups is the number of gradient replicas of the full model.
        if not is_zero:
            nreplicas = replicated_info.nranks
        else:
            nreplicas = replicated_info.nreplicated * params_info.zero_ngroups
        nreplicas2localparams[nreplicas].extend(params_info.params)
        processed_seqs[seq] = replicated_info
    return nreplicas2localparams


def _multi_tensor_total_norm(grads, chunk_size=2048 * 32) -> torch.Tensor:
    """
    Returns:
        Total norm of the input tensors in float32.
    """
    per_device_grads = {}
    norms = []
    for grad in grads:
        device = grad.device
        cur_device_grads = per_device_grads.get(device)
        if cur_device_grads is None:
            cur_device_grads = []
            per_device_grads[device] = cur_device_grads
        cur_device_grads.append(grad)
    for device in per_device_grads.keys():
        cur_device_grads = per_device_grads[device]
        if device.type == "cuda":
            # TODO(msb) return has_inf
            has_inf = torch.zeros((1, 1), dtype=torch.int, device=device)
            with torch.cuda.device(device):
                norm = multi_tensor_l2norm(
                    chunk_size, has_inf, [cur_device_grads], False
                )
            norms.append(norm[0].to(torch.cuda.current_device()))
        else:
            assert False, 'non cuda device is not supported.'
            norms += [torch.norm(g, p=2, dtype=torch.float32) for g in cur_device_grads]
    assert len(norms) == 1
    total_norm = torch.norm(torch.stack(norms))
    return total_norm


@torch.no_grad()
def calcuate_gnorm(params: List[torch.Tensor], device: Optional[torch.device] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Calculate the gradient norm of the given parameters.

    Args:
        params (List[torch.Tensor],): list of parameters
        device (Optional[torch.device]): device to calculate the gradient norm. Default is the device of the first parameter

    Returns:
        Tuple[torch.Tensor, List[torch.Tensor]]: Tuple of the gradient norm and the list of gradients.
    """
    def grad_exists(p):
        return p is not None and getattr(p, "grad", None) is not None
    if device is None:
        # assume all weights are on the same device
        device = params[0].device
    params = list(filter(grad_exists, params))
    grads = []
    for p in params:
        grads.append(p.grad.detach())
    if len(grads) == 0:
        total_norm = torch.tensor(0.0, dtype=torch.float32, device=device)  # alway use float32
    elif len(grads) == 1:
        total_norm = torch.norm(grads[0], p=2, dtype=torch.float32)
    else:
        if multi_tensor_l2norm_available:
            total_norm = _multi_tensor_total_norm(grads).to(device)
        else:
            # torch.nn.utils.clip_grad_norm_ way to calculate the norm
            # norms = torch._foreach_norm(grads, 2.0)
            # total_norm = torch.linalg.vector_norm(torch.stack([norm.to(device) for norm in norms]), 2.0)
            total_norm = torch.norm(
                torch.stack(
                    [torch.norm(g, p=2, dtype=torch.float32).to(device) for g in grads]
                )
            )

    return total_norm, grads


@torch.no_grad()
def clip_grads(grads: List[torch.Tensor], gnorm, max_norm: float) -> None:
    """
    Clip gradients.

    Args:
        grads: list of gradients
        gnorm: the norm of all the gradients (maybe in different devices)
        max_norm: max norm value

    Returns:
        None
    """
    max_norm = float(max_norm)
    clip_coef = (max_norm / (gnorm + 1e-6)).clamp_(max=1)
    for g in grads:
        g.mul_(clip_coef)


@torch.no_grad()
def clip_gnorm(
    nreplicas2localparams: Dict[int, List[torch.Tensor]],
    max_norm: Optional[float] = None
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Calculate gnorm and clip gradients

    Args:
        nreplicas2localparams: a dict mapping from number_of_replicas to a list of local params.
            For example, nreplicas2localparams[2] contains all the parameters that have replicated 2 times.
        max_norm: max norm value. If None or <= 0, no clipping will be performed.

    Returns:
        Tuple[torch.Tensor, List[torch.Tensor]]: Tuple of The gradient norm and the list of gradients.
    """
    # assume all weights are on the same device
    for localparams in nreplicas2localparams.values():
        if len(localparams) == 0:
            continue
        device = localparams[0].device
        break
    else:
        raise RuntimeError('no parameters found')

    total_grad_square = torch.tensor(0.0, dtype=torch.float64, device=device)
    grads = []
    for nreplicas, localparams in nreplicas2localparams.items():
        if len(localparams) == 0:
            continue
        # compute gnorm
        local_gnorm, local_grads = calcuate_gnorm(localparams, device)
        total_grad_square += local_gnorm.to(dtype=torch.float64).pow_(2).div_(nreplicas)
        grads.extend(local_grads)
    dist.all_reduce(total_grad_square)
    total_norm = total_grad_square.sqrt_().to(torch.float32)

    if max_norm is not None and max_norm > 0:
        clip_grads(grads, total_norm, max_norm)

    return total_norm, grads
