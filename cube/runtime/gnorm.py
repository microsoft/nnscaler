# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import List, Dict, Tuple, TYPE_CHECKING
from dataclasses import dataclass
from collections import defaultdict

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from cube.runtime.module import CubeModule

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
            slicers = cube_model.fullmap[name][1]
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
    because a parameter may be replicated (not data parallelism) which is supported by cube.
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
        for name, param in zip(params_info.param_names, params_info.params):
            assert param.requires_grad
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
