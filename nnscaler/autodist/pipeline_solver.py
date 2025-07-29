from .model_graph import ModelGraph, estimate_mem_lower_bound, IntervalInfo
from .spmd_solver import SPMDSolver
from .descs import *
from .autodist_config import AutoDistConfig

import os
import time
import json
import copy
import math
import multiprocessing
import logging
from typing import List, Dict, Tuple
from pathlib import Path

__all__ = [
    'calc_optimal_pp_plan',
]

_logger = logging.getLogger(__name__)


def _dev_num2mesh_desc(dev_num: int, base_col: int) -> MeshDesc:
    if dev_num <= base_col:
        return MeshDesc(1, dev_num)
    else:
        assert dev_num % base_col == 0
        return MeshDesc(dev_num // base_col, base_col)


def _calc_legal_tp_degrees(max_tp_degree: int) -> List[int]:
    ret = []
    tp_degree = 1
    while tp_degree <= max_tp_degree:
        ret.append(tp_degree)
        tp_degree = tp_degree * 2
    return ret


def _collect_tp_intervals(
    model_graph: ModelGraph,
    cfg: AutoDistConfig,
    tp_degree: int,
    stage_num: int,
    interval_groups: List[List[IntervalInfo]],
    spmd_solver: SPMDSolver,
) -> List[int]:
    '''
    collect intervals for given tp_degree and stage_num
    no need to calculate all possible intervals
        1. some intervals may not fit into the memory
        2. some intervals are sub-optimal
            we want to make pipeline stages as balanced as possible
            ideally, we want to make the time of each stage equal.
            to be robust, we can constrain the average time of each stage
            is within a certain range, like no more than 200% of the global
            average time
        3. some intervals are identical (exactly the same ops and topology)

    Args:
        model_graph: the graph in AutoDist
        cfg: the AutoDistConfig
        tp_degree: the tensor parallelism degree
        stage_num: the pipeline stage number
        interval_groups: a list of groups. identical intervals are in a group
        spmd_solver: the solver for tensor parallelism

    Returns:
        selected_groups: the indices of selected interval groups
    '''

    def calc_min_mem(start, end):
        param_mem, buffer_mem, activation_mem = model_graph.query_mem(
            start, end)
        if cfg.zero_stage == 1:
            zero_group_size = tp_degree * cfg.world_size // cfg.mesh_desc.ngpus // cfg.zero_ngroups
        elif cfg.zero_stage == 0:
            zero_group_size = tp_degree
        else:
            raise RuntimeError(f'invalid zero stage {cfg.zero_stage}')
        return estimate_mem_lower_bound(
            param_mem=param_mem,
            buffer_mem=buffer_mem,
            activation_mem=activation_mem * stage_num,
            plan_ngpus=tp_degree,
            zero_group_size=zero_group_size,
            cfg=cfg,
        )

    idxs = [0] + model_graph.get_pipeline_pivots() + [model_graph.op_num]
    global_fw_span = model_graph.query_fw_span(
        0, model_graph.op_num - 1) / model_graph.autodist_config.mesh_desc.ngpus
    min_fw_span = global_fw_span * cfg.max_pipeline_unbalance_ratio
    max_fw_span = global_fw_span / cfg.max_pipeline_unbalance_ratio
    selected_groups = []
    for i, group in enumerate(interval_groups):
        start, end = group[0].start, group[0].end
        if calc_min_mem(start, end) > cfg.memory_constraint:
            continue
        if spmd_solver.estimate_min_mem(start, end) > cfg.memory_constraint:
            continue
        local_fw_span = model_graph.query_fw_span(start, end) / tp_degree
        if local_fw_span < min_fw_span or local_fw_span > max_fw_span:
            continue
        selected_groups.append(i)
    return selected_groups


def _compute_tp_info(
    model_graph: ModelGraph,
    cfg: AutoDistConfig,
    legal_tp_degrees: List[int],
) -> Dict[Tuple[int, int, int, int], SPMDSearchOutput]:
    '''
    Pre-compute the optimal spmd plan and store the result in a dict.
    The key of the dict is (tp_degree, stage_num, start, end),
    which means the optimal spmd plan for the interval [start, end]
    with tp_degree devices and stage_num pipeline stages.

    Args:
        model_graph: the graph in AutoDist
        cfg: the AutoDistConfig
        legal_tp_degrees: the legal tensor parallelism device numbers

    Returns:
        tp_info: the dict that stores the optimal spmd plan for each interval
    '''

    _logger.info('start to compute tp info')
    interval_groups = model_graph.group_pipeline_intervals()
    # if there is no solution for (tp_degree, stage_num, start, end),
    # there is no solution for (tp_degree, stage_num + 1, start, end)
    no_solution_states = set()

    def process_case(device_num, stage_num):
        solver = SPMDSolver(graph=model_graph,
                            mesh_desc=_dev_num2mesh_desc(
                                device_num, cfg.mesh_desc.col),
                            autodist_config=cfg,
                            stage_num=stage_num)

        selected_group_idxs = _collect_tp_intervals(
            model_graph,
            cfg,
            device_num,
            stage_num,
            interval_groups,
            solver,
        )
        intervals = []
        for i in selected_group_idxs:
            start, end = interval_groups[i][0].start, interval_groups[i][0].end
            if (start, end, device_num) in no_solution_states:
                continue
            intervals.append((start, end))
        _logger.info(
            f'process case: tp {device_num}, s {stage_num}, {len(intervals)} intervals'
        )
        solver_ret = solver.solve(intervals, 1)
        return intervals, solver_ret

    def _calc_upper_bound(tp_degree: int):
        # bubble time percentage <= bubble_ratio:
        # (stage_num - 1) / (stage_num - 1 + micro_batch_num) <= bubble_ratio
        # stage_num <= 1 + bubble_ratio * micro_batch_num / (1 - bubble_ratio)
        bubble_ratio = cfg.max_pipeline_bubble_ratio
        micro_batch_num = cfg.update_freq
        upper_bound = math.floor(bubble_ratio /
                                 (1 - bubble_ratio) * micro_batch_num + 1)
        return min(cfg.mesh_desc.ngpus - tp_degree + 1, upper_bound)

    # TODO(yizhu1): use multiprocessing to speed up
    tp_info = {}
    for tp_degree in legal_tp_degrees:
        for stage_num in range(1, _calc_upper_bound(tp_degree) + 1):
            intervals, solver_ret = process_case(tp_degree, stage_num)
            for interval, spmd_descs in zip(intervals, solver_ret):
                start, end = interval
                if spmd_descs:
                    for group in interval_groups:
                        if group[0].start == start and group[0].end == end:
                            for interval in group:
                                tp_info[(tp_degree, stage_num, interval.start,
                                         interval.end)] = spmd_descs[0]
                else:
                    no_solution_states.add((start, end, tp_degree))
                    _logger.info(
                        f'fail to find a valid plan for {start}, {end}')
    _logger.info('finish computing tp info')
    return tp_info


def calc_optimal_pp_plan(
        model_graph: ModelGraph,
        autodist_config: AutoDistConfig) -> PipelineSearchOutput:
    # TODO: based on experience, tensor parallelism should <= 8
    legal_tp_degrees = _calc_legal_tp_degrees(
        min(8, autodist_config.mesh_desc.col))

    tp_info = _compute_tp_info(model_graph, autodist_config, legal_tp_degrees)
    '''
    T: dynamic programming table
    T[s, pp, tp, i]: optimal time of a pipeline state, where
        - s: stage number
        - pp: device number used for this state
        - tp: device number used for the 1st pipeline stage in this state
        - i: start operator index
    Transitions of T:
        - leaf state:
            tp == pp and s == 1: means current tp is the last one in the pipeline
            T[1, pp, tp, i] = tp[tp][s][i][end_op_idx]
        - non-leaf state:
            T[s, pp, tp, i] = min(max(T[s-1, pp-tp, tp', j+1], tp[tp][s][i][j]))
    store optimal path during dynamic programming in T as well
    '''
    ngpus = autodist_config.mesh_desc.ngpus
    pp_idxs = [0] + model_graph.get_pipeline_pivots() + [model_graph.op_num]
    T = {}
    for s in range(1, ngpus + 1):
        for pp in range(s, ngpus + 1):
            for tp in range(1, pp - s + 1 + 1):
                if tp not in legal_tp_degrees:
                    continue
                for ii in range(len(pp_idxs) - 1 - 1, 0 - 1, -1):
                    i = pp_idxs[ii]
                    cur_idx = (s, pp, tp, i)
                    T[cur_idx] = [float('inf'), (-1, -1, -1, -1)]

                    if tp == pp and s == 1:
                        tp_idx = (tp, s, i, model_graph.op_num - 1)
                        if tp_idx in tp_info:
                            T[cur_idx][0] = tp_info[tp_idx].all_time
                        continue

                    for jj in range(len(pp_idxs) - 1 - 1, ii, -1):
                        j = pp_idxs[jj]
                        next_pp = pp - tp
                        for next_tp in range(1, next_pp - (s - 1) + 1 + 1):
                            if next_tp not in legal_tp_degrees:
                                continue
                            prev_idx = (s - 1, next_pp, next_tp, j)
                            if prev_idx not in T:
                                continue
                            prev_tp_idx = (tp, s, i, j - 1)
                            if prev_tp_idx not in tp_info:
                                continue
                            lhs, _ = T[prev_idx]
                            rhs = tp_info[prev_tp_idx].all_time
                            val = max(lhs, rhs)
                            if T[cur_idx][0] > val:
                                T[cur_idx] = [val, prev_idx]

    best_time = float('inf')
    best_state = (-1, -1, -1, -1)
    micro_batch_num = autodist_config.update_freq
    for stage_num in range(1, ngpus + 1):
        for pp_dev_num in range(stage_num, ngpus + 1):
            for tp_degree in range(1, pp_dev_num - stage_num + 1 + 1):
                if tp_degree not in legal_tp_degrees:
                    continue

                cur_idx = (stage_num, pp_dev_num, tp_degree, 0)
                if cur_idx not in T:
                    continue
                cur_time = T[cur_idx][0] * (micro_batch_num - 1 + stage_num)
                if best_time > cur_time:
                    best_time, best_state = cur_time, cur_idx

    _logger.info(
        f'best time/s: {best_time}, state (s, pp, tp, i): {best_state}')
    if best_state == (-1, -1, -1, -1):
        raise RuntimeError('fail to find a valid pipeline plan')

    spmd_outs = []

    def build_answer(s, pp, tp, i):
        _, prev_idx = T[(s, pp, tp, i)]
        if prev_idx[0] == -1:
            tp_idx = (tp, s, i, pp_idxs[-1] - 1)
        else:
            j_plus_1 = prev_idx[3]
            tp_idx = (tp, s, i, j_plus_1 - 1)
        spmd_outs.append(tp_info[tp_idx])
        if prev_idx[0] != -1:
            build_answer(*prev_idx)

    build_answer(*best_state)

    spmd_descs = [spmd_out.desc for spmd_out in spmd_outs]
    pp_desc = PipelineParallelDesc(spmd_descs, [], autodist_config.mesh_desc)
    stage_mems = [spmd_out.memory for spmd_out in spmd_outs]
    stage_all_times = [spmd_out.all_time for spmd_out in spmd_outs]
    stage_comp_times = [spmd_out.comp_time for spmd_out in spmd_outs]
    return PipelineSearchOutput(pp_desc, best_time, stage_mems, stage_all_times,
                                stage_comp_times)
