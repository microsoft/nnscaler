#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from nnscaler.autodist.cube_operator import CubeOperator
from nnscaler.graph.function.dimops import DimAnno, IRDimops

import itertools
from typing import List, Tuple


def calc_factors(val: int, num: int) -> List[Tuple[int, ...]]:
    """
    Calculate all possible factors of val that can be divided into num parts.
    NOTE: 6=2*3 and 6=3*2 are considered the same.
    """
    plans = []

    def backtrace(target: int, remaining: int, path: List[int]):
        if remaining == 1:
            if target != 1:
                plans.append(path + [target])
            else:
                if target != 1 or path:
                    raise RuntimeError(f'invalid target {target}, path {path}')
                plans.append([1])
            return

        for i in range(2, target):
            if target % i == 0:
                backtrace(target // i, remaining - 1, path + [i])

    backtrace(val, num, [])

    visited = set()
    for plan in plans:
        plan.sort()
        visited.add(tuple(plan))
    return list(visited)


_factor_cache = {}


def calc_factors_cached(val: int, num: int) -> List[List[int]]:
    if (val, num) not in _factor_cache:
        _factor_cache[(val, num)] = calc_factors(val, num)
    return _factor_cache[(val, num)]


def generate_partitions(
        dim_ids: List[str],
        device_num: int) -> List[Tuple[Tuple[str, ...], Tuple[int, ...]]]:
    """
    Generate all possible partitions of dim_ids into device_num parts.

    Args:
        dim_ids: a list of dimension names.
        device_num: the number of devices.

    Returns:
        A list of possible partitions.

    Example:
        dim_ids = ['a', 'b'], device_num = 4
        possible partitions:
            (('a', 'b'), (2, 2))
            (('b', 'a'), (2, 2))
            (('a',), (4,))
            (('b',), (4,))
    """
    candidates = []
    for i in range(1, device_num + 1):
        if i > len(dim_ids):
            break
        factors = calc_factors_cached(device_num, i)
        if not factors:
            break
        for factor in factors:
            visited = set()
            for factor_permutation in itertools.permutations(factor):
                if factor_permutation not in visited:
                    visited.add(factor_permutation)
                    for dim_permutation in itertools.permutations(dim_ids, i):
                        if -1 in dim_permutation and dim_permutation[0] != -1:
                            continue
                        candidates.append((dim_permutation, factor_permutation))
    return candidates


class OpPartition:
    """
    OpPartition represents a partition plan for a CubeOperator.
    It is defined by a list of partition_dims and a list of partition_nums.

    If there is a matrix multiplication operator with annotation 'm k+, k+ n -> m n'
    where m=512, k=1024, n=2048, a partition plan can be:
    partition_dims = [-1, 'm', 'k'], partition_nums = [2, 2, 2].
    It means that the operator will be split into 8 sub-operators with shape
    m=256, k=512, n=2048.
    NOTE:
    - if -1 in partition_dims, it should be placed at the first position.
    - the example partition above is different from [-1, 'k', 'm'], [2, 2, 2]
    """

    def __init__(self, partition_dims: Tuple[str, ...],
                 partition_nums: Tuple[int, ...], operator: CubeOperator):
        self.operator = operator
        self.partition_dims = partition_dims
        self.partition_nums = partition_nums
        self.is_partial_val = False

        if len(partition_dims) != len(partition_nums):
            raise ValueError(
                'partition_dims and partition_nums should have the same length')
        if len(partition_dims) != 1:
            raise ValueError('only support split along one dimension for now')

        if isinstance(self.operator.ir_cell, IRDimops):
            if partition_dims[0] != -1:
                idx, dim = operator.dim_id2pos(partition_dims[0])
                if not operator.ir_cell.algorithms('dim').satisfy(
                        idx, dim, partition_nums[0]):
                    raise ValueError(
                        f'invalid partition plan {partition_dims}, {partition_nums} for {operator.op_name}'
                    )
                # Store the first node among partition results of the full cube node.
                # Other nodes are not stored because
                # 1. they share the same shape with the first node.
                # 2. we can calculate th intra-communication cost without knowing the device assignment now,
                #    since operator is constrained to be partitioned along one dimension.
                # It is used to query the computation cost in the cost database.
                self.ir_cell = operator.ir_cell.algorithms('dim').instantiate(
                    idx, dim, partition_nums[0])[0]
            else:
                self.ir_cell = operator.ir_cell

            for dim, num in zip(partition_dims, partition_nums):
                if dim == -1:
                    continue
                if operator.get_reduce_type(dim) == DimAnno.ReduceType.Sum and \
                 num > 1:
                    self.is_partial_val = True
                    break
        else:
            if partition_dims[0] != -1:
                raise ValueError('only support replicated for non-dimops')
            self.ir_cell = operator.ir_cell

    def is_replicated(self):
        return len(self.partition_dims) == 1 and self.partition_dims[0] == -1

    def __repr__(self):
        return f'OpPartition({self.partition_dims}, {self.partition_nums})'
