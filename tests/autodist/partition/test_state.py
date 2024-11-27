#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from nnscaler.autodist.op_partition import calc_factors, generate_partitions


def test_calc_factors():
    assert calc_factors(1, 1) == [(1,)]
    assert calc_factors(2, 1) == [(2,)]
    assert calc_factors(2, 2) == []
    assert calc_factors(4, 2) == [(2, 2)]
    assert calc_factors(6, 2) == [(2, 3)]
    assert calc_factors(8, 2) == [(2, 4)]
    assert calc_factors(8, 3) == [(2, 2, 2)]
    assert calc_factors(16, 3) == [(2, 2, 4)]


def test_generate_partitions():
    # [['a'], [2]], [['b'], [2]]
    assert len(generate_partitions(['a', 'b'], 2)) == 2
    # [['a'], [4]], [['b'], [4]], [['a', 'b'], [2, 2]], [['b', 'a'], [2, 2]]
    assert len(generate_partitions(['a', 'b'], 4)) == 4
    # [['a'], [4]], [['b'], [4]], [['c'], [4]]
    # [['a', 'b'], [2, 2]], [['a', 'c'], [2, 2]]
    # [['b', 'a'], [2, 2]], [['b', 'c'], [2, 2]]
    # [['c', 'a'], [2, 2]], [['c', 'b'], [2, 2]
    assert len(generate_partitions(['a', 'b', 'c'], 4)) == 9
    # [['a'], [8]], [['b'], [8]]
    # [['a', 'b'], [2, 4]], [['b', 'a'], [2, 4]]
    # [['a', 'b'], [4, 2]], [['b', 'a'], [4, 2]]
    assert len(generate_partitions(['a', 'b'], 8)) == 6
    # [['a'], [8]], [['b'], [8]], [['c'], [8]]
    # [['a', 'b'], [2, 4]], [['a', 'c'], [2, 4]]
    # [['b', 'a'], [2, 4]], [['b', 'c'], [2, 4]]
    # [['c', 'a'], [2, 4]], [['c', 'b'], [2, 4]]
    # [['a', 'b'], [4, 2]], [['a', 'c'], [4, 2]]
    # [['b', 'a'], [4, 2]], [['b', 'c'], [4, 2]]
    # [['c', 'a'], [4, 2]], [['c', 'b'], [4, 2]]
    # [['a', 'b', 'c'], [2, 2, 2]], [['a', 'c', 'b'], [2, 2, 2]]
    # [['b', 'a', 'c'], [2, 2, 2]], [['b', 'c', 'a'], [2, 2, 2]]
    # [['c', 'a', 'b'], [2, 2, 2]], [['c', 'b', 'a'], [2, 2, 2]]
    assert len(generate_partitions(['a', 'b', 'c'], 8)) == 21
    # [['a'], [8]], [['b'], [8]], [[-1], [8]]
    # [[-1, 'a'], [2, 4]], [[-1, 'b'], [2, 4]]
    # [['a', 'b'], [2, 4]], [['b', 'a'], [2, 4]]
    # [[-1, 'a'], [4, 2]], [[-1, 'b'], [4, 2]]
    # [['a', 'b'], [4, 2]], [['b', 'a'], [4, 2]]
    # [[-1, 'a', 'b'], [2, 2, 2]], [[-1, 'b', 'a'], [2, 2, 2]]
    assert len(generate_partitions(['a', 'b', -1], 8)) == 13
