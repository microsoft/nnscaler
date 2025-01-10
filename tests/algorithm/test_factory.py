#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import pytest
from nnscaler.algorithm.factory import GenericDistAlgo, _DistAlgorithmFactory


def test_mro():
    factory = _DistAlgorithmFactory()
    factory._algos.clear()

    class A:
        pass

    class B(A):
        pass

    class C(B, A):
        pass

    class D(C):
        pass

    class AlgoA(GenericDistAlgo):
        pass

    class AlgoB(GenericDistAlgo):
        pass

    class AlgoC(GenericDistAlgo):
        pass

    class AlgoA2(GenericDistAlgo):
        pass

    class AlgoB2(GenericDistAlgo):
        pass

    class AlgoC2(GenericDistAlgo):
        pass

    factory.register(A, AlgoA, 'tag')
    factory.register(B, AlgoB, 'tag')
    factory.register(C, AlgoC, 'tag')

    # different tag with diffent algorithm
    factory.register(A, AlgoA2, 'tag2')
    factory.register(B, AlgoB2, 'tag2')
    factory.register(C, AlgoC2, 'tag2')

    # different tag with the same algorithm
    factory.register(A, AlgoA, 'tag3')
    factory.register(B, AlgoB, 'tag3')
    factory.register(C, AlgoC, 'tag3')

    assert factory.algorithms(D) == [AlgoC, AlgoC2, AlgoB, AlgoB2, AlgoA, AlgoA2]
    assert factory.algorithms(C) == [AlgoC, AlgoC2, AlgoB, AlgoB2, AlgoA, AlgoA2]
    assert factory.algorithms(B) == [AlgoB, AlgoB2, AlgoA, AlgoA2]
    assert factory.algorithms(A) == [AlgoA, AlgoA2]

    assert factory.algorithm(D, 'tag3') == AlgoC
    assert factory.algorithm(D, 'tag2') == AlgoC2
    assert factory.algorithm(D, 'tag') == AlgoC
    with pytest.raises(ValueError):
        factory.algorithm(D, 'tag4')

    assert factory.algorithm(C, 'tag3') == AlgoC
    assert factory.algorithm(C, 'tag2') == AlgoC2
    assert factory.algorithm(C, 'tag') == AlgoC
    with pytest.raises(ValueError):
        factory.algorithm(C, 'tag4')

    assert factory.algorithm(B, 'tag3') == AlgoB
    assert factory.algorithm(B, 'tag2') == AlgoB2
    assert factory.algorithm(B, 'tag') == AlgoB

    assert factory.algorithm(A, 'tag3') == AlgoA
    assert factory.algorithm(A, 'tag2') == AlgoA2
    assert factory.algorithm(A, 'tag') == AlgoA


