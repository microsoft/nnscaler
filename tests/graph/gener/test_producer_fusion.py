#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from nnscaler.ir.tensor import IRFullTensor
import nnscaler.graph.function.function as F
from nnscaler.graph import IRGraph

from nnscaler.graph.gener.gen import IRAdapterGener


def _tensor(shape, requires_grad=True):
    return IRFullTensor(shape, requires_grad=requires_grad).tosub()


def test_gener_producer_fusion_replicate():

    data = _tensor([128, 128], False)
    w1 = _tensor([128, 128])
    out1 = _tensor([128, 128])
    l1 = F.Linear(data, w1)
    l1.set_output(0, out1)

    w2 = _tensor([128, 128])
    out2 = _tensor([128, 128])
    l2 = F.Linear(l1.output(0), w2)
    l2.set_output(0, out2)

    loss = _tensor([1])
    sum = F.Sum(l2.output(0))
    sum.set_output(0, loss)

    nodes = [l1, l2, sum]
    graph = IRGraph(nodes, [data], [loss], 'genmodel')
    graph.backward(loss)

    graph.assign(l1, 0)

    s1, s2 = graph.partition(l2, l2.algorithm('dim'), idx=0, dim=0, num=2)
    r1, r2 = graph.replicate(s1, 2)
    graph.assign(r1, 0)
    graph.assign(r2, 0)
    s3, s4 = graph.partition(s2, s2.algorithm('dim'), idx=0, dim=1, num=2)
    graph.assign(s3, 1)
    graph.assign(s4, 1)

    graph.assign(sum, 0)

    # print(graph.extra_repr())
    IRAdapterGener.local_producer_fusion(graph, out2.parent)
    # print(graph.extra_repr())

    assert len(graph.select(name='accum')) == 1
    accum = graph.select(name='accum')[0]

    mms = graph.select(name='linear')[1:]
    assert len(mms) == 4

    new_t = mms[0].output(0)
    old_t = l2.output(0)
    assert new_t.parent != old_t.parent
    for mm in mms:
        if mm.device == (0,):
            assert mm.output(0).parent == new_t.parent
        if mm.device == (1,):
            assert mm.output(0).parent == old_t.parent
    assert accum.output(0).parent == new_t.parent
