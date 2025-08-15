#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from typing import List, Tuple
import nnscaler
from nnscaler.ir.tensor import IRFullTensor
from nnscaler.graph.gener.rvd.layout import RVDLayout, RVDInspector
from nnscaler.graph.gener.rvd.intra import IntraPathFinder, IntraAutoPlacer, IntraTransition
import numpy as np


import pytest


@pytest.fixture(autouse=True)
def enable_reduce_scatter_adapter():
    from nnscaler.flags import CompileFlag
    old = CompileFlag.disable_reduce_scatter_adapter
    CompileFlag.disable_reduce_scatter_adapter = False
    yield
    CompileFlag.disable_reduce_scatter_adapter = old


def factors(k: int, num: int) -> List[Tuple[int]]:
    """
    get all possible sequence k1 * k2 * .. k_{num} = k
    """
    if num == 1: return [(k,)]
    res = []
    for i in range(1, k):
        if k % i != 0: continue
        for sub_res in factors(k // i, num - 1):
            res.append((i,) + sub_res)
    return res


def test_intra_transition(tmp_path):
    fshape = [256, 256]
    ftensor = IRFullTensor(shape=fshape, name='tensor', requires_grad=False)

    src = (1, 2, 1, 4)
    dst = (1, 1, 1, 8)

    devs = list(range(8))
    src_rvd = RVDLayout.grid(ftensor, r=src[0], v=src[1], dims=src[2:], devices=devs)

    rets = IntraTransition.transition(src_rvd, dst)
    assert len(rets) == 1
    ret = rets[0]
    assert ret[0].vec == dst
    assert len(ret[1]) == 4  # one prim will handle 2 devices
    # v->d will generate reduce_scatter
    assert all(p.signature == 'nnscaler.runtime.adapter.reduce_scatter' for p in ret[1])
    for idx, (layout, prims) in enumerate(rets):
        RVDInspector.draw(src_rvd, layout, tmp_path / 'rvd-trans-{idx}.png')


def test_transition_space(tmp_path):
    fshape = [256, 256]
    ftensor = IRFullTensor(shape=fshape, name='tensor', requires_grad=False)

    src = (1, 2, 1, 4)
    dst = (1, 1, 1, 8)
    devs = list(range(8))

    choices = IntraPathFinder.get_device_space(ftensor, [src, dst], placement=devs)
    assert len(choices) == 1
    # 0/4, 1/5, 2/6, 3/7 have the same indmap, different valmap
    # reduce_scatter will be generated
    assert choices.pop() == (0, 4, 1, 5, 2, 6, 3, 7)

    # draw output
    for idx, choice in enumerate(choices):
        src_rvd = RVDLayout.grid(ftensor, r=src[0], v=src[1], dims=src[2:], devices=devs)
        dst_rvd = RVDLayout.grid(ftensor, r=dst[0], v=dst[1], dims=dst[2:], devices=choice)
        RVDInspector.draw(src_rvd, dst_rvd, tmp_path / f'rvd-{idx}.png')


def test_one_f_case():
    """
    Test complex RVD transformation: (1,4,1,1,2) → (2,1,2,1,2)
    
    Note: This test case has multiple optimal paths with equal costs, so the specific
    path selection depends on algorithm implementation details (e.g., Dijkstra's 
    tie-breaking behavior). Once a path is chosen, the device assignments are 
    deterministic based on the RVD layout structure.
    
    Current path (one of multiple possible optimal paths):
    - Step 1: (1,4,1,1,2) → (1,2,2,1,2) - v2d transformation via reduce_scatter
    - Step 2: (1,2,2,1,2) → (2,1,2,1,2) - v2r transformation via all_reduce
    
    Alternative path with same cost exists:
    - (1,4,1,1,2) → (1,1,4,1,2) → (2,1,2,1,2)
    
    The test verifies the current algorithm's choice and the resulting device mappings.
    If the path selection algorithm changes, this test may need to be updated to 
    reflect the new chosen path, but the device assignments for any given path 
    should remain deterministic.
    """
    fshape = [128, 256, 512]

    src_r, src_v, src_d = 1,4,(1,1,2)
    dst_r, dst_v, dst_d = 2,1,(2,1,2)
    src_rvd = (src_r, src_v) + src_d
    dst_rvd = (dst_r, dst_v) + dst_d
    ndevs = src_r * src_v * np.prod(np.array(src_d))

    ftensor = IRFullTensor(shape=fshape, name='tensor', requires_grad=False)

    pdevs = list(range(ndevs))
    fp_rvd = RVDLayout.grid(ftensor, r=src_r, v=src_v, dims=src_d, devices=pdevs)

    cdevs = list(range(ndevs))
    fc_rvd = RVDLayout.grid(ftensor, r=dst_r, v=dst_v, dims=dst_d, devices=cdevs)

    rvds = IntraPathFinder.get_optimal_path(ftensor, src_rvd, dst_rvd)
    # reduce-scatter(v2d) and then all-reduce(v2r)
    # Note: This is one of multiple possible optimal paths
    assert rvds == ((1, 4, 1, 1, 2), (1, 2, 2, 1, 2), (2, 1, 2, 1, 2))

    fprims = IntraPathFinder.path(fp_rvd, fc_rvd)
    assert len(fprims) == 8
    
    # Step 1: (1, 4, 1, 1, 2) => (1, 2, 2, 1, 2) via reduce_scatter
    # Once the path is determined, device mappings are deterministic based on RVDLayout.grid
    # here the device align is found with `inner_transpose` alternative.
    assert fprims[0].signature == 'nnscaler.runtime.adapter.reduce_scatter'
    assert fprims[0].device == [0, 2]  # Deterministic given the chosen path and layout
    assert fprims[1].signature == 'nnscaler.runtime.adapter.reduce_scatter'
    assert fprims[1].device == [1, 3]  # Deterministic given the chosen path and layout
    assert fprims[2].signature == 'nnscaler.runtime.adapter.reduce_scatter'
    assert fprims[2].device == [4, 6]  # Deterministic given the chosen path and layout
    assert fprims[3].signature == 'nnscaler.runtime.adapter.reduce_scatter'
    assert fprims[3].device == [5, 7]  # Deterministic given the chosen path and layout
    
    # Step 2: (1, 2, 2, 1, 2) => (2, 1, 2, 1, 2) via all_reduce
    # These device assignments follow from the intermediate layout structure
    assert fprims[4].signature == 'nnscaler.runtime.adapter.all_reduce'
    assert fprims[4].device == [0, 4]  # Deterministic based on intermediate RVD layout
    assert fprims[5].signature == 'nnscaler.runtime.adapter.all_reduce'
    assert fprims[5].device == [1, 5]  # Deterministic based on intermediate RVD layout
    assert fprims[6].signature == 'nnscaler.runtime.adapter.all_reduce'
    assert fprims[6].device == [2, 6]  # Deterministic based on intermediate RVD layout
    assert fprims[7].signature == 'nnscaler.runtime.adapter.all_reduce'
    assert fprims[7].device == [3, 7]  # Deterministic based on intermediate RVD layout


def test_f_reducescatter_alltoall():
    # this functio is trying to reproduce the case where reduce-scatter + all2all are used
    # which sometimes can lead some bugs
    # but currently we still can't reproduce the bug
    # this test case is for reference
    fshape = [8, 8]

    src_r, src_v, src_d = 1,2,(1,2)
    dst_r, dst_v, dst_d = 1,1,(4,1)
    src_rvd = (src_r, src_v) + src_d
    dst_rvd = (dst_r, dst_v) + dst_d
    ndevs = src_r * src_v * np.prod(np.array(src_d))
    assert ndevs == 4

    ftensor = IRFullTensor(shape=fshape, name='tensor', requires_grad=False)

    pdevs = list(range(ndevs))
    assert pdevs == [0, 1, 2, 3]
    fp_rvd = RVDLayout.grid(ftensor, r=src_r, v=src_v, dims=src_d, devices=pdevs)
    fp_subtensors = {
        f.device: f for f in fp_rvd.mat.flatten()
    }

    cdevs = list(range(ndevs))
    assert cdevs == [0, 1, 2, 3]
    fc_rvd = RVDLayout.grid(ftensor, r=dst_r, v=dst_v, dims=dst_d, devices=cdevs)
    fc_subtensors = {
        f.device: f for f in fc_rvd.mat.flatten()
    }

    rvds = IntraPathFinder.get_optimal_path(ftensor, src_rvd, dst_rvd)
    # reduce-scatter(v2d) and then all-2-all(d2d)
    assert rvds == ((1, 2, 1, 2), (1, 1, 2, 2), (1, 1, 4, 1))

    fprims = IntraPathFinder.path(fp_rvd, fc_rvd)
    assert len(fprims) == 4
    # (1, 2, 1, 2) => (1, 1, 2, 2)
    assert fprims[0].signature == 'nnscaler.runtime.adapter.reduce_scatter'
    assert fprims[0].device == [0, 2]
    assert fprims[0]._inputs[0].device == (0,)
    assert fprims[0]._inputs[0].indmap == ((0,8), (0,4))
    assert fprims[0]._inputs[0].valmap == (0, 2)
    assert fprims[0]._inputs[0] == fp_subtensors[fprims[0]._inputs[0].device]

    assert fprims[0]._inputs[1].device == (2,)
    assert fprims[0]._inputs[1].indmap == ((0,8), (0,4))
    assert fprims[0]._inputs[1].valmap == (1, 2)
    assert fprims[0]._inputs[1] == fp_subtensors[fprims[0]._inputs[1].device]

    assert fprims[0]._outputs[0].device == (0,)
    assert fprims[0]._outputs[0].indmap == ((0,4), (0,4))
    assert fprims[0]._outputs[0].valmap == (0, 1)

    assert fprims[0]._outputs[1].device == (2,)
    assert fprims[0]._outputs[1].indmap == ((4,8), (0,4))
    assert fprims[0]._outputs[1].valmap == (0, 1)

    assert fprims[1].signature == 'nnscaler.runtime.adapter.reduce_scatter'
    assert fprims[1].device == [1, 3]
    assert fprims[1]._inputs[0].device == (1,)
    assert fprims[1]._inputs[0].indmap == ((0,8), (4,8))
    assert fprims[1]._inputs[0].valmap == (0, 2)
    assert fprims[1]._inputs[0] == fp_subtensors[fprims[1]._inputs[0].device]

    assert fprims[1]._inputs[1].device == (3,)
    assert fprims[1]._inputs[1].indmap == ((0,8), (4,8))
    assert fprims[1]._inputs[1].valmap == (1, 2)
    assert fprims[1]._inputs[1] == fp_subtensors[fprims[1]._inputs[1].device]

    assert fprims[1]._outputs[0].device == (1,)
    assert fprims[1]._outputs[0].indmap == ((0,4), (4,8))
    assert fprims[1]._outputs[0].valmap == (0, 1)

    assert fprims[1]._outputs[1].device == (3,)
    assert fprims[1]._outputs[1].indmap == ((4,8), (4,8))
    assert fprims[1]._outputs[1].valmap == (0, 1)

    # (1, 1, 2, 2) => (1, 1, 4, 1)  d2d
    assert fprims[2].signature == 'nnscaler.runtime.adapter.all_to_all'
    assert fprims[2].device == [0, 1]
    assert fprims[2]._inputs[0] == fprims[0]._outputs[0]
    assert fprims[2]._inputs[0].device == fprims[0]._outputs[0].device
    assert fprims[2]._inputs[1] == fprims[1]._outputs[0]
    assert fprims[2]._inputs[1].device == fprims[1]._outputs[0].device

    assert fprims[2]._outputs[0].device == (0,)
    assert fprims[2]._outputs[1].device == (1,)
    assert fprims[2]._outputs[0] == fc_subtensors[fprims[2]._outputs[0].device]
    assert fprims[2]._outputs[1] == fc_subtensors[fprims[2]._outputs[1].device]

    assert fprims[3].signature == 'nnscaler.runtime.adapter.all_to_all'
    assert fprims[3].device == [2, 3]
    assert fprims[3]._inputs[0] == fprims[0]._outputs[1]
    assert fprims[3]._inputs[0].device == fprims[0]._outputs[1].device
    assert fprims[3]._inputs[1] == fprims[1]._outputs[1]
    assert fprims[3]._inputs[1].device == fprims[1]._outputs[1].device

    assert fprims[3]._outputs[0].device == (2,)
    assert fprims[3]._outputs[1].device == (3,)
    assert fprims[3]._outputs[0] == fc_subtensors[fprims[3]._outputs[0].device]
    assert fprims[3]._outputs[1] == fc_subtensors[fprims[3]._outputs[1].device]


def test_simple_v2r_transformation():
    """
    Test simple value-to-replica transformation: (1,8,1,1,1) → (8,1,1,1,1)
    This is a stable test case with deterministic path selection.
    Only one optimal path exists, making this test robust to algorithm changes.
    """
    fshape = [32, 64, 128]
    src_rvd = (1, 8, 1, 1, 1)
    dst_rvd = (8, 1, 1, 1, 1)
    ndevs = 8

    ftensor = IRFullTensor(shape=fshape, name='tensor', requires_grad=False)
    pdevs = list(range(ndevs))

    src_r, src_v = src_rvd[0], src_rvd[1]
    src_d = src_rvd[2:]
    dst_r, dst_v = dst_rvd[0], dst_rvd[1]
    dst_d = dst_rvd[2:]

    fp_rvd = RVDLayout.grid(ftensor, r=src_r, v=src_v, dims=src_d, devices=pdevs)
    fc_rvd = RVDLayout.grid(ftensor, r=dst_r, v=dst_v, dims=dst_d, devices=pdevs)

    # Get path and communication operations
    rvds = IntraPathFinder.get_optimal_path(ftensor, src_rvd, dst_rvd)
    fprims = IntraPathFinder.path(fp_rvd, fc_rvd)

    # Verify path properties
    assert rvds[0] == src_rvd
    assert rvds[-1] == dst_rvd
    assert len(rvds) == 2  # Direct transformation
    assert len(fprims) == 1  # Single operation

    # Verify operation types (more stable than specific devices)
    expected_ops = {'all_reduce': 1}
    actual_ops = {}
    for p in fprims:
        op_type = p.signature.split('.')[-1]
        actual_ops[op_type] = actual_ops.get(op_type, 0) + 1
    assert actual_ops == expected_ops

    # Verify device coverage
    all_devices = set()
    for p in fprims:
        all_devices.update(p.device)
    assert all_devices == set(range(8))


def test_simple_r2d_transformation():
    """
    Test simple replica-to-dimension transformation: (8,1,1,1,1) → (1,1,8,1,1)
    This is a stable test case with deterministic path selection.
    Only one optimal path exists, making this test robust to algorithm changes.
    """
    fshape = [32, 64, 128]
    src_rvd = (8, 1, 1, 1, 1)
    dst_rvd = (1, 1, 8, 1, 1)
    ndevs = 8

    ftensor = IRFullTensor(shape=fshape, name='tensor', requires_grad=False)
    pdevs = list(range(ndevs))

    src_r, src_v = src_rvd[0], src_rvd[1]
    src_d = src_rvd[2:]
    dst_r, dst_v = dst_rvd[0], dst_rvd[1]
    dst_d = dst_rvd[2:]

    fp_rvd = RVDLayout.grid(ftensor, r=src_r, v=src_v, dims=src_d, devices=pdevs)
    fc_rvd = RVDLayout.grid(ftensor, r=dst_r, v=dst_v, dims=dst_d, devices=pdevs)

    # Get path and communication operations
    rvds = IntraPathFinder.get_optimal_path(ftensor, src_rvd, dst_rvd)
    fprims = IntraPathFinder.path(fp_rvd, fc_rvd)

    # Verify path properties
    assert rvds[0] == src_rvd
    assert rvds[-1] == dst_rvd
    assert len(rvds) == 2  # Direct transformation
    assert len(fprims) == 1  # Single operation

    # Verify operation types (more stable than specific devices)
    expected_ops = {'chunk': 1}
    actual_ops = {}
    for p in fprims:
        op_type = p.signature.split('.')[-1]
        actual_ops[op_type] = actual_ops.get(op_type, 0) + 1
    assert actual_ops == expected_ops

    # Verify device coverage
    all_devices = set()
    for p in fprims:
        all_devices.update(p.device)
    assert all_devices == set(range(8))


def test_partial_v2r_transformation():
    """
    Test partial value-to-replica transformation: (1,4,1,1,2) → (4,1,1,1,2)
    This is a stable test case with deterministic path selection.
    The transformation is simple and direct, making the path selection robust.
    """
    fshape = [32, 64, 128]
    src_rvd = (1, 4, 1, 1, 2)
    dst_rvd = (4, 1, 1, 1, 2)
    ndevs = 8

    ftensor = IRFullTensor(shape=fshape, name='tensor', requires_grad=False)
    pdevs = list(range(ndevs))

    src_r, src_v = src_rvd[0], src_rvd[1]
    src_d = src_rvd[2:]
    dst_r, dst_v = dst_rvd[0], dst_rvd[1]
    dst_d = dst_rvd[2:]

    fp_rvd = RVDLayout.grid(ftensor, r=src_r, v=src_v, dims=src_d, devices=pdevs)
    fc_rvd = RVDLayout.grid(ftensor, r=dst_r, v=dst_v, dims=dst_d, devices=pdevs)

    # Get path and communication operations
    rvds = IntraPathFinder.get_optimal_path(ftensor, src_rvd, dst_rvd)
    fprims = IntraPathFinder.path(fp_rvd, fc_rvd)

    # Verify path properties
    assert rvds[0] == src_rvd
    assert rvds[-1] == dst_rvd
    assert len(rvds) == 2  # Direct transformation
    assert len(fprims) == 2  # Two parallel operations

    # Verify operation types (more stable than specific devices)
    expected_ops = {'all_reduce': 2}
    actual_ops = {}
    for p in fprims:
        op_type = p.signature.split('.')[-1]
        actual_ops[op_type] = actual_ops.get(op_type, 0) + 1
    assert actual_ops == expected_ops

    # Verify device coverage
    all_devices = set()
    for p in fprims:
        all_devices.update(p.device)
    assert all_devices == set(range(8))

    # Verify each operation involves exactly 4 devices (value=4)
    for p in fprims:
        assert len(p.device) == 4


def print_prims(fp_rvd, fc_rvd, fprims):
    print('fp_rvd:')
    for f in fp_rvd.mat.flatten():
        print(f'\tdevice({f.device[0]}): indmap({f.indmap}) | valmap({f.valmap})')

    print('prims:')
    for f in fprims:
        print(f.signature)
        for i, t in enumerate(f._inputs):
            print(f'\tinput {i}: device({t.device[0]}) | indmap({t.indmap}) | valmap({t.valmap})')
        for i, t in enumerate(f._outputs):
            print(f'\toutput {i}: device({t.device[0]}) | indmap({t.indmap}) | valmap({t.valmap})')

    print('fc_rvd:')
    for f in fc_rvd.mat.flatten():
        print(f'\tdevice({f.device[0]}): indmap({f.indmap}) | valmap({f.valmap})')


def test_align():
    fshape = [8, 8]

    src_r, src_v, src_d = 1,2,(1,2)
    dst_r, dst_v, dst_d = 1,1,(4,1)

    ftensor = IRFullTensor(shape=fshape, name='tensor', requires_grad=False)

    pdevs = [0, 2, 1, 3]
    fp_rvd = RVDLayout.grid(ftensor, r=src_r, v=src_v, dims=src_d, devices=pdevs)

    cdevs = [0, 1, 2, 3]
    fc_rvd = RVDLayout.grid(ftensor, r=dst_r, v=dst_v, dims=dst_d, devices=cdevs)

    rvds = ((1,2,1,2), (1, 1, 1, 4), (1, 1, 4, 1))
    align, all_prims = IntraPathFinder.device_align(fp_rvd, fc_rvd, rvds)
    assert True


def test_all_f_cases_fix_placement():
    fshape = [128, 256, 512]
    ftensor = IRFullTensor(shape=fshape, name='tensor', requires_grad=False)

    ndevs = 8
    ndims = len(fshape) + 2
    for src_rvd in factors(ndevs, ndims):
        for dst_rvd in factors(ndevs, ndims):
            if src_rvd == dst_rvd or src_rvd[1] < dst_rvd[1]: continue
            print(f'test generating | source rvd: {src_rvd}, destination rvd: {dst_rvd}')
            pdevs = list(range(ndevs))
            fp_rvd = RVDLayout.grid(ftensor, r=src_rvd[0], v=src_rvd[1], dims=src_rvd[2:], devices=pdevs)
            fptensors = fp_rvd.subtensors

            cdevs = list(range(ndevs))
            fc_rvd = RVDLayout.grid(ftensor, r=dst_rvd[0], v=dst_rvd[1], dims=dst_rvd[2:],devices=cdevs)
            fctensors = fc_rvd.subtensors

            fprims = IntraPathFinder.path(fp_rvd, fc_rvd)
    # the above code will not raise any exception
    assert True


def test_all_f_cases_auto_placement():
    fshape = [128, 256, 512]
    ftensor = IRFullTensor(shape=fshape, name='tensor', requires_grad=False)

    ndevs = 8
    ndims = len(fshape) + 2
    for src_rvd in factors(ndevs, ndims):
        for dst_rvd in factors(ndevs, ndims):
            if src_rvd == dst_rvd or src_rvd[1] < dst_rvd[1]: continue
            print(f'test generating | source rvd: {src_rvd}, destination rvd: {dst_rvd}')
            pdevs = list(range(ndevs))
            fp_rvd = RVDLayout.grid(ftensor, r=src_rvd[0], v=src_rvd[1], dims=src_rvd[2:], devices=pdevs)

            placement, cost = IntraAutoPlacer.advice(
                ftensor.shape,
                src_rvd, dst_rvd, None, None,
                src_placement=pdevs
            )
            fc_rvd = RVDLayout.grid(ftensor, r=dst_rvd[0], v=dst_rvd[1], dims=dst_rvd[2:],devices=placement)

            fprims = IntraPathFinder.path(fp_rvd, fc_rvd)
            print(f'cost: {cost}')
    # the above code will not raise any exception
    assert True


def test_one_fb_case():
    fshape = [128, 256, 512]

    # forward
    fsrc_r, fsrc_v, fsrc_d = 2,2,(1,1,2)
    fdst_r, fdst_v, fdst_d = 2,1,(1,1,4)
    bsrc_r, bsrc_v, bsrc_d = 1,2,(1,1,4)
    bdst_r, bdst_v, bdst_d = 4,1,(1,1,2)
    ndevs = fsrc_r * fsrc_v * np.prod(np.array(fsrc_d))

    ftensor = IRFullTensor(shape=fshape, name='tensor', requires_grad=True)
    btensor: IRFullTensor = ftensor.grad

    # forward producer / backward consumer
    fpdevs = list(range(ndevs))
    fp_rvd = RVDLayout.grid(ftensor, r=fsrc_r, v=fsrc_v, dims=fsrc_d, devices=fpdevs)
    # print('forward producer tensor:')
    # for t in fp_rvd.mat.flatten():
    #     print('\t'+tensor_vd_repr(t))
    bc_rvd = RVDLayout.grid(btensor, r=bdst_r, v=bdst_v, dims=bdst_d, devices=fpdevs)

    # forward consumer / backward producer
    fcdevs, _ = IntraAutoPlacer.advice(
        fshape, (fsrc_r, fsrc_v) + fsrc_d, (fdst_r, fdst_v) + fdst_d,
        (bsrc_r, bsrc_v) + bsrc_d, (bdst_r, bdst_v) + bdst_d, fpdevs)

    assert fcdevs == (0, 2, 1, 3, 4, 6, 5, 7)

    fc_rvd = RVDLayout.grid(ftensor, r=fdst_r, v=fdst_v, dims=fdst_d, devices=fcdevs)
    # print('forward consumer tensor:')
    # for t in fc_rvd.mat.flatten():
    #     print('\t'+tensor_vd_repr(t))
    bp_rvd = RVDLayout.grid(btensor, r=bsrc_r, v=bsrc_v, dims=bsrc_d, devices=fcdevs)

    fprims = IntraPathFinder.path(fp_rvd, fc_rvd)
    bprims = IntraPathFinder.path(bp_rvd, bc_rvd)

    assert len(fprims) == 4
    assert fprims[0].signature == 'nnscaler.runtime.adapter.reduce_scatter'
    assert fprims[0].device == [0, 2]
    assert fprims[1].signature == 'nnscaler.runtime.adapter.reduce_scatter'
    assert fprims[1].device == [1, 3]
    assert fprims[2].signature == 'nnscaler.runtime.adapter.reduce_scatter'
    assert fprims[2].device == [4, 6]
    assert fprims[3].signature == 'nnscaler.runtime.adapter.reduce_scatter'
    assert fprims[3].device == [5, 7]

    assert len(bprims) == 6
    assert bprims[0].signature == 'nnscaler.runtime.adapter.reduce_scatter'
    assert bprims[0].device == [0, 4]
    assert bprims[1].signature == 'nnscaler.runtime.adapter.reduce_scatter'
    assert bprims[1].device == [2, 6]
    assert bprims[2].signature == 'nnscaler.runtime.adapter.reduce_scatter'
    assert bprims[2].device == [1, 5]
    assert bprims[3].signature == 'nnscaler.runtime.adapter.reduce_scatter'
    assert bprims[3].device == [3, 7]
    assert bprims[4].signature == 'nnscaler.runtime.adapter.all_gather'
    assert bprims[4].device == [0, 2, 4, 6]
    assert bprims[5].signature == 'nnscaler.runtime.adapter.all_gather'
    assert bprims[5].device == [1, 3, 5, 7]


def test_all_fb_cases_fix_placement():
    fshape = [128, 256, 512]
    ndevs = 8

    ftensor = IRFullTensor(shape=fshape, name='tensor', requires_grad=True)
    btensor: IRFullTensor = ftensor.grad

    ndims = len(fshape) + 2
    for fp_rvd in factors(ndevs, ndims):

        fdevs = list(range(ndevs))
        fp = RVDLayout.grid(ftensor, r=fp_rvd[0], v=fp_rvd[1], dims=fp_rvd[2:], devices=fdevs)

        for fc_rvd in factors(ndevs, ndims):
            if fc_rvd[1] != 1: continue
            fc = RVDLayout.grid(ftensor, r=fc_rvd[0], v=fc_rvd[1], dims=fc_rvd[2:], devices=fdevs)

            # case1: forward replica -> backward replica
            bp_rvd = fc_rvd
            bc_rvd = (fp_rvd[0] * fp_rvd[1], 1) + fp_rvd[2:]
            print(f'test generating | fp rvd: {fp_rvd}, fc rvd: {fc_rvd}, bp rvd: {bp_rvd}, bc rvd: {bc_rvd}')

            bp = RVDLayout.grid(btensor, r=bp_rvd[0], v=bp_rvd[1], dims=bp_rvd[2:], devices=fdevs)
            bc = RVDLayout.grid(btensor, r=bc_rvd[0], v=bc_rvd[1], dims=bc_rvd[2:], devices=fdevs)

            fprims = IntraPathFinder.path(fp, fc)
            bprims = IntraPathFinder.path(bp, bc)

            # case2: forward replica -> backward accum
            bp_rvd = (1, fc_rvd[0] * fc_rvd[1]) + fc_rvd[2:]
            bc_rvd = (fp_rvd[0] * fp_rvd[1], 1) + fp_rvd[2:]
            print(f'test generating | fp rvd: {fp_rvd}, fc rvd: {fc_rvd}, bp rvd: {bp_rvd}, bc rvd: {bc_rvd}')

            bp = RVDLayout.grid(btensor, r=bp_rvd[0], v=bp_rvd[1], dims=bp_rvd[2:], devices=fdevs)
            bc = RVDLayout.grid(btensor, r=bc_rvd[0], v=bc_rvd[1], dims=bc_rvd[2:], devices=fdevs)

            fprims = IntraPathFinder.path(fp, fc)
            bprims = IntraPathFinder.path(bp, bc)

    # the above code will not raise any exception
    assert True


def test_all_fb_cases_advisor():
    fshape = [128, 256, 512]
    ndevs = 8

    ftensor = IRFullTensor(shape=fshape, name='tensor', requires_grad=True)
    btensor: IRFullTensor = ftensor.grad

    ndims = len(fshape) + 2
    for fp_rvd in factors(ndevs, ndims):

        fdevs = list(range(ndevs))
        fp = RVDLayout.grid(ftensor, r=fp_rvd[0], v=fp_rvd[1], dims=fp_rvd[2:], devices=fdevs)

        for fc_rvd in factors(ndevs, ndims):
            if fc_rvd[1] != 1: continue

            # case1: forward replica -> backward replica
            bp_rvd = fc_rvd
            bc_rvd = (fp_rvd[0] * fp_rvd[1], 1) + fp_rvd[2:]
            print(f'test generating | fp rvd: {fp_rvd}, fc rvd: {fc_rvd}, bp rvd: {bp_rvd}, bc rvd: {bc_rvd}')

            placement, cost = IntraAutoPlacer.advice(
                fshape, fp_rvd, fc_rvd, bp_rvd, bc_rvd, fdevs)

            fc = RVDLayout.grid(ftensor, r=fc_rvd[0], v=fc_rvd[1], dims=fc_rvd[2:], devices=placement)
            bp = RVDLayout.grid(btensor, r=bp_rvd[0], v=bp_rvd[1], dims=bp_rvd[2:], devices=placement)
            bc = RVDLayout.grid(btensor, r=bc_rvd[0], v=bc_rvd[1], dims=bc_rvd[2:], devices=fdevs)

            fprims = IntraPathFinder.path(fp, fc)
            bprims = IntraPathFinder.path(bp, bc)

            # case2: forward replica -> backward accum
            bp_rvd = (1, fc_rvd[0] * fc_rvd[1]) + fc_rvd[2:]
            bc_rvd = (fp_rvd[0] * fp_rvd[1], 1) + fp_rvd[2:]
            print(f'test generating | fp rvd: {fp_rvd}, fc rvd: {fc_rvd}, bp rvd: {bp_rvd}, bc rvd: {bc_rvd}')

            placement, cost = IntraAutoPlacer.advice(
                fshape, fp_rvd, fc_rvd, bp_rvd, bc_rvd, fdevs)

            fc = RVDLayout.grid(ftensor, r=fc_rvd[0], v=fc_rvd[1], dims=fc_rvd[2:], devices=placement)
            bp = RVDLayout.grid(btensor, r=bp_rvd[0], v=bp_rvd[1], dims=bp_rvd[2:], devices=placement)
            bc = RVDLayout.grid(btensor, r=bc_rvd[0], v=bc_rvd[1], dims=bc_rvd[2:], devices=fdevs)

            fprims = IntraPathFinder.path(fp, fc)
            bprims = IntraPathFinder.path(bp, bc)

    # the above code will not raise any exception
    assert True
