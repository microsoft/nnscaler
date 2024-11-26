#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""
PYTHONPATH=.:$PYTHONPATH torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    examples/mlp/train.py --policy PASMegatronTP
"""

from pathlib import Path
import tempfile
from typing import Dict, TypedDict
import pytest
import torch
from torch import nn
import torch.distributed

import nnscaler
from nnscaler.runtime.gnorm import calcuate_gnorm
from nnscaler.runtime.utils import microbatches
from nnscaler.runtime.module import ParallelModule
from nnscaler.parallel import ComputeConfig, build_optimizer, parallelize, merge_state_dicts
from .common import assert_equal, init_distributed, PASMegatron, init_random
from ..launch_torchrun import clone_to_cpu_recursively, launch_torchrun
from ..utils import replace_all_device_with, clear_dir_on_rank0

from .test_checkpoint import End2EndMLP


DATA_SIZE = 64
MBS = 2           # microbatch size
DIM = 16
LAYERS = 16

class MLP(nn.Module):
    def __init__(self, dim: int = DIM, nlayers: int = LAYERS):
        init_random()
        super().__init__()
        self.layers = torch.nn.ModuleList([])
        for _ in range(nlayers):
            self.layers.append(nn.Linear(dim, dim, bias=False))
        self.loss_fn = nn.BCELoss()

    def forward(self, data: Dict[str, torch.Tensor]):
        x = data['data']
        for layer in self.layers:
            x = layer(x)
        x = torch.sigmoid(x)
        loss = self.loss_fn(x, data['target'])
        return loss


def dummy_data():
    return {
        'data': torch.randn(
            MBS, DIM, device=torch.cuda.current_device()),
        'target': torch.rand(
            MBS, DIM, device=torch.cuda.current_device())
    }


def _train_cube(model: ParallelModule, mbs, num_replicas, rank):
    init_random()
    optimizer = build_optimizer(model, torch.optim.Adam, lr=0.01)
    data = []
    init_random()
    for _ in range(DATA_SIZE):
        data.append(dummy_data())
    data = [data[i] for i in range(rank, DATA_SIZE, num_replicas)]
    chunks = [data[i:i + mbs] for i in range(0, len(data), mbs)]
    results = []
    for _, x in enumerate(chunks):
        model.train()
        losses = model.train_step(x)
        optimizer.step()
        gnorm = optimizer.clip_gnorm()
        grads = {n: p.grad for n, p in model.named_parameters()}
        model._add_extra_state(grads, '')
        weights = {n: p.data for n, p in model.named_parameters()}
        model._add_extra_state(weights, '')
        results.append(clone_to_cpu_recursively([grads, weights, gnorm]))
        optimizer.zero_grad()
    return results


def _train_ga(model, update_freq, data_size=DATA_SIZE):
    init_random()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    data = []
    init_random()
    for _ in range(data_size):
        data.append(dummy_data())
    results = []
    for i, x in enumerate(data):
        model.train()
        loss = model(x)
        loss.backward()
        if i % update_freq == update_freq - 1:
            optimizer.step()
            gnorm = calcuate_gnorm(list(model.parameters()))[0]
            grads = {n: p.grad for n, p in model.named_parameters()}
            weights = {n: p.data for n, p in model.named_parameters()}
            results.append(clone_to_cpu_recursively([grads, weights, gnorm]))
            optimizer.zero_grad()
    return results


def gpu_worker_cube_general(runtime_ngpus, plan_ngpus, policy, nstages=None, nmicros=None, model_cls=MLP, async_reducer=False, use_zero=False, use_bucket=False, zero_use_reduce_scatter=False, pipeline_scheduler='1f1b'):
    init_distributed()
    init_random()
    nstages = nstages or plan_ngpus
    nmicros = nmicros or plan_ngpus
    with clear_dir_on_rank0(Path(tempfile.gettempdir()) / 'cube_test_end2end') as tempdir:
        init_random()
        model = model_cls()
        model = parallelize(
            model,
            {'data': dummy_data()},
            pas_policy=policy,
            compute_config= ComputeConfig(
                plan_ngpus, runtime_ngpus,
                use_end2end=True,
                use_zero=use_zero,
                zero_use_reduce_scatter=zero_use_reduce_scatter,
                use_async_reducer=async_reducer,
                reducer_bucket_cap_mb=1e-6 if use_bucket else 0, # 1e-6 to make sure one parameter per bucket
                pas_config=dict(
                    pipeline_nmicros=nmicros,
                    pipeline_nstages=nstages,
                    pipeline_scheduler=pipeline_scheduler
                ),
            ),
            gen_savedir=tempdir
        )
        model.cuda()
        train_result = _train_cube(model, nmicros, runtime_ngpus // plan_ngpus, torch.distributed.get_rank() // plan_ngpus)

        with torch.inference_mode():
            model.eval()
            init_random()
            infer_data = []
            for _ in range(nmicros):
                infer_data.append(dummy_data())
            infer_result = clone_to_cpu_recursively(model.infer_step(infer_data))

        return train_result, infer_result, clone_to_cpu_recursively(infer_data)


def gpu_worker_cube(runtime_ngpus, plan_ngpus, policy, nstages=None, nmicros=None, model_cls=MLP, pipeline_scheduler='1f1b'):
    return gpu_worker_cube_general(runtime_ngpus, plan_ngpus, policy, nstages, nmicros, model_cls, False, False, False, False, pipeline_scheduler)


class CubeOptions(TypedDict):
    use_zero: bool = False
    use_async_reducer: bool = False
    use_bucket: bool = False
    zero_use_reduce_scatter: bool = False


def gpu_work_cube_tp_2_4(option: CubeOptions):
    return gpu_worker_cube_general(4, 2, 'tp',
        use_zero=option['use_zero'],
        use_bucket=option['use_bucket'],
        async_reducer=option['use_async_reducer'],
        zero_use_reduce_scatter=option['zero_use_reduce_scatter'],
    )


def merge_cube_result(cube_results, zero_use_reduce_scatter=False):
    cube_result = []
    for i in range(len(cube_results[0])):
        for rank in cube_results:
            assert torch.equal(cube_results[rank][i][2], cube_results[0][i][2])
        if not zero_use_reduce_scatter:
            cube_result.append([
                merge_state_dicts([cube_results[rank][i][0] for rank in cube_results])[0],
                merge_state_dicts([cube_results[rank][i][1] for rank in cube_results])[0],
                cube_results[0][i][2]
            ])
        else:
            # grads are not merged for zero_use_reduce_scatter
            # as they are different in different ranks
            cube_result.append([
                merge_state_dicts([cube_results[rank][i][1] for rank in cube_results])[0],
                cube_results[0][i][2]
            ])
    return cube_result


def allclose(a, b, atol=1e-6, rtol=1e-6):
    assert len(a) == len(b)
    for step in range(len(a)):
        # grads and weights (grads can be absent in case of zero_use_reduce_scatter)
        assert len(a[step]) == len(b[step])
        for i in range(len(a[step]) - 1):
            assert len(a[step][i]) == len(b[step][i])
            for k in a[step][i].keys():
                assert torch.allclose(a[step][i][k].cpu(), b[step][i][k].cpu(), atol=atol, rtol=rtol)
        # gnorm is last element
        assert torch.allclose(a[step][-1].cpu(), b[step][-1].cpu(), atol=atol, rtol=rtol)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4, reason='lack of gpu devices')
def test_end2end():
    torch.cuda.set_device(0)
    torch.set_default_device(f'cuda:0')
    model = MLP()
    ga4_result = _train_ga(model, 4)  # micro_batch_size = 4
    assert len(ga4_result) == 16
    # will be used for comparision when zero_use_reduce_scatter is True
    ga4_result_without_grads = []
    for i in range(len(ga4_result)):
        ga4_result_without_grads.append([ga4_result[i][1], ga4_result[i][2]])

    cube2_results = launch_torchrun(4, gpu_worker_cube, 4, 2, 'hybrid') # micro_batch_size = 4
    for _, v in cube2_results.items():
        # all losses should be scalar tensor
        assert all(i.shape == () for i in v[1])
    cube2_result = merge_cube_result({k: v[0] for k, v in cube2_results.items()})
    assert len(cube2_result) == 16
    allclose(cube2_result, ga4_result)

    cube4_results = launch_torchrun(4, gpu_worker_cube, 4, 4, PASMegatron)  # micro_batch_size = 4
    for _, v in cube2_results.items():
        # all losses should be scalar tensor
        assert all(i.shape == () for i in v[1])
    cube4_result = merge_cube_result({k: v[0] for k, v in cube4_results.items()})
    assert len(cube4_result) == 16
    allclose(cube4_result, ga4_result)

    cube2_results_non_pipeline = {}
    for use_async_reducer in [False, True]:
        for use_zero in [False, True]:
            for use_bucket in [False, True]:
                zero_use_reduce_scatter = False
                cube2_results_non_pipeline[(use_zero, use_async_reducer, use_bucket, zero_use_reduce_scatter)] = launch_torchrun(
                    4, gpu_work_cube_tp_2_4,
                    CubeOptions(use_zero=use_zero,
                        use_async_reducer=use_async_reducer,
                        use_bucket=use_bucket,
                        zero_use_reduce_scatter=zero_use_reduce_scatter
                    )
                )
                if not use_zero:
                    cube2_results_non_pipeline[(use_zero, use_async_reducer, use_bucket, not zero_use_reduce_scatter)] = \
                    cube2_results_non_pipeline[(use_zero, use_async_reducer, use_bucket, zero_use_reduce_scatter)]
                else:
                    cube2_results_non_pipeline[(use_zero, use_async_reducer, use_bucket, not zero_use_reduce_scatter)] = launch_torchrun(
                        4, gpu_work_cube_tp_2_4,
                        CubeOptions(use_zero=use_zero,
                            use_async_reducer=use_async_reducer,
                            use_bucket=use_bucket,
                            zero_use_reduce_scatter=not zero_use_reduce_scatter
                        )
                    )

    for r in cube2_results_non_pipeline.values():
        for _, v in r.items():
            # all losses should be scalar tensor
            assert all(i.shape == () for i in v[1])

    cube2_result_non_pipeline = {
        kk: merge_cube_result({k: v[0] for k, v in vv.items()}, zero_use_reduce_scatter=kk[3])
        for kk, vv in cube2_results_non_pipeline.items()
    }

    for r in cube2_result_non_pipeline.values():
        assert len(r) == 16

    for use_async_reducer in [False, True]:
        for use_zero in [False, True]:
            for use_bucket in [False, True]:
                for zero_use_reduce_scatter in [False, True]:
                    allclose(cube2_result_non_pipeline[(use_zero, use_async_reducer, use_bucket, zero_use_reduce_scatter)],
                             ga4_result if not zero_use_reduce_scatter else ga4_result_without_grads,
                             atol=1e-5, rtol=1e-5) # looks tp introduces more error

    for use_zero in [False, True]:
        for zero_use_reduce_scatter in [False, True]:
            # when use_bucket, it should be the same for both async and non-async
            use_async_reducer = True
            use_bucket = True
            assert_equal(cube2_result_non_pipeline[(use_zero, use_async_reducer, use_bucket, zero_use_reduce_scatter)],
                        cube2_result_non_pipeline[(use_zero, not use_async_reducer, use_bucket, zero_use_reduce_scatter)])

    infer_results = {k: v[1] for k, v in cube2_results_non_pipeline[(False, False, False, False)].items()}
    infer_datas = {k: v[2] for k, v in cube2_results_non_pipeline[(False, False, False, False)].items()}
    assert len(infer_results) == 4
    assert len(infer_datas) == 4
    infer_result = infer_results[0]
    infer_data = infer_datas[0]
    for k in infer_results:
        assert_equal(infer_results[k], infer_result)
    for k in infer_datas:
        assert_equal(infer_datas[k], infer_data)

    for i, data in enumerate(infer_data):
        with torch.inference_mode():
            model.eval()
            loss = model({key: v.cuda() for key, v in data.items()})
            assert torch.allclose(loss.cpu(), infer_result[i].cpu(), atol=1e-6, rtol=1e-6)


class MLPShared(End2EndMLP):
    def __init__(self):
        super().__init__()
        self.layers[5].weight = self.layers[0].weight  # shared weights across stages


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4, reason='lack of gpu devices')
def test_pipeline_shared():
    torch.cuda.set_device(0)
    torch.set_default_device(f'cuda:0')
    model = MLPShared()
    ga4_result = _train_ga(model, 4)  # micro_batch_size = 4
    assert len(ga4_result) == 16
    for step in range(len(ga4_result)):
        # fake shared weights for later compare
        ga4_result[step][0]['layers.5.weight'] = ga4_result[step][0]['layers.0.weight']
        ga4_result[step][1]['layers.5.weight'] = ga4_result[step][1]['layers.0.weight']

    with pytest.raises(ValueError, match='is not supported in training mode'):
        ComputeConfig(
            2, 2,
            inference_only=False,
            use_end2end=True).apply_pipeline_scheduler(
            None, pipeline_nmicros=2, pipeline_nstages=2,
            pipeline_scheduler='infer_pipe'
        )
    with pytest.raises(ValueError, match='is not supported in inference mode'):
        ComputeConfig(
            2, 2,
            inference_only=True,
            use_end2end=True).apply_pipeline_scheduler(
            None, pipeline_nmicros=2, pipeline_nstages=2,
            pipeline_scheduler='1f1b'
        )

    for ps in ['1f1b', '1f1b_plus','gpipe']:
        # 'chimera_direct' needs more gpus
        # 'infer_pipe' only work for inference
        # None looks doesn't work
        cube2_results = launch_torchrun(4, gpu_worker_cube, 4, 2, 'hybrid', None, None, MLPShared, ps) # micro_batch_size = 4
        cube2_result = merge_cube_result({k: v[0] for k, v in cube2_results.items()})
        assert len(cube2_result) == 16
        allclose(cube2_result, ga4_result)

    # TODO: fix `chimera_direct`
    # cube4_results = launch_torchrun(4, gpu_worker_cube, 4, 4, PASMegatron, None, None, MLPShared, 'chimera_direct')  # micro_batch_size = 4
    # cube4_result = merge_cube_result({k: v[0] for k, v in cube4_results.items()})
    # assert len(cube4_result) == 16
    # allclose(cube4_result, ga4_result)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 8, reason='lack of gpu devices')
def test_pipeline():
    torch.cuda.set_device(0)
    torch.set_default_device(f'cuda:0')
    model = MLP()
    ga4_result = _train_ga(model, 4)  # micro_batch_size = 4
    assert len(ga4_result) == 16

    # pp_size = 2
    # tp_size = 2
    # scale unit size = 4
    cube8_results = launch_torchrun(8, gpu_worker_cube, 8, 4, PASMegatron, 2, 2)  # micro_batch_size = 4
    cube8_result = merge_cube_result({k: v[0] for k, v in cube8_results.items()})
    assert len(cube8_result) == 16
    allclose(cube8_result, ga4_result, atol=1e-5, rtol=1e-5) # looks tp introduces more error

    # TODO: scalar type support
    # `v[1].reshape(())` to unify torch.shape == [] or torch.shape == [1]
    infer_results = {k: tuple(i.reshape(()) for i in v[1]) for k, v in cube8_results.items()}
    infer_datas = {k: v[2] for k, v in cube8_results.items()}
    assert len(infer_results) == 8
    assert len(infer_datas) == 8
    infer_result = infer_results[0]
    infer_data = infer_datas[0]
    for k in infer_results:
        assert_equal(infer_results[k], infer_result)
    for k in infer_datas:
        assert_equal(infer_datas[k], infer_data)

    for i, data in enumerate(infer_data):
        with torch.inference_mode():
            model.eval()
            loss = model({key: v.cuda() for key, v in data.items()})
            assert torch.allclose(loss.cpu(), infer_result[i].cpu(), atol=1e-6, rtol=1e-6)


def _train_cube_one_sample(model: ParallelModule, mbs):
    init_random()
    optimizer = build_optimizer(model, torch.optim.Adam, lr=0.01)
    data = []
    init_random()
    data_size = mbs
    for _ in range(data_size):
        data.append(dummy_data())
    chunks = [data[i:i + mbs] for i in range(0, len(data), mbs)]
    results = []
    for _, x in enumerate(chunks):
        model.train()
        losses = model.train_step(x, [False, True], scale_fn=lambda t: t * 2.0)
        optimizer.step()
        gnorm = optimizer.clip_gnorm()
        grads = {n: p.grad for n, p in model.named_parameters()}
        model._add_extra_state(grads, '')
        weights = {n: p.data for n, p in model.named_parameters()}
        model._add_extra_state(weights, '')
        results.append(clone_to_cpu_recursively([grads, weights, gnorm]))
        optimizer.zero_grad()
    return results


def gpu_worker_cube_one_sample():
    init_distributed()
    init_random()
    with clear_dir_on_rank0(Path(tempfile.gettempdir()) / 'cube_test_end2end') as tempdir:
        init_random()
        model = MLP()
        model = parallelize(
            model,
            {'data': dummy_data()},
            pas_policy='hybrid',
            compute_config= ComputeConfig(
                2, 2,
                use_end2end=True,
                pas_config=dict(
                    pipeline_nmicros=2, pipeline_nstages=2,
                    pipeline_scheduler='1f1b'
                ),
            ),
            gen_savedir=tempdir
        )
        model.cuda()
        train_result = _train_cube_one_sample(model, 2)
        return train_result


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason='lack of gpu devices')
def test_loss_scaling():
    torch.cuda.set_device(0)
    torch.set_default_device(f'cuda:0')
    model = MLP()
    ga4_result = _train_ga(model, 1, 1)
    assert len(ga4_result) == 1
    ga4_grads = ga4_result[0][0]
    scaled_ga4_grads = {n: g * 2.0 for n, g in ga4_grads.items()}

    cube2_results = launch_torchrun(2, gpu_worker_cube_one_sample)
    cube2_result = merge_cube_result({k: v for k, v in cube2_results.items()})
    assert len(cube2_result) == 1
    cube2_grads = cube2_result[0][0]
    assert len(cube2_grads) == len(scaled_ga4_grads)
    for k in cube2_grads:
        assert torch.allclose(cube2_grads[k].cpu(), scaled_ga4_grads[k].cpu(), atol=1e-6, rtol=1e-6)
