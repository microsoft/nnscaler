#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from pathlib import Path
import tempfile
import pytest
import torch
from torch import nn

from nnscaler import parallelize, ComputeConfig, ParallelModule

from nnscaler.parallel import build_optimizer, sync_grad_when, merge_state_dicts
from tests.launch_torchrun import launch_torchrun
from tests.launch_torchrun import clone_to_cpu_recursively
from tests.parallel_module.common import assert_equal, init_distributed
from tests.utils import clear_dir_on_rank0, init_random
from .test_wholemodule import FcRelu_4_4


class OrigModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_relu1 = FcRelu_4_4()
        self.fc_relu2 = FcRelu_4_4()
        self.linear3 = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.fc_relu1(x)
        x = self.fc_relu2(x)
        x = self.linear3(x)
        x = self.sigmoid(x)
        return x


def _to_cube_model(module, pas, compute_config, cube_savedir, instance_name):
    return parallelize(
        module,
        {'x': torch.tensor([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])},
        pas,
        compute_config,
        gen_savedir=cube_savedir,
        instance_name=instance_name
    )


def _create_modules(pas, compute_config, cube_savedir, name_prefix=''):
    init_random()
    whole_module = _to_cube_model(
        OrigModule(), pas, compute_config, cube_savedir, f'{name_prefix}whole'
    ).cuda()
    init_random()
    sub_module = OrigModule().cuda()
    sub_module.fc_relu1 = _to_cube_model(
        sub_module.fc_relu1, pas, compute_config, cube_savedir, f'{name_prefix}fc_relu1'
    ).cuda()
    sub_module.fc_relu2 = _to_cube_model(
        sub_module.fc_relu2, pas, compute_config, cube_savedir, f'{name_prefix}fc_relu2'
    ).cuda()
    return whole_module, sub_module


def _train(model: ParallelModule, update_freq):
    init_random()

    loss_fn = nn.BCELoss()
    optimizer = build_optimizer(model, torch.optim.Adam, lr=0.1)
    data = []
    DATA_SIZE = 20
    UPDATE_FREQ = update_freq
    for _ in range(DATA_SIZE):
        data.append((
            torch.randn((2, 4), device='cuda', dtype=torch.float32),
            torch.rand((2, 1), device='cuda', dtype=torch.float32),
        ))
    results = []
    for i, (x, y) in enumerate(data):
        model.train()
        with sync_grad_when(i % UPDATE_FREQ == UPDATE_FREQ - 1):
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
        if i % UPDATE_FREQ == UPDATE_FREQ - 1:
            optimizer.step()
            optimizer.zero_grad()
            results.append(clone_to_cpu_recursively([y_pred, model.state_dict()]))
    return results


def _gpu_worker(pas, ngpus, update_freq):
    init_distributed()
    with clear_dir_on_rank0(Path(tempfile.gettempdir()) / 'cube_test_async') as tempdir:
        whole_module_async, sub_module_async = _create_modules(
            pas, ComputeConfig(
                1, ngpus, use_async_reducer=True,
                reducer_bucket_cap_mb=1e-6
            ),
            tempdir,
            'async_',
        )
        whole_module_sync, sub_module_sync = _create_modules(
            pas, ComputeConfig(
                1, ngpus, use_async_reducer=False,
                reducer_bucket_cap_mb=100
            ),
            tempdir,
            'sync_',
        )
        whole_async_results = _train(whole_module_async, update_freq)
        whole_sync_results = _train(whole_module_sync, update_freq)
        sub_async_results = _train(sub_module_async, update_freq)
        sub_sync_results = _train(sub_module_sync, update_freq)
        return (
            whole_async_results,
            whole_sync_results,
            sub_async_results,
            sub_sync_results
        )


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason='lack of gpu devices')
@pytest.mark.parametrize('update_freq', [1, 4])
def test_dp2(update_freq):
    results = launch_torchrun(2, _gpu_worker, 'dp', 2, update_freq)
    whole_async0, whole_sync0, sub_async0, sub_sync0 = results[0]
    whole_async1, whole_sync1, sub_async1, sub_sync1 = results[1]

    assert len(whole_async0) == len(whole_sync0) == len(sub_async0) == len(sub_sync0)

    for iter in range(len(whole_async0)): # for each iteration
        iter_whole_async0 = whole_async0[iter]
        iter_whole_sync0 = whole_sync0[iter]
        iter_sub_async0 = sub_async0[iter]
        iter_sub_sync0 = sub_sync0[iter]

        iter_whole_async1 = whole_async1[iter]
        iter_whole_sync1 = whole_sync1[iter]
        iter_sub_async1 = sub_async1[iter]
        iter_sub_sync1 = sub_sync1[iter]

        # pred
        assert torch.equal(iter_whole_async0[0], iter_whole_async1[0])
        assert torch.equal(iter_sub_async0[0], iter_sub_async1[0])
        assert torch.equal(iter_whole_sync0[0], iter_whole_sync1[0])
        assert torch.equal(iter_sub_sync0[0], iter_sub_sync1[0])

        assert torch.equal(iter_whole_async0[0], iter_whole_sync0[0])
        assert torch.equal(iter_sub_async0[0], iter_sub_sync0[0])
        assert torch.equal(iter_whole_async0[0], iter_sub_async0[0])

        # weights
        whole_async_weights, _ = merge_state_dicts([iter_whole_async0[1], iter_whole_async1[1]])
        whole_sync_weights, _ = merge_state_dicts([iter_whole_sync0[1], iter_whole_sync1[1]])
        sub_async_weights, _ = merge_state_dicts([iter_sub_async0[1], iter_sub_async1[1]])
        sub_sync_weights, _ = merge_state_dicts([iter_sub_sync0[1], iter_sub_sync1[1]])

        assert_equal(whole_async_weights, whole_sync_weights)
        assert_equal(sub_async_weights, sub_sync_weights)

        assert set(whole_async_weights.keys()) == set(sub_async_weights.keys())

        for key in whole_async_weights.keys():
            assert torch.equal(whole_async_weights[key], sub_async_weights[key])


class OrigModuleEnd2End(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.orig_module = OrigModule()
        self.loss_fn = nn.BCELoss()

    def forward(self, data):
        x = data['data']
        x = self.orig_module(x)
        loss = self.loss_fn(x, data['target'])
        return loss


def _train_pp(model: ParallelModule, num_replicas, rank):
    mbs = model.nmicros_per_scheduler_step
    assert model.use_scheduler

    init_random()

    optimizer = build_optimizer(model, torch.optim.Adam, lr=0.1)
    data = []
    DATA_SIZE = 64
    for _ in range(DATA_SIZE):
        data.append({
            'data': torch.randn((2, 4), device='cuda', dtype=torch.float32),
            'target': torch.rand((2, 1), device='cuda', dtype=torch.float32),
        })
    data = [data[i] for i in range(rank, DATA_SIZE, num_replicas)]
    chunks = [data[i:i + mbs] for i in range(0, len(data), mbs)]
    results = []
    for _, x in enumerate(chunks):
        model.train()
        _ = model.train_step(x)
        optimizer.step()
        optimizer.zero_grad()
        results.append(clone_to_cpu_recursively(model.state_dict()))
    return results


def _gpu_worker_pp(pas, pp_ngpus, runtime_ngpus, update_freq):
    init_distributed()
    with clear_dir_on_rank0(Path(tempfile.gettempdir()) / 'cube_test_pp_async') as tempdir:
        init_random()
        whole_module_async = parallelize(
            OrigModuleEnd2End(), {
                'data': {
                    'data': torch.randn(2, 4, device=torch.cuda.current_device()),
                    'target': torch.rand(2, 1, device=torch.cuda.current_device())
                }
            },
            pas, ComputeConfig(
            pp_ngpus, runtime_ngpus, use_async_reducer=True,
            reducer_bucket_cap_mb=1e-6,
            use_end2end=True,
            pas_config=dict(
                    pipeline_nmicros=update_freq,
                    pipeline_nstages=pp_ngpus,
                    pipeline_scheduler='1f1b',
                )
            ),
            gen_savedir=tempdir,
            instance_name='async_pp_whole'
        ).cuda()

        init_random()
        whole_module_sync = parallelize(
            OrigModuleEnd2End(), {
                'data': {
                    'data': torch.randn(2, 4, device=torch.cuda.current_device()),
                    'target': torch.rand(2, 1, device=torch.cuda.current_device())
                }
            }, pas,
            ComputeConfig(
                pp_ngpus, runtime_ngpus, use_async_reducer=False,
                reducer_bucket_cap_mb=1e-6,
                use_end2end=True,
                pas_config=dict(
                    pipeline_nmicros=update_freq,
                    pipeline_nstages=pp_ngpus,
                    pipeline_scheduler='1f1b',
                )
            ),
            gen_savedir=tempdir,
            instance_name='sync_pp_whole'
        ).cuda()

        whole_async_results = _train_pp(whole_module_async, runtime_ngpus // pp_ngpus, torch.distributed.get_rank() // pp_ngpus)
        whole_sync_results = _train_pp(whole_module_sync, runtime_ngpus // pp_ngpus, torch.distributed.get_rank() // pp_ngpus)

        return (
            whole_async_results,
            whole_sync_results,
        )


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4, reason='lack of gpu devices')
def test_pp2():
    results = launch_torchrun(4, _gpu_worker_pp, 'pp', 2, 4, 4)
    whole_async0, whole_sync0 = results[0]
    whole_async1, whole_sync1 = results[1]
    whole_async2, whole_sync2 = results[2]
    whole_async3, whole_sync3 = results[3]

    assert len(whole_async0) == len(whole_sync0)

    for iter in range(len(whole_async0)): # for each iteration
        assert_equal(
            merge_state_dicts(
                [whole_async0[iter], whole_async1[iter], whole_async2[iter], whole_async3[iter]]
            ),
            merge_state_dicts(
                [whole_sync0[iter], whole_sync1[iter], whole_sync2[iter], whole_sync3[iter]]
            )
        )


def _gpu_worker_interleaved_pp(tempdir, tp_size=1):
    init_distributed()
    pp_size = 2
    stages = 4
    plan_ngpus = pp_size * tp_size
    runtime_ngpus = 4
    update_freq = 8
    # the generated train_step:
    # Please note
    # 1. the assignment of runtime flags (starting with `nnscaler.flags.RuntimeFlag`)
    # 2. Each gpus will hold 2 segments(stages) of pipeline (`segment53` and `segment69`)
    # def _train_step(model, dataloader_126):
    #     _ = None
    #     nnscaler.flags.RuntimeFlag.skip_zero_grad = False
    #     model.zero_grad()
    #     data_125 = next(*(dataloader_126, ))
    #     add_1_92 = nnscaler.runtime.executor.fexecute('segment53', model.segment53, *(data_125, ), requires_grad=True)
    #     _ = nnscaler.runtime.executor.aexecute(model.adapter256, *(add_1_92, ), requires_grad=False)
    #     data_316 = next(*(dataloader_126, ))
    #     add_1_321 = nnscaler.runtime.executor.fexecute('segment53', model.segment53, *(data_316, ), requires_grad=True)
    #     _ = nnscaler.runtime.executor.aexecute(model.adapter256, *(add_1_321, ), requires_grad=False)
    #     add_3_102 = nnscaler.runtime.executor.aexecute(model.adapter212, *(), requires_grad=True)
    #     add_5_112 = nnscaler.runtime.executor.fexecute('segment69', model.segment69, *(add_3_102, ), requires_grad=True)
    #     add_3_340 = nnscaler.runtime.executor.aexecute(model.adapter212, *(), requires_grad=True)
    #     _ = nnscaler.runtime.executor.aexecute(model.adapter286, *(add_5_112, ), requires_grad=False)
    #     add_5_360 = nnscaler.runtime.executor.fexecute('segment69', model.segment69, *(add_3_340, ), requires_grad=True)
    #     _ = nnscaler.runtime.executor.aexecute(model.adapter286, *(add_5_360, ), requires_grad=False)
    #     gadd_5_157 = nnscaler.runtime.executor.aexecute(model.adapter297, *(), requires_grad=False)
    #     data_378 = next(*(dataloader_126, ))
    #     add_1_383 = nnscaler.runtime.executor.fexecute('segment53', model.segment53, *(data_378, ), requires_grad=True)
    #     _ = nnscaler.runtime.executor.aexecute(model.adapter256, *(add_1_383, ), requires_grad=False)
    #     nnscaler.flags.RuntimeFlag.skip_reducer = True
    #     gadd_3_147 = nnscaler.runtime.executor.backward('segment69', (add_3_102, ), (add_5_112, ), (gadd_5_157, ))
    #     del add_5_112, gadd_5_157
    #     gadd_5_361 = nnscaler.runtime.executor.aexecute(model.adapter297, *(), requires_grad=False)
    #     _ = nnscaler.runtime.executor.aexecute(model.adapter223, *(gadd_3_147, ), requires_grad=False)
    #     del add_3_102, gadd_3_147
    #     data_407 = next(*(dataloader_126, ))
    #     add_1_412 = nnscaler.runtime.executor.fexecute('segment53', model.segment53, *(data_407, ), requires_grad=True)
    #     _ = nnscaler.runtime.executor.aexecute(model.adapter256, *(add_1_412, ), requires_grad=False)
    #     add_3_419 = nnscaler.runtime.executor.aexecute(model.adapter212, *(), requires_grad=True)
    #     nnscaler.flags.RuntimeFlag.skip_reducer = True
    #     gadd_3_341 = nnscaler.runtime.executor.backward('segment69', (add_3_340, ), (add_5_360, ), (gadd_5_361, ))
    #     del add_5_360, gadd_5_361
    #     _ = nnscaler.runtime.executor.aexecute(model.adapter223, *(gadd_3_341, ), requires_grad=False)
    #     del add_3_340, gadd_3_341
    #     gadd_1_137 = nnscaler.runtime.executor.aexecute(model.adapter267, *(), requires_grad=False)
    #     add_5_451 = nnscaler.runtime.executor.fexecute('segment69', model.segment69, *(add_3_419, ), requires_grad=True)
    #     add_3_459 = nnscaler.runtime.executor.aexecute(model.adapter212, *(), requires_grad=True)
    #     _ = nnscaler.runtime.executor.aexecute(model.adapter286, *(add_5_451, ), requires_grad=False)
    #     nnscaler.flags.RuntimeFlag.skip_reducer = True
    #     _ = nnscaler.runtime.executor.backward('segment53', (), (add_1_92, ), (gadd_1_137, ))
    #     del add_1_92, gadd_1_137
    #     gadd_1_322 = nnscaler.runtime.executor.aexecute(model.adapter267, *(), requires_grad=False)
    #     add_5_493 = nnscaler.runtime.executor.fexecute('segment69', model.segment69, *(add_3_459, ), requires_grad=True)
    #     _ = nnscaler.runtime.executor.aexecute(model.adapter286, *(add_5_493, ), requires_grad=False)
    #     nnscaler.flags.RuntimeFlag.skip_reducer = True
    #     _ = nnscaler.runtime.executor.backward('segment53', (), (add_1_321, ), (gadd_1_322, ))
    #     del add_1_321, gadd_1_322
    #     gadd_5_452 = nnscaler.runtime.executor.aexecute(model.adapter297, *(), requires_grad=False)
    #     data_520 = next(*(dataloader_126, ))
    #     add_1_525 = nnscaler.runtime.executor.fexecute('segment53', model.segment53, *(data_520, ), requires_grad=True)
    #     _ = nnscaler.runtime.executor.aexecute(model.adapter256, *(add_1_525, ), requires_grad=False)
    #     nnscaler.flags.RuntimeFlag.skip_reducer = True
    #     gadd_3_420 = nnscaler.runtime.executor.backward('segment69', (add_3_419, ), (add_5_451, ), (gadd_5_452, ))
    #     del add_5_451, gadd_5_452
    #     gadd_5_494 = nnscaler.runtime.executor.aexecute(model.adapter297, *(), requires_grad=False)
    #     _ = nnscaler.runtime.executor.aexecute(model.adapter223, *(gadd_3_420, ), requires_grad=False)
    #     del add_3_419, gadd_3_420
    #     data_549 = next(*(dataloader_126, ))
    #     add_1_554 = nnscaler.runtime.executor.fexecute('segment53', model.segment53, *(data_549, ), requires_grad=True)
    #     _ = nnscaler.runtime.executor.aexecute(model.adapter256, *(add_1_554, ), requires_grad=False)
    #     add_3_561 = nnscaler.runtime.executor.aexecute(model.adapter212, *(), requires_grad=True)
    #     nnscaler.flags.RuntimeFlag.skip_reducer = True
    #     gadd_3_460 = nnscaler.runtime.executor.backward('segment69', (add_3_459, ), (add_5_493, ), (gadd_5_494, ))
    #     del add_5_493, gadd_5_494
    #     _ = nnscaler.runtime.executor.aexecute(model.adapter223, *(gadd_3_460, ), requires_grad=False)
    #     del add_3_459, gadd_3_460
    #     gadd_1_384 = nnscaler.runtime.executor.aexecute(model.adapter267, *(), requires_grad=False)
    #     add_5_593 = nnscaler.runtime.executor.fexecute('segment69', model.segment69, *(add_3_561, ), requires_grad=True)
    #     add_3_601 = nnscaler.runtime.executor.aexecute(model.adapter212, *(), requires_grad=True)
    #     _ = nnscaler.runtime.executor.aexecute(model.adapter286, *(add_5_593, ), requires_grad=False)
    #     nnscaler.flags.RuntimeFlag.skip_reducer = True
    #     _ = nnscaler.runtime.executor.backward('segment53', (), (add_1_383, ), (gadd_1_384, ))
    #     del add_1_383, gadd_1_384
    #     gadd_1_413 = nnscaler.runtime.executor.aexecute(model.adapter267, *(), requires_grad=False)
    #     add_5_635 = nnscaler.runtime.executor.fexecute('segment69', model.segment69, *(add_3_601, ), requires_grad=True)
    #     _ = nnscaler.runtime.executor.aexecute(model.adapter286, *(add_5_635, ), requires_grad=False)
    #     nnscaler.flags.RuntimeFlag.skip_reducer = True
    #     _ = nnscaler.runtime.executor.backward('segment53', (), (add_1_412, ), (gadd_1_413, ))
    #     del add_1_412, gadd_1_413
    #     gadd_5_594 = nnscaler.runtime.executor.aexecute(model.adapter297, *(), requires_grad=False)
    #     data_662 = next(*(dataloader_126, ))
    #     add_1_667 = nnscaler.runtime.executor.fexecute('segment53', model.segment53, *(data_662, ), requires_grad=True)
    #     _ = nnscaler.runtime.executor.aexecute(model.adapter256, *(add_1_667, ), requires_grad=False)
    #     nnscaler.flags.RuntimeFlag.skip_reducer = True
    #     gadd_3_562 = nnscaler.runtime.executor.backward('segment69', (add_3_561, ), (add_5_593, ), (gadd_5_594, ))
    #     del add_5_593, gadd_5_594
    #     gadd_5_636 = nnscaler.runtime.executor.aexecute(model.adapter297, *(), requires_grad=False)
    #     _ = nnscaler.runtime.executor.aexecute(model.adapter223, *(gadd_3_562, ), requires_grad=False)
    #     del add_3_561, gadd_3_562
    #     data_691 = next(*(dataloader_126, ))
    #     add_1_696 = nnscaler.runtime.executor.fexecute('segment53', model.segment53, *(data_691, ), requires_grad=True)
    #     _ = nnscaler.runtime.executor.aexecute(model.adapter256, *(add_1_696, ), requires_grad=False)
    #     add_3_703 = nnscaler.runtime.executor.aexecute(model.adapter212, *(), requires_grad=True)
    #     nnscaler.flags.RuntimeFlag.skip_reducer = True
    #     gadd_3_602 = nnscaler.runtime.executor.backward('segment69', (add_3_601, ), (add_5_635, ), (gadd_5_636, ))
    #     del add_5_635, gadd_5_636
    #     _ = nnscaler.runtime.executor.aexecute(model.adapter223, *(gadd_3_602, ), requires_grad=False)
    #     del add_3_601, gadd_3_602
    #     gadd_1_526 = nnscaler.runtime.executor.aexecute(model.adapter267, *(), requires_grad=False)
    #     add_5_735 = nnscaler.runtime.executor.fexecute('segment69', model.segment69, *(add_3_703, ), requires_grad=True)
    #     add_3_743 = nnscaler.runtime.executor.aexecute(model.adapter212, *(), requires_grad=True)
    #     _ = nnscaler.runtime.executor.aexecute(model.adapter286, *(add_5_735, ), requires_grad=False)
    #     nnscaler.flags.RuntimeFlag.skip_reducer = True
    #     _ = nnscaler.runtime.executor.backward('segment53', (), (add_1_525, ), (gadd_1_526, ))
    #     del add_1_525, gadd_1_526
    #     gadd_1_555 = nnscaler.runtime.executor.aexecute(model.adapter267, *(), requires_grad=False)
    #     add_5_777 = nnscaler.runtime.executor.fexecute('segment69', model.segment69, *(add_3_743, ), requires_grad=True)
    #     _ = nnscaler.runtime.executor.aexecute(model.adapter286, *(add_5_777, ), requires_grad=False)
    #     nnscaler.flags.RuntimeFlag.skip_reducer = True
    #     _ = nnscaler.runtime.executor.backward('segment53', (), (add_1_554, ), (gadd_1_555, ))
    #     del add_1_554, gadd_1_555
    #     gadd_5_736 = nnscaler.runtime.executor.aexecute(model.adapter297, *(), requires_grad=False)
    #     nnscaler.flags.RuntimeFlag.skip_reducer = True
    #     gadd_3_704 = nnscaler.runtime.executor.backward('segment69', (add_3_703, ), (add_5_735, ), (gadd_5_736, ))
    #     del add_5_735, gadd_5_736
    #     gadd_5_778 = nnscaler.runtime.executor.aexecute(model.adapter297, *(), requires_grad=False)
    #     _ = nnscaler.runtime.executor.aexecute(model.adapter223, *(gadd_3_704, ), requires_grad=False)
    #     del add_3_703, gadd_3_704
    #     nnscaler.flags.RuntimeFlag.skip_reducer = False
    #     gadd_3_744 = nnscaler.runtime.executor.backward('segment69', (add_3_743, ), (add_5_777, ), (gadd_5_778, ))
    #     del add_5_777, gadd_5_778
    #     _ = nnscaler.runtime.executor.aexecute(model.adapter223, *(gadd_3_744, ), requires_grad=False)
    #     del add_3_743, gadd_3_744
    #     gadd_1_668 = nnscaler.runtime.executor.aexecute(model.adapter267, *(), requires_grad=False)
    #     nnscaler.flags.RuntimeFlag.skip_reducer = True
    #     _ = nnscaler.runtime.executor.backward('segment53', (), (add_1_667, ), (gadd_1_668, ))
    #     del add_1_667, gadd_1_668
    #     gadd_1_697 = nnscaler.runtime.executor.aexecute(model.adapter267, *(), requires_grad=False)
    #     nnscaler.flags.RuntimeFlag.skip_reducer = False
    #     _ = nnscaler.runtime.executor.backward('segment53', (), (add_1_696, ), (gadd_1_697, ))
    #     del add_1_696, gadd_1_697
    #     binary_cross_entropy_82 = nnscaler.runtime.executor.aexecute(model.adapter236, *(), requires_grad=True)
    #     binary_cross_entropy_391 = nnscaler.runtime.executor.aexecute(model.adapter236, *(), requires_grad=True)
    #     binary_cross_entropy_502 = nnscaler.runtime.executor.aexecute(model.adapter236, *(), requires_grad=True)
    #     binary_cross_entropy_533 = nnscaler.runtime.executor.aexecute(model.adapter236, *(), requires_grad=True)
    #     binary_cross_entropy_644 = nnscaler.runtime.executor.aexecute(model.adapter236, *(), requires_grad=True)
    #     binary_cross_entropy_675 = nnscaler.runtime.executor.aexecute(model.adapter236, *(), requires_grad=True)
    #     binary_cross_entropy_786 = nnscaler.runtime.executor.aexecute(model.adapter236, *(), requires_grad=True)
    #     binary_cross_entropy_809 = nnscaler.runtime.executor.aexecute(model.adapter236, *(), requires_grad=True)
    #     _ = nnscaler.runtime.executor.aexecute(model.reducer548, *(), requires_grad=False)
    #     return binary_cross_entropy_82, binary_cross_entropy_391, binary_cross_entropy_502, binary_cross_entropy_533, binary_cross_entropy_644, binary_cross_entropy_675, binary_cross_entropy_786, binary_cross_entropy_809

    init_random()
    whole_module_async = parallelize(
        OrigModuleEnd2End(), {
            'data': {
                'data': torch.randn(2, 4, device=torch.cuda.current_device()),
                'target': torch.rand(2, 1, device=torch.cuda.current_device())
            }
        },
        'hybrid', ComputeConfig(
        plan_ngpus, runtime_ngpus, use_async_reducer=True,
        reducer_bucket_cap_mb=1e-6,
        use_end2end=True,
        pas_config=dict(
                pipeline_nmicros=update_freq,
                pipeline_nstages=stages,
                pipeline_scheduler='1f1b_interleaved',
                pp_size=pp_size,
            )
        ),
        gen_savedir=tempdir,
        instance_name='async_interleaved_pp_whole'
    ).cuda()

    init_random()
    whole_module_sync = parallelize(
        OrigModuleEnd2End(), {
            'data': {
                'data': torch.randn(2, 4, device=torch.cuda.current_device()),
                'target': torch.rand(2, 1, device=torch.cuda.current_device())
            }
        }, 'hybrid',
        ComputeConfig(
            plan_ngpus, runtime_ngpus, use_async_reducer=False,
            reducer_bucket_cap_mb=1e-6,
            use_end2end=True,
            pas_config=dict(
                pipeline_nmicros=update_freq,
                pipeline_nstages=stages,
                pipeline_scheduler='1f1b_interleaved',
                pp_size=pp_size,
            )
        ),
        gen_savedir=tempdir,
        instance_name='sync_interleaved_pp_whole'
    ).cuda()

    whole_async_results = _train_pp(whole_module_async, runtime_ngpus // pp_size, torch.distributed.get_rank() // pp_size)
    whole_sync_results = _train_pp(whole_module_sync, runtime_ngpus // pp_size, torch.distributed.get_rank() // pp_size)

    return (
        whole_async_results,
        whole_sync_results,
    )


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4, reason='lack of gpu devices')
@pytest.mark.parametrize('tp_size', [1, 2])
def test_interleaved_pp(tmp_path, tp_size):
    results = launch_torchrun(4, _gpu_worker_interleaved_pp, tmp_path, tp_size)
    whole_async0, whole_sync0 = results[0]
    whole_async1, whole_sync1 = results[1]
    whole_async2, whole_sync2 = results[2]
    whole_async3, whole_sync3 = results[3]

    assert len(whole_async0) == len(whole_sync0)

    for iter in range(len(whole_async0)): # for each iteration
        assert_equal(
            merge_state_dicts(
                [whole_async0[iter], whole_async1[iter], whole_async2[iter], whole_async3[iter]]
            ),
            merge_state_dicts(
                [whole_sync0[iter], whole_sync1[iter], whole_sync2[iter], whole_sync3[iter]]
            )
        )
