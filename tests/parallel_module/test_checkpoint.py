#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import tempfile
import itertools
import re
from pathlib import Path
import shutil
import pytest
from typing import Dict, Tuple, List
from dataclasses import dataclass, replace

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

import numpy as np

from nnscaler.parallel import ComputeConfig, parallelize, build_optimizer, merge_state_dicts, load_merged_state_dict
from nnscaler.runtime.module import ParallelModule, ExtraState
from nnscaler.runtime.gnorm import calcuate_gnorm

from .common import CubeLinear, init_random, init_distributed, PASMegatron
from ..launch_torchrun import launch_torchrun, clone_to_cpu_recursively
from ..utils import replace_all_device_with, clear_dir_on_rank0


class FcRelu(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.fc1 = CubeLinear(in_features, in_features, bias=bias)
        self.relu1 = nn.ReLU()
        self.fc2 = CubeLinear(in_features, out_features, bias=bias)
        self.relu2 = nn.ReLU()


    def forward(self, x):
        return self.relu2(self.fc2(self.relu1(self.fc1(x))))


class FcRelu_4_4(FcRelu):
    def __init__(self):
        super().__init__(4, 4)
        self.register_buffer('buffer', torch.ones(1, 4))
    def forward(self, x):
        return super().forward(x + self.buffer)


def _to_cube_model(module, pas, compute_config, cube_savedir, instance_name, dummy_input = None):
    return parallelize(
        module,
        dummy_input if dummy_input is not None else {'x': torch.tensor([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])},
        pas,
        compute_config,
        gen_savedir=cube_savedir,
        instance_name=instance_name
    )


def pipeline_dummy_data():
    return {
        'data': torch.randn(
            2, 16, device=torch.cuda.current_device()),
        'target': torch.rand(
            2, 16, device=torch.cuda.current_device())
    }


class End2EndMLP(nn.Module):
    def __init__(self):
        init_random()
        super().__init__()
        self.layers = torch.nn.ModuleList([])
        for _ in range(8):
            self.layers.append(nn.Linear(16, 16, bias=False))
        self.loss_fn = nn.BCELoss()

    def forward(self, data: Dict[str, torch.Tensor]):
        x = data['data']
        for layer in self.layers:
            x = layer(x)
        x = torch.sigmoid(x)
        loss = self.loss_fn(x, data['target'])
        return loss

    @classmethod
    def to_pipeline_module(cls, compute_config: ComputeConfig, cube_savedir,
        instance_name='pipeline', scheduler='1f1b'
    ):
        assert compute_config.runtime_ngpus == 4
        assert compute_config.plan_ngpus == 2
        compute_config = replace(compute_config,
            use_end2end=True,
            pas_config=dict(
                pipeline_nmicros=2,
                pipeline_nstages=2,
                pipeline_scheduler=scheduler
            )
        )
        return parallelize(
            cls,
            {'data': pipeline_dummy_data()},
            PASMegatron,
            compute_config,
            gen_savedir=cube_savedir,
            instance_name=instance_name
        )

    @classmethod
    def gen_pipeline_data(cls, data_size, start, end, rank, num_replicas):
        data = []
        for _ in range(data_size):
            data.append(pipeline_dummy_data())
        data = data[start:end]
        data = [data[i] for i in range(rank, len(data), num_replicas)]
        data = [(data[i:i + 2], None) for i in range(0, len(data), 2)]
        return data

    @classmethod
    def gen_raw_data(cls, data_size, start, end, rank, num_replicas):
        data = []
        for _ in range(data_size):
            data.append(pipeline_dummy_data())
        data = data[start:end]
        data = [(data[i], None) for i in range(rank, len(data), num_replicas)]
        return data


class End2EndMLPWithUnusedAndShared(End2EndMLP):
    def __init__(self):
        super().__init__()
        self.linear0_unused = nn.Linear(4, 4)  # unused weights
        self.layers[5].weight = self.layers[0].weight  # shared weights across stages


def train_step(model, x, y, optimizer):
    model.train()
    if isinstance(model, ParallelModule) and model.use_scheduler:
        # actually train_step will return two losses (for each input)
        # here we fake one loss to y_pred, so we don't need to change the check logic
        y_pred, loss = model.train_step(x)
        # workaround scalar tensor bug
        y_pred = y_pred.reshape(())
        loss = loss.reshape(())
    elif isinstance(model, End2EndMLP):
        y_pred = model(x)
        loss = y_pred
        loss.backward()
    else:
        loss_fn = nn.BCELoss()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
    optimizer.step()
    return y_pred, loss


def gendata(model, data_size, start, end, rank, num_replicas):
    data = []
    init_random()
    if isinstance(model, ParallelModule) and model.use_scheduler:
        data = End2EndMLP.gen_pipeline_data(data_size, start, end, rank, num_replicas)
    elif isinstance(model, End2EndMLP):
        data = End2EndMLP.gen_raw_data(data_size, start, end, rank, num_replicas)
    else:
        for _ in range(data_size):
            data.append((
                torch.randn((2, 4), device='cuda', dtype=torch.float32),
                torch.rand((2, 1), device='cuda', dtype=torch.float32),
            ))
        data = data[start:end]  # continue from last training
        data = [data[i] for i in range(rank, len(data), num_replicas)]
    return data


def _create_cube_module(pas, compute_config: ComputeConfig, cube_savedir, module_type='whole'):
    init_random()
    if module_type == 'whole':
        class CompiledModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(4, 4)
                self.fc_relu1 = FcRelu_4_4()
                self.linear2 = nn.Linear(4, 4)
                self.fc_relu2 = FcRelu_4_4()
                self.linear3 = nn.Linear(4, 1)
                self.sigmoid = nn.Sigmoid()
            def forward(self, x):
                x = self.linear1(x)
                x = self.fc_relu1(x)
                x = self.linear2(x)
                x = self.fc_relu2(x)
                x = self.linear3(x)
                x = self.sigmoid(x)
                return x
        CompiledModule = _to_cube_model(CompiledModule, pas, compute_config, cube_savedir, f'whole-{compute_config.inference_only}')
    elif module_type == 'pipeline':
        CompiledModule = End2EndMLP.to_pipeline_module(compute_config, cube_savedir,
            f'pipeline-{compute_config.inference_only}',
            scheduler='infer_pipe' if compute_config.inference_only else '1f1b'
        )
    elif module_type == 'sub':
        class CompiledModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(4, 4)
                self.fc_relu1 = _to_cube_model(FcRelu_4_4(), pas, compute_config, cube_savedir, f'fc_relu1-{compute_config.inference_only}')
                self.linear2 = nn.Linear(4, 4)
                self.fc_relu2 = _to_cube_model(FcRelu_4_4(), pas, compute_config, cube_savedir, f'fc_relu2-{compute_config.inference_only}')
                self.linear3 = nn.Linear(4, 1)
                self.sigmoid = nn.Sigmoid()
            def forward(self, x):
                x = self.linear1(x)
                x = self.fc_relu1(x)
                x = self.linear2(x)
                x = self.fc_relu2(x)
                x = self.linear3(x)
                x = self.sigmoid(x)
                return x
    elif module_type == 'start':
        class CompiledModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = _to_cube_model(CubeLinear(4, 4, bias=True),
                    pas, compute_config, cube_savedir, f'start_linear1-{compute_config.inference_only}'
                )
                self.linear2 = CubeLinear(4, 1, bias=True)
                self.sigmoid = nn.Sigmoid()
            def forward(self, x):
                x = self.linear1(x)
                x = self.linear2(x)
                x = self.sigmoid(x)
                return x
    elif module_type == 'end':
        # parallel module as the last module
        class CompiledModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = CubeLinear(4, 4, bias=True)
                self.linear2 = _to_cube_model(CubeLinear(4, 4, bias=True),
                    pas, compute_config, cube_savedir, f'end_linear2-{compute_config.inference_only}'
                )
                self.sigmoid = nn.Sigmoid()
            def forward(self, x):
                x = self.linear1(x)
                x = self.linear2(x)
                x = torch.sum(x, dim=1, keepdim=True)
                x = self.sigmoid(x)
                return x
    elif module_type == 'small':
        # num of parameter elements is small
        class CompiledModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = CubeLinear(4, 4, bias=True)
                self.linear2 = _to_cube_model(CubeLinear(4, 1, bias=True),
                    pas, compute_config, cube_savedir, f'small_linear2-{compute_config.inference_only}'
                )
                # the following tests depend on the rngstate in PASRandomSPMD
                if not compute_config.inference_only:
                    assert len(self.linear2.reducers) == 1
                    assert len(self.linear2.reducers[0].ranks) == 4
                self.sigmoid = nn.Sigmoid()
            def forward(self, x):
                x = self.linear1(x)
                x = self.linear2(x)
                x = self.sigmoid(x)
                return x
    init_random()
    compiled_module = CompiledModule().cuda()
    return compiled_module

DATA_SIZE = 256

@dataclass
class StepResult:
    pred: torch.Tensor
    loss: torch.Tensor
    grads: Dict[str, torch.Tensor]
    gnorm: torch.Tensor
    weights: Dict[str, torch.Tensor]


def assert_model_state_dict_equal(state_dict1: dict, state_dict2: dict):
    assert set(state_dict1.keys()) == set(state_dict2.keys())
    for index in state_dict1.keys():
        if index.endswith('CUBE_EXTRA_STATE'):
            continue
        assert torch.equal(state_dict1[index].cpu(), state_dict2[index].cpu())


def _train(model: torch.nn.Module, num_replicas, rank, start, end, ckpt_dir, inference_module: torch.nn.Module = None, check_merge_log=False):
    ckpt_file_template = 'ckpt_{rank}_{start}.pth'
    ckpt_merged_file_template = 'ckpt_merged_{start}.pth'
    temp_inferenece_ckpt_file_template = 'inference-{rank}.pth'
    ckpt_start_file = ckpt_dir / ckpt_file_template.format(
        rank=torch.distributed.get_rank(),
        start=start
    )
    ckpt_start_merged_file = ckpt_dir / ckpt_merged_file_template.format(
        start=start
    )
    temp_inferenece_ckpt_file = ckpt_dir / temp_inferenece_ckpt_file_template.format(rank=torch.distributed.get_rank())

    init_random()

    loss_fn = nn.BCELoss()
    optimizer = build_optimizer(model, torch.optim.Adam, lr=0.01)
    if ckpt_start_file.exists():
        ckpt_dict = torch.load(ckpt_start_file, weights_only=False)
        model_state_dict = ckpt_dict['model']
        for name, m in model.named_modules():
            prefix = f'{name}.' if name else ''
            if isinstance(m, ParallelModule):
                assert f'{prefix}CUBE_EXTRA_STATE' in model_state_dict
        optimizer_state_dict = ckpt_dict['optimizer']
        assert 'CUBE_EXTRA_STATE' in optimizer_state_dict
        model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(optimizer_state_dict)

        assert ckpt_start_merged_file.exists()
        merged_ckpt_dict = torch.load(ckpt_start_merged_file)
        merged_model_state_dict = merged_ckpt_dict['model']
        merged_opt_state_dict = merged_ckpt_dict['optimizer']

        # In most cases, we can't load state_dict directly
        # because they are different models, and the names of parameters are changed.
        # inference_module.load_state_dict(model_state_dict, strict=False)
        # assert not check_model_state_dict_equal(inference_module.state_dict(), model_state_dict)

        # inference model can be loaded from merged state_dict
        load_merged_state_dict(inference_module, merged_model_state_dict)
        torch.save(inference_module.state_dict(), temp_inferenece_ckpt_file)
        torch.distributed.barrier()
        inference_ckpt_files = [ckpt_dir / temp_inferenece_ckpt_file_template.format(rank=i) for i in range(torch.distributed.get_world_size())]
        inference_state_dicts = [torch.load(f, weights_only=False) for f in inference_ckpt_files]
        merged_inference_state_dict, _ = merge_state_dicts(inference_state_dicts)
        assert_model_state_dict_equal(merged_model_state_dict, merged_inference_state_dict)

        model_from_merged = type(model)()
        optimizer_from_merged = build_optimizer(model_from_merged, torch.optim.Adam, lr=0.01)
        load_merged_state_dict(
            model_from_merged, merged_model_state_dict,
            optimizer_from_merged, merged_opt_state_dict,
        )

        # check merged model
        result_orig_model_state_dict = model.state_dict()
        result_merged_model_state_dict = model_from_merged.state_dict()
        assert_model_state_dict_equal(result_orig_model_state_dict, result_merged_model_state_dict)

        result_orig_opt_state_dict = optimizer.state_dict()
        result_merged_opt_state_dict = optimizer_from_merged.state_dict()
        assert set(result_orig_opt_state_dict.keys()) == set(result_merged_opt_state_dict.keys())
        assert result_orig_opt_state_dict['CUBE_EXTRA_STATE'] == result_merged_opt_state_dict['CUBE_EXTRA_STATE']
        assert result_orig_opt_state_dict['param_groups'] == result_merged_opt_state_dict['param_groups']
        assert set(result_orig_opt_state_dict['state']) == set(result_merged_opt_state_dict['state'])
        for index in result_orig_opt_state_dict['state']:
            for key in ('step', 'exp_avg', 'exp_avg_sq'):
                assert torch.equal(result_orig_opt_state_dict['state'][index][key], result_merged_opt_state_dict['state'][index][key])
    torch.distributed.barrier()
    data = gendata(model, DATA_SIZE, start, end, rank, num_replicas)
    results = []
    for i, (x, y) in enumerate(data):
        y_pred, loss = train_step(model, x, y, optimizer)
        grads = {n: p.grad.clone() for n, p in model.named_parameters()}
        gnorm = optimizer.clip_gnorm()
        results.append(clone_to_cpu_recursively([y_pred, loss, grads, gnorm]))
        optimizer.zero_grad()
        weights = {n: p.data.clone() for n, p in model.named_parameters()}
        results[-1].append(clone_to_cpu_recursively(weights))
        results[-1] = StepResult(*results[-1])

    ckpt_file = ckpt_dir / ckpt_file_template.format(
        rank=torch.distributed.get_rank(),
        start=end
    )
    ckpt_merged_file = ckpt_dir / ckpt_merged_file_template.format(
        start=end
    )
    model_state_dict = model.state_dict()
    for name, m in model.named_modules():
        if isinstance(m, ParallelModule):
            prefix = f'{name}.' if name else ''
            assert f'{prefix}CUBE_EXTRA_STATE' in model_state_dict
            extra_state1 = ExtraState(**model_state_dict[f'{prefix}CUBE_EXTRA_STATE'])
            assert extra_state1.compute_config
            if extra_state1.compute_config.use_zero:
                assert extra_state1.model_idx2opt_idx
                assert extra_state1.opt_idx2ranks
            assert extra_state1.origin_param_names
    optimizer_state_dict = optimizer.state_dict()
    assert 'CUBE_EXTRA_STATE' in optimizer_state_dict
    torch.save({
        'model': model_state_dict,
        'optimizer': optimizer_state_dict
    }, ckpt_file)
    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        ckpt_files = [ckpt_dir / ckpt_file_template.format(rank=i, start=end) for i in range(torch.distributed.get_world_size())]
        ckpt_state_dicts = [torch.load(f, weights_only=False) for f in ckpt_files]
        model_state_dicts = [ckpt['model'] for ckpt in ckpt_state_dicts]
        optimizer_state_dicts = [ckpt['optimizer'] for ckpt in ckpt_state_dicts]
        if check_merge_log:
            from nnscaler.runtime.module import _logger
            import logging
            from io import StringIO
            string_stream = StringIO()
            old = _logger.level
            _logger.setLevel(logging.DEBUG)
            handler = logging.StreamHandler(string_stream)
            handler.setLevel(logging.DEBUG)
            _logger.addHandler(handler)
            merged_model_state_dicts, merged_optimizer_state_dict = merge_state_dicts(model_state_dicts, optimizer_state_dicts)
            logs = string_stream.getvalue()
            # check some zero merging is skipped due to replicate
            assert 'skip merging duplicated optimizer state for param' in logs
            assert 'skip merging duplicated model state for param' in logs
            _logger.removeHandler(handler)
            _logger.setLevel(old)
        else:
            merged_model_state_dicts, merged_optimizer_state_dict = merge_state_dicts(model_state_dicts, optimizer_state_dicts)
        torch.save({
            'model': merged_model_state_dicts,
            'optimizer': merged_optimizer_state_dict
        }, ckpt_merged_file)
    torch.distributed.barrier()
    return results


def _gpu_worker(module_type, use_zero, pas, plan_ngpus, runtime_ngpus, per_resume_update_count, resume_count, check_module=None):
    init_distributed()
    compiled_results = []
    with clear_dir_on_rank0(Path(tempfile.gettempdir()) / 'cube_test_ckpt') as tempdir:
        for i in range(resume_count):
            start = i * per_resume_update_count
            end = (i + 1) * per_resume_update_count
            compiled_module = _create_cube_module(pas,
                ComputeConfig(plan_ngpus, runtime_ngpus, use_zero=use_zero),
                tempdir,
                module_type,
            )
            compiled_inference_module = _create_cube_module(pas,
                ComputeConfig(plan_ngpus, runtime_ngpus, use_zero=use_zero, inference_only=True),
                tempdir,
                module_type,
            )
            if check_module:
                check_module(compiled_module)
            compiled_results.extend(_train(
                compiled_module,
                runtime_ngpus // plan_ngpus,
                torch.distributed.get_rank() // plan_ngpus,
                start, end, tempdir,
                inference_module=compiled_inference_module
            ))
        return compiled_results


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4, reason='lack of gpu devices')
@pytest.mark.parametrize('module_type', ['sub', 'whole', 'start', 'end', 'small', 'pipeline'])
@pytest.mark.parametrize('use_zero', [True, False])
def test_checkpoint(module_type, use_zero):
    plan_ngpus = 2
    runtime_ngpus = 4
    cube_results = launch_torchrun(4, _gpu_worker, module_type, use_zero, 'tp', plan_ngpus, runtime_ngpus, 32, 1)
    rcube_results = launch_torchrun(4, _gpu_worker, module_type, use_zero, 'tp', plan_ngpus, runtime_ngpus, 16, 2)

    results0, results1,  results2, results3 = cube_results[0], cube_results[1], cube_results[2], cube_results[3]
    rresults0, rresults1,  rresults2, rresults3 = rcube_results[0], rcube_results[1], rcube_results[2], rcube_results[3]

    # pred, loss
    for r0, r1 in [(results0, results1), (results2, results3),
                   (rresults0, rresults1), (rresults2, rresults3),
                   (results0, rresults0), (results2, rresults2)
        ]:
        # have the same input
        assert len(r0) == len(r1)  # iteration count
        for i in range(len(r0)):
            a, b = r0[i], r1[i]
            assert torch.equal(a.pred, b.pred)  # pred
            assert torch.equal(a.loss, b.loss)  # loss
            assert torch.equal(a.gnorm, b.gnorm)  # gnorm

    # grad, weights
    for r0, r1 in [(results0, results2), (results1, results3),
                   (rresults0, rresults2), (rresults1, rresults3),
                   (results0, rresults0), (results1, rresults1)
        ]:
        # in the same shard, grads and weights are the same
        assert len(r0) == len(r1)
        for i in range(len(r0)):
            a, b = r0[i], r1[i]
            assert torch.equal(a.gnorm, b.gnorm)  # gnorm
            for k in a.grads.keys(): # grad
                assert torch.equal(a.grads[k], b.grads[k])
            for k in a.weights.keys():  # weights
                assert torch.equal(a.weights[k], b.weights[k])


def assert_intra_reducer(module: ParallelModule):
    assert module.compute_config.plan_ngpus == module.compute_config.runtime_ngpus
    assert len(module.reducers) > 0
    # so we have both parameters in reducers and not in reducers
    # (assume one reducer gives one bucket, which is true in general.)
    assert len(module.parameters_for_optimizer()) > len(module.reducers)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason='lack of gpu devices')
@pytest.mark.parametrize('module_type', ['whole'])
@pytest.mark.parametrize('use_zero', [True, False])
def test_checkpoint_intra_reducer(module_type, use_zero):
    """
    Test when:
    Some of the parameters will be added to reducers,
    but some of the parameters are not.
    """
    plan_ngpus = 2
    runtime_ngpus = 2
    cube_results = launch_torchrun(2, _gpu_worker, module_type, use_zero, 'tp', plan_ngpus, runtime_ngpus, 32, 1, assert_intra_reducer)
    rcube_results = launch_torchrun(2, _gpu_worker, module_type, use_zero, 'tp', plan_ngpus, runtime_ngpus, 16, 2, assert_intra_reducer)
    results0 = cube_results[0]
    rresults0 = rcube_results[0]

    # pred, loss
    for r0, r1 in [(results0, rresults0)]:
        # have the same input
        assert len(r0) == len(r1)  # iteration count
        for i in range(len(r0)):
            a, b = r0[i], r1[i]
            assert torch.equal(a.pred, b.pred)  # pred
            assert torch.equal(a.loss, b.loss)  # loss
            assert torch.equal(a.gnorm, b.gnorm)  # gnorm

    # grad, weights
    for r0, r1 in [(results0, rresults0)]:
        # in the same shard, grads and weights are the same
        assert len(r0) == len(r1)
        for i in range(len(r0)):
            a, b = r0[i], r1[i]
            assert torch.equal(a.gnorm, b.gnorm)  # gnorm
            for k in a.grads.keys(): # grad
                assert torch.equal(a.grads[k], b.grads[k])
            for k in a.weights.keys():  # weights
                assert torch.equal(a.weights[k], b.weights[k])


def _gpu_merge_worker():
    init_distributed()
    with clear_dir_on_rank0(Path(tempfile.gettempdir()) / 'cube_test_ckpt_merge') as tempdir:
        compiled_module = _create_cube_module('data',
            ComputeConfig(2, 2, use_zero=True),
            tempdir,
            'whole',
        )
        _train(
            compiled_module,
            1,
            0,
            0,
            8,
            tempdir,
            check_merge_log=True
        )


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason='lack of gpu devices')
def test_checkpoint_merge():
    launch_torchrun(2, _gpu_merge_worker)
