#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from contextlib import contextmanager
import os
from pathlib import Path
import math
from typing import Dict, List

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, Timer
from lightning.fabric.utilities.cloud_io import _load as pl_load

import pytest
from unittest.mock import Mock, patch

import nnscaler
from nnscaler.parallel import ComputeConfig
from nnscaler.integration.lightning.pytorch import NnScalerStrategy, NnScalerPrecision
import nnscaler.runtime

from nnscaler.cli.trainer import Trainer as CliTrainer
from nnscaler.cli.trainer_args import CheckpointConfig, DataloaderConfig, DatasetConfig, DatasetSamplerConfig, HookConfig, ModelConfig, TrainerArgs, OptimizerConfig, LRSchedulerConfig

from ....launch_torchrun import launch_torchrun
from ....utils import init_random
from ....parallel_module.common import assert_close, assert_equal
from .simple_datamodules import ClassifDataModule, SklearnDataset
from .simple_models import BoringModel, ClassificationModel, ClassificationModelWithLRScheduler


def fit_worker(tmp_path):
    dm = ClassifDataModule()
    model = ClassificationModel()
    compute_config=ComputeConfig(1, 1)
    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=2,
        enable_progress_bar=False,
        accelerator="gpu", devices=1,
        gradient_clip_val=None,
        strategy=NnScalerStrategy(compute_config=compute_config, pas_policy='tp', gen_savedir=tmp_path),
        plugins=[NnScalerPrecision('32-true')]
    )
    trainer.fit(model, datamodule=dm)
    trainer.validate(model, datamodule=ClassifDataModule())


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason='lack of gpu devices')
def test_multi_gpu_model_only(tmp_path):
    launch_torchrun(1, fit_worker, tmp_path)


def ckpt_path_epoch_restored_worker(tmp_path):
    """Verify resuming from checkpoint runs the right number of epochs."""

    class TestModel(BoringModel):
        # Model that tracks epochs and batches seen
        num_epochs_end_seen = 0
        num_batches_seen = 0
        num_on_load_checkpoint_called = 0

        def on_train_epoch_end(self):
            self.num_epochs_end_seen += 1

        def on_train_batch_start(self, *_):
            self.num_batches_seen += 1

        def on_load_checkpoint(self, _):
            self.num_on_load_checkpoint_called += 1

    model = TestModel()
    max_epochs = 2
    compute_config=ComputeConfig(2, 2)
    trainer = Trainer(
        max_epochs=max_epochs,
        limit_train_batches=0.65,
        limit_val_batches=1,
        callbacks=ModelCheckpoint(dirpath=tmp_path, save_top_k=-1),
        default_root_dir=tmp_path,
        val_check_interval=1.0,
        enable_progress_bar=False,
        logger=False,
        enable_model_summary=False,
        strategy=NnScalerStrategy(compute_config=compute_config, pas_policy='tp', gen_savedir=tmp_path),
        plugins=[NnScalerPrecision('32-true')]
    )
    trainer.fit(model)

    assert model.num_epochs_end_seen == max_epochs
    assert model.num_batches_seen == trainer.num_training_batches * max_epochs == trainer.global_step
    assert model.num_on_load_checkpoint_called == 0

    checkpoints = sorted(list(set(Path(trainer.checkpoint_callback.dirpath).glob("*.ckpt"))))

    assert len(checkpoints) == max_epochs
    for ckpt in checkpoints:
        model = TestModel()
        state = pl_load(ckpt / '0.pt')
        # Resume training
        trainer = Trainer(
            default_root_dir=tmp_path, max_epochs=2,
            enable_progress_bar=False,
            strategy=NnScalerStrategy(
                compute_config=compute_config,
                pas_policy='tp',
                gen_savedir=tmp_path
            ),
            plugins=[NnScalerPrecision('32-true')]
        )
        trainer.fit(model, ckpt_path=ckpt)
        assert state["global_step"] + model.num_batches_seen == trainer.global_step
        assert model.num_on_load_checkpoint_called == 1


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason='lack of gpu devices')
def test_ckpt_path_epoch_restored(tmp_path):
    launch_torchrun(2, ckpt_path_epoch_restored_worker, tmp_path)


def trainer_accumulate_grad_batches_zero_grad(tmp_path, accumulate_grad_batches):
    with patch("torch.optim.SGD.zero_grad") as sgd_zero_grad:
        model = BoringModel()
        trainer = Trainer(
            num_nodes=1,
            devices=2,
            default_root_dir=tmp_path,
            num_sanity_val_steps=0,
            limit_train_batches=20,
            limit_val_batches=1,
            max_epochs=1,
            enable_progress_bar=False,
            enable_model_summary=False,
            accumulate_grad_batches=accumulate_grad_batches,
            strategy=NnScalerStrategy(compute_config=ComputeConfig(1, 2), pas_policy='tp', gen_savedir=tmp_path),
            plugins=[NnScalerPrecision('32-true')]
        )
        assert trainer.accumulate_grad_batches == accumulate_grad_batches
        trainer.fit(model)
        assert sgd_zero_grad.call_count == math.ceil(trainer.limit_train_batches / accumulate_grad_batches)


@pytest.mark.parametrize("accumulate_grad_batches", [1, 2, 3])
@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason='lack of gpu devices')
def test_trainer_accumulate_grad_batches_zero_grad(tmp_path, accumulate_grad_batches):
    launch_torchrun(2, trainer_accumulate_grad_batches_zero_grad, tmp_path, accumulate_grad_batches)


# hack to satisfy cli requirements
_correctnes_worker_datamodule: ClassifDataModule = None
_correctnes_worker_model: ClassificationModel = None
_correctnes_worker_update_history = []
_correctnes_worker_train_loss_history = []
_correctnes_worker_single_loss_history = []
_correctnes_worker_val_loss_history = []


def get_full_qualified_name(class_or_func):
    return f'{class_or_func.__module__}.{class_or_func.__qualname__}'


def correctnes_worker_cli_dataset(stage):
    if stage == 'train':
        return SklearnDataset(_correctnes_worker_datamodule.x_train,
                            _correctnes_worker_datamodule.y_train,
                            _correctnes_worker_datamodule._x_type,
                            _correctnes_worker_datamodule._y_type
                )
    elif stage == 'val':
        return SklearnDataset(_correctnes_worker_datamodule.x_valid,
                            _correctnes_worker_datamodule.y_valid,
                            _correctnes_worker_datamodule._x_type,
                            _correctnes_worker_datamodule._y_type
                )
    else:
        raise ValueError(f'Unknown stage: {stage}')


class CorrectnessWorkerM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m =_correctnes_worker_model
        self.m.log = lambda *args, **kwargs: None
        del self.m.train_acc
        self.m.train_acc = lambda *args, **kwargs: None

    def forward(self, batch):
        return self.m.training_step(batch, 0)['loss']


def on_before_grad_clip(trainer: Trainer):
    grads = {n: p.grad.cpu() for n, p in trainer.model.named_parameters()}
    weights = {n: p.data.cpu() for n, p in trainer.model.named_parameters()}
    _correctnes_worker_update_history.append((grads, weights))


def after_aggregate_train_step_outputs(trainer: Trainer, aggregated_outputs, train_loss):
    _correctnes_worker_train_loss_history.append(train_loss)


def on_train_step_end(trainer: 'Trainer', outputs) -> None:
    _correctnes_worker_single_loss_history.append(outputs[0].item())


_mocked_params: Dict[int, List[torch.Tensor]] = {}
@contextmanager
def mock_reducer_add_param():
    """
    Reorder the parameters in the reducer to match the order in the model
    """
    from nnscaler.runtime.adapter.reducer import Reducer
    from nnscaler.runtime.module import CubeModule
    def add_param(self, param):
        if id(self) not in _mocked_params:
            _mocked_params[id(self)] = []
        _mocked_params[id(self)].append(param)
    old_add_param = Reducer.add_param
    old_add_reducer = CubeModule.add_reducer
    Reducer.add_param = add_param
    def add_reducer(self, reducer):
        register_parameters = {}
        for idx, p in enumerate(self.parameters()):
            register_parameters[id(p)] = idx
        if id(reducer) in _mocked_params:
            _mocked_params[id(reducer)].sort(key=lambda x: register_parameters[id(x)])
            for p in _mocked_params[id(reducer)]:
                old_add_param(reducer, p)
            _mocked_params.pop(id(reducer))
        old_add_reducer(self, reducer)
    CubeModule.add_reducer = add_reducer
    yield
    Reducer.add_param = old_add_param
    CubeModule.add_reducer = old_add_reducer


@mock_reducer_add_param()
def correctnes_worker_cli(
    tmp_path,
    gradient_clip_val,
    with_lr_scheduler,
    precision='32-true',
    with_tp=False
):

    def on_val_step_end(trainer: Trainer, outputs) -> None:
        _correctnes_worker_val_loss_history.append(outputs[0].item())

    assert precision == '32-true'
    global _correctnes_worker_datamodule
    global _correctnes_worker_model
    init_random()
    dm = ClassifDataModule()
    _correctnes_worker_datamodule = dm
    lr_config = None
    init_random()
    _correctnes_worker_model = ClassificationModel()
    if with_lr_scheduler:
        lr_config = LRSchedulerConfig(
            type=torch.optim.lr_scheduler.StepLR,
            args={
                'step_size': 1,
            }
        )

    if with_tp:
        compute_config=ComputeConfig(2, 4, use_end2end=True)
        policy = 'tp'
    else:
        compute_config=ComputeConfig(1, 2, use_end2end=True)
        policy = 'dp'

    tmp_path = Path(tmp_path) / 'cli'
    train_args = TrainerArgs(
        compute_config=compute_config,
        gen_savedir=tmp_path / 'code',
        micro_batch_size=_correctnes_worker_model.batch_size,
        global_batch_size=_correctnes_worker_model.batch_size*2,
        max_epochs=2,
        pas_policy=policy,
        instance_name=f'cli_{policy}',
        enable_progress_bar=False,
        model=ModelConfig(
            type=CorrectnessWorkerM,
        ),
        dataset=DatasetConfig(
            type=correctnes_worker_cli_dataset,
            train_args={
                'stage': 'train'
            },
            val_args={
                'stage': 'val'
            },
        ),
        dataset_sampler=DatasetSamplerConfig(
            type='torch.utils.data.DistributedSampler',
            val_args={
                'shuffle': False, # lightning doesn't shuffle val set
            },
        ),
        optimizer=OptimizerConfig(
            type=torch.optim.Adam,
            args={
                'lr': _correctnes_worker_model.lr
            },
            clip_gnorm=gradient_clip_val,
        ),
        checkpoint=CheckpointConfig(
            no_save=True,
        ),
        lr_scheduler=lr_config,
        hook=dict(
            before_gnorm_clip=on_before_grad_clip,
            after_aggregate_train_step_outputs=after_aggregate_train_step_outputs,
            on_train_step_end=on_train_step_end,
            on_val_step_end=on_val_step_end,
        ),
    )
    trainer = CliTrainer(
        train_args=train_args,
    )
    _correctnes_worker_update_history.clear()
    _correctnes_worker_train_loss_history.clear()
    _correctnes_worker_single_loss_history.clear()
    _correctnes_worker_val_loss_history.clear()
    trainer.run()
    return _correctnes_worker_update_history, trainer.model.fullmap, \
        _correctnes_worker_val_loss_history, \
        _correctnes_worker_train_loss_history, \
        _correctnes_worker_single_loss_history


@mock_reducer_add_param()
def correctnes_worker_nnscaler(tmp_path, gradient_clip_val, with_lr_scheduler,
    precision='32-true',
    with_tp=False, with_empty_scaler=False
):
    init_random()
    dm = ClassifDataModule()
    init_random()
    if with_lr_scheduler:
        model = ClassificationModelWithLRScheduler()
    else:
        model = ClassificationModel()
    if with_tp:
        compute_config=ComputeConfig(2, 4)
        policy = 'tp'
        devices = 4
    else:
        compute_config=ComputeConfig(1, 2)
        policy = 'dp'
        devices = 2
    scaler = None
    if with_empty_scaler or precision == '16-mixed':
        scaler = torch.cuda.amp.GradScaler(enabled=(precision == '16-mixed'))
    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=2,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        accelerator="gpu", devices=devices,
        gradient_clip_val=gradient_clip_val,
        strategy=NnScalerStrategy(
            compute_config=compute_config, pas_policy=policy, gen_savedir=tmp_path,
            instance_name=policy
        ),
        plugins=[NnScalerPrecision(precision, scaler=scaler)]
    )
    trainer.fit(model, datamodule=dm)
    return model.update_history, model.nnscaler_pmodule.fullmap, model.val_loss_history, model.loss_history


@mock_reducer_add_param()
def correctnes_worker_nnscaler_checkpoint(tmp_path, gradient_clip_val, with_lr_scheduler,
    precision='32-true',
    with_tp=False, with_empty_scaler=False
):
    init_random()
    dm = ClassifDataModule()
    init_random()
    if with_lr_scheduler:
        model = ClassificationModelWithLRScheduler()
        state_dict_type = 'sharded'
    else:
        model = ClassificationModel()
        state_dict_type = 'deduped'
    if gradient_clip_val:
        do_merge = True
    else:
        do_merge = False
    if with_tp:
        compute_config=ComputeConfig(2, 4)
        policy = 'tp'
        devices = 4
    else:
        compute_config=ComputeConfig(1, 2)
        policy = 'dp'
        devices = 2
    scaler = None
    if with_empty_scaler or precision == '16-mixed':
        scaler = torch.cuda.amp.GradScaler(enabled=(precision == '16-mixed'))
    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=1,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        callbacks=[ModelCheckpoint(dirpath=tmp_path, save_top_k=1, save_last=True)],
        accelerator="gpu", devices=devices,
        gradient_clip_val=gradient_clip_val,
        strategy=NnScalerStrategy(
            compute_config=compute_config, pas_policy=policy, gen_savedir=tmp_path,
            instance_name=policy + '_resume',
            state_dict_type=state_dict_type
        ),
        plugins=[NnScalerPrecision(precision, scaler=scaler)]
    )
    trainer.fit(model, datamodule=dm)

    torch.distributed.barrier()
    if do_merge:
        resume_ckpt = Path(tmp_path) / 'merged.ckpt'
        ckpt_last_dir = Path(tmp_path) / 'last.ckpt'
        ckpt_last_files = list(ckpt_last_dir.glob('**/*.pt'))
        if torch.distributed.get_rank() == 0:
            NnScalerStrategy.merge_checkpoint(ckpt_last_files, resume_ckpt)
    else:
        resume_ckpt = Path(tmp_path) / 'last.ckpt'
    torch.distributed.barrier()

    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=2,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        callbacks=[ModelCheckpoint(dirpath=tmp_path, save_top_k=1, save_last=True)],
        accelerator="gpu", devices=devices,
        gradient_clip_val=gradient_clip_val,
        strategy=NnScalerStrategy(
            compute_config=compute_config, pas_policy=policy, gen_savedir=tmp_path,
            instance_name=policy + '_resume',
            state_dict_type=state_dict_type
        ),
        plugins=[NnScalerPrecision(precision, scaler=scaler)]
    )
    trainer.fit(model, datamodule=dm, ckpt_path=resume_ckpt)
    return model.update_history, model.nnscaler_pmodule.fullmap, model.val_loss_history, model.loss_history


def correctnes_worker_ddp(tmp_path, gradient_clip_val, with_lr_scheduler, precision='32-true'):
    init_random()
    dm = ClassifDataModule()
    init_random()
    if with_lr_scheduler:
        model = ClassificationModelWithLRScheduler()
    else:
        model = ClassificationModel()
    trainer = Trainer(
        default_root_dir=tmp_path,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        precision=precision,
        max_epochs=2,
        accelerator="gpu", devices=2,
        gradient_clip_val=gradient_clip_val,
        strategy='ddp',
    )
    trainer.fit(model, datamodule=dm)
    return {'update': model.update_history, 'loss': model.loss_history, 'val_loss': model.val_loss_history}


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason='lack of gpu devices')
@pytest.mark.parametrize("gradient_clip_val", [None, 0.1])  # 0.1 is chosen to only clip the first update
@pytest.mark.parametrize("with_lr_scheduler", [False, True])
def test_correctness(tmp_path, gradient_clip_val, with_lr_scheduler):
    def _merge_results(returns):
        results = [returns[i][0] for i in range(len(returns))]
        fullmaps = [returns[i][1] for i in range(len(returns))]
        weight_results = []
        grad_results = []
        for i in range(len(results[0])):
            weight_results.append(
                nnscaler.runtime.module.ParallelModule.merge_state_dicts(
                    fullmaps,
                    [result[i][1] for result in results]
                )[0]
            )
            grad_results.append(
                nnscaler.runtime.module.ParallelModule.merge_state_dicts(
                    fullmaps,
                    [result[i][0] for result in results]
                )[0]
            )
        return weight_results, grad_results

    def _assert_loss_equal(returns0, returns1, loss_idx0=-1, loss_idx1=-1, val_loss_idx0=-2, val_loss_idx1=-2):
        # TODO: val_loss check
        assert len(returns0) == len(returns1)
        for i in range(len(returns0)):
            assert returns0[i][loss_idx0] ==  returns1[i][loss_idx1]
            assert returns0[i][val_loss_idx0] ==  returns1[i][val_loss_idx1]

    # Test 16-mixed with and without gradient clipping
    # when gradient clipping is on, the following check will fail
    # TODO: fix the test when gradient clipping is on
    if not gradient_clip_val:
        ddp_results = launch_torchrun(2, correctnes_worker_ddp, tmp_path, gradient_clip_val, with_lr_scheduler, '16-mixed')

        nnscaler_returns = launch_torchrun(2, correctnes_worker_nnscaler, tmp_path, gradient_clip_val, with_lr_scheduler, '16-mixed', False, True)
        nnscaler_merged_weight_results_fp16, nnscaler_merged_grad_results_fp16 = _merge_results(nnscaler_returns)

        for i in range(len(ddp_results[0])):
            assert_close(nnscaler_merged_weight_results_fp16[i], ddp_results[0]['update'][i][1])
            assert_close(nnscaler_merged_grad_results_fp16[i], ddp_results[0]['update'][i][0])
            assert_equal(ddp_results[1]['update'][i], ddp_results[0]['update'][i])

    nnscaler_returns_ckpt = launch_torchrun(2, correctnes_worker_nnscaler_checkpoint, tmp_path, gradient_clip_val, with_lr_scheduler)
    nnscaler_merged_weight_results_ckpt, nnscaler_merged_grad_results_ckpt = _merge_results(nnscaler_returns_ckpt)

    nnscaler_returns = launch_torchrun(2, correctnes_worker_nnscaler, tmp_path, gradient_clip_val, with_lr_scheduler)
    nnscaler_merged_weight_results, nnscaler_merged_grad_results = _merge_results(nnscaler_returns)
    _assert_loss_equal(nnscaler_returns_ckpt, nnscaler_returns)

    nnscaler_returns = launch_torchrun(2, correctnes_worker_nnscaler, tmp_path, gradient_clip_val, with_lr_scheduler, '32-true', False, True)
    nnscaler_merged_weight_results_scaler, nnscaler_merged_grad_results_scaler = _merge_results(nnscaler_returns)
    _assert_loss_equal(nnscaler_returns_ckpt, nnscaler_returns)

    cli_returns = launch_torchrun(2, correctnes_worker_cli, tmp_path, gradient_clip_val, with_lr_scheduler)
    cli_merged_weight_results, cli_merged_grad_results = _merge_results(cli_returns)
    # remove leading 'm.' in names
    cli_merged_weight_results = [{k[2:]: v for k, v in x.items()} for x in cli_merged_weight_results]
    cli_merged_grad_results = [{k[2:]: v for k, v in x.items()} for x in cli_merged_grad_results]
    _assert_loss_equal(cli_returns, nnscaler_returns, val_loss_idx0=-3)
    assert cli_returns[0][-2] == cli_returns[1][-2]
    assert [(x+y)/2 for x, y in zip(cli_returns[0][-1],cli_returns[1][-1])] == cli_returns[0][-2]

    assert len(nnscaler_merged_weight_results) == len(nnscaler_merged_weight_results_ckpt)
    assert len(nnscaler_merged_weight_results) == len(nnscaler_merged_weight_results_scaler)
    assert len(nnscaler_merged_weight_results) == len(cli_merged_weight_results)

    assert len(nnscaler_merged_grad_results) == len(nnscaler_merged_grad_results_ckpt)
    assert len(nnscaler_merged_grad_results) == len(nnscaler_merged_grad_results_scaler)
    assert len(nnscaler_merged_grad_results) == len(cli_merged_grad_results)

    for i in range(len(nnscaler_merged_weight_results_scaler)):
        assert_equal(nnscaler_merged_weight_results[i], nnscaler_merged_weight_results_scaler[i])
        assert_equal(nnscaler_merged_weight_results[i], nnscaler_merged_weight_results_ckpt[i])
        assert_equal(nnscaler_merged_weight_results[i], cli_merged_weight_results[i])

        assert_equal(nnscaler_merged_grad_results[i], nnscaler_merged_grad_results_scaler[i])
        assert_equal(nnscaler_merged_grad_results[i], nnscaler_merged_grad_results_ckpt[i])
        assert_equal(nnscaler_merged_grad_results[i], cli_merged_grad_results[i])

    ddp_results = launch_torchrun(2, correctnes_worker_ddp, tmp_path, gradient_clip_val, with_lr_scheduler)
    if not gradient_clip_val:
        _assert_loss_equal(ddp_results, nnscaler_returns, loss_idx0='loss', val_loss_idx0='val_loss')
    for i in range(len(ddp_results[0])):
        if gradient_clip_val: # currently it is not exactly the same when gradient clipping is on
            assert_close(nnscaler_merged_weight_results[i], ddp_results[0]['update'][i][1])
            assert_close(nnscaler_merged_grad_results[i], ddp_results[0]['update'][i][0])
        else:
            assert_equal(nnscaler_merged_weight_results[i], ddp_results[0]['update'][i][1])
            assert_equal(nnscaler_merged_grad_results[i], ddp_results[0]['update'][i][0])
        assert_equal(ddp_results[1]['update'][i], ddp_results[0]['update'][i])

    if torch.cuda.device_count() >= 4:
        nnscaler_returns = launch_torchrun(4, correctnes_worker_nnscaler, tmp_path, gradient_clip_val, with_lr_scheduler, '32-true', True)
        nnscaler_merged_weight_results, nnscaler_merged_grad_results = _merge_results(nnscaler_returns)

        for i in range(len(ddp_results[0])):
            assert_close(nnscaler_merged_weight_results[i], ddp_results[0]['update'][i][1])
            assert_close(nnscaler_merged_grad_results[i], ddp_results[0]['update'][i][0])

        cli_returns = launch_torchrun(4, correctnes_worker_cli, tmp_path, gradient_clip_val, with_lr_scheduler, '32-true', True)
        cli_merged_weight_results, cli_merged_grad_results = _merge_results(cli_returns)
        # remove leading 'm.' in names
        cli_merged_weight_results = [{k[2:]: v for k, v in x.items()} for x in cli_merged_weight_results]
        cli_merged_grad_results = [{k[2:]: v for k, v in x.items()} for x in cli_merged_grad_results]

        for i in range(len(nnscaler_merged_weight_results)):
            assert_equal(nnscaler_merged_weight_results[i], cli_merged_weight_results[i])
            assert_equal(nnscaler_merged_grad_results[i], cli_merged_grad_results[i])
