#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from pathlib import Path
import shutil

import torch
import pytest
import torch.distributed

from nnscaler.cli.trainer import Trainer
from nnscaler.cli.trainer_args import AggregatedOutputs, TrainerArgs
from tests.parallel_module.common import assert_equal
from tests.utils import replace_all_device_with
from ..launch_torchrun import launch_torchrun


def trainer_logging_worker(save_dir):
    save_dir = Path(save_dir)
    config_path = str(Path(__file__).with_name('trainer_args.yaml').resolve())
    gen_savedir = save_dir / 'gen'
    log_savedir = save_dir / 'log'
    tb_log_savedir = log_savedir / 'tensorboard'
    wandb_log_savedir = log_savedir / 'wandb'
    # train 4 epcho in one time
    trainer = Trainer([
        '-f', config_path,
        '--max_epochs', '2',
        '--gen_savedir', str(gen_savedir),
        '--compute_config.plan_ngpus', '2',
        '--compute_config.runtime_ngpus', '4',
        '--checkpoint.no_save', 'true',
        '--log.0.type', 'nnscaler.cli.loggers.TensorBoardLogger',
        '--log.0.args.name', 'test-cli',
        '--log.0.args.root_dir', str(tb_log_savedir),
        '--log.1.type', 'nnscaler.cli.loggers.WandbLogger',
        '--log.1.args.name', 'test-cli',
        '--log.1.args.dir', str(wandb_log_savedir),
        '--log.1.args.project', 'nnscaler',
        '--log.1.args.mode', 'offline',
    ])
    trainer.run()

    torch.distributed.barrier()

    if torch.distributed.get_rank() == 0:
        assert (tb_log_savedir / 'test-cli').exists()
        for tag in ['val', 'train']:
            tfevents = list((tb_log_savedir / 'test-cli' / tag).glob('events.out.tfevents.*'))
            assert len(tfevents) == 1
            assert tfevents[0].stat().st_size > 1000

        assert (wandb_log_savedir / 'wandb').exists()
        wandb_offline_dir = list((wandb_log_savedir / 'wandb').glob('offline-run-*'))
        assert len(wandb_offline_dir) == 1
        wandb_run_db = list(wandb_offline_dir[0].glob('run-*.wandb'))
        assert len(wandb_run_db) == 1
        assert wandb_run_db[0].stat().st_size > 1000


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4, reason='lack of gpu devices')
def test_trainer_logging(tmp_path):
    launch_torchrun(4, trainer_logging_worker, tmp_path)


@replace_all_device_with('cpu')
def test_trainer_compile_worker(tmp_path):
    save_dir = Path(tmp_path)
    config_path = str(Path(__file__).with_name('trainer_args.yaml').resolve())
    gen_savedir = save_dir / 'gen'
    # compile only
    trainer = Trainer([
        '-f', config_path,
        '--max_epochs', '2',
        '--gen_savedir', str(gen_savedir),
        '--compute_config.plan_ngpus', '2',
        '--compute_config.runtime_ngpus', '4',
        '--checkpoint.no_save', 'true',
        '--run_mode', 'compile',
        '--broadcast_strategy', 'none',
    ])
    trainer.run()

    assert set([f.name for f in gen_savedir.glob('**/*.py')]) == set(['gencode0.py', 'gencode1.py', 'gencode2.py', 'gencode3.py'])
    shutil.rmtree(gen_savedir)


def trainer_resume_worker(save_dir, save_type, bf16):
    save_dir = Path(save_dir)
    config_path = str(Path(__file__).with_name('trainer_args.yaml').resolve())
    gen_savedir = save_dir / 'gen'
    ckpt_savedir = save_dir / 'ckpt'
    # train 4 epcho in one time
    optimizer_type = 'nnscaler.runtime.f16_optimizer.MixedPrecisionAdam' \
        if bf16 == 'Mixed' \
        else 'torch.optim.Adam'
    use_zero = save_type == 'sharded'

    trainer = Trainer([
        '-f', config_path,
        '--precision', 'bf16' if bf16 else 'none',
        '--optimizer.type', optimizer_type,
        '--max_epochs', '4',
        '--enable_progress_bar', 'false',
        '--gen_savedir', str(gen_savedir),
        '--compute_config.plan_ngpus', '2',
        '--compute_config.runtime_ngpus', '4',
        '--compute_config.use_zero', str(use_zero),
        '--checkpoint.save_type', save_type,
        '--checkpoint.save_dir', str(ckpt_savedir),
        '--checkpoint.resume_from', 'last',
        '--checkpoint.keep_last_n_checkpoints', '30',
    ])
    trainer.run()
    ckpt_files = set(ckpt_savedir.glob('**/*.ckpt'))
    assert len(ckpt_files)/4 == min(30, trainer.total_train_steps_per_epoch * 4) + 2 # 2 for best/last

    torch.distributed.barrier()
    # train 4 epcho two times (resume from last)
    ckpt0_savedir = save_dir / 'ckpt0'
    # first two epochs
    trainer = Trainer([
        '-f', config_path,
        '--precision', 'bf16' if bf16 else 'none',
        '--optimizer.type', optimizer_type,
        '--max_epochs', '2',
        '--enable_progress_bar', 'false',
        '--gen_savedir', str(gen_savedir),
        '--compute_config.plan_ngpus', '2',
        '--compute_config.runtime_ngpus', '4',
        '--compute_config.use_zero', str(use_zero),
        '--checkpoint.save_type', save_type,
        '--checkpoint.save_dir', str(ckpt0_savedir),
        '--checkpoint.resume_from', 'last',
        '--checkpoint.keep_last_n_checkpoints', '30',
    ])
    trainer.run()
    ckpt0_files0 = {f: f.stat().st_mtime_ns for f in ckpt0_savedir.glob('**/*.ckpt')}
    assert len(ckpt0_files0)/4 == min(30, trainer.total_train_steps_per_epoch * 2) + 2 # 2 for best/last

    # resume from last without update max_epochs
    trainer = Trainer([
        '-f', config_path,
        '--precision', 'bf16' if bf16 else 'none',
        '--optimizer.type', optimizer_type,
        '--max_epochs', '2',
        '--enable_progress_bar', 'false',
        '--gen_savedir', str(gen_savedir),
        '--compute_config.plan_ngpus', '2',
        '--compute_config.runtime_ngpus', '4',
        '--compute_config.use_zero', str(use_zero),
        '--checkpoint.save_type', save_type,
        '--checkpoint.save_dir', str(ckpt0_savedir),
        '--checkpoint.resume_from', 'last',
        '--checkpoint.keep_last_n_checkpoints', '30',
    ])
    trainer.run()
    ckpt0_files0_x = {f: f.stat().st_mtime_ns for f in ckpt0_savedir.glob('**/*.ckpt')}
    # nothing should be updated in this case.
    assert ckpt0_files0 == ckpt0_files0_x

    # create merged checkpoint
    ckpt1_savedir = save_dir / 'ckpt1'
    ckpt1_savedir.mkdir(parents=True, exist_ok=True)
    if trainer.rank == 0:
        Trainer.merge_checkpoint(list((ckpt0_savedir / 'last').glob('*.ckpt')), ckpt1_savedir / 'merged.pt')

    torch.distributed.barrier()
    # continue with the last two epochs (resume for sharded/deduped checkpoint)
    trainer = Trainer([
        '-f', config_path,
        '--precision', 'bf16' if bf16 else 'none',
        '--optimizer.type', optimizer_type,
        '--max_epochs', '4',
        '--enable_progress_bar', 'false',
        '--gen_savedir', str(gen_savedir),
        '--compute_config.plan_ngpus', '2',
        '--compute_config.runtime_ngpus', '4',
        '--compute_config.use_zero', str(use_zero),
        '--checkpoint.save_type', save_type,
        '--checkpoint.save_dir', str(ckpt0_savedir),
        '--checkpoint.resume_from', 'last',
        '--checkpoint.keep_last_n_checkpoints', '30',
    ])
    trainer.run()
    left_files = {
        f: f.stat().st_mtime_ns for f in ckpt0_files0.keys()
        if f.exists() and f.parent.name not in ['last', 'best']
    }
    assert left_files  # some checkpoints are removed
    for f, s in left_files.items(): # make sure the old checkpoints are not overwritten
        assert ckpt0_files0[f] == s

    ckpt0_files1 = set(ckpt0_savedir.glob('**/*.ckpt'))
    assert len(ckpt0_files1)/4 == min(30, trainer.total_train_steps_per_epoch * 4) + 2 # 2 for best/last

    torch.distributed.barrier()

    # continue with the last two epochs (resume for merged)
    trainer = Trainer([
        '-f', config_path,
        '--precision', 'bf16' if bf16 else 'none',
        '--optimizer.type', optimizer_type,
        '--max_epochs', '4',
        '--gen_savedir', str(gen_savedir),
        '--compute_config.plan_ngpus', '2',
        '--compute_config.runtime_ngpus', '4',
        '--compute_config.use_zero', str(use_zero),
        '--checkpoint.save_type', save_type,
        '--checkpoint.save_dir', str(ckpt1_savedir),
        '--checkpoint.resume_from', str(ckpt1_savedir / 'merged.pt'),
        '--checkpoint.keep_last_n_checkpoints', '30',
    ])
    trainer.run()
    left_files = {
        f: f.stat().st_mtime_ns for f in ckpt0_files0.keys()
        if f.exists() and f.parent.name not in ['last', 'best']
    }
    assert left_files  # some checkpoints are removed
    for f, s in left_files.items(): # make sure the old checkpoints are not overwritten
        assert ckpt0_files0[f] == s

    ckpt0_files1 = set(ckpt0_savedir.glob('**/*.ckpt'))
    assert len(ckpt0_files1)/4 == min(30, trainer.total_train_steps_per_epoch * 4) + 2 # 2 for best/last

    torch.distributed.barrier()

    if torch.distributed.get_rank() == 0:
        assert {f.parent.name for f in ckpt_files} == {f.parent.name for f in ckpt0_files1}
        for i in range(4):
            x = torch.load(ckpt_savedir / 'last' / f'{i}.ckpt')
            y = torch.load(ckpt0_savedir / 'last' / f'{i}.ckpt')
            z = torch.load(ckpt1_savedir / 'last' / f'{i}.ckpt')
            assert_equal(x['model'], y['model'])
            assert_equal(x['optimizer'], y['optimizer'])
            assert_equal(x['lr_scheduler'], y['lr_scheduler'])
            assert_equal(x['model'], z['model'])
            assert_equal(x['optimizer'], z['optimizer'])
            assert_equal(x['lr_scheduler'], z['lr_scheduler'])

        if save_type == 'deduped':
            assert (ckpt_savedir / 'last/0.ckpt').stat().st_size > (ckpt_savedir / 'last/2.ckpt').stat().st_size
            assert (ckpt_savedir / 'last/1.ckpt').stat().st_size > (ckpt_savedir / 'last/3.ckpt').stat().st_size
        else:
            assert (ckpt_savedir / 'last/0.ckpt').stat().st_size == (ckpt_savedir / 'last/2.ckpt').stat().st_size
            assert (ckpt_savedir / 'last/1.ckpt').stat().st_size == (ckpt_savedir / 'last/3.ckpt').stat().st_size


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4, reason='lack of gpu devices')
@pytest.mark.parametrize('save_type', ['sharded', 'deduped'])
@pytest.mark.parametrize('bf16', [True, False, 'Mixed'])
def test_trainer_resume(tmp_path, save_type, bf16):
    launch_torchrun(4, trainer_resume_worker, tmp_path, save_type, bf16)


def trainer_last_checkpoint_worker(save_dir):
    save_dir = Path(save_dir)
    config_path = str(Path(__file__).with_name('trainer_args.yaml').resolve())
    gen_savedir = save_dir / 'gen'
    ckpt_savedir = save_dir / 'ckpt'

    for i in range (100): # make a lot of fake checkpoints
        (ckpt_savedir / f'0000-{i*15:04d}').mkdir(parents=True, exist_ok=True)

    trainer = Trainer([
        '-f', config_path,
        '--max_epochs', '1',
        '--global_batch_size', '4',  # mini_batch_size=2, update_freq=2
        '--gen_savedir', str(gen_savedir),
        '--compute_config.plan_ngpus', '1',
        '--compute_config.runtime_ngpus', '1',
        '--val_every_n_train_steps', '1',
        '--checkpoint.every_n_train_steps', '15',
        '--checkpoint.save_dir', str(ckpt_savedir),
    ])
    trainer.run()

    torch.distributed.barrier()
    # make sure the last checkpoint is saved.
    assert (ckpt_savedir / '0000-0025' / f'{trainer.rank}.ckpt').exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='lack of gpu devices')
def test_trainer_last_checkpoint(tmp_path):
    launch_torchrun(1, trainer_last_checkpoint_worker, tmp_path)


_train_losses = []
_val_losses = []


def after_aggregate_train_step_outputs(trainer: 'Trainer', aggregated_outputs: 'AggregatedOutputs', train_loss: float, idx: int) -> None:
    _train_losses.append(train_loss)


def after_aggregate_val_step_outputs(trainer: 'Trainer', aggregated_outputs: 'AggregatedOutputs', val_loss: float, idx: int) -> None:
    _val_losses.append(val_loss)


def trainer_loss_reduction_worker(save_dir):
    save_dir = Path(save_dir)
    config_path = str(Path(__file__).with_name('trainer_args.yaml').resolve())
    gen_savedir = save_dir / 'gen'

    _train_losses.clear()
    _val_losses.clear()
    trainer = Trainer([
        '-f', config_path,
        '--enable_progress_bar', 'false',
        '--max_epochs', '1',
        '--global_batch_size', '4',  # mini_batch_size=2, update_freq=2
        '--gen_savedir', str(gen_savedir),
        '--compute_config.plan_ngpus', '1',
        '--compute_config.runtime_ngpus', '1',
        '--val_every_n_train_steps', '1',
        '--checkpoint.no_save', 'true',
        '--optimizer.loss_reduction', 'mean',
        '--hook.after_aggregate_train_step_outputs',
            'tests.cli.test_trainer.after_aggregate_train_step_outputs',
        '--hook.after_aggregate_val_step_outputs',
            'tests.cli.test_trainer.after_aggregate_val_step_outputs',
    ])
    trainer.run()

    # get a copy
    train_loss_mean = _train_losses[:]
    val_loss_mean = _val_losses[:]

    torch.distributed.barrier()

    _train_losses.clear()
    _val_losses.clear()

    trainer = Trainer([
        '-f', config_path,
        '--enable_progress_bar', 'false',
        '--max_epochs', '1',
        '--global_batch_size', '4',  # mini_batch_size=2, update_freq=2
        '--gen_savedir', str(gen_savedir),
        '--compute_config.plan_ngpus', '1',
        '--compute_config.runtime_ngpus', '1',
        '--val_every_n_train_steps', '1',
        '--checkpoint.no_save', 'true',
        '--optimizer.loss_reduction', 'sum',
        '--hook.after_aggregate_train_step_outputs',
            'tests.cli.test_trainer.after_aggregate_train_step_outputs',
        '--hook.after_aggregate_val_step_outputs',
            'tests.cli.test_trainer.after_aggregate_val_step_outputs',
    ])
    trainer.run()
    torch.distributed.barrier()

    assert len(train_loss_mean) == len(_train_losses)
    assert len(val_loss_mean) == len(_val_losses)
    for i in range(len(train_loss_mean)):
        assert train_loss_mean[i] == _train_losses[i] / 2  # 2 is update freq

    for i in range(len(val_loss_mean)):
        assert val_loss_mean[i] == _val_losses[i]

    torch.distributed.barrier()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='lack of gpu devices')
def test_trainer_loss_reduction(tmp_path):
    launch_torchrun(1, trainer_loss_reduction_worker, tmp_path)


_before_step_grads = None
def before_gnorm_clip(trainer: 'Trainer') -> None:
    global _before_step_grads
    _before_step_grads = {i: g.grad.clone() for i, g in enumerate(trainer.optimizer.param_groups[0]['params'])}


def aggregate_outputs(loss_outputs, sync_group) -> 'AggregatedOutputs':
    # loss is the first element of the output (or the only element)
    losses = [
        loss if isinstance(loss, torch.Tensor)
        else loss[0]
        for loss in loss_outputs
    ]
    loss_sum = torch.sum(torch.stack(losses), dtype=torch.float64)
    torch.distributed.all_reduce(loss_sum, group=sync_group)
    num_batches = torch.tensor(len(losses), device=torch.cuda.current_device())
    torch.distributed.all_reduce(num_batches, group=sync_group)
    num_tokens = num_batches * 2  # fake value

    return AggregatedOutputs(
        loss_sum = loss_sum.item(),
        num_batches=num_batches.item(),
        num_tokens=num_tokens.item(),
    )


def trainer_per_token_worker(save_dir):
    save_dir = Path(save_dir)
    config_path = str(Path(__file__).with_name('trainer_args.yaml').resolve())
    gen_savedir = save_dir / 'gen'

    _train_losses.clear()
    _val_losses.clear()
    trainer = Trainer([
        '-f', config_path,
        '--enable_progress_bar', 'false',
        '--max_epochs', '1',
        '--global_batch_size', '8',
        '--grad_accumulation_steps', '2',
        '--gen_savedir', str(gen_savedir),
        '--max_train_steps', '1',
        '--compute_config.plan_ngpus', '1',
        '--compute_config.runtime_ngpus', '2',
        '--val_every_n_train_steps', '1',
        '--checkpoint.no_save', 'true',
        '--optimizer.grad_reduction', 'mean',
        '--optimizer.aggregate_outputs_fn', 'tests.cli.test_trainer.aggregate_outputs',
        '--hook.before_gnorm_clip',
            'tests.cli.test_trainer.before_gnorm_clip',
    ])
    trainer.run()

    # get a copy
    grads = _before_step_grads

    torch.distributed.barrier()

    trainer = Trainer([
        '-f', config_path,
        '--enable_progress_bar', 'false',
        '--max_epochs', '1',
        '--global_batch_size', '8',
        '--grad_accumulation_steps', '2',
        '--gen_savedir', str(gen_savedir),
        '--max_train_steps', '1',
        '--compute_config.plan_ngpus', '1',
        '--compute_config.runtime_ngpus', '2',
        '--val_every_n_train_steps', '1',
        '--checkpoint.no_save', 'true',
        '--optimizer.grad_reduction', 'per-token-mean',
        '--optimizer.aggregate_outputs_fn', 'tests.cli.test_trainer.aggregate_outputs',
        '--hook.before_gnorm_clip',
            'tests.cli.test_trainer.before_gnorm_clip',
    ])
    trainer.run()

    torch.distributed.barrier()

    assert set(grads.keys()) == set(_before_step_grads.keys())
    for n, p in grads.items():
        assert torch.equal(p / 2, _before_step_grads[n])

    torch.distributed.barrier()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='lack of gpu devices')
def test_trainer_per_token(tmp_path):
    launch_torchrun(2, trainer_per_token_worker, tmp_path)


def test_dataset_empty_train_args():
    def _empty_train_args():
        from .common import SimpleDataset
        return SimpleDataset(10)

    config_path = str(Path(__file__).with_name('trainer_args.yaml').resolve())
    train_args = TrainerArgs.from_cli([
        '-f', config_path,
        '--compute_config.plan_ngpus', '1',
        '--compute_config.runtime_ngpus', '2',
    ])
    train_args.dataset.type = _empty_train_args
    train_args.dataset.train_args = {}
    train_args.dataset.val_args = {}
    assert train_args.create_dataset() is not None
    assert train_args.create_dataset('val') is None


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4 or torch.cuda.device_count() >= 8, reason='lack of gpu devices')
@pytest.mark.parametrize('use_bf16', [True, False])
@pytest.mark.parametrize('zero_ngroups', [None, '1', '2'])
def test_trainer_grad_sync_check_4gpu(tmp_path, use_bf16, zero_ngroups):
    launch_torchrun(4, trainer_grad_sync_check, tmp_path, use_bf16, zero_ngroups, '4')


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 8, reason='lack of gpu devices')
@pytest.mark.parametrize('use_bf16', [True, False])
@pytest.mark.parametrize('zero_ngroups', [None, '1', '2', '4'])
def test_trainer_grad_sync_check_8gpu(tmp_path, use_bf16, zero_ngroups):
    launch_torchrun(8, trainer_grad_sync_check, tmp_path, use_bf16, zero_ngroups, '8')


def trainer_grad_sync_check(save_dir, use_bf16, zero_ngroups, runtime_ngpus):
    save_dir = Path(save_dir)
    config_path = str(Path(__file__).with_name('trainer_args.yaml').resolve())
    gen_savedir = save_dir / 'gen'
    ckpt_savedir = save_dir / 'ckpt'
    optimizer_type = 'torch.optim.Adam'
    use_zero = False if zero_ngroups is None else True
    zero_ngroups = '1' if zero_ngroups is None else zero_ngroups

    trainer = Trainer([
        '-f', config_path,
        '--precision', 'bf16' if use_bf16 else 'none',
        '--optimizer.type', optimizer_type,
        '--max_epochs', '1',
        '--enable_progress_bar', 'false',
        '--gen_savedir', str(gen_savedir),
        '--compute_config.plan_ngpus', '2',
        '--compute_config.runtime_ngpus', runtime_ngpus,
        '--compute_config.use_zero', str(use_zero),
        '--compute_config.zero_ngroups', zero_ngroups,
        '--checkpoint.save_dir', str(ckpt_savedir),
        '--debug.check_gradient_sync_cross_devices', 'true',
    ])
    trainer.run()
    torch.distributed.barrier()
