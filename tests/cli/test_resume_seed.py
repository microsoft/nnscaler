#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import os
import pytest
import torch
from nnscaler.cli.trainer import Trainer
from nnscaler.cli.trainer_args import *
from tests.launch_torchrun import launch_torchrun


@pytest.mark.skipif(not torch.cuda.is_available(), reason='no gpu')
def test_resume_seed(tmp_path):
    launch_torchrun(1, resume_seed_worker, tmp_path)


def resume_seed_worker(tmp_path):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)

    # compile separately because run multiple trainers in one process will confuse `gen_reuse`
    _compile(tmp_path)
    _test_resume_seed(tmp_path, steps_per_epoch=100, max_steps=20, resume_at=10)
    _test_resume_seed(tmp_path, steps_per_epoch=5, max_steps=20, resume_at=10)


def _test_resume_seed(tmp_path, steps_per_epoch, max_steps, resume_at):
    # no resume
    model_1 = _train(tmp_path, steps_per_epoch, max_train_steps=max_steps, resume_from=None)
    weight_1 = next(model_1.parameters()).data

    # resume
    _train(tmp_path, steps_per_epoch, max_train_steps=resume_at, resume_from=None)
    model_2 = _train(tmp_path, steps_per_epoch, max_train_steps=max_steps, resume_from='last')
    weight_2 = next(model_2.parameters()).data

    assert torch.equal(weight_1, weight_2)

    ## resume without resuming seeds
    _train(tmp_path, steps_per_epoch, max_train_steps=resume_at, resume_from=None)
    _remove_rng_states(tmp_path)
    model_3 = _train(tmp_path, steps_per_epoch, max_train_steps=max_steps, resume_from='last')
    weight_3 = next(model_3.parameters()).data

    assert not torch.equal(weight_1, weight_3)


def _compile(tmp_path):
    trainer_args = TrainerArgs(
        compute_config=ComputeConfig(plan_ngpus=1, runtime_ngpus=1, use_end2end=True),
        gen_reuse='override',
        gen_savedir=tmp_path/'src',
        run_mode='compile',
        model=ModelConfig(type=Model),
        optimizer=OptimizerConfig(type=torch.optim.AdamW),
        dataset=DatasetConfig(type=RandomDataset, train_args={'length': 100}),
        max_train_steps=1,
        enable_progress_bar=False,
        seed=0,
    )
    trainer = Trainer(train_args=trainer_args)
    trainer.run()


def _train(tmp_path, steps_per_epoch, max_train_steps, resume_from):
    trainer_args = TrainerArgs(
        gen_savedir=tmp_path/'src',
        compute_config=ComputeConfig(plan_ngpus=1, runtime_ngpus=1, use_end2end=True),
        model=ModelConfig(type=Model),
        optimizer=OptimizerConfig(type=torch.optim.AdamW),
        dataset=DatasetConfig(type=RandomDataset, train_args={'length': steps_per_epoch}),
        checkpoint=CheckpointConfig(resume_from=ResumeOptions(checkpoint=resume_from), save_dir=tmp_path/'checkpoints'),
        max_train_steps=max_train_steps,
        enable_progress_bar=False,
        seed=0,
    )
    trainer = Trainer(train_args=trainer_args)
    trainer.run()
    return trainer.model


def _remove_rng_states(tmp_path):
    ckpt_path = tmp_path / 'checkpoints/last/0.ckpt'
    ckpt = torch.load(ckpt_path, weights_only=False)
    ckpt['rng_states'] = None
    torch.save(ckpt, ckpt_path)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(100, 10)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, data):
        x = data['x']
        x = self.linear(x)
        x = self.dropout(x)
        return torch.nn.functional.cross_entropy(x, data['y'])


class RandomDataset:
    def __init__(self, length):
        self.length = length

    def __getitem__(self, i):
        return {
            'x': torch.rand(100),
            'y': torch.randint(10, tuple()),
        }

    def __len__(self):
        return self.length
