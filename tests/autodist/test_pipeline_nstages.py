#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import os
import pytest
import torch
from nnscaler.cli.trainer import Trainer
from nnscaler.cli.trainer_args import *
from tests.launch_torchrun import launch_torchrun


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason='need 2 gpus')
def test_1_stage(tmp_path):
    launch_torchrun(1, _compile_worker, tmp_path, 1)
    # for TP, the scripts should be identical (except tensor names)
    lines0 = _count_gencode_lines(tmp_path, 0)
    lines1 = _count_gencode_lines(tmp_path, 1)
    assert lines0 == lines1


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason='need 2 gpus')
def test_2_stages(tmp_path):
    launch_torchrun(1, _compile_worker, tmp_path, 2)
    # for PP, since we have 3 linears, the scripts should be different
    lines0 = _count_gencode_lines(tmp_path, 0)
    lines1 = _count_gencode_lines(tmp_path, 1)
    assert lines0 != lines1


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason='need 2 gpus')
def test_auto_stages(tmp_path):
    launch_torchrun(1, _compile_worker, tmp_path, 'auto')
    # just check it does not throw
    # because both results are possible theoretically


def _compile_worker(tmp_path, nstages):
    _compile(tmp_path, nstages)


def _compile(tmp_path, nstages):
    trainer_args = TrainerArgs(
        compute_config=ComputeConfig(
            plan_ngpus=2,
            runtime_ngpus=2,
            use_end2end=True,
            pas_config={
                'pipeline_pivots': 'Linear',
                'pipeline_nstages': nstages,
                'max_pipeline_bubble_ratio': 0.99,  # force autodist to accept unbalanced stages
            },
        ),
        gen_reuse='override',
        gen_savedir=tmp_path/'src',
        run_mode='compile',
        model=ModelConfig(type=Model),
        optimizer=OptimizerConfig(type=torch.optim.AdamW),
        dataset=DatasetConfig(type=RandomDataset, train_args={'length': 100}),
        max_train_steps=1,
    )
    trainer = Trainer(train_args=trainer_args)
    trainer.run()


def _count_gencode_lines(tmp_path, index):
    script = 'tests/autodist/test_pipeline_nstages'
    path = f'_parallel_modules/{script}/Model/_/gencode{index}.py'
    text = Path(tmp_path, 'src', path).read_text()
    return text.count('\n')


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)
        self.linear2 = torch.nn.Linear(10, 10)
        self.linear3 = torch.nn.Linear(10, 10)

    def forward(self, data):
        x = data['x']
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return torch.nn.functional.cross_entropy(x, data['y'])


class RandomDataset:
    def __init__(self, length):
        self.length = length

    def __getitem__(self, i):
        return {
            'x': torch.rand(10),
            'y': torch.randint(10, tuple()),
        }

    def __len__(self):
        return self.length
