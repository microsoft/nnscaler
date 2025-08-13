#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from pathlib import Path
import pytest

import nnscaler
from nnscaler.cli.trainer_args import load_type, ComputeConfig, OptionalComputeConfig, TrainerArgs


def test_load_type():
    assert load_type(int) == int
    assert load_type('int') == int
    assert load_type(int.to_bytes) == int.to_bytes
    assert load_type('int.to_bytes') == int.to_bytes
    assert load_type('nnscaler.cli.trainer_args.TrainerArgs') == nnscaler.cli.trainer_args.TrainerArgs
    assert load_type('nnscaler.cli.trainer_args.TrainerArgs.from_cli') == nnscaler.cli.trainer_args.TrainerArgs.from_cli

    with pytest.raises(RuntimeError):
        load_type('not_exist_name')

    with pytest.raises(RuntimeError):
        load_type('not_exist_namespace.not_exist_name')

    with pytest.raises(RuntimeError):
        load_type('nnscaler.not_exist_name')

    with pytest.raises(RuntimeError):
        load_type('nnscaler.cli.trainer_args.TrainerArgs.not_exist_name')


def test_compute_config_merge():
    cc = ComputeConfig(1, 2, constant_folding=True, use_end2end=True, use_zero=True)
    occ = OptionalComputeConfig(constant_folding=False, use_zero=False)
    rcc = occ.resolve(cc)
    assert rcc == ComputeConfig(1, 2, constant_folding=False, use_end2end=False)

    occ2 = OptionalComputeConfig(zero_ngroups=-1)
    with pytest.raises(ValueError):
        occ2.resolve(cc)


def test_arg_merge_resolve():
    config_path = str(Path(__file__).with_name('trainer_args.yaml').resolve())
    args = TrainerArgs.from_cli(['-f', config_path,
        '--vars.dim', '22',
        '--vars.hello', '$(compute_config.plan_ngpus)',
        '--global_batch_size!'
    ])
    assert args.vars['dim'] == 22
    assert args.dataset.train_args['dim'] == 22
    assert args.dataset.val_args['dim'] == 22
    assert args.vars['hello'] == args.compute_config.plan_ngpus


def gen_instance_name(stem):
    return f'instance_{stem}'


def test_dyn_str_config():
    config_path = str(Path(__file__).with_name('trainer_args.yaml').resolve())
    args = TrainerArgs.from_cli(['-f', config_path,
        '--instance_name.__type', 'tests.cli.test_train_args.gen_instance_name',
        '--instance_name.stem', 'p$(compute_config.plan_ngpus)',
        '--compute_config.plan_ngpus', '1',
        '--global_batch_size!',
    ])
    assert args.instance_name == 'instance_p1'
