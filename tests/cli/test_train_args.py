#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import pytest

import nnscaler
from nnscaler.cli.trainer_args import load_type


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
