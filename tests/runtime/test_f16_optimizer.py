#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from pathlib import Path
import shutil

import torch
import pytest
import torch.distributed

from nnscaler.cli.trainer import Trainer
from tests.parallel_module.common import assert_close
from ..launch_torchrun import launch_torchrun


def trainer_worker(save_dir):
    save_dir = Path(save_dir)
    config_path = str(Path(__file__).with_name('test_f16_optimizer_trainer_args.yaml').resolve())
    gen_savedir = save_dir / 'gen'

    # train with normal mixed optimizer
    ckpt0_savedir = save_dir / 'ckpt0'
    trainer = Trainer([
        '-f', config_path,
        '--optimizer.type', 'nnscaler.runtime.f16_optimizer.MixedPrecisionAdam',
        '--gen_savedir', str(gen_savedir),
        '--checkpoint.save_dir', str(ckpt0_savedir),
    ])
    trainer.run()
    torch.distributed.barrier()

    # train with normal optimizer
    ckpt1_savedir = save_dir / 'ckpt1'
    trainer = Trainer([
        '-f', config_path,
        '--optimizer.type', 'torch.optim.Adam',
        '--gen_savedir', str(gen_savedir),
        '--checkpoint.save_dir', str(ckpt1_savedir),
    ])
    trainer.run()
    torch.distributed.barrier()

    if torch.distributed.get_rank() == 0:
        for i in range(2):
            x = torch.load(ckpt0_savedir / 'last' / f'{i}.ckpt', weights_only=False)
            y = torch.load(ckpt1_savedir / 'last' / f'{i}.ckpt', weights_only=False)
            # actually they are not close
            # assert_close(x['model'], y['model'])
            # assert_close(x['optimizer'], y['optimizer'])
            # assert_close(x['lr_scheduler'], y['lr_scheduler'])
            assert x['optimizer']['state'][0]['exp_avg'].dtype == torch.float32
            assert 'fp32_params' in x['optimizer']['state'][0]
            assert x['optimizer']['state'][0]['fp32_params'].dtype == torch.float32
            assert y['optimizer']['state'][0]['exp_avg'].dtype == torch.bfloat16
            assert 'fp32_params' not in y['optimizer']['state'][0]


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason='lack of gpu devices')
def test_bf16(tmp_path):
    launch_torchrun(2, trainer_worker, tmp_path)
