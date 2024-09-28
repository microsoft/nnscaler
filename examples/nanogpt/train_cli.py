#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""
Run training with this command in this directory:
```
DETERMINISTIC=1 torchrun --standalone --nproc_per_node=1 \
     ../../nnscaler/cli/train.py -f train_cli_args.yaml
```
"""

import math
import os
from pathlib import Path
import pickle
import random
import sys
from typing import TYPE_CHECKING
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import lightning as L
import nnscaler
if TYPE_CHECKING:
    from nnscaler.cli.trainer import Trainer
    from nnscaler.cli.trainer_args import TrainerArgs

nanogpt_path = Path(__file__).absolute().with_name('nanoGPT')
sys.path.append(str(nanogpt_path))

from model import GPTConfig, GPT


def init_env(train_args: 'TrainerArgs'):
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    if os.environ.get('DETERMINISTIC') is not None:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)

    # be consistent with nanogpt settings
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn


def on_train_step_end(trainer: 'Trainer', outputs, batches, idx: int) -> None:
    if torch.distributed.get_rank() == 0:
        print(f'# train_loss {idx:03d}', outputs[0].item())


def on_val_step_end(trainer: 'Trainer', outputs, batches, idx: int) -> None:
    if torch.distributed.get_rank() == 0:
        print(f'# val_loss {idx:03d}', outputs[0].item())


# poor man's data loader
class NanoGptDataset(Dataset):
    def __init__(self, data_dir, split, block_size):
        self.split = split
        self.block_size = block_size
        self.data_dir = data_dir
        data = np.memmap(os.path.join(self.data_dir, f'{split}.bin'), dtype=np.uint16, mode='r')
        self.len = len(data) - self.block_size

    def __getitems__(self, indices):
        x, y = self.get_batch(self.split, indices)
        return (
            x.clone().detach(),  # theoretically unnecessary, for robustness
            y.clone().detach(),
        )

    def __len__(self):
        return self.len

    def get_batch(self, split, ix):
        # We recreate np.memmap every batch to avoid a memory leak, as per
        # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
        data = np.memmap(os.path.join(self.data_dir, f'{split}.bin'), dtype=np.uint16, mode='r')
        x = torch.stack([torch.from_numpy((data[i:i+self.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+self.block_size]).astype(np.int64)) for i in ix])
        return x, y


def _create_nano_gpt_model(
    init_from,
    *,
    n_layer,
    n_head,
    n_embd,
    block_size,
    bias,
    dropout,
    meta_path=None,
):
    # reset seeds to ensure the same initialization with nanogpt
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    # attempt to derive vocab_size from the dataset
    meta_vocab_size = None
    if meta_path and os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        meta_vocab_size = meta['vocab_size']
        print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

    # model init
    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                    bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
    if init_from == 'scratch':
        # init a new model from scratch
        print("Initializing a new model from scratch")
        # determine the vocab size we'll use for from-scratch training
        if meta_vocab_size is None:
            print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
        model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
    elif init_from == 'resume':
        print(f"Resuming training")
        # resume training from a checkpoint. (handled by lightning)
        if meta_vocab_size is None:
            print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
        model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
    elif init_from.startswith('gpt2'):
        print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
        # initialize from OpenAI GPT-2 weights
        override_args = dict(dropout=dropout)
        model = GPT.from_pretrained(init_from, override_args)
        # read off the created config params, so we can store them into checkpoint correctly
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = getattr(model.config, k)

    # crop down the model block size if desired, using model surgery
    if block_size < model.config.block_size:
        model.crop_block_size(block_size)
        model_args['block_size'] = block_size # so that the checkpoint will have the right value

    return model


class Model(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = _create_nano_gpt_model(*args, **kwargs)

    def forward(self, batch):
        _logits, loss = self.model(batch[0], batch[1])
        return loss


class Scheduler(torch.optim.lr_scheduler.LRScheduler):
    def __init__(self, optimizer, warmup_iters, learning_rate, lr_decay_iters, min_lr):
        self.it = 0  # must before super().__init__()
        self.warmup_iters = warmup_iters
        self.learning_rate = learning_rate
        self.lr_decay_iters = lr_decay_iters
        self.min_lr = min_lr
        super().__init__(optimizer)

    def get_lr(self):
        lr = self._get_lr(self.it)
        self.it += 1
        return [lr for _ in self.optimizer.param_groups]

    # learning rate decay scheduler (cosine with warmup)
    def _get_lr(self, it):
        # 1) linear warmup for warmup_iters steps
        if it < self.warmup_iters:
            return self.learning_rate * it / self.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > self.lr_decay_iters:
            return self.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - self.warmup_iters) / (self.lr_decay_iters -  self.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return self.min_lr + coeff * ( self.learning_rate - self.min_lr)
