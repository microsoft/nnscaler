#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import math
import os
from pathlib import Path
import pickle
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import lightning as L
import nnscaler.integration.lightning.pytorch

nanogpt_path = Path(__file__).absolute().with_name('nanoGPT')
sys.path.append(str(nanogpt_path))

from model import GPTConfig, GPT

torch.manual_seed(0)

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# system
dtype = 'float32'

# nnscaler
use_nnscaler = True
plan_ngpus = 1
runtime_ngpus = -1  # use -1 for WORLD_SIZE since nanoGPT's argparse require it to have static type

deterministic = os.environ.get('DETERMINISTIC') is not None

# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(nanogpt_path / 'configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

if deterministic:
    # seed is set at the top of the file
    dropout = 0.0  # must set before model init
    grad_clip = 0.0
    torch.use_deterministic_algorithms(True)  # NOTE: requires env CUBLAS_WORKSPACE_CONFIG=":4096:8"
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

# various inits, derived attributes, I/O setup

torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]

# poor man's data loader
data_dir = os.path.join(nanogpt_path, 'data', dataset)
def get_batch(split, ix):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
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

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

## Lightning Wrappers ##

class NanoGptDataset(Dataset):
    def __init__(self, split):
        self.split = split
        data = np.memmap(os.path.join(data_dir, f'{split}.bin'), dtype=np.uint16, mode='r')
        self.len = len(data) - block_size

    def __getitems__(self, indices):
        x, y = get_batch(self.split, indices)
        return (
            x.clone().detach(),  # theoretically unnecessary, for robustness
            y.clone().detach(),
        )

    def __len__(self):
        return self.len

class Scheduler(torch.optim.lr_scheduler.LRScheduler):
    def __init__(self, optimizer):
        self.it = 0  # must before super().__init__()
        super().__init__(optimizer)

    def get_lr(self):
        lr = get_lr(self.it)
        self.it += 1
        return [lr for _ in self.optimizer.param_groups]

class LitModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = model
        self.dummy_forward_args_fn = lambda batch: {'x': batch[0], 'y': batch[1]}

    def forward(self, x, y):
        _logits, loss = self.model(x, y)
        return loss

    def step(self, batch, batch_idx, log_name):
        x, y = batch
        loss = self(x, y)
        self.log(log_name, loss, logger=True, on_epoch=True, sync_dist=True)
        return {'loss': loss}

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, log_name='train_loss')

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, log_name='val_loss')

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, log_name='test_loss')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate, betas=(beta1, beta2), fused=True)
        scheduler = Scheduler(optimizer)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            },
        }

## Training Loop ##

def main():
    global runtime_ngpus

    precision = {'float32': '32-true', 'bfloat16': 'bf16-true', 'float16': '16-true'}[dtype]

    if use_nnscaler:
        if not os.getenv('WORLD_SIZE'):
            print('[ERROR] nnScaler must be launched with torchrun')
            print('Example usage for single GPU:')
            print('    torchrun --standalone --nproc_per_node=1 train.py nanoGPT/config/train_shakespeare_char.py')
            exit(1)

        if runtime_ngpus == -1:
            runtime_ngpus = int(os.getenv('WORLD_SIZE'))

        compute_config = nnscaler.ComputeConfig(plan_ngpus, runtime_ngpus, constant_folding=True)
        strategy = nnscaler.integration.lightning.pytorch.NnScalerStrategy(
            compute_config=compute_config,
            pas_policy='autodist',
            reuse='override',
        )
        plugins = [nnscaler.integration.lightning.pytorch.NnScalerPrecision(precision)]
        precision = None

    else:
        strategy = 'ddp'
        plugins = None

    lightning_model = LitModel()

    trainer = L.Trainer(
        strategy=strategy,
        precision=precision,
        max_steps=max_iters,
        limit_train_batches=eval_interval,
        limit_val_batches=eval_iters,
        limit_test_batches=eval_iters,
        accumulate_grad_batches=gradient_accumulation_steps,
        gradient_clip_val=(grad_clip if grad_clip != 0.0 else None),
        plugins=plugins,
    )

    trainer.fit(
        lightning_model,
        DataLoader(NanoGptDataset('train'), batch_size=batch_size, shuffle=True),
        DataLoader(NanoGptDataset('val'), batch_size=batch_size, shuffle=True),
        ckpt_path=('last' if init_from == 'resume' else None),
    )

if __name__ == '__main__':
    main()
