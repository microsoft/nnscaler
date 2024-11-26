#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

# ref: https://medium.com/@supersjgk/fine-tuning-vision-transformer-with-hugging-face-and-pytorch-df19839d5396

"""
Run example:

First go to `examples/vit` directory, Then

1. Use transformer.train() to train the model
    a. `python examples/vit/vit_cli.py`: will use `dp` to train the model.
        It will utilize all your GPUs in current node.
        You can specify the GPUs with `CUDA_VISIBLE_DEVICES` env variable.
    b. `torchrun --nproc_per_node=<gpus> --nnodes=<nodes> examples/vit/vit_cli.py`: will use `ddp` to train the model.
2. Use nnscaler to train the model
    `torchrun --nproc_per_node=<gpus> --nnodes=<nodes> $(which nnscaler-train) -f train_cli_args.yaml`

Here in order to be consistent with transformers,
we use dataloader/scheduler from transformers.
See `accelerator_dataloader_fn` and `scheduler_fn` functions below for details.
If you don't need to be consistent with transformers, you can use your own dataloader/scheduler.

Please note `nnscaler` only supports 1 optimizer parameter group for now. So we also disable multiple parameter groups in the code when you use `transformers.train()`.
See `SingleParamGroupTrainer` in the code for details.

The loss of nnscaler will be exactly the same as transformers given gnorm clipping is disabled.
When gnorm clipping is enabled, the loss will be slightly different
due to the difference in gnorm calculation.

"""

import random
from typing import TYPE_CHECKING
import time
import os

from datasets import load_dataset

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Normalize, Resize, ToTensor, Compose
from transformers import ViTImageProcessor, ViTForImageClassification, get_linear_schedule_with_warmup

import nnscaler

if TYPE_CHECKING:
    from nnscaler.cli.trainer import Trainer
    from nnscaler.cli.trainer_args import TrainerArgs


VIT_MODEL_NAME = "google/vit-base-patch16-224"


_trainer: 'Trainer' = None


def init_env(trainer: 'Trainer'):
    global _trainer
    # save trainer for later use (e.g. to get max_train_steps)
    _trainer = trainer
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    if os.environ.get('DETERMINISTIC') is not None:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)

    if int(os.environ.get('NNSCALER_DEBUG', 0)):
        import debugpy
        # 5678 is the default attach port in the VS Code debug configurations.
        # Unless a host and port are specified, host defaults to 127.0.0.1
        # see https://code.visualstudio.com/docs/python/debugging for more details
        debugpy.listen(5678)
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
        debugpy.breakpoint()
        print('Resume on this line')


def on_train_step_end(trainer: 'Trainer', outputs, batches, idx: int) -> None:
    if torch.distributed.get_rank() == 0:
        print(f'# train_loss {idx:03d}', outputs[0].item())


def on_val_step_end(trainer: 'Trainer', outputs, batches, idx: int) -> None:
    if torch.distributed.get_rank() == 0:
        print(f'# val_loss {idx:03d}', outputs[0].item())


datasets = None
itos = None
stoi = None
vit_processor = None


def init_dataset():
    global datasets, itos, stoi, vit_processor

    vit_processor = ViTImageProcessor.from_pretrained(VIT_MODEL_NAME)

    mu, sigma = vit_processor.image_mean, vit_processor.image_std #get default mu,sigma
    size = vit_processor.size

    norm = Normalize(mean=mu, std=sigma) #normalize image pixels range to [-1,1]

    # resize 3x32x32 to 3x224x224 -> convert to Pytorch tensor -> normalize
    _transf = Compose([
        Resize(size['height']),
        ToTensor(),
        norm
    ])

    # apply transforms to PIL Image and store it to 'pixels' key
    def transf(arg):
        arg['pixels'] = [_transf(image.convert('RGB')) for image in arg['img']]
        return arg

    trainds,  = load_dataset("cifar10", split=["train[:5000]"])

    itos = dict((k,v) for k,v in enumerate(trainds.features['label'].names))
    stoi = dict((v,k) for k,v in enumerate(trainds.features['label'].names))

    splits = trainds.train_test_split(test_size=0.1, shuffle=False)
    trainds = splits['train']
    valds = splits['test']

    trainds.set_transform(transf)
    valds.set_transform(transf)

    datasets = {
        'train': trainds,
        'val': valds,
    }


def cifar10_dataset(split):
    if not datasets:
        init_dataset()
    return datasets[split]


def cifar10_collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixels'] for x in batch]),
        'labels': torch.tensor([x['label'] for x in batch]),
    }


def accelerator_dataloader_fn(dataset, batch_size, collate_fn, num_workers=0, drop_last=False, **kwargs):
    from accelerate import Accelerator
    from accelerate.utils import GradientAccumulationPlugin, DataLoaderConfiguration
    from torch.utils.data.sampler import RandomSampler
    from transformers.trainer_utils import seed_worker

    accelerator = Accelerator(
        gradient_accumulation_plugin=GradientAccumulationPlugin(num_steps=1),
        dataloader_config=DataLoaderConfiguration(even_batches=True, use_seedable_sampler=True)
    )

    sampler = RandomSampler(dataset)

    dataloader_params = {
        "batch_size": batch_size,
        "collate_fn": collate_fn,
        "num_workers": num_workers,
        "drop_last": drop_last,
        "sampler": sampler,
        "worker_init_fn": seed_worker,
    }
    return accelerator.prepare(DataLoader(dataset, **dataloader_params))


def vit_model():
    with torch.random.fork_rng():
        torch.manual_seed(0)
        return ViTForImageClassification.from_pretrained(
            VIT_MODEL_NAME, num_labels=10,
            ignore_mismatched_sizes=True,
            id2label=itos,
            label2id=stoi
        )


class VModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = vit_model()

    def forward(self, data):
        outputs = self.model(pixel_values=data['pixel_values'], labels=data['labels'])
        return outputs.loss


def scheduler_fn(optimizer, num_warmup_steps):
    return get_linear_schedule_with_warmup(optimizer, num_warmup_steps, _trainer.max_train_steps)


if __name__ == '__main__':
    from transformers import TrainingArguments, Trainer
    from sklearn.metrics import accuracy_score
    from nnscaler.cli.trainer_args import TrainerArgs
    from pathlib import Path
    import yaml

    with open(Path(__file__).absolute().with_name('train_cli_args.yaml'), 'r') as f:
        trainer_args = yaml.safe_load(f)

    init_env(None)
    init_dataset()

    args = TrainingArguments(
        f"test-cifar-10",
        save_strategy="epoch",
        evaluation_strategy="epoch",
        learning_rate=float(trainer_args['optimizer']['args']['lr']),
        per_device_train_batch_size=int(trainer_args['micro_batch_size']),
        max_grad_norm=float(trainer_args['optimizer']['clip_gnorm']),
        per_device_eval_batch_size=4,
        num_train_epochs=int(trainer_args['max_epochs']),
        weight_decay=float(trainer_args['optimizer']['args']['weight_decay']),
        warmup_steps=int(trainer_args['lr_scheduler']['args']['num_warmup_steps']),
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_dir='logs',
        logging_steps=1,
        remove_unused_columns=False,
        seed=0,
    )

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return dict(accuracy=accuracy_score(predictions, labels))

    model = vit_model()
    class SingleParamGroupTrainer(Trainer):
        """
        For parity check reason,
        we need to override the `create_optimizer` method to use only one param group for optimizer
        """
        def get_decay_parameter_names(self, model) -> list[str]:
            # make all parameters decay
            return [n for n, _ in model.named_parameters()]

    trainer = SingleParamGroupTrainer(
        model,
        args,
        train_dataset=datasets['train'],
        eval_dataset=datasets['val'],
        data_collator=cifar10_collate_fn,
        compute_metrics=compute_metrics,
        tokenizer=vit_processor,
    )
    trainer.train()
