#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import argparse
from datetime import datetime
import os
from pathlib import Path

import datasets
import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
)

from nnscaler.cli.loggers.tensorboard import TensorBoardLogger
from nnscaler.cli.trainer import Trainer
from nnscaler.cli.trainer_args import (
    CheckpointConfig,
    ComputeConfig,
    DataloaderConfig,
    DatasetConfig,
    LogConfig,
    ModelConfig,
    OptimizerConfig,
    TrainerArgs,
)
from nnscaler.runtime.f16_optimizer import MixedPrecisionAdamW
import nnscaler.utils

import torch._dynamo  # FIXME: a workaround to avoid tracing the dynamic import


model_id = 'meta-llama/Meta-Llama-3-8B-Instruct'
tokenizer_id = model_id
dataset_id = 'bookcorpus/bookcorpus'


def prepare_data(max_seq_len, dataset_path=None):
    if dataset_path is None:
        dataset_path = f'./bookcorpus-{max_seq_len}'

    dataset = datasets.load_dataset(dataset_id)['train']
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    
    def _tokenize(sample):
        text = tokenizer.bos_token + sample['text'] + tokenizer.eos_token
        input_ids = tokenizer.encode(text, add_special_tokens=False)
        return {'input_ids': input_ids}
    
    tokenized_dataset = dataset.map(
        _tokenize,
        remove_columns=dataset.column_names,
        num_proc=32,
    )
    
    def _concat_split(samples):
        buffer = []
        resized_ids = []
        for input_ids in samples['input_ids']:
            buffer.extend(input_ids)
            while len(buffer) >= max_seq_len:
                resized_ids.append(buffer[:max_seq_len])
                buffer = buffer[max_seq_len:]
        return {'input_ids': resized_ids}
    
    final_dataset = tokenized_dataset.map(
        _concat_split,
        remove_columns=tokenized_dataset.column_names,
        num_proc=32,
        batched=True,
        batch_size=10000,
    )
    
    final_dataset.save_to_disk(dataset_path)
    return dataset_path


class WrapperModel(torch.nn.Module):
    def __init__(self, model_id, from_scratch=False, num_hidden_layers=None):
        super().__init__()

        if num_hidden_layers is not None:
            from_scratch = True

        if from_scratch:
            config = AutoConfig.from_pretrained(model_id)
            if num_hidden_layers:
                config.num_hidden_layers = num_hidden_layers
            self.model = AutoModelForCausalLM.from_config(config)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_id)

    def forward(self, data):
        result = self.model(
            input_ids=data['input_ids'],
            labels=data['labels'],
        )
        return result.loss


def main():
    nnscaler.utils.set_default_logger_level('INFO')

    ## Parse Args ##

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--prepare_data',
        action='store_true',
        help='prepare dataset',
    )
    parser.add_argument(
        '--max_train_steps',
        type=int,
        default=1000,
        help='specify max training steps',
    )
    parser.add_argument(
        '--mini',
        action='store_true',
        help='equals to "--from_scratch=True --num_hidden_layers=4 --max_seq_len=4096" (overrides these parameters)',
    )
    parser.add_argument(
        '--resume_from',
        help='load specified checkpoint',
    )
    parser.add_argument(
        '--merge_checkpoint',
        help='merge specified checkpoint',
    )
    parser.add_argument(
        '--from_scratch',
        action='store_true',
        help='train from scratch instead of finetune from huggingface checkpoint',
    )
    parser.add_argument(
        '--num_hidden_layers',
        type=int,
        help="specify the model's layer number",
    )
    parser.add_argument(
        '--max_seq_len',
        type=int,
        default=8192,
        help="specify max sequence length",
    )
    parser.add_argument(
        '--dataset_path',
        help='specify dataset path (default to "./bookcorpus-{max_seq_len}")',
    )
    args = parser.parse_args()

    if args.mini:
        args.from_scratch = True
        args.num_hidden_layers = 4
        args.max_seq_len = 4096

    ## Special Commands ##

    if args.prepare_data:
        dataset_path = prepare_data(args.max_seq_len, args.dataset_path)
        print(f'Dataset saved to {dataset_path}')
        return

    if args.merge_checkpoint:
        checkpoint_files = sorted(Path(args.merge_checkpoint).iterdir())
        Trainer.merge_checkpoint(checkpoint_files, './checkpoints/merged.ckpt')
        print('Checkpoint merged to ./checkpoints/merged.ckpt')
        return

    ## Setup Dataset ##

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    def collate(samples):
        if len(samples) == 0:
            return {}

        mini_batch = data_collator(samples)

        input_ids = mini_batch['input_ids']
        seq_len = input_ids.size(-1)

        shift_labels = mini_batch['labels'][..., 1:]
        labels = torch.nn.functional.pad(shift_labels, (0, 1), 'constant', -100).contiguous()

        return {
            'input_ids': input_ids,
            'labels': labels,
        }

    ## Config Trainer ##

    world_size = int(os.getenv('WORLD_SIZE'))
    compute_config = ComputeConfig(
        plan_ngpus=1,
        runtime_ngpus=world_size,
        constant_folding=True,
        use_zero=True,
        use_end2end=True,
        pas_config={  # to reduce memory usage
            'recompute_modules': 'LlamaDecoderLayer',
            'transient_mem_coef': 0.5,
        }
    )

    model_config = ModelConfig(
        type=WrapperModel,
        args={
            'model_id': model_id,
            'from_scratch': args.from_scratch,
            'num_hidden_layers': args.num_hidden_layers,
        },
    )

    optimizer_config = OptimizerConfig(
        type=MixedPrecisionAdamW,
        args={'lr': 2e-5, 'fused': True},
        clip_gnorm=1.0,
    )

    dataset_path = args.dataset_path
    if dataset_path is None:
        dataset_path = f'./bookcorpus-{args.max_seq_len}'
    dataset_config = DatasetConfig(
        type=datasets.load_from_disk,
        train_args={'dataset_path': dataset_path},
    )

    dataloader_config = DataloaderConfig(
        train_args={'collate_fn': collate, 'drop_last': True},
    )

    checkpoint_config = CheckpointConfig(
        every_n_epochs=1,
        save_type='deduped',
        resume_from=args.resume_from,
    )

    timestamp = datetime.now().strftime('%y%m%d%H%M%S')
    log_config = LogConfig(
        type=TensorBoardLogger,
        args={
            'name': f'llama3-example-{timestamp}',
            'root_dir': 'runs',
        },
    )

    trainer_args = TrainerArgs(
        compute_config=compute_config,
        pas_policy='autodist',
        model=model_config,
        optimizer=optimizer_config,
        dataset=dataset_config,
        dataloader=dataloader_config,
        checkpoint=checkpoint_config,
        log=[log_config],
        precision='bf16',
        grad_accumulation_steps=8,
        max_train_steps=args.max_train_steps,
        seed=0,
    )

    trainer = Trainer(train_args=trainer_args)
    trainer.run()


if __name__ == '__main__':
    main()
