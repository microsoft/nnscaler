#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import torch

from datasets import load_from_disk
from transformers import DataCollatorForLanguageModeling

from ...transformers_utils import get_tokenizer


IGNORE_IDX=-100


def get_dataset(dataset_path, tokenizer_name_or_path):
    dataset = load_from_disk(dataset_path)
    tokenizer = get_tokenizer(tokenizer_name_or_path)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    def collate_fn(samples):
        if len(samples) == 0:
            return {}

        mini_batch = dict(data_collator(samples))
        seq_len = mini_batch['input_ids'].size(-1)
        shift_labels = mini_batch['input_ids'][..., 1:]
        mini_batch['labels'] = torch.nn.functional.pad(shift_labels, (0, 1), 'constant', IGNORE_IDX).contiguous()

        # cast `nsentences` and `ntokens` to tensor since current pipeline parallelism can only transfer data in tensor format
        return {
            "nsentences": torch.tensor(len(samples), dtype=torch.long),
            "ntokens": torch.tensor(len(samples) * seq_len, dtype=torch.long),
            "net_input": mini_batch,
            "target": mini_batch.pop('labels'),
        }

    return dataset, collate_fn
