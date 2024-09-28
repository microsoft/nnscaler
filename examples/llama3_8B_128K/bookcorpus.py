#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import argparse
from typing import List, Dict

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer
import numpy
import torch


def get_tokenizer(model_path):
    return AutoTokenizer.from_pretrained(model_path)


def tokenize(sample: Dict[str, str], tokenizer: PreTrainedTokenizer, text_key: str):
    input_ids = tokenizer.encode(tokenizer.bos_token + sample[text_key] + tokenizer.eos_token, add_special_tokens=False)
    return {"input_ids": input_ids}


def concate_split(samples: Dict[str, List[List[int]]], sample_len: int, text_key: str):
    buffer = samples[text_key][0]
    resized_ids = []
    length = []
    for in_ids in samples[text_key]:
        buffer.extend(in_ids)
        while len(buffer) >= sample_len:
            resized_ids.append(buffer[:sample_len])
            length.append(sample_len)
            buffer = buffer[sample_len:]
    return {"input_ids": resized_ids, "length": length}


def create_dataset(tokenizer: PreTrainedTokenizer, raw_dataset: Dataset, text_key: str, sample_len: int = 8 * 1024, batch_size=10000):
    tokenized_dataset = raw_dataset.map(tokenize, remove_columns=raw_dataset.column_names, num_proc=32,
                                        fn_kwargs={'tokenizer': tokenizer, 'text_key': text_key})
    return tokenized_dataset.map(concate_split, remove_columns=tokenized_dataset.column_names, num_proc=32, batched=True,
                                 batch_size=batch_size, fn_kwargs={'sample_len': sample_len, 'text_key': 'input_ids'})


if __name__ == '__main__':
    # python bookcorpus.py --data_path_or_name "bookcorpus/bookcorpus" --tokenizer_path_or_name "meta-llama/Llama-2-7b-hf" --save_path "bookcorpus-llama2-2k-hf" --sequence_length 2048
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path_or_name', help='the path or name of the raw dataset, for exmaple, "bookcorpus/bookcorpus"', type=str, required=True)
    parser.add_argument('--tokenizer_path_or_name', help='the tokenizer path or name, for example, "meta-llama/Llama-2-7b-hf"', type=str, required=True)
    parser.add_argument('--save_path', help='the path to save the tokenized dataset', type=str, required=True)
    parser.add_argument('--sequence_length', help='the length of each sample in the tokenized dataset, usually set to the max sequence length', type=int, required=True)
    args = parser.parse_args()
    data_path_or_name = args.data_path_or_name
    tokenizer_path_or_name = args.tokenizer_path_or_name
    save_path = args.save_path
    sequence_length = args.sequence_length

    raw_dataset = load_dataset(data_path_or_name)["train"]
    tokenizer = get_tokenizer(tokenizer_path_or_name)
    dataset = create_dataset(tokenizer, raw_dataset, "text", sequence_length)
    dataset.save_to_disk(save_path)
    # used by fairseq dataset
    sizes = numpy.array(dataset["length"])
    torch.save(sizes, f"{save_path}/fairseq_dataset_sizes.np")

    print(f"If you want to use this dataset, please set `DATASET_TYPE=hf_disk` and `DATASET_PATH={save_path}` in run script")
