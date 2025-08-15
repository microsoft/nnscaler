#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import argparse
from datasets import load_from_disk, concatenate_datasets, load_dataset
from pathlib import Path
from transformers import PreTrainedTokenizer
from typing import List, Dict

from examples.transformers_utils import get_tokenizer


ROOT_SAVE_DIR = Path(__file__).parent
MAX_WORKERS = 32


def tokenize(samples: Dict[str, List], tokenizer: PreTrainedTokenizer, min_length=None, max_length=None):
    def condition(text):
        return (min_length is None or len(text) >= min_length) and (max_length is None or len(text) <= max_length)

    input_ids_list = []
    for text in samples["text"]:
        if condition(text):
            input_ids_list.append(tokenizer.encode(tokenizer.bos_token + text + tokenizer.eos_token, add_special_tokens=False))
    return {"input_ids": input_ids_list, "length": [len(input_ids) for input_ids in input_ids_list]}


def cat(samples: Dict[str, List], max_seq_len=128 * 1024, context_len=128 * 1024):
    input_ids_list = []
    position_ids_list = []

    input_ids_buffer = []
    position_ids_buffer = []

    for input_ids in samples["input_ids"]:
        input_ids = input_ids[:context_len]
        input_ids_buffer.extend(input_ids)
        position_ids_buffer.extend(range(len(input_ids)))

        if len(input_ids_buffer) >= max_seq_len:
            input_ids_list.append(input_ids_buffer[:max_seq_len])
            position_ids_list.append(position_ids_buffer[:max_seq_len])

            input_ids_buffer = []
            position_ids_buffer = []

    return {"input_ids": input_ids_list, "position_ids": position_ids_list}


if __name__ == "__main__":
    root_save_dir = ROOT_SAVE_DIR
    max_workers = MAX_WORKERS

    parser = argparse.ArgumentParser(description="Set the tokenizer name or path.")
    parser.add_argument(
        '--tokenizer_name_or_path',
        type=str,
        default='meta-llama/Meta-Llama-3-8B',
        help='Path to the tokenizer model or name of the tokenizer.'
    )
    args = parser.parse_args()
    tokenizer_name_or_path = args.tokenizer_name_or_path

    max_seq_len = 128 * 1024
    short_context_len = 8 * 1024
    long_context_len = 128 * 1024
    short_len_split = [64, 2 * 1024, 4 * 1024, 9 * 1024]
    long_len_split = [8 * 1024, 32 * 1024, 64 * 1024, 128 * 1024, 200 * 1024]

    tokenizer = get_tokenizer(tokenizer_name_or_path)

    fweb_dataset = load_dataset(str(root_save_dir / "fineweb-edu/sample/10BT"), num_proc=max_workers)["train"]
    fweb_dataset = fweb_dataset.map(tokenize,
                                    fn_kwargs={"tokenizer": tokenizer, "max_length": 5 * short_len_split[-1]},
                                    batched=True,
                                    num_proc=max_workers,
                                    batch_size=10000,
                                    remove_columns=fweb_dataset.column_names)

    sub_fweb_datasets = []
    fweb_sample_size = [8000, 8000, 16000]
    for left, right, size in zip(short_len_split[:-1], short_len_split[1:], fweb_sample_size):
        assert left < right
        fweb_dataset_idx = [idx for idx, length in enumerate(fweb_dataset["length"]) if left < length <= right]
        sub_fweb_dataset = fweb_dataset.select(fweb_dataset_idx)
        sub_fweb_dataset = fweb_dataset.map(cat,
                                            fn_kwargs={"max_seq_len": max_seq_len, "context_len": short_context_len},
                                            batched=True,
                                            num_proc=max_workers,
                                            batch_size=10000,
                                            remove_columns=sub_fweb_dataset.column_names)
        sub_fweb_dataset = sub_fweb_dataset.select(range(size))
        print(f"Short context [cat]: {left} - {right}, sample size: {len(sub_fweb_dataset)}")
        sub_fweb_datasets.append(sub_fweb_dataset)
    concatenate_datasets(sub_fweb_datasets).save_to_disk(root_save_dir / "fineweb-edu-sample-10BT-short-context")
    del sub_fweb_datasets

    for split in ["arxiv", "common_crawl", "wikipedia"]:
        rp_dataset = load_dataset(str(root_save_dir / "RedPajama-Data-1T" / f"{split}"), num_proc=max_workers)["train"]
        rp_dataset = rp_dataset.map(tokenize,
                                    batched=True,
                                    fn_kwargs={"tokenizer": tokenizer, "min_length": 4 * long_len_split[0], "max_length": 5 * long_len_split[-1]},
                                    num_proc=max_workers,
                                    batch_size=10000,
                                    remove_columns=rp_dataset.column_names)
        rp_dataset_idx = [idx for idx, length in enumerate(rp_dataset["length"]) if long_len_split[0] < length <= long_len_split[-1]]
        rp_dataset = rp_dataset.select(rp_dataset_idx)
        print(f"Long context [{split} filter]: {long_len_split[0]} - {long_len_split[-1]}, sample size: {len(rp_dataset)}")
        rp_dataset.save_to_disk(root_save_dir / f"RedPajama-Data-1T-{split}-long-context-filtered")
        del rp_dataset

    sub_rp_datasets = []
    rp_sample_size = {
        "arxiv": [3000, 4000, 8000, 3000],
        "common_crawl": [2000, 3000, 8000, 5000],
        "wikipedia": [3000, 1000, 0, 0]
    }
    for split in ["arxiv", "common_crawl", "wikipedia"]:
        rp_dataset = load_from_disk(root_save_dir / f"RedPajama-Data-1T-{split}-long-context-filtered")
        for left, right, size in zip(long_len_split[:-1], long_len_split[1:], rp_sample_size[split]):
            assert left < right
            rp_dataset_idx = [idx for idx, length in enumerate(rp_dataset["length"]) if left < length <= right]
            sub_rp_dataset = rp_dataset.select(rp_dataset_idx)
            sub_rp_dataset = sub_rp_dataset.map(cat,
                                                fn_kwargs={"max_seq_len": max_seq_len, "context_len": long_context_len},
                                                batched=True,
                                                num_proc=max_workers,
                                                batch_size=10000,
                                                remove_columns=sub_rp_dataset.column_names)
            sub_rp_dataset = sub_rp_dataset.select(range(size))
            print(f"Long context [{split} cat]: {left} - {right}, sample size: {len(sub_rp_dataset)}")
            sub_rp_datasets.append(sub_rp_dataset)
    concatenate_datasets(sub_rp_datasets).save_to_disk(root_save_dir / "RedPajama-Data-1T-long-context")
    del sub_rp_datasets

    # create final mix context window dataset
    concatenate_datasets([
        load_from_disk(root_save_dir / "fineweb-edu-sample-10BT-short-context"),
        load_from_disk(root_save_dir / "RedPajama-Data-1T-long-context"),
    ]).save_to_disk(root_save_dir / f"mix-context-win-short-{short_context_len}-long-{long_context_len}")
