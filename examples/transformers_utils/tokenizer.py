#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from transformers import AutoTokenizer


IGNORE_IDX = -100


def get_tokenizer(tokenizer_name_or_path,
                  model_max_length=None,):

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)
    special_tokens_dict = dict()
    assert tokenizer.bos_token is not None and tokenizer.eos_token is not None
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = tokenizer.unk_token if tokenizer.unk_token else tokenizer.eos_token
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = tokenizer.pad_token if tokenizer.pad_token else tokenizer.eos_token

    tokenizer.add_special_tokens(special_tokens_dict)
    if model_max_length:
        tokenizer.model_max_length = model_max_length
    return tokenizer
