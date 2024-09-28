#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import argparse
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def main(args):
    config = AutoConfig.from_pretrained(args.model_id)
    config.num_hidden_layers = 4
    config.use_cache = False
    config._attn_implementation = 'flash_attention_2'
    model = AutoModelForCausalLM.from_config(config)
    model.save_pretrained(args.output_id)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.save_pretrained(args.output_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_id',
        default=None,
        type=str,
        help='huggingface model id / path',
    )
    parser.add_argument(
        '--output_id',
        default=None,
        type=str,
        help='output model id / path',
    )
    args = parser.parse_args()

    if args.model_id is None:
        raise ValueError('model_id is required')
    if args.output_id is None:
        raise ValueError('output_id is required')

    main(args)
