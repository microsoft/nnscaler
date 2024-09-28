#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import argparse
import os

import torch

from nnscaler.cli.trainer import Trainer
from nnscaler.utils import set_default_logger_level


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ckpt_dir',
        default=None,
        type=str,
        help='path to the sharded checkpoint directory',
    )
    parser.add_argument(
        '--output_fname',
        default=None,
        type=str,
        help='output filename',
    )
    args = parser.parse_args()

    if args.ckpt_dir is None:
        raise ValueError('ckpt_dir is required')
    if args.output_fname is None:
        raise ValueError('output_fname is required')

    set_default_logger_level('INFO')
    ckpt_files = [os.path.join(args.ckpt_dir, f) for f in os.listdir(args.ckpt_dir) if f.endswith('.ckpt')]
    Trainer.merge_checkpoint(ckpt_files, args.output_fname)
