#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import logging

import nnscaler
from nnscaler.cli.trainer import Trainer


def main():
    nnscaler.utils.set_default_logger_level(level=logging.INFO)
    trainer = Trainer()
    try:
        trainer.run()
    finally:
        nnscaler.uninit()


if __name__ == '__main__':
    main()
