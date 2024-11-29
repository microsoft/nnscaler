#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import torch
import os
import sys
import shutil
from datetime import datetime
import subprocess
import torch
import logging
from pathlib import Path
from nnscaler.autodist.util import get_node_arch, get_default_profile_path


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("nnscaler.comm_profiler")


def main():
    default_path = get_default_profile_path()

    if not default_path.is_dir():
        default_path.mkdir(parents=True)
        logger.info(f'create folder: {default_path}')
    else:
        logger.info(f'folder already exists: {default_path}')

    comm_path = default_path / 'comm'

    if comm_path.is_dir():
        logger.info(f'back up legacy comm info: {comm_path}')
        shutil.move(
                comm_path,
                default_path / f'comm_back_{str(datetime.now().timestamp())}')
    comm_path.mkdir(parents=True, exist_ok=True)

    logger.info(f'CUDA device num: {torch.cuda.device_count()}')
    profiler_fname = Path(__file__).parent / 'comm_profile.py'
    device_num = 2
    while device_num <= torch.cuda.device_count():
        command = f'torchrun --master_port 21212 --nproc_per_node={device_num} {profiler_fname} --comm_profile_dir={comm_path}'
        output = subprocess.check_output(command, shell=True, text=True)
        device_num = device_num * 2

    logger.info(f'comm profile done')


if __name__ == '__main__':
    main()
