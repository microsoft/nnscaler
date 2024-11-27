#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import os
from .launch_torchrun import launch_torchrun

def worker_fn():
    rank = int(os.environ["RANK"])
    return rank


def test_torchrun():
    outputs = launch_torchrun(2, worker_fn)
    assert outputs == {0: 0, 1: 1}
