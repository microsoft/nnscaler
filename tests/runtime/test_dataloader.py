#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import torch
from nnscaler.runtime.utils import MicroBatchDataLoader, microbatches


import pytest

def mock_dataloader_sample():
    tokens = torch.randint(0, 1000, (2, 1024))
    labels = torch.randint(0, 1000, (2,))
    ntokens = 2048
    return tokens, labels, ntokens


def test_microbatch_dataloader():

    samples = [mock_dataloader_sample() for _ in range(4)]
    dataloader = microbatches(samples)

    assert isinstance(dataloader, MicroBatchDataLoader)
    assert len(dataloader) == 4

    sample = next(dataloader)
    assert isinstance(sample, tuple)
    assert isinstance(sample[0], torch.Tensor)
    assert isinstance(sample[1], torch.Tensor)
    assert isinstance(sample[2], int)
    _ = next(dataloader)
    _ = next(dataloader)
    _ = next(dataloader)
    with pytest.raises(StopIteration):
        _ = next(dataloader)


def test_microbatch_dataloaser_with_cycle():

    samples = [mock_dataloader_sample() for _ in range(4)]
    dataloader = microbatches(samples, cycle=True)

    assert isinstance(dataloader, MicroBatchDataLoader)
    assert len(dataloader) == 4

    # no stop iteration should be raised
    for _ in range(16):
        sample = next(dataloader)
        assert isinstance(sample, tuple)
        assert isinstance(sample[0], torch.Tensor)
        assert isinstance(sample[1], torch.Tensor)
        assert isinstance(sample[2], int)
    