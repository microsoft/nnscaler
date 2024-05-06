# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

r"""Runtime Utilities"""

from typing import Any
import logging

import torch.utils.data as data

_logger = logging.getLogger(__name__)


def create_dummy_dataloader(sample: Any, 
                            batch_size: int, drop_last=True, 
                            **dataloader_config) -> data.DataLoader:
    """Create a dummy dataloader

    The function is mainly used for performance test.
    
    Args:
        sample (Any): a data sample without batch size dimension.
            The sample can be a single tensor/object or tuple/list of tensors/objects
        batch_size (int): batch size
        drop_last (bool): whether to drop last batch to make batch size consistent.
        dataloader_config (dict): kwargs for dataloader initialization.

    Returns:
        dataloader (torch.utils.data.DataLoader):
            returns 
    """

    class DummyDataset(data.Dataset):

        def __init__(self, sample: Any):

            self.sample = sample

        def __len__(self):
            return 1024000
        
        def __getitem__(self, key: int):
            return self.sample

    dataset = DummyDataset(sample)
    dataloader = data.DataLoader(
        dataset, batch_size=batch_size, drop_last=drop_last, 
        **dataloader_config)
    return dataloader
