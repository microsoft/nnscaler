#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import torch
from torch.utils.data import DataLoader, Dataset

from tests.parallel_module.test_end2end import MLP

class SimpleDataset(Dataset):
    def __init__(self, dim: int, size: int = 100):
        torch.manual_seed(0)
        self.data = torch.randn(size, dim)
        self.target = torch.rand(size, dim)

    def __getitem__(self, idx: int):
        return {
            'data': self.data[idx],
            'target': self.target[idx]
        }

    def __len__(self):
        return len(self.data)
