#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from typing import Dict

from nnscaler.cli.trainer_args import TrainerArgs
from tests.parallel_module.test_end2end import MLP
from tests.utils import init_random as init_random_fn


class MixModuleMLP(nn.Module):
    def __init__(self, dim: int, nlayers: int, init_random: bool = True):
        super().__init__()
        if init_random:
            init_random_fn()
        self.layers = torch.nn.ModuleList([])
        for _ in range(nlayers):
            self.layers.append(nn.Linear(dim, dim, bias=False))

    def forward(self, input):
        x = input
        for layer in self.layers:
            x = layer(x)
        return x


class MixModuleMLP2(MixModuleMLP):
    pass


class MixModuleMLP3(MixModuleMLP):
    pass


class MixModuleMLP4(MixModuleMLP):
    pass


class MixModuleMLPWithLoss(nn.Module):
    def __init__(self, dim: int, nlayers: int, init_random: bool = True):
        super().__init__()
        self.mlp = MixModuleMLP(dim, nlayers, init_random=init_random)
        self.loss_fn = nn.BCELoss()

    def forward(self, input, target):
        x = self.mlp(input)
        x = torch.sigmoid(x)
        loss = self.loss_fn(x, target)
        return loss


class MixedModule(torch.nn.Module):
    def __init__(self, dim: int, nlayers: int, init_random: bool = True):
        super().__init__()
        self.mlp0 = MixModuleMLP(dim, nlayers, init_random=init_random)
        self.mlp1 = MixModuleMLP2(dim, nlayers, init_random=init_random)
        self.mlp2 = MixModuleMLP3(dim, nlayers, init_random=init_random)
        self.mlploss = MixModuleMLPWithLoss(dim, nlayers, init_random=init_random)

    def forward(self, data: Dict[str, torch.Tensor]):
        x = data['data']
        target = data['target']
        x = self.mlp0(x)
        x = self.mlp1(x)
        x = self.mlp2(x)
        return self.mlploss(x, target)


def forward_args_gen_fn(trainer_args: TrainerArgs):
    return {
        'input':
            torch.randn(trainer_args.dataset.train_args['size'], trainer_args.dataset.train_args['dim']),
        'target':
            torch.rand(trainer_args.dataset.train_args['size'], trainer_args.dataset.train_args['dim']),
    }


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
