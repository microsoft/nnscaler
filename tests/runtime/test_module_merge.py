#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import torch
import nnscaler
import os

from functools import partial
import pytest

from nnscaler.ir.operator import IRFwOperation
from nnscaler.runtime.device import DeviceGroup
from nnscaler.compiler import compile
from nnscaler.utils import load_model
from ..launch_torchrun import torchrun


@pytest.fixture(autouse=True, scope='module')
def clean_checkpoints():
    yield
    i = 0
    while True:
        try:
            os.remove(f'checkpoint-shard{i}.pt')
            i += 1
        except Exception:
            break


class Module(torch.nn.Module):
    def __init__(self):
        super(Module, self).__init__()

        self.register_buffer('buffer0', torch.randn(8, 8))
        self.param0 = torch.nn.Parameter(torch.randn(8, 8))
        self.param1 = torch.nn.Parameter(torch.randn(8, 8))
        self.register_buffer('buffer1', torch.randn(8, 8))
        self.param2 = torch.nn.Parameter(torch.randn(8, 8))

    def forward(self, x):
        x = x * self.param0
        x = x + self.buffer0
        x = x * self.param1
        x = x + self.buffer1
        x = x * self.param2
        return torch.sum(x)

def tp_policy(graph, resource):
    for idx, node in enumerate(graph.select(ntype=IRFwOperation)):
        if node.name == 'add':
            sub_nodes = graph.partition(
                node, node.algorithm('dim'), idx=1, dim=idx % 2, num=resource.ngpus)
        else:
            sub_nodes = graph.replicate(node, times=resource.ngpus)
        for devid, node in enumerate(sub_nodes):
            graph.assign(node, devid)
    return graph


def assert_same_state(origin, merged):
    assert set(origin.keys()) == set(merged.keys()), \
        f"state keys are not equal: origin: {origin.keys()}, merged: {merged.keys()}"
    for name in origin.keys():
        if isinstance(origin[name], dict):
            assert_same_state(origin[name], merged[name])
        elif isinstance(origin[name], torch.Tensor):
            assert torch.equal(origin[name].cpu(), merged[name].cpu()), \
                f"state {name} is not equal: origin:\n{origin[name]}\nmerged:\n{merged[name]}"
        else:
            assert origin[name] == merged[name], \
                f"state {name} is not equal: origin:\n{origin[name]}\nmerged:\n{merged[name]}"


def merge_model_states_test():
    nnscaler.init()

    model = Module()
    sample = torch.randn(8, 8, device=torch.cuda.current_device())

    full_model_state = model.state_dict()

    @compile(model, sample, PAS=tp_policy)
    def train_iter(model, sample):
        loss = model(sample)
        loss.backward()
        return loss
    cube_model = load_model()

    state_dict = cube_model.state_dict()
    torch.save({'state_dict': state_dict, 'fullmap': cube_model.fullmap},
               f'checkpoint-shard{DeviceGroup().rank}.pt')
    torch.distributed.barrier()
    if DeviceGroup().rank == 0:
        model_states = []
        fullmaps = []
        for i in range(DeviceGroup().world_size):
            checkpoint = torch.load(f'checkpoint-shard{i}.pt')
            model_states.append(checkpoint['state_dict'])
            fullmaps.append(checkpoint['fullmap'])
        merged_state_dict = cube_model.merge_model_state_dicts(model_states, fullmaps)
        assert_same_state(full_model_state, merged_state_dict)


test_merge_model_states = partial(torchrun, 2, merge_model_states_test)


def merge_optimizer_states_test():
    nnscaler.init()

    torch.manual_seed(0)
    model = Module().cuda()
    full_optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    sample = torch.randn(8, 8, device=torch.cuda.current_device())

    full_model_state = model.state_dict()
    full_optim_state = full_optimizer.state_dict()

    @compile(model, sample, PAS=tp_policy)
    def train_iter(model, sample):
        loss = model(sample)
        loss.backward()
        return loss

    cube_model = load_model()
    optimizer = torch.optim.Adam(cube_model.parameters(), lr=0.01)

    # test for initial state
    model_state_dict = cube_model.state_dict()
    optim_state_dict = optimizer.state_dict()
    states = {
        'model': model_state_dict,
        'optimizer': optim_state_dict,
        'fullmap': cube_model.fullmap
    }
    torch.save(states, f'checkpoint-shard{DeviceGroup().rank}.pt')
    torch.distributed.barrier()

    if DeviceGroup().rank == 0:
        states = []
        for i in range(DeviceGroup().world_size):
            checkpoint = torch.load(f'checkpoint-shard{i}.pt')
            states.append((checkpoint['model'], checkpoint['optimizer'], checkpoint['fullmap']))
        merged_model_states, merged_optim_states = cube_model.merge_partial_states(states)
        assert_same_state(full_model_state, merged_model_states)
        assert_same_state(full_optim_state, merged_optim_states)
    torch.distributed.barrier()

    # test after training

    model.cuda()
    for _ in range(2):
        # full model
        loss = model(sample)
        loss.backward()
        full_optimizer.step()
        full_optimizer.zero_grad()

        # cube model
        loss = train_iter(cube_model, sample)
        optimizer.step()
        optimizer.zero_grad()

    model_state_dict = cube_model.state_dict()
    optim_state_dict = optimizer.state_dict()
    states = {
        'model': model_state_dict,
        'optimizer': optim_state_dict,
        'fullmap': cube_model.fullmap
    }

    torch.save(states, f'checkpoint-shard{DeviceGroup().rank}.pt')
    torch.distributed.barrier()

    full_model_state = model.state_dict()
    full_optim_state = full_optimizer.state_dict()

    if DeviceGroup().rank == 0:
        states = []
        for i in range(DeviceGroup().world_size):
            checkpoint = torch.load(f'checkpoint-shard{i}.pt')
            states.append((checkpoint['model'], checkpoint['optimizer'], checkpoint['fullmap']))
        merged_model_states, merged_optim_states = cube_model.merge_partial_states(states)
        assert_same_state(full_model_state, merged_model_states)
        assert_same_state(full_optim_state, merged_optim_states)


test_merge_optim_states = partial(torchrun, 2, merge_optimizer_states_test)
