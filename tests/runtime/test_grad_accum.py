#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import torch
import pytest
from functools import partial

import nnscaler
from nnscaler.utils import accum_mode
from nnscaler.runtime.module import CubeModule
from ..launch_torchrun import torchrun
from ..utils import init_parameter, assert_parity


class MLP(CubeModule):
    def __init__(self, ngpus, async_op, dim=512, nlayers=4,):
        super().__init__()
        ranks = list(range(ngpus))
        self.init_group(ranks=ranks)

        self.layers = torch.nn.ModuleList([])
        for _ in range(nlayers):
            self.layers.append(torch.nn.Linear(dim, dim, bias=False))

        self.wreducer1 = nnscaler.runtime.adapter.Reducer(ranks=ranks, reduce_op='sum', async_op=async_op, zero=False,
                                                      max_bucket_size_bytes=137217728, zero_ngroups=1)
        for param in self.parameters():
            self.wreducer1.add_param(param)
        self.add_reducer(self.wreducer1)

    def forward(self, data):
        x = data
        for layer in self.layers:
            x = layer(x)
        loss = torch.sum(x)
        return loss


class BaseMLP(torch.nn.Module):
    def __init__(self, dim=512, nlayers=4,):
        super().__init__()
        self.layers = torch.nn.ModuleList([])
        for _ in range(nlayers):
            self.layers.append(torch.nn.Linear(dim, dim, bias=False))

    def forward(self, data):
        x = data
        for layer in self.layers:
            x = layer(x)
        loss = torch.sum(x)
        return loss


def get_dummy_data(batch_size: int = 256):
    torch.random.manual_seed(0)
    return torch.randn(
        [batch_size, 512], dtype=torch.float32,
        device=torch.cuda.current_device())


def baseline(accum_times: int = 4):
    model = BaseMLP()
    init_parameter(model)
    model = model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    losses = []
    for _ in range(3):
        for _ in range(accum_times):
            x = get_dummy_data()
            loss = model(x)
            loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss = loss.item()
        while abs(loss) > 10.0:
            loss /= 10.0
        losses.append(loss)
    return losses


def reducer_sync_test(accum_times: int = 4):
    ngpus = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    model = MLP(ngpus, async_op=False)
    init_parameter(model)
    model = model.cuda()
    for reducer in model.reducers:
        reducer.build_buckets()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    losses = []
    for _ in range(3):
        model.zero_grad()
        for _ in range(accum_times):
            x = get_dummy_data()
            x = x.chunk(ngpus, dim=0)[rank]
            loss = model(x)
            loss.backward()

        torch.distributed.all_reduce(loss)
        for reducer in model.reducers:
            reducer.sync_grads()
        optimizer.step()
        optimizer.zero_grad()
        loss = loss.item()
        while abs(loss) > 10.0:
            loss /= 10.0
        losses.append(loss)
    return losses


def reducer_async_test_wrong(accum_times: int = 4):
    ngpus = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    model = MLP(ngpus, async_op=True)
    init_parameter(model)
    model = model.cuda()
    for reducer in model.reducers:
        reducer.build_buckets()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    losses = []
    for _ in range(3):
        model.zero_grad()
        for _ in range(accum_times):
            x = get_dummy_data()
            x = x.chunk(ngpus, dim=0)[rank]
            loss = model(x)
            loss.backward()

        torch.distributed.all_reduce(loss)
        for reducer in model.reducers:
            reducer.sync_grads()
        optimizer.step()
        optimizer.zero_grad()
        loss = loss.item()
        while abs(loss) > 10.0:
            loss /= 10.0
        losses.append(loss)
    return losses


def reducer_async_test_correct(accum_times: int = 4):
    ngpus = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    model = MLP(ngpus, async_op=True)
    init_parameter(model)
    model = model.cuda()
    for reducer in model.reducers:
        reducer.build_buckets()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    losses = []
    for _ in range(3):
        model.zero_grad()
        for step in range(accum_times):
            with accum_mode(begin=(step == 0), end=(step == accum_times - 1)):
                x = get_dummy_data()
                x = x.chunk(ngpus, dim=0)[rank]
                loss = model(x)
                loss.backward()

        torch.distributed.all_reduce(loss)
        for reducer in model.reducers:
            reducer.sync_grads()
        optimizer.step()
        optimizer.zero_grad()
        loss = loss.item()
        while abs(loss) > 10.0:
            loss /= 10.0
        losses.append(loss)
    return losses


def accum_test():
    nnscaler.init()
    print('starting reducer sync')
    assert_parity(baseline, partial(reducer_sync_test, 4))
    print('starting reducer async')
    assert_parity(baseline, partial(reducer_async_test_correct, 4))
    # FIXME: this will hang:
    # print('starting reducer async wrong')
    # with pytest.raises(RuntimeError):
    #     assert_parity(baseline, partial(reducer_async_test_wrong, 4))

test_accum_2gpu = partial(torchrun, 2, accum_test)

