#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""
This test is to verify the correctness of the gradient norm algorithm for nnscaler.

To avoid other potential parity issues that may have influence the gradient value,
we use weight data as gradient, and calculate its norm to verify the correctness
of gnorm calculation.
"""
import torch
from functools import partial

import nnscaler
from nnscaler.compiler import compile
from nnscaler.utils import load_model
from nnscaler.ir.operator import IRFwOperation
from nnscaler.runtime.module import CubeModule
from nnscaler.runtime.gnorm import prepare_for_grad_clip, clip_gnorm
from nnscaler.flags import CompileFlag

from ..launch_torchrun import torchrun
from ..utils import init_parameter


class Module(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(16, 16, bias=False)
        self.linear2 = torch.nn.Linear(16, 16, bias=False)
        self.linear3 = torch.nn.Linear(16, 16, bias=False)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return torch.sum(x)


def tensor_parallelism(graph, node: IRFwOperation, idx, dim, num):
    sub_nodes = graph.partition(
        node, node.algorithm('dim'), idx=idx, dim=dim, num=num)
    for idx, sub_node in enumerate(sub_nodes):
        graph.assign(sub_node, idx)
    return sub_nodes


def cal_wnorm_baseline(model):
    wnorm = torch.norm(
        torch.stack([torch.norm(p, p=2, dtype=torch.float32) for p in model.parameters()])
    )
    return wnorm


def cal_wnorm_cube(model: CubeModule):
    for p in model.parameters_for_optimizer():
        p.grad = p.data
        # p.grad.copy_(p.data)
    nreplicas2localparams = prepare_for_grad_clip(model, is_zero=CompileFlag.use_zero)
    wnorm, _ = clip_gnorm(nreplicas2localparams, None)
    # maps = {tid: [t.size() for t in ts] for tid, ts in nreplicas2localparams.items()}
    # print(f'cube nrepicas len: {maps}')
    return wnorm

# su_num: scale unit number
def dp_policy(graph, resource, su_num):
    ngpus = resource.ngpus // su_num
    for node in graph.select(ntype=IRFwOperation):
        tensor_parallelism(graph, node, idx=0, dim=0, num=ngpus)
    return graph

def pp_policy(graph, resource, su_num):
    ngpus = resource.ngpus // su_num
    devid = 0
    for node in graph.select(ntype=IRFwOperation):
        graph.assign(node, devid)
        devid = (devid + 1) % ngpus
    return graph


def model_test(policy, su_num: int = 1, use_zero: bool = False):
    # su_num: scale unit number
    nnscaler.init()
    CompileFlag.use_zero = use_zero

    model = Module().cuda()
    init_parameter(model)

    # get baseline weight norm
    wnorm_baseline = cal_wnorm_baseline(model)

    sample = torch.randn(16, 16).cuda()
    @compile(model, sample, PAS=partial(policy, su_num=su_num),
                  scale=su_num > 1)
    def train_iter(model, data):
        loss = model(data)
        loss.backward()
        return loss

    model = load_model()

    # train_iter(model, sample)  # link .grad to reducer buffer
    wnorm_cube = cal_wnorm_cube(model)

    for rank in range(torch.distributed.get_world_size()):
        if rank == torch.distributed.get_rank():
            print(f'rank: {rank}: baseline wnorm: {wnorm_baseline}')
            print(f'rank: {rank}: cube     wnorm: {wnorm_cube}')
        torch.distributed.barrier()

    assert wnorm_cube == wnorm_baseline


test_norm_case1_dp = partial(torchrun, 2, model_test, dp_policy)
test_norm_case1_dp_su = partial(torchrun, 4, model_test, dp_policy, 2)
test_norm_case1_dp_zero = partial(torchrun, 2, model_test, dp_policy, 1, True)

test_norm_case2_pp = partial(torchrun, 2, model_test, pp_policy)
test_norm_case2_pp_su = partial(torchrun, 4, model_test, pp_policy, 2)
test_norm_case2_pp_su_zero = partial(torchrun, 4, model_test, dp_policy, 2, True)