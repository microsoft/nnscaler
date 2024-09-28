#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import torch
import nnscaler
from nnscaler.graph import IRGraph
from nnscaler.ir.operator import IRFwOperation
from nnscaler.parallel import parallelize, ComputeConfig, ReuseType
import torch.distributed as dist
from flash_attn import flash_attn_func

import nnscaler.graph
import nnscaler.graph.function
from ring_attn import wrap_ring_attn_func

import random

def set_seed(rank, seed=42):
    seed = rank + seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def log(msg, a, rank0_only=False):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if rank0_only:
        if rank == 0:
            print(
                f"{msg}: "
                f"max {a.abs().max().item()}, "
                f"mean {a.abs().mean().item()}",
                flush=True,
            )
        return

    for i in range(world_size):
        if i == rank:
            if rank == 0:
                print(f"{msg}:")
            print(
                f"[{rank}] "
                f"max {a.abs().max().item()}, "
                f"mean {a.abs().mean().item()}",
                flush=True,
            )
        dist.barrier()

class TestModule(torch.nn.Module):
    def __init__(self):
        super(TestModule, self).__init__()

    def forward(self, _in0, _in1, _in2):
        out = wrap_ring_attn_func(_in0, _in1, _in2)
        return out

def policy(graph: IRGraph, resource: ComputeConfig) -> IRGraph:
    ngpus = resource.plan_ngpus
    partitioned = False
    for idx, node in enumerate(graph.select(ntype=IRFwOperation)):
        if not partitioned and node.signature == 'ring_attn.wrap_ring_attn_func':
            print('Partitioned node: ', node)
            sub_nodes = graph.partition(
                node, node.algorithms('dim'), idx=0, dim=1, num=ngpus)
            partitioned = True
        else:
            sub_nodes = graph.replicate(node, times=ngpus)
        for idx, sub_node in enumerate(sub_nodes):
            graph.assign(sub_node, idx)
    assert partitioned, f'expect ring_attn_func in graph, but not found.'
    return graph

if __name__ == "__main__":
    nnscaler.init()
    rank_id = torch.distributed.get_rank()
    world_size = dist.get_world_size()

    set_seed(rank_id)
    bsz = 1
    seqlen = 8192
    nheads = 24
    d = 128

    device = torch.device(f"cuda:{rank_id}")
    # dtype = torch.float16
    dtype = torch.bfloat16

    q = torch.randn(bsz, seqlen, nheads, d, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(bsz, seqlen, nheads, d, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(bsz, seqlen, nheads, d, device=device, dtype=dtype, requires_grad=True)

    dist.broadcast(q, src=0)
    dist.broadcast(k, src=0)
    dist.broadcast(v, src=0)
    dist.barrier()

    single_out = wrap_ring_attn_func(q, k, v)
    single_out.retain_grad()
    single_loss = single_out.sum()
    single_loss.backward()

    model = TestModule()

    _in0 = q.detach().clone().requires_grad_()
    _in1 = k.detach().clone().requires_grad_()
    _in2 = v.detach().clone().requires_grad_()

    parallel_model = parallelize(model, dummy_forward_args={"_in0": _in0, "_in1": _in1, "_in2": _in2}, pas_policy=policy,
                                 compute_config=ComputeConfig(world_size, world_size), reuse=ReuseType.OVERRIDE)
    parallel_model = parallel_model.cuda()


    parallel_model.train()

    _in0 = q.detach().clone().requires_grad_()
    _in1 = k.detach().clone().requires_grad_()
    _in2 = v.detach().clone().requires_grad_()

    para_out = parallel_model(_in0, _in1, _in2)
    para_loss = para_out.sum()
    para_loss.backward()
    parallel_model.sync_grad()

    log("single out", single_out, rank0_only=True)
    log("multi  out", para_out, rank0_only=True)
    log("out   diff", single_out - para_out, rank0_only=True)

    log("single  dq", q.grad, rank0_only=True)
    log("multi   dq", _in0.grad, rank0_only=True)
    log("dq    diff", q.grad - _in0.grad, rank0_only=True)

    log("single  dk", k.grad, rank0_only=True)
    log("multi   dk", _in1.grad, rank0_only=True)
    log("dk    diff", k.grad - _in1.grad, rank0_only=True)

    log("single  dv", v.grad, rank0_only=True)
    log("multi   dv", _in2.grad, rank0_only=True)
    log("dv    diff", v.grad - _in2.grad, rank0_only=True)
