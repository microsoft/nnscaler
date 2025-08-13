#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import torch
import argparse
import nnscaler
from nnscaler.graph import IRGraph
from nnscaler.ir.operator import IRFwOperation
from nnscaler.parallel import parallelize, ComputeConfig, ReuseType
import torch.distributed as dist

import nnscaler.graph
import nnscaler.graph.function
from ring_attention import wrap_ring_attn_func
from ring_attention.core.utils import set_seed, log


class TestModule(torch.nn.Module):
    def __init__(self):
        super(TestModule, self).__init__()

    def forward(self, q, k, v):
        out = wrap_ring_attn_func(q, k, v)
        return out


def policy(graph: IRGraph, resource: ComputeConfig) -> IRGraph:
    ngpus = resource.plan_ngpus
    partitioned = False
    for idx, node in enumerate(graph.select(ntype=IRFwOperation)):
        if not partitioned and node.signature == 'ring_attention.ring_attn.wrap_ring_attn_func':
            print('\nPartitioned node: ', node, '\n')
            sub_nodes = graph.partition(node, node.algorithm('dim'), idx=0, dim=1, num=ngpus)
            partitioned = True
        else:
            sub_nodes = graph.replicate(node, times=ngpus)
        for idx, sub_node in enumerate(sub_nodes):
            graph.assign(sub_node, idx)
    assert partitioned, f'expect ring_attn_func in graph, but not found.'
    return graph


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["fp16", "bf16"],
        help="Data type for inputs",
    )
    args = parser.parse_args()

    nnscaler.init()
    rank_id = torch.distributed.get_rank()
    world_size = dist.get_world_size()

    set_seed(rank_id)
    bsz = 1
    seqlen = 8192
    nheads = 24
    d = 128

    device = torch.device(f"cuda:{rank_id}")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    q = torch.randn(bsz, seqlen, nheads, d, device=device, dtype=dtype)
    k = torch.randn(bsz, seqlen, nheads, d, device=device, dtype=dtype)
    v = torch.randn(bsz, seqlen, nheads, d, device=device, dtype=dtype)

    dist.broadcast(q, src=0)
    dist.broadcast(k, src=0)
    dist.broadcast(v, src=0)
    dist.barrier()

    q.requires_grad = True
    k.requires_grad = True
    v.requires_grad = True

    single_out = wrap_ring_attn_func(q, k, v)
    single_out.retain_grad()
    single_loss = single_out.sum()
    single_loss.backward()

    model = TestModule()

    qq = q.detach().clone().requires_grad_()
    kk = k.detach().clone().requires_grad_()
    vv = v.detach().clone().requires_grad_()

    parallel_model = parallelize(model, dummy_forward_args={"q": qq, "k": kk, "v": vv}, pas_policy=policy,
                                 compute_config=ComputeConfig(world_size, world_size), reuse=ReuseType.OVERRIDE)
    parallel_model = parallel_model.cuda()


    parallel_model.train()

    qq = q.detach().clone().requires_grad_()
    kk = k.detach().clone().requires_grad_()
    vv = v.detach().clone().requires_grad_()

    para_out = parallel_model(qq, kk, vv)
    para_loss = para_out.sum()
    para_loss.backward()
    parallel_model.sync_grad()

    log("single out", single_out, rank0_only=True)
    log("multi  out", para_out, rank0_only=True)
    log("out   diff", single_out - para_out, rank0_only=True)

    log("single  dq", q.grad, rank0_only=True)
    log("multi   dq", qq.grad, rank0_only=True)
    log("dq    diff", q.grad - qq.grad, rank0_only=True)

    log("single  dk", k.grad, rank0_only=True)
    log("multi   dk", kk.grad, rank0_only=True)
    log("dk    diff", k.grad - kk.grad, rank0_only=True)

    log("single  dv", v.grad, rank0_only=True)
    log("multi   dv", vv.grad, rank0_only=True)
    log("dv    diff", v.grad - vv.grad, rank0_only=True)

    dist.destroy_process_group()
