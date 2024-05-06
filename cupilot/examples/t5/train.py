from typing import List

import logging
import more_itertools as mitr
from functools import partial
import torch

from examples.t5.model import Config, T5, get_t5_dummy_dataloader
from examples.adam import Adam

import cube
from cube.profiler.timer import CudaTimer, print_each_rank
from cube.profiler.memory import memory_summary
from cube.runtime.device import DeviceGroup

from cube.graph import IRGraph

import cupilot
from cupilot.solver.block import IRBlock
from cupilot.constraints import Constraints
from cupilot.parallel.spmd import tensor_parallelism, replicate
from cupilot.estimator.profiler import Estimator

import argparse
parser = argparse.ArgumentParser(parents=[cupilot.build_parser()], description='T5 Train')
parser.add_argument('--mbs', type=int, default=1, help='micro-batch size')
parser.add_argument('--gbs', type=int, default=256, help='global batch size')
# arch
parser.add_argument('--layers', type=int, required=True, help='number of encoder / decoder layers')
parser.add_argument('--hidden', type=int, required=True)
parser.add_argument('--heads', type=int, required=True)
parser.add_argument('--seqlen', type=int, required=True)
parser.add_argument('--vocab', type=int, required=True)
# policy
parser.add_argument('--policy', type=str, choices=['alpa', 'cupilot', 'megatron'], required=True)
# log save
parser.add_argument('--save-spec', type=str, default=None,
                    help='folder for save searched results.')
parser.add_argument('--load-spec', type=str, default=None,
                    help='load searched tetris schedule from file')
args = parser.parse_args()

Estimator.register_rule('embedding', 1, 0, 16)
Estimator.register_rule('linear', 1, 0, 16)


cube.init()
print_each_rank(str(args), rank_only=0)

cube.set_logger_level(logging.WARN)
for name in logging.root.manager.loggerDict:
    if name.startswith('cupilot'):
        logging.getLogger(name).setLevel(logging.INFO)
logging.getLogger('cupilot.solver.spmd').setLevel(logging.DEBUG) 
logging.getLogger('cube.compiler').setLevel(logging.INFO)


def setup_policy(args):

    config = cupilot.build_config(parser)
    if args.policy == 'alpa':
        print_each_rank(config, rank_only=0)

    if args.policy == 'alpa':

        def constraint_fn(graph: IRGraph, resource, blocks: List[IRBlock], constraints: Constraints):
            
            for embed in graph.select(name='embedding'):
                constraints.add_trans_constraints(embed, (1, 0), None)
            for linear in graph.select(name='linear'):
                constraints.add_trans_constraints(linear, (1, 0), None)
            return blocks

        policy = partial(cupilot.policy,
                         nmicros=args.gbs//args.mbs,
                         mbs=args.mbs,
                         constrain_fn=constraint_fn,
                         load_spec_file=args.load_spec,
                         save_spec_file=args.save_spec,
                         config=config)

    elif args.policy == 'cupilot':

        def constraint_fn(graph: IRGraph, resource, blocks: List[IRBlock], constraints: Constraints):
            
            devices = list(range(resource.ngpus))
            head, tail = blocks[0], blocks[-1]

            def fn(segment):
                for node in segment.nodes():
                    if node.name in ('embedding', 'linear') :
                        tensor_parallelism(graph, node, idx=1, dim=0, devs=devices)
                    elif node.name == 'sum':
                        tensor_parallelism(graph, node, idx=0, dim=2, devs=devices)
                    else:
                        replicate(graph, node, devs=devices)

            head.mark_standalone(fn)
            tail.mark_standalone(fn)
            return blocks

        policy = partial(cupilot.policy,
                         nmicros=args.gbs//args.mbs,
                         mbs=args.mbs,
                         constrain_fn=constraint_fn,
                         load_spec_file=args.load_spec,
                         save_spec_file=args.save_spec,
                         config=config)
    
    elif args.policy == 'megatron':
        
        from cube.ir.operator import IRFwOperation
        def constraint_fn(graph: IRGraph, resource, blocks: List[IRBlock], constraints: Constraints):
            for node in graph.select(ntype=IRFwOperation):
                tp_size = resource.ngpus // args.max_pp
                if node.name in ('embedding', 'linear'):
                    constraints.add_trans_constraints(node, (1,0), tp_size)
                if node.name == 'self_attention':
                    constraints.add_trans_constraints(node, (1,0), tp_size)
                if node.name == 'cross_attention':
                    constraints.add_trans_constraints(node, (2,0), tp_size)
            
            assert isinstance(args.max_pp, int)
            head, transformers, tail = blocks[0], blocks[1:-1], blocks[-1]
            transformers[0] = IRBlock.merge([head, transformers[0]])
            transformers[-1] = IRBlock.merge([transformers[-1], tail])
            blocks = [IRBlock.merge(list(sub_blks)) for sub_blks in mitr.divide(args.max_pp, transformers)]
            return blocks

        policy = partial(cupilot.policy,
                         nmicros=args.gbs//args.mbs,
                         mbs=args.mbs,
                         constrain_fn=constraint_fn,
                         load_spec_file=args.load_spec,
                         save_spec_file=args.save_spec,
                         config=config)

    else:
        raise ValueError(f"{args.policy}: policy not found")
    return policy


def train():

    # setup model arg
    cfg = Config(
        args.vocab,
        args.hidden,
        args.hidden // args.heads,
        args.hidden * 4,
        args.layers,
        args.heads,
        args.seqlen)
    assert args.hidden % args.heads == 0

    print_each_rank(f"{cfg}", rank_only=0)

    if DeviceGroup().local_rank == 0:
        model = T5(cfg).half()
    else:
        model = None
    dataloader = get_t5_dummy_dataloader(args.mbs, cfg)

    if torch.distributed.get_rank() == 0:
        nparams = 0
        for param in model.parameters():
            nparams += param.nelement()
        print(f'full model parameter: {nparams}')
        print(f'TFLOPs: {round(model.flops(args.gbs) / 1e12, 2)}')

    policy = setup_policy(args)

    @cube.compile(model, dataloader, PAS=policy, load_content=False)
    def train_iter(model, dataloader):
        datas = next(dataloader)
        loss = model(*datas)
        loss.backward()
        # return loss
    model = cube.load_model(load_content=False)

    optimizer = Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.999))

    torch.distributed.barrier()
    print_each_rank('model weight consumpition:')
    memory_summary()
    nparams = 0
    for param in model.parameters():
        nparams += param.nelement()
    print_each_rank(f'loaded model parameter: {nparams}')

    CudaTimer(enable=False).warmup()
    dataloader = iter(dataloader)
    iter_num, warmup = 3, 2
    for step in range(iter_num):

        if step == warmup:
            CudaTimer(enable=True, predefined=False).start('e2e')

        # training
        train_iter(model, dataloader)
        optimizer.step()
        optimizer.zero_grad()

        if step == 0:
            print_each_rank('passed first iteration')
        if (step + 1) % 2 == 0:
            print_each_rank(f'iter [{step + 1}/{iter_num}]', rank_only=0)
        
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    CudaTimer().stop('e2e')
    print_each_rank('e2e time (ms) per iteration: {} ms'.format(
        CudaTimer().duration(iter_num-warmup, field_name='e2e')))
    CudaTimer().print_all(times=iter_num-warmup)
    memory_summary()


if __name__ == '__main__':

    train()