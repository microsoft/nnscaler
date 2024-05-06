import logging
from functools import partial
import more_itertools
import torch

from examples.gpt.model import Config, GPT, get_gpt_dummy_dataloader

import cube
from cube.profiler.timer import CudaTimer, print_each_rank
from cube.profiler.memory import memory_summary
from cube.runtime.device import DeviceGroup

from cube.graph import IRGraph
from cube.ir.operator import IRFwOperation
from cube.graph.function.anchor import IRGraphAnchor

import cupilot
from cupilot.parallel.spmd import tensor_parallelism, replicate

import argparse
parser = argparse.ArgumentParser(parents=[cupilot.build_parser()],
                                 description='GPT Train')
parser.add_argument('--fp16', action='store_true', default=False)
parser.add_argument('--mbs', type=int, default=1, help='micro-batch size')
parser.add_argument('--gbs', type=int, default=256, help='global batch size')
# arch
parser.add_argument('--layers', type=int, required=True, help='number of encoder / decoder layers')
parser.add_argument('--hidden', type=int, required=True)
parser.add_argument('--heads', type=int, required=True)
parser.add_argument('--seqlen', type=int, required=True)
parser.add_argument('--vocab', type=int, default=51200)
# policy
parser.add_argument('--policy', type=str, choices=['alpa', 'cupilot', 'megatron'], required=True)
# log save
parser.add_argument('--save-spec', type=str, default=None,
                    help='folder for save searched results.')
parser.add_argument('--load-spec', type=str, default=None,
                    help='load searched tetris schedule from file')
args = parser.parse_args()


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
    if args.policy != 'megatron':
        print_each_rank(config, rank_only=0)

    if args.policy == 'alpa':
        policy = partial(cupilot.policy,
                         nmicros=args.gbs//args.mbs,
                         mbs=args.mbs,
                         load_spec_file=args.load_spec,
                         save_spec_file=args.save_spec,
                         config=config)

    elif args.policy == 'cupilot':

        def constraint_fn(graph: IRGraph, resource):
            nodes = graph.nodes()
            embeds = graph.select(name='embedding')

            start, stop = nodes.index(embeds[0]), nodes.index(embeds[-1])
            segment = graph.group(nodes[start:stop+1])
            cupilot.mark_standalone(segment)

            devices = list(range(resource.ngpus))
            for node in segment.nodes():
                if node.name == 'embedding':
                    tensor_parallelism(graph, node, idx=1, dim=0, devs=devices)
                else:
                    replicate(graph, node, devices)

            return graph

        policy = partial(cupilot.policy,
                         nmicros=args.gbs//args.mbs,
                         mbs=args.mbs,
                         constrain_fn=constraint_fn,
                         load_spec_file=args.load_spec,
                         save_spec_file=args.save_spec,
                         config=config)
    
    elif args.policy == 'megatron':
        pass
    
    else:
        raise ValueError(f"{args.policy}: policy not found")
    return policy


def train():

    # setup model arg
    cfg = Config(
        args.hidden,
        args.layers,
        args.heads,
        args.hidden * 4,
        args.vocab,
        args.seqlen
    )
    assert args.hidden % args.heads == 0

    print_each_rank(f"{cfg}", rank_only=0)

    if DeviceGroup().local_rank == 0:
        model = GPT(cfg)
        model = model.half() if args.fp16 else model
    else:
        model = None
    dataloader = get_gpt_dummy_dataloader(args.mbs, cfg)

    if torch.distributed.get_rank() == 0:
        nparams = 0
        for param in model.parameters():
            nparams += param.nelement()
        print(f'full model parameter: {nparams}')

    policy = setup_policy(args)

    @cube.compile(model, dataloader, PAS=policy, load_content=False)
    def train_iter(model, dataloader):
        datas = next(dataloader)
        loss = model(*datas)
        loss.backward()
        # return loss
    model = cube.load_model(load_content=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.999))

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

    CudaTimer().stop('e2e')
    print_each_rank('e2e time (ms) per iteration: {} ms'.format(
        CudaTimer().duration(iter_num-warmup, field_name='e2e')))
    CudaTimer().print_all(times=iter_num-warmup)
    memory_summary()


if __name__ == '__main__':

    train()