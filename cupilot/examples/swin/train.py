from typing import List
import logging
from functools import partial
import torch
import more_itertools as mitr
import math

from examples.swin.model import SwinTransformer, build_config
from examples.swin.model import create_swin_dummy_dataloader
from examples.swin.blocks import init_relative_position_index
from examples.adam import Adam

import cube
from cube.profiler.timer import CudaTimer, print_each_rank
from cube.profiler.memory import memory_summary
from cube.runtime.device import DeviceGroup
from cube.graph import IRGraph

import cupilot
from cupilot.constraints import Constraints
from cupilot.estimator.profiler import Estimator
from cupilot.parallel.spmd import (
    nested_tensor_parallelism, tensor_parallelism, replicate
)
from cupilot.solver.block import IRBlock

# this node should be skipped as the profiler cannot get correct index range.
# Estimator.register_skip_node('get_position_bias')
Estimator.register_skip_node('view')
Estimator.register_skip_node('add')
Estimator.register_rule('window_attn', 1, 0, 4)


import argparse
parser = argparse.ArgumentParser(parents=[cupilot.build_parser()], description='SwinTransformer Pretraining')
parser.add_argument('--mbs', type=int, default=1, help='micro-batch size')
parser.add_argument('--gbs', type=int, default=256, help='global batch size')
# arch
parser.add_argument('--hidden', type=int, required=True)
parser.add_argument('--heads', type=int, required=True)
parser.add_argument('--layers', type=int, required=True)
parser.add_argument('--img-size', type=int, required=True)
parser.add_argument('--window-size', type=int, required=True)
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
# logging.getLogger('cube.algorithm.ops.dimops').setLevel(logging.INFO)


def setup_policy(args):

    config = cupilot.build_config(parser)
    if args.policy != 'megatron':
        print_each_rank(config, rank_only=0)

    if args.policy == 'alpa':
        
        def constrain_fn(graph: IRGraph, resource,
                         blocks: List[IRBlock], constraints: Constraints):
            """Due to memory fragmentation, we constrain the first 4 transformer layers"""
            head = blocks.pop(0)
            blocks[0] = IRBlock.merge([head, blocks[0]])
            for blk in blocks[:4]:
                blk.min_tp = 2
            return blocks

        policy = partial(cupilot.policy,
                         nmicros=args.gbs//args.mbs,
                         mbs=args.mbs,
                         constrain_fn=constrain_fn,
                         load_spec_file=args.load_spec,
                         save_spec_file=args.save_spec,
                         config=config)

    elif args.policy == 'cupilot':

        def constrain_fn(graph: IRGraph, resource, 
                         blocks: List[IRBlock], constraints: Constraints):
            """apply coshard to attention and feedforward"""
            if DeviceGroup().world_size == 4:
                tp_size, colocate = 2, 4
            if DeviceGroup().world_size == 8:
                tp_size, colocate = 4, 6
            if DeviceGroup().world_size == 16:
                tp_size, colocate = 4, 4
            if DeviceGroup().world_size == 32:
                tp_size, colocate = 8, 4
            devices = [[i] * colocate for i in range(tp_size)]
            devices = tuple(mitr.flatten(devices))
        
            # we only consider the first 4 transformer layers (including head)
            blocks = [IRBlock.merge(blocks[:5])] + blocks[5:]
            for node in blocks[0].nodes:
                if node.name in ('window_attn', 'feedforward'):
                    if node.recompute is None:
                        graph.recompute([node])
                    constraints.add_trans_constraints(node, algo=(1, 0), num=tp_size*colocate)
                    constraints.add_place_constraints(node, devices=devices)

            return blocks

        policy = partial(cupilot.policy,
                         nmicros=args.gbs//args.mbs,
                         mbs=args.mbs,
                         constrain_fn=constrain_fn,
                         load_spec_file=args.load_spec,
                         save_spec_file=args.save_spec,
                         config=config)
    
    elif args.policy == 'megatron':
        def constrain_fn(graph: IRGraph, resource, 
                         blocks: List[IRBlock], constraints: Constraints):

            from cube.ir.operator import IRFwOperation
            for node in graph.select(ntype=IRFwOperation):
                tp_size = resource.ngpus // args.max_pp
                if node.name in ('embedding', 'linear'):
                    constraints.add_trans_constraints(node, (1,0), tp_size)
                if node.name == 'window_attn':
                    constraints.add_trans_constraints(node, (1,0), tp_size)
                if node.name == 'feedforward':
                    constraints.add_trans_constraints(node, (1,0), tp_size)

            head, transformers = blocks[0], blocks[1:]
            transformers[0] = IRBlock.merge([head, transformers[0]])
            blocks = [IRBlock.merge(list(sub_blks)) for sub_blks in mitr.divide(args.max_pp, transformers)]
            return blocks
            
        policy = partial(cupilot.policy,
                         nmicros=args.gbs//args.mbs,
                         mbs=args.mbs,
                         constrain_fn=constrain_fn,
                         load_spec_file=args.load_spec,
                         save_spec_file=args.save_spec,
                         config=config)
    
    else:
        raise ValueError(f"{args.policy}: policy not found")
    return policy


def train():

    cfg = build_config(args.layers, args.hidden, args.heads,
                       args.img_size, args.window_size)
    print_each_rank(f"{cfg}", rank_only=0)

    if DeviceGroup().local_rank == 0:
        print('initing model...')
        model = SwinTransformer(cfg).half()
        flops = model.flops(args.gbs)
        print(f'flops: {flops}')
    else:
        model = None
    dataloader = create_swin_dummy_dataloader(
        cfg, args.mbs, torch.float16)

    if torch.distributed.get_rank() == 0:
        nparams = 0
        for param in model.parameters():
            nparams += param.nelement()
        print(f'full model parameter: {nparams}')

    policy = setup_policy(args)

    load_content = False

    @cube.compile(model, dataloader, PAS=policy, load_content=load_content, override=True)
    def train_iter(model, dataloader):
        images = next(dataloader)
        loss = model(images)
        loss.backward()
        # return loss
    model = cube.load_model(load_content=False)

    if not load_content:
        for name, buffer in model.named_buffers():
            if 'rp_index' in name:
                window_size = int(math.sqrt(buffer.size(0)))
                buffer.copy_(init_relative_position_index(window_size).cuda())

    # adam optimizer for mixed precision training
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
        
        torch.cuda.empty_cache()

    CudaTimer().stop('e2e')
    e2e = CudaTimer().duration(iter_num-warmup, field_name='e2e')
    wps = args.gbs / (e2e / 1000)
    if DeviceGroup().local_rank == 0:
        tflops = round(flops / 1e12 / (e2e / 1000), 2)  # aggregated tera FLOPS
    else:
        tflops = '--'
    print_each_rank(f'e2e time per iteration: {round(e2e,2)} ms | {round(wps,2)} images/s | TFLOPS: {tflops}')
    CudaTimer().print_all(times=iter_num-warmup)
    memory_summary()


if __name__ == '__main__':

    train()