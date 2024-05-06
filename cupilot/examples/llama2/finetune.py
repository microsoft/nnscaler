from typing import List
import logging
from functools import partial
import torch
import more_itertools as mitr

from examples.llama2.model import LlamaForCausalLM
from examples.llama2.model import build_llama_config
from examples.llama2.model import create_llama_dummy_dataloader
from examples.adam import Adam

import cube
from cube.profiler.timer import CudaTimer, print_each_rank
from cube.profiler.memory import memory_summary
from cube.runtime.device import DeviceGroup

from cube.graph import IRGraph

import cupilot
from cupilot.constraints import Constraints
from cupilot.estimator.profiler import Estimator
from cupilot.solver.block import IRBlock

# Estimator.register_rule('attention', 1, 0, 4)
# Estimator.register_rule('mlp', 0, 1, 4)
Estimator.register_rule('linear', 1, 0, 4)
Estimator.register_rule('compute_loss', 0, 1, 4)


import argparse
parser = argparse.ArgumentParser(parents=[cupilot.build_parser()], description='Llama2 Finetune')
parser.add_argument('--mbs', type=int, default=1, help='micro-batch size')
parser.add_argument('--gbs', type=int, default=256, help='global batch size')
# arch
parser.add_argument('--arch', type=str, required=True, choices=['1.3b', '7b', '13b', '34b'],
                    help='number of encoder / decoder layers')
parser.add_argument('--vocab', type=int, default=32000,
                    help='vocab size')
parser.add_argument('--seqlen', type=int, default=2048,
                    help='sequence (context) length')
# policy
parser.add_argument('--policy', type=str,
                    choices=['alpa', 'cupilot', 'megatron'], required=True)
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

# always use fp16 for computation
torch.set_default_dtype(torch.float16)

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

        def constraint_fn(graph: IRGraph, resource,
                          blocks: List[IRBlock], constraints: Constraints):
            """apply coshard to attention and feedforward"""
            tp_size = 4
            nstages = resource.ngpus // tp_size

            embed, transformers, tail = blocks[0], blocks[1:-1], blocks[-1]
            assert len(transformers) % nstages == 0
            # add staged_spmd search constraints: the stage can only be uniformly grouped
            transformers = [IRBlock.merge(blks) for blks in mitr.divide(nstages, transformers)]
            transformers[0] = IRBlock.merge([embed, transformers[0]])
            transformers[-1] = IRBlock.merge([transformers[-1], tail])
            blocks = transformers
            return blocks

        policy = partial(cupilot.policy,
                         nmicros=args.gbs//args.mbs,
                         mbs=args.mbs,
                         constrain_fn=constraint_fn,
                         load_spec_file=args.load_spec,
                         save_spec_file=args.save_spec,
                         config=config)
    
    elif args.policy == 'megatron':
        from examples.llama2.policy.megatron import megatron_policy
        policy = partial(megatron_policy,
                         nmicros=args.gbs//args.mbs,
                         pp_size=config.max_pp_size,
                         tp_size=config.max_tp_size,
                         recompute=config.recompute)
    
    else:
        raise ValueError(f"{args.policy}: policy not found")
    return policy


def train():

    cfg = build_llama_config(args.arch, args.seqlen)
    print_each_rank(f"{cfg}", rank_only=0)

    if DeviceGroup().local_rank == 0:
        print('initing model...')
        model = LlamaForCausalLM(cfg).half()
        flops = model.flops(args.gbs)
    else:
        model = None
    dataloader = create_llama_dummy_dataloader(cfg, args.mbs)

    if torch.distributed.get_rank() == 0:
        nparams = 0
        for param in model.parameters():
            nparams += param.nelement()
        print(f'full model parameter: {nparams}')

    curr_mem = round(torch.cuda.memory_allocated() / (1 << 30), 2)
    print_each_rank(f'current memory: (data) {curr_mem} GB')

    policy = setup_policy(args)

    @cube.compile(model, dataloader, PAS=policy, load_content=False)
    def train_iter(model, dataloader):
        datas = next(dataloader)
        loss = model(*datas)
        loss.backward()
        # return loss
    model = cube.load_model(load_content=False)

    curr_mem = round(torch.cuda.memory_allocated() / (1 << 30), 2)
    print_each_rank(f'current memory: (data+model) {curr_mem} GB')

    # adam optimizer for mixed precision training
    # optimizer = Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.999))
    optimizer = Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.999))

    torch.distributed.barrier()

    nparams = 0
    for param in model.parameters():
        nparams += param.nelement()
    print_each_rank(f'loaded model parameter: {nparams}')
    memory_summary()

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

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        if step == 0:
            print_each_rank('passed first iteration')
        if (step + 1) % 2 == 0:
            print_each_rank(f'iter [{step + 1}/{iter_num}]', rank_only=0)

    CudaTimer().stop('e2e')
    e2e = CudaTimer().duration(iter_num-warmup, field_name='e2e')
    if DeviceGroup().local_rank == 0:
        tflops = round(flops / 1e12 / (e2e / 1000), 2)  # aggregated tera FLOPS
    else:
        tflops = '--'
    wps = args.seqlen * args.gbs / (e2e / 1000)
    print_each_rank(f'e2e time per iteration: {round(e2e,2)} ms | {round(wps,2)} wps | tflops: {tflops}')
    CudaTimer().print_all(times=iter_num-warmup)
    memory_summary()
    print_each_rank(f'current memory: (param + opt) {torch.cuda.memory_allocated() / (1 << 30)} GB')


if __name__ == '__main__':

    train()