from typing import List
import logging
from functools import partial
import torch
import more_itertools as mitr
import math

from examples.swin.parallel_ds import (
    SwinTransformer, build_config,
    create_swin_dummy_dataloader, init_relative_position_index,
    MPU
)
from examples.adam import Adam

import cube
from cube.profiler.timer import CudaTimer, print_each_rank
from cube.profiler.memory import memory_summary
from cube.runtime.device import DeviceGroup
from cube.graph import IRGraph

from cupilot.solver.block import IRBlock



import argparse
parser = argparse.ArgumentParser(description='SwinTransformer Pretraining')
parser.add_argument('--mbs', type=int, default=1, help='micro-batch size')
parser.add_argument('--gbs', type=int, default=256, help='global batch size')
# arch
parser.add_argument('--hidden', type=int, required=True)
parser.add_argument('--heads', type=int, required=True)
parser.add_argument('--layers', type=int, required=True)
parser.add_argument('--img-size', type=int, required=True)
parser.add_argument('--window-size', type=int, required=True)
# policy
parser.add_argument('--tp', type=int, default=1,
                    help='number of stages for pipeline parallelism')
parser.add_argument('--offload', type=str, default="none",
                    help="offload parameters and optimizer states into a device, can be cpu or none. Only used in zero 3")
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

import deepspeed

deepspeed.init_distributed()
cube.init()
print_each_rank(str(args), rank_only=0)
ngpus = torch.distributed.get_world_size()
mpu = MPU(dp=ngpus // args.tp, tp=args.tp)

zero3_config = {
    "train_batch_size": args.gbs,
    "train_micro_batch_size_per_gpu": args.mbs,
    "zero_optimization": {
        "stage": 3,
        "offload_param": { # zero-3 only
            "device": args.offload
        },
        "offload_optimizer": {  # zero-2 and zero-3
            "device": args.offload
        },
        "contiguous_gradients": False,
        "overlap_comm": False,
    },
    "mp_size": args.tp,
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 0.00015,
            "betas": [0.9, 0.95]
        }
    },
    "fp16": {
        "enabled": True,
        "autocast": False,
    },
    "wall_clock_breakdown": True,
    "steps_per_print": 1,
}

def train():

    cfg = build_config(args.layers, args.hidden, args.heads,
                       args.img_size, args.window_size)
    print_each_rank(f"{cfg}", rank_only=0)

    print('initing model...')
    model = SwinTransformer(cfg).half()
    flops = model.flops(args.gbs)
    print(f'flops: {flops}')

    dataloader = create_swin_dummy_dataloader(
        cfg, args.mbs, torch.float16)
    
    model, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        dist_init_required=False,
        config=zero3_config,
        mpu=mpu
    )
    grad_accum = model.gradient_accumulation_steps()
    print_each_rank(f"DeepSpeed engine created, grad accum times: {grad_accum}")

    if torch.distributed.get_rank() == 0:
        nparams = 0
        for param in model.parameters():
            nparams += param.nelement()
        print(f'full model parameter: {nparams}')

    def train_iter(model, dataloader):
        images = next(dataloader)
        loss = model(images)
        model.backward(loss)
        # return loss

    # if not load_content:
    #     for name, buffer in model.named_buffers():
    #         if 'rp_index' in name:
    #             window_size = int(math.sqrt(buffer.size(0)))
    #             buffer.copy_(init_relative_position_index(window_size).cuda())

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
        for _ in range(model.gradient_accumulation_steps()):
            train_iter(model, dataloader)

        model.step()

        if step == 0:
            print_each_rank('passed first iteration')
        if (step + 1) % 2 == 0:
            print_each_rank(f'iter [{step + 1}/{iter_num}]', rank_only=0)
        
        torch.cuda.synchronize()
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