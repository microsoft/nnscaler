from typing import List

import logging
import torch

import cube
from cube.profiler.timer import CudaTimer, print_each_rank
from cube.profiler.memory import memory_summary


from examples.t5.parallel_ds import Config, T5, get_t5_dummy_dataloader, MPU
from examples.adam import Adam

import argparse
parser = argparse.ArgumentParser(description='T5 Train')
parser.add_argument('--mbs', type=int, default=1, help='micro-batch size')
parser.add_argument('--gbs', type=int, default=256, help='global batch size')
# arch
parser.add_argument('--layers', type=int, required=True, help='number of encoder / decoder layers')
parser.add_argument('--hidden', type=int, required=True)
parser.add_argument('--heads', type=int, required=True)
parser.add_argument('--seqlen', type=int, required=True)
parser.add_argument('--vocab', type=int, required=True)
parser.add_argument('--cpu-offload', action='store_true', default=False)

parser.add_argument('--tp', type=int, default=1,
                    help='number of stages for pipeline parallelism')
parser.add_argument('--offload-param', type=str, default="none",
                    help="offload parameters into a device, can be cpu or none. Only used in zero 3")
parser.add_argument('--offload-opt', type=str, default="none",
                    help="offload optimizer into a device, can be cpu or none. Used in zero 2 and 3")
args = parser.parse_args()

cube.set_logger_level(logging.WARN)
for name in logging.root.manager.loggerDict:
    if name.startswith('cupilot'):
        logging.getLogger(name).setLevel(logging.INFO)
logging.getLogger('cupilot.solver.spmd').setLevel(logging.DEBUG) 
logging.getLogger('cube.compiler').setLevel(logging.INFO)

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
            "device": args.offload_param
        },
        "offload_optimizer": {  # zero-2 and zero-3
            "device": args.offload_opt
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

    model = T5(cfg).half()
    # with torch.no_grad():
    #     for p in model.parameters():
    #         torch.nn.init.normal_(p, mean=0.0, std=0.02)
    model, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        dist_init_required=False,
        config=zero3_config,
        mpu=mpu
    )
    grad_accum = model.gradient_accumulation_steps()
    print_each_rank(f"DeepSpeed engine created, grad accum times: {grad_accum}")

    dataloader = get_t5_dummy_dataloader(args.mbs, cfg)

    if torch.distributed.get_rank() == 0:
        nparams = 0
        for param in model.parameters():
            nparams += param.nelement()
        print(f'full model parameter: {nparams}')
        print(f'TFLOPs: {round(model.flops(args.gbs) / 1e12, 2)}')


    def train_iter(model, dataloader):
        datas = next(dataloader)
        loss = model(*datas)
        model.backward(loss)
        return loss
    
    # optimizer = Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.999))

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
    print_each_rank('e2e time (ms) per iteration: {} ms'.format(
        CudaTimer().duration(iter_num-warmup, field_name='e2e')))
    CudaTimer().print_all(times=iter_num-warmup)
    memory_summary()


if __name__ == '__main__':

    train()