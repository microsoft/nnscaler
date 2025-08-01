#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""
PYTHONPATH=..:$PYTHONPATH OMP_NUM_THREADS=4 torchrun  \
    --nproc_per_node=8  \
    gen_model/gen_llama3_default.py --policy hybrid \
        --layers 2 \
        --hidden 1024\
        --heads 32\
        --dp_size 2 \
        --pp_size 2 \
        --tp_size 2 \
        --gbs 16 \
        --mbs 4 
PYTHONPATH=..:$PYTHONPATH OMP_NUM_THREADS=4 torchrun  \
    --nproc_per_node=1  \
    gen_model/gen_llama3_default.py --policy dp \
        --layers 2 \
        --hidden 1024\
        --heads 32\
        --dp_size 1 \
        --pp_size 1 \
        --tp_size 1 \
        --gbs 16 \
        --mbs 4 
"""

import sys

sys.path.append("..")
sys.setrecursionlimit(10000)

import os
import shutil
import torch

from Verdict.gen_model.model.llama3 import Transformer, ModelArgs


import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="llama3")
    parser.add_argument("--policy", type=str, help='policy choice, starting with "PAS"')
    parser.add_argument(
        "--dp_size", type=int, default=1, help="size of data parallelism"
    )
    parser.add_argument(
        "--pp_size", type=int, default=1, help="size of pipeline parallelism"
    )
    parser.add_argument(
        "--tp_size", type=int, default=1, help="size of tensor parallelism"
    )
    parser.add_argument(
        "--zero", action="store_true", default=False, help="use zero1 for the training"
    )
    parser.add_argument("--mbs", type=int, default=4, help="micro batch size")
    parser.add_argument(
        "--fp16", action="store_true", default=False, help="use fp16 for the training"
    )

    parser.add_argument("--gbs", type=int, help="global batch size")
    parser.add_argument(
        "--layers",
        type=int,
        help="number of transformer layers",
    )
    parser.add_argument("--hidden", type=int, help="hidden size")
    parser.add_argument(
        "--heads",
        type=int,
        help="number of attention heads",
    )
    parser.add_argument("--seqlen", type=int, help="sequence length")
    args = parser.parse_args()
    return args


def render_model_args(args):
    model_args = ModelArgs()
    model_args.max_batch_size = (
        args.mbs if args.pp_size > 1 else args.gbs // args.dp_size
    )
    model_args.n_layers = args.layers or model_args.n_layers
    model_args.dim = args.hidden or model_args.dim
    model_args.n_heads = args.heads or model_args.n_heads
    model_args.max_seq_len = args.seqlen or model_args.max_seq_len
    return model_args


def create_path(args, model_args: ModelArgs):
    dp = args.dp_size
    pp = args.pp_size
    tp = args.tp_size
    gbs = args.gbs
    mbs = model_args.max_batch_size
    layers = model_args.n_layers
    hi = model_args.dim
    h = model_args.n_heads
    sql = model_args.max_seq_len
    nm = gbs // dp // mbs if args.policy in ["hybrid"] else 1
    strategy = "default"
    fname = f"llama3_{strategy}_dp{dp}_pp{pp}_tp{tp}_nm{nm}_gbs{gbs}_ly{layers}_h{h}_hi{hi}_sq{sql}"
    dst = f"gen_model/mgeners/{fname}.pkl"
    return dst


def gen_model(args, model_args, dst):
    import nnscaler
    import examples.vision.swin.policy.gallery as gallery
    from nnscaler.parallel import parallelize, ComputeConfig
    from fairscale.nn.model_parallel.initialize import initialize_model_parallel

    dp = args.dp_size
    pp = args.pp_size
    tp = args.tp_size
    gbs = args.gbs
    mbs = model_args.max_batch_size
    nm = gbs // dp // mbs if args.policy in ["hybrid"] else 1
    sql = model_args.max_seq_len
    ngpus = dp * pp * tp
    policy_name = "pas_" + args.policy

    # initialize
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group("nccl")
    model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
    initialize_model_parallel(model_parallel_size)

    # sanity check
    nnscaler.init()
    # if torch.distributed.get_world_size() != ngpus:
    #     raise ValueError("world size should be equal to dp_size * pp_size * tp_size")
    if gbs % mbs != 0:
        raise ValueError("global batch size should be divisible by micro batch size")

    # model
    model = Transformer(model_args)
    model = model if not args.fp16 else model.half()

    # dummy_input
    input_ids = torch.randint(1, 1000, size=(mbs, sql), dtype=torch.int64)
    position_ids = 0
    dummy_input = {"tokens": input_ids, "start_pos": position_ids}

    # get policy
    if policy_name in gallery.__dict__:
        policy = gallery.__dict__[policy_name]
    else:
        policy = args.policy  # use the builtin policies

    # compute_config
    compute_config = ComputeConfig(
        plan_ngpus=pp * tp,
        # runtime_ngpus=torch.distributed.get_world_size(),
        runtime_ngpus=ngpus,
        use_zero=args.zero,
        use_end2end=True,
        constant_folding=True,
        use_pipeline=pp > 1,
        pipeline_nmicros=nm,
        pipeline_nstages=pp,
        pas_config={
            # customized settings that can affect code generation.
            "_pas_name": args.policy,
            "_gbs": gbs,
            "_pp_size": pp,
            "_tp_size": tp,
            "_dp_size": dp,
        },
        user_config={
            "mbs": mbs,
        },
    )
    # print(compute_config)
    # print(input_ids.shape)
    # exit(0)

    # parallelization
    pmodel = parallelize(
        module_or_module_class=model,
        dummy_input=dummy_input,
        pas_policy=policy,
        compute_config=compute_config,
        gen_savedir="./.nnscaler",
        reuse="override",
        # instance_name: Optional[str] = None,
        load_module=False,
        # module_dtype:  Optional[torch.dtype] = None,
        # module_fn: Optional[Callable[[], torch.nn.Module]] = None,
        # init_module_params: bool = True,
        # broadcast_strategy: Union[str, BroadcastGenFilesStrategy] = 'none',
    )

    file = "mgener.pkl"
    try:
        shutil.move(file, dst)
    except:
        pass


if __name__ == "__main__":
    args = parse_arguments()
    model_args = render_model_args(args)
    dst = create_path(args, model_args)

    print(f"👉 Model destination: {dst}")
    if os.path.exists(dst):
        print(f"💾 Using cached model.")
    else:
        print(f"🏎️ Start generating NNScaler model...")
        gen_model(args, model_args, dst)
    print(f"✅ Model ready.")
