#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""
PYTHONPATH=.:$PYTHONPATH OMP_NUM_THREADS=4 torchrun  \
    --nproc_per_node=1  \
    genmodel/gen_mlp.py --policy dp \
        --dim 1024 \
        --layers 10 \
        --dp_size 1 \
        --pp_size 1 \
        --tp_size 1 \
        --gbs 1024 \
        --mbs 1024

PYTHONPATH=.:$PYTHONPATH OMP_NUM_THREADS=4 torchrun  \
    --nproc_per_node=8  \
    genmodel/gen_mlp.py --policy tp \
        --dim 1024 \
        --layers 10 \
        --dp_size 1 \
        --pp_size 1 \
        --tp_size 8 \
        --gbs 1024 \
        --mbs 1024

PYTHONPATH=.:$PYTHONPATH OMP_NUM_THREADS=4 torchrun  \
    --nproc_per_node=8  \
    genmodel/gen_mlp.py --policy hybrid \
        --dim 1024 \
        --layers 10 \
        --dp_size 2  \
        --pp_size 2 \
        --tp_size 2 \
        --gbs 1024 \
        --mbs 256

        
PYTHONPATH=.:$PYTHONPATH OMP_NUM_THREADS=4 torchrun  \
    --nproc_per_node=1  \
    genmodel/gen_mlp.py --policy dp \
        --dim 1024 \
        --layers 10 \
        --dp_size 1  \
        --pp_size 1 \
        --tp_size 1 \
        --gbs 1024 \
        --mbs 1024


PYTHONPATH=.:$PYTHONPATH OMP_NUM_THREADS=4 torchrun  \
    --nproc_per_node=8  \
    genmodel/gen_mlp.py --policy hybrid \
        --dim 1024 \
        --layers 2 \
        --dp_size 2  \
        --pp_size 2 \
        --tp_size 2 \
        --gbs 16 \
        --mbs 4

PYTHONPATH=.:$PYTHONPATH OMP_NUM_THREADS=4 torchrun  \
    --nproc_per_node=1  \
    genmodel/gen_mlp.py --policy dp \
        --dim 1024 \
        --layers 1 \
        --dp_size 1  \
        --pp_size 1 \
        --tp_size 1 \
        --gbs 16 \
        --mbs 16

PYTHONPATH=.:$PYTHONPATH OMP_NUM_THREADS=4 torchrun  \
    --nproc_per_node=3  \
    genmodel/gen_mlp.py --policy tp \
        --layers 2 \
        --dp_size 1 \
        --pp_size 1 \
        --tp_size 3 \
        --gbs 16 \
        --mbs 16

PYTHONPATH=.:$PYTHONPATH OMP_NUM_THREADS=4 torchrun  \
    --nproc_per_node=4  \
    genmodel/gen_mlp.py --policy hybrid \
        --layers 2 \
        --dp_size 1 \
        --pp_size 2 \
        --tp_size 2 \
        --gbs 16 \
        --mbs 16

PYTHONPATH=.:$PYTHONPATH OMP_NUM_THREADS=4 torchrun  \
    --nproc_per_node=2  \
    genmodel/gen_mlp.py --policy tp \
        --layers 2 \
        --dp_size 1 \
        --pp_size 1 \
        --tp_size 2 \
        --gbs 16 \
        --mbs 16
"""

import torch

import nnscaler
import examples.vision.swin.policy.gallery as gallery
from nnscaler.parallel import parallelize, ComputeConfig

from genmodel.model.mlp import MLP


import argparse
parser = argparse.ArgumentParser(description='MLP example')
parser.add_argument('--policy', type=str, help='policy choice, starting with "PAS"')
parser.add_argument('--dim', type=int, default=1024, help='model hidden size')
parser.add_argument('--layers', type=int, default=16, help='number of linear layers')
parser.add_argument('--gbs', type=int, default=4, help='global batch size')
parser.add_argument('--mbs', type=int, default=4, help='micro batch size')
parser.add_argument('--fp16', action='store_true', default=False, help='use fp16 for the training')
parser.add_argument('--dp_size', type=int, default=1, help='size of data parallelism')
parser.add_argument('--pp_size', type=int, default=1, help='size of pipeline parallelism')
parser.add_argument('--tp_size', type=int, default=1, help='size of tensor parallelism')
parser.add_argument('--zero', action='store_true', default=False, help='use zero1 for the training')
args = parser.parse_args()


nnscaler.init()
if torch.distributed.get_world_size() != args.dp_size * args.pp_size * args.tp_size:
    raise ValueError('world size should be equal to dp_size * pp_size * tp_size')
if args.gbs % args.mbs != 0:
    raise ValueError('global batch size should be divisible by micro batch size')


# model
model = MLP(dim=args.dim, nlayers=args.layers)

# dummy_input
def dummy_data():
    return torch.randn(
        args.mbs, args.dim, device=torch.cuda.current_device())
dummy_input = {"data": dummy_data()}

# get policy
policy_name = 'pas_' + args.policy
if policy_name in gallery.__dict__:
    policy = gallery.__dict__[policy_name]
else:
    policy = args.policy # use the builtin policies

# compute_config
compute_config=ComputeConfig(
    plan_ngpus=args.pp_size * args.tp_size,
    runtime_ngpus=torch.distributed.get_world_size(),
    use_zero=args.zero,
    use_end2end=True,
    constant_folding=True,
    use_pipeline=args.pp_size > 1,
    pipeline_nmicros=args.gbs // args.dp_size // args.mbs,
    pipeline_nstages=args.pp_size,
    pas_config={
        # customized settings that can affect code generation.
        '_pas_name': args.policy,
        '_gbs': args.gbs,
        '_pp_size': args.pp_size,
        '_tp_size': args.tp_size,
        '_dp_size': args.dp_size,
        # for autodist only
        'update_freq': args.gbs // args.mbs,
        'use_fp16': args.fp16,
    },
    user_config={
        'mbs': args.mbs,
        'fp16': args.fp16,
    }
)


# parallelization
pmodel = parallelize(
    module_or_module_class=model,
    dummy_input=dummy_input,
    pas_policy=policy,
    compute_config=compute_config,
    gen_savedir='./.nnscaler',
    reuse="override",
    # instance_name: Optional[str] = None,
    # load_module: bool = True,
    # module_dtype:  Optional[torch.dtype] = None,
    # module_fn: Optional[Callable[[], torch.nn.Module]] = None,
    # init_module_params: bool = True,
    # broadcast_strategy: Union[str, BroadcastGenFilesStrategy] = 'none',
)


import shutil, os

dp = args.dp_size
pp = args.pp_size
tp = args.tp_size
gbs = args.gbs
mbs = args.mbs
dim = args.dim
layers = args.layers
nm = gbs//dp//mbs if args.policy in ["hybrid", "pp"] else 1
fname = f"mlp_mgener_dp{args.dp_size}_pp{args.pp_size}_tp{args.tp_size}_nm{nm}_gbs{gbs}_dim{dim}_ly{layers}"

file = "mgener.pkl"
dst = f"genmodel/mgeners/{fname}.pkl"
try:
    shutil.move(file, dst)
    print("MGENER:", dst)
except:
    pass