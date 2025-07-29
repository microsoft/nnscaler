"""

PYTHONPATH=.:$PYTHONPATH OMP_NUM_THREADS=4 torchrun  \
    --nproc_per_node=1  \
    genmodel/gen_llama3.py --policy dp \
        --layers 1 \
        --hidden 4096\
        --heads 32\
        --dp_size 1 \
        --pp_size 1 \
        --tp_size 1 \
        --gbs 16 \
        --mbs 16 

"""
import sys
sys.setrecursionlimit(10000)

import os
import shutil
import torch

import nnscaler
import examples.vision.swin.policy.gallery as gallery
from nnscaler.parallel import parallelize, ComputeConfig

from genmodel.model.llama3 import Transformer, ModelArgs

from fairscale.nn.model_parallel.initialize import initialize_model_parallel

if not torch.distributed.is_initialized():
    torch.distributed.init_process_group("nccl")
model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
initialize_model_parallel(model_parallel_size)

model_args = ModelArgs()

import argparse
parser = argparse.ArgumentParser(description='llama3')
parser.add_argument('--policy', type=str,
                    help='policy choice, starting with "PAS"')
parser.add_argument('--dp_size', type=int, default=1,
                    help='size of data parallelism')
parser.add_argument('--pp_size', type=int, default=1,
                    help='size of pipeline parallelism')
parser.add_argument('--tp_size', type=int, default=1,
                    help='size of tensor parallelism')
parser.add_argument('--zero', action='store_true',
                    default=False, help='use zero1 for the training')
parser.add_argument('--mbs', type=int, default=4, help='micro batch size')
parser.add_argument('--fp16', action='store_true',
                    default=False, help='use fp16 for the training')

parser.add_argument('--gbs', type=int, default=model_args.max_batch_size, help='global batch size')
parser.add_argument('--layers', type=int, default=model_args.n_layers,
                    help='number of transformer layers')
parser.add_argument('--hidden', type=int, default=model_args.dim, help='hidden size')
parser.add_argument('--heads', type=int, default=model_args.n_heads,
                    help='number of attention heads')
parser.add_argument('--seqlen', type=int, default=model_args.max_seq_len, help='sequence length')
args = parser.parse_args()

model_args.max_batch_size = args.gbs
model_args.n_layers = args.layers
model_args.dim = args.hidden
model_args.n_heads = args.heads
model_args.max_seq_len = args.seqlen

nnscaler.init()
if torch.distributed.get_world_size() != args.dp_size * args.pp_size * args.tp_size:
    raise ValueError(
        'world size should be equal to dp_size * pp_size * tp_size')
if args.gbs % args.mbs != 0:
    raise ValueError(
        'global batch size should be divisible by micro batch size')

# model
model = Transformer(model_args)
model = model if not args.fp16 else model.half()

# dummy_input
input_ids = torch.randint(
            1, 1000, size=(model_args.max_batch_size, model_args.max_seq_len),
            dtype=torch.int64)
position_ids = 0
dummy_input = {"tokens": input_ids, "start_pos": position_ids}
# get policy
policy_name = 'pas_' + args.policy
if policy_name in gallery.__dict__:
    policy = gallery.__dict__[policy_name]
else:
    policy = args.policy  # use the builtin policies

# compute_config
compute_config = ComputeConfig(
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
    },
    user_config={
        'mbs': args.mbs,
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


dp = args.dp_size
pp = args.pp_size
tp = args.tp_size
gbs = args.gbs
mbs = args.mbs
layers = args.layers
hidden = args.hidden
heads = args.heads
seqlen = args.seqlen
nm = gbs//dp//mbs if args.policy in ["hybrid", "pp"] else 1
fname = f"llama3_mgener_dp{args.dp_size}_pp{args.pp_size}_tp{args.tp_size}_nm{nm}_gbs{gbs}_ly{layers}_h{heads}_hi{hidden}_sq{seqlen}"

file = "mgener.pkl"
dst = f"genmodel/mgeners/{fname}.pkl"
try:
    shutil.move(file, dst)
    print("MGENER:", dst)
except:
    pass
