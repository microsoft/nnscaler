#!/bin/bash

LOG_DIR=logs
LOGS=${LOG_DIR}/swin
mkdir -p $LOGS

echo NODE_RANK=$NODE_RANK
echo MASTER_ADDR=$MASTER_ADDR

GPU=V100
NGPUS=16
TOTAL_GPUS=32

echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export PYTHONPATH=.:$PYTHONPATH
export OMP_NUM_THREADS=4

# cube flags
export USE_JIT_PARSER=1
export DISABLE_INTER_RVD=1

GBS=256 # global batch size

LAYERS=56
HEADS=32
HIDDEN=1536

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# =============================================================================

POLICY=cupilot

torchrun --nproc_per_node=$NGPUS --nnodes=2 \
    --master_addr=$MASTER_ADDR --node_rank=$NODE_RANK \
    examples/swin/train.py \
        --mbs 1 --gbs $GBS --policy $POLICY \
        --layers $LAYERS --hidden $HIDDEN --heads $HEADS --img-size 1536 --window-size 48 \
        --max-pp 4 \
        --recompute --db-cache swin_${GPU}_db.json --save-spec swin.cupilot.16gpus.json \
    2>&1 | tee ${LOGS}/${TOTAL_GPUS}gpus.$POLICY.layer${LAYERS}.hidden${HIDDEN}.heads${HEADS}.log

# ========================================================================================

GBS=16
POLICY=alpa
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

torchrun --nproc_per_node=$NGPUS --nnodes=2 \
    --master_addr=$MASTER_ADDR --node_rank=$NODE_RANK \
    examples/swin/train.py \
        --mbs 1 --gbs $GBS --policy $POLICY \
        --layers $LAYERS --hidden $HIDDEN --heads $HEADS --img-size 1536 --window-size 48 \
        --max-pp 1 \
        --recompute --db-cache swin_${GPU}_db.json --save-spec swin.alpa.16gpus.json \
    2>&1 | tee ${LOGS}/${TOTAL_GPUS}gpus.$POLICY.layer${LAYERS}.hidden${HIDDEN}.heads${HEADS}.log

# deepspeed

torchrun --nproc_per_node=$NGPUS --nnodes=2 \
    --master_addr=$MASTER_ADDR --node_rank=$NODE_RANK \
    examples/swin/train_ds.py \
        --mbs 1 --gbs $GBS \
        --layers $LAYERS --hidden $HIDDEN --heads $HEADS --img-size 1536 --window-size 48 \
        --tp 8 --offload cpu \
    2>&1 | tee ${LOGS}/${TOTAL_GPUS}gpus.ds.layer${LAYERS}.hidden${HIDDEN}.heads${HEADS}.log
