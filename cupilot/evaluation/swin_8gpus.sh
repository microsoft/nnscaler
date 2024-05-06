#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

LOG_DIR=logs
LOGS=${LOG_DIR}/swin
mkdir -p $LOGS

GPU=V100
NGPUS=8
TOTAL_GPUS=8

echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export PYTHONPATH=.:$PYTHONPATH
export OMP_NUM_THREADS=4

# cube flags
export USE_JIT_PARSER=1
export DISABLE_INTER_RVD=1
# export ASYNC_COMM=1

GBS=256 # global batch size


# ================================= swin =============================

LAYERS=40
HEADS=24
HIDDEN=768

POLICY=cupilot

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024

torchrun --nproc_per_node=$NGPUS \
    examples/swin/train.py \
        --mbs 1 --gbs $GBS --policy $POLICY \
        --layers $LAYERS --hidden $HIDDEN --heads $HEADS --img-size 1536 --window-size 48 \
        --max-pp 2 --dev0-mem-limit 23 \
        --recompute --db-cache swin_${GPU}_db.json --save-spec temp.json \
    2>&1 | tee ${LOGS}/${TOTAL_GPUS}gpus.$POLICY.layer${LAYERS}.hidden${HIDDEN}.heads${HEADS}.log

# ========================================================================================

POLICY=alpa

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

torchrun --nproc_per_node=$NGPUS \
    examples/swin/train.py \
        --mbs 1 --gbs $GBS --policy $POLICY \
        --layers $LAYERS --hidden $HIDDEN --heads $HEADS --img-size 1536 --window-size 48 \
        --max-pp 1 \
        --recompute --db-cache swin_${GPU}_db.json --save-spec temp.json \
    2>&1 | tee ${LOGS}/${TOTAL_GPUS}gpus.$POLICY.layer${LAYERS}.hidden${HIDDEN}.heads${HEADS}.log


# deepspeed

torchrun --nproc_per_node=$NGPUS \
    examples/swin/train_ds.py \
        --mbs 1 --gbs $GBS \
        --layers $LAYERS --hidden $HIDDEN --heads $HEADS --img-size 1536 --window-size 48 \
        --tp 4 --offload cpu \
    2>&1 | tee ${LOGS}/${TOTAL_GPUS}gpus.ds.layer${LAYERS}.hidden${HIDDEN}.heads${HEADS}.log

