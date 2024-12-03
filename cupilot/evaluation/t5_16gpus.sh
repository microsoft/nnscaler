#!/bin/bash

LOG_DIR=logs
LOGS=${LOG_DIR}/t5
mkdir -p $LOGS

GPU=V100
NGPUS=16
TOTAL_GPUS=16

echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export PYTHONPATH=.:$PYTHONPATH
export OMP_NUM_THREADS=4

# cube flags
export USE_JIT_PARSER=1
export ASYNC_COMM=0
export DISABLE_INTER_RVD=1

GBS=256 # global batch size


# ================================= T5 =============================

LAYERS=32
HEADS=64
HIDDEN=4096
VOCAB_K=1536
VOCAB=`expr ${VOCAB_K} \* 1000`

POLICY=cupilot

torchrun --nproc_per_node=$NGPUS \
    examples/t5/train.py \
        --mbs 1 --gbs $GBS --policy $POLICY \
        --layers $LAYERS --hidden $HIDDEN --heads $HEADS --seqlen 1024 --vocab $VOCAB \
        --recompute --db-cache t5_${GPU}_db.json --save-spec t5.cupilot.16gpus.json \
        --max-pp 4 --max-tp 4 --max-dp 1 \
        --order-plan mllm.4stages.sched.json \
    2>&1 | tee ${LOGS}/${TOTAL_GPUS}gpus.$POLICY.vocab${VOCAB_K}k.layer${LAYERS}.hidden${HIDDEN}.heads${HEADS}.log


sleep 10


POLICY=alpa

torchrun --nproc_per_node=$NGPUS \
    examples/t5/train.py \
        --mbs 1 --gbs $GBS --policy $POLICY \
        --layers $LAYERS --hidden $HIDDEN --heads $HEADS --seqlen 1024 --vocab $VOCAB \
        --max-pp 1 \
        --recompute --db-cache t5_${GPU}_db.json --save-spec t5.alpa.16gpus.json \
    2>&1 | tee ${LOGS}/${TOTAL_GPUS}gpus.$POLICY.vocab${VOCAB_K}k.layer${LAYERS}.hidden${HIDDEN}.heads${HEADS}.log
