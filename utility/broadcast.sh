#!/bin/bash
# broadcast the folder to all workers under the same workspace

set -ex

WORKSPACE=/workspace
FOLDER=MagicCube

WORKER_PREFIX=node-
WORKER_NUM=2

for ((i=1; i<=${WORKER_NUM}; i++)); do
    WORKER=${WORKER_PREFIX}${i}
    scp -r ${WORKSPACE}/${SYNC_FOLDER} ${WORKER}:${WORKSPACE}
done
