#!/bin/bash
# gather the folder to all workers to node-0 under the same workspace

set -ex

WORKSPACE=/workspace
FOLDER=MagicCube

WORKER_PREFIX=node-
WORKER_NUM=2

for ((i=1; i<${WORKER_NUM}; i++)); do
    WORKER=${WORKER_PREFIX}${i}
    scp -r ${WORKER}:${WORKSPACE}/${FOLDER} ${WORKSPACE}/${FOLDER}-${WORKER}
done
