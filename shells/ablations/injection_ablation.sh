#!/bin/bash
# Injection-mode ablation: r=4 loop, s=10, budget=1e18, sweep input injection.
# Array job: task 0 -> passthrough, task 1 -> additive.
#
# Usage:
#   ./shells/_submit.sh shells/hyperparams/injection_ablation.sh -- --array=0-1

cd "${REPO_ROOT:-$HOME/looped-lm-scaling}"
uv sync
source .venv/bin/activate

source "${MACHINE_CONFIG:-shells/_machine_config.sh}"
validate_config || exit 1

NPROC_PER_NODE=${NUM_GPUS:-1}

TARGET_FLOPS=1e18
SIZE=10
NUM_RECUR=4
N_PRELUDE=2
N_RECUR_BLOCK=4
N_CODA=2
MATRIX_LR=0.014
DEVICE_BATCH_SIZE=${DEVICE_BATCH_SIZE:-32}

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
INJECTIONS=(passthrough additive)
INJECTION=${INJECTIONS[$TASK_ID]}

TAG="inject_${INJECTION}_${TARGET_FLOPS}_loop_s${SIZE}"

torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train \
    --target-flops $TARGET_FLOPS \
    --model-tag $TAG \
    --run $TAG \
    --device-batch-size $DEVICE_BATCH_SIZE \
    --eval-every=-1 \
    --core-metric-every=-1 \
    --sample-every=-1 \
    --save-every=${SAVE_EVERY:--1} \
    --num-recur=$NUM_RECUR \
    --bptt-k=$NUM_RECUR \
    --input-injection=$INJECTION \
    --n-prelude=$N_PRELUDE \
    --n-recur-block=$N_RECUR_BLOCK \
    --n-coda=$N_CODA \
    --matrix-lr=$MATRIX_LR \
    --size $SIZE
