#!/bin/bash

cd "${REPO_ROOT:-$HOME/looped-lm-scaling}"
uv sync
source .venv/bin/activate

source "${MACHINE_CONFIG:-shells/_machine_config.sh}"
validate_config || exit 1

# Number of processes/GPUs to use (from _machine_config.sh, defaults to 1)
NPROC_PER_NODE=${NUM_GPUS:-1}

# --- Experiment config ---
TARGET_FLOPS=2.15e18
SIZE=12
NUM_RECUR=4
BPTT_K=4
DEVICE_BATCH_SIZE=32
TAG_SUFFIX=""
# --- Less common ---
INPUT_INJECTION=inject_init_prelude  # inject_init_prelude | passthrough | additive

# --- Derived config ---
TAG="r${NUM_RECUR}_${TARGET_FLOPS}_s${SIZE}${TAG_SUFFIX}"
RUN=$TAG

# Run base training, capture output for CSV logging
TRAIN_LOG=$(mktemp)
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train \
    --target-flops $TARGET_FLOPS \
    --model-tag $TAG \
    --run $RUN \
    --device-batch-size $DEVICE_BATCH_SIZE \
    --core-metric-every=-1 \
    --core-metric-max-per-task=-1 \
    --num-recur=$NUM_RECUR \
    --bptt-k=$BPTT_K \
    --input-injection=$INPUT_INJECTION \
    --save-every=-1 \
    --size $SIZE 2>&1 | tee "$TRAIN_LOG"
TRAIN_EXIT=${PIPESTATUS[0]}
