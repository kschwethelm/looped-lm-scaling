#!/bin/bash
# LR sweep for HyperP transfer law experiments.
# Array job: each task runs one LR value.
#
# Environment variables:
#   LOOPED    - true/false (default: true)
#   SIZE      - model size (default: 10)
#   RATIO     - target param-data ratio (default: 10)
#   LRS       - comma-separated LR grid (default: 0.008,0.010,0.012,0.014,0.016,0.018,0.020,0.024)
#   DEVICE_BATCH_SIZE - override device batch size (default: 32)
#
# Usage:
#   # Reference sweep at s10, ratio=10 (8 LRs)
#   LOOPED=true ./shells/_submit.sh shells/hyperparams/sweep_lr.sh -- --array=0-7
#
#   # Data scaling: ratio=20, narrower grid
#   RATIO=20 LRS=0.008,0.011,0.014,0.016,0.018 LOOPED=true \
#     ./shells/_submit.sh shells/hyperparams/sweep_lr.sh -- --array=0-4
#
#   # Width transfer: test s8 and s14
#   SIZE=8  LOOPED=true ./shells/_submit.sh shells/hyperparams/sweep_lr.sh -- --array=0-4
#   SIZE=14 LOOPED=true ./shells/_submit.sh shells/hyperparams/sweep_lr.sh -- --array=0-4

cd "${REPO_ROOT:-$HOME/looped-lm-scaling}"
uv sync
source .venv/bin/activate

source "${MACHINE_CONFIG:-shells/_machine_config.sh}"
validate_config || exit 1

NPROC_PER_NODE=${NUM_GPUS:-1}

# --- Experiment config ---
SIZE=${SIZE:-10}
DEVICE_BATCH_SIZE=${DEVICE_BATCH_SIZE:-32}
TOTAL_BATCH_SIZE=262144  # 256K tokens
TARGET_PARAM_DATA_RATIO=${RATIO:-10}

# Looped vs non-looped
LOOPED=${LOOPED:-true}
if [ "$LOOPED" = true ]; then
    N_PRELUDE=2
    N_RECUR_BLOCK=4
    N_CODA=2
    NUM_RECUR=4
    INJECTION=inject_init_prelude
    ARCH_TAG="loop"
else
    N_PRELUDE=0
    N_RECUR_BLOCK=20
    N_CODA=0
    NUM_RECUR=1
    INJECTION=passthrough
    ARCH_TAG="noloop"
fi

# LR grid (base LR before data/batch scaling — base_train.py applies
# data_lr_scale and batch_lr_scale automatically, so effective LR differs
# at non-reference ratios, e.g. RATIO=20 → effective ≈ 0.80 × base)
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
IFS=',' read -ra LR_ARRAY <<< "${LRS:-0.008,0.010,0.012,0.014,0.016,0.018,0.020,0.024}"
MATRIX_LR=${LR_ARRAY[$TASK_ID]}

TAG="sweep_lr_${ARCH_TAG}_s${SIZE}_r${TARGET_PARAM_DATA_RATIO}_lr${MATRIX_LR}"

torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train \
    --target-param-data-ratio $TARGET_PARAM_DATA_RATIO \
    --total-batch-size $TOTAL_BATCH_SIZE \
    --model-tag $TAG \
    --run $TAG \
    --device-batch-size $DEVICE_BATCH_SIZE \
    --eval-every=500 \
    --core-metric-every=-1 \
    --sample-every=-1 \
    --save-every=-1 \
    --num-recur=$NUM_RECUR \
    --bptt-k=$NUM_RECUR \
    --input-injection=$INJECTION \
    --n-prelude=$N_PRELUDE \
    --n-recur-block=$N_RECUR_BLOCK \
    --n-coda=$N_CODA \
    --matrix-lr=$MATRIX_LR \
    --size $SIZE
