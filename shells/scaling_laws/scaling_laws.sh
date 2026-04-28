#!/bin/bash
# ISOFLOPs scaling law: sweep model sizes at a fixed FLOPs budget.
# Array job: each task trains one model size.
#
# Environment variables:
#   ARCH         - loop8 | loop4 | loop2 | noloop (default: loop4)
#   TARGET_FLOPS - FLOPs budget (required)
#   SIZES        - comma-separated model sizes (default: auto from TARGET_FLOPS)
#
# Architectures (all 20 effective layers):
#   loop8:  2 prelude + 2 recur-block x 8 recurrences + 2 coda
#   loop4:  2 prelude + 4 recur-block x 4 recurrences + 2 coda
#   loop2:  2 prelude + 8 recur-block x 2 recurrences + 2 coda
#   noloop: 20 blocks, no recurrence
#
# Usage:
#   TARGET_FLOPS=4.64e17 ARCH=loop4  ./shells/_submit.sh shells/scaling_laws/scaling_laws.sh -- --array=0-3
#   TARGET_FLOPS=4.64e17 ARCH=loop2  ./shells/_submit.sh shells/scaling_laws/scaling_laws.sh -- --array=0-3
#   TARGET_FLOPS=4.64e18 ARCH=noloop ./shells/_submit.sh shells/scaling_laws/scaling_laws.sh -- --array=0-5

cd "${REPO_ROOT:-$HOME/looped-lm-scaling}"
uv sync
source .venv/bin/activate

source "${MACHINE_CONFIG:-shells/_machine_config.sh}"
validate_config || exit 1

NPROC_PER_NODE=${NUM_GPUS:-1}

# --- Experiment config ---
TARGET_FLOPS=${TARGET_FLOPS:?ERROR: Set TARGET_FLOPS env var}
MATRIX_LR=0.014

# Model sizes: explicit override or auto from FLOPs budget
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
if [ -n "$SIZES" ]; then
    IFS=',' read -ra SIZE_ARRAY <<< "$SIZES"
else
    case "$TARGET_FLOPS" in
        4.64e17) SIZE_ARRAY=(6 8 10 12) ;;
        1e18)    SIZE_ARRAY=(6 8 10 12 14) ;;
        2.15e18) SIZE_ARRAY=(8 10 12 14 16) ;;
        4.64e18) SIZE_ARRAY=(10 12 14 16 18 20) ;;
        1e19)    SIZE_ARRAY=(12 16 18 20 24) ;;
        2.15e19) SIZE_ARRAY=(18 24 28 34) ;;
        *)       echo "ERROR: No default sizes for TARGET_FLOPS=$TARGET_FLOPS. Set SIZES env var."; exit 1 ;;
    esac
fi
SIZE=${SIZE_ARRAY[$TASK_ID]}

# Device batch size: reduce for large models to avoid OOM (overridable via env)
if [ -z "${DEVICE_BATCH_SIZE:-}" ]; then
    if [ "$SIZE" -ge 30 ]; then
        DEVICE_BATCH_SIZE=8
    elif [ "$SIZE" -ge 18 ]; then
        DEVICE_BATCH_SIZE=16
    else
        DEVICE_BATCH_SIZE=32
    fi
fi

# Architecture selector
ARCH=${ARCH:-loop4}
case "$ARCH" in
    loop4)
        N_PRELUDE=2; N_RECUR_BLOCK=4;  N_CODA=2; NUM_RECUR=4
        INJECTION=inject_init_prelude
        ARCH_TAG="loop"  # historical tag, keeps existing runs' naming
        ;;
    loop2)
        N_PRELUDE=2; N_RECUR_BLOCK=8;  N_CODA=2; NUM_RECUR=2
        INJECTION=inject_init_prelude
        ARCH_TAG="loop2"
        ;;
    loop8)
        N_PRELUDE=2; N_RECUR_BLOCK=2;  N_CODA=2; NUM_RECUR=8
        INJECTION=inject_init_prelude
        ARCH_TAG="loop8"
        ;;
    noloop)
        N_PRELUDE=0; N_RECUR_BLOCK=20; N_CODA=0; NUM_RECUR=1
        INJECTION=passthrough
        ARCH_TAG="noloop"
        ;;
    *)
        echo "ERROR: Unknown ARCH=$ARCH (expected loop8|loop4|loop2|noloop)"; exit 1 ;;
esac

TAG="isoflops_${TARGET_FLOPS}_${ARCH_TAG}_s${SIZE}"

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
