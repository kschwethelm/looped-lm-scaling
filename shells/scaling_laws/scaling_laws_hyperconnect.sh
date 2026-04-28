#!/bin/bash
# Hyperconnections scaling law sweep.
#
# Environment variables:
#   ARCH         - loop2 | loop4 | loop8 (required)
#   TARGET_FLOPS - FLOPs budget (required)
#   NUM_LANES    - number of lanes (default: 2)
#   SIZES        - comma-separated model sizes (default: auto from TARGET_FLOPS)
#
# Tag convention: isoflops_hc${NUM_LANES}_${TARGET_FLOPS}_${ARCH_TAG}_s${SIZE}
#
# Usage:
#   TARGET_FLOPS=1e18    ARCH=loop4 ./shells/_submit.sh shells/scaling_laws/scaling_laws_hyperconnect.sh -- --array=0-4
#   TARGET_FLOPS=2.15e19 ARCH=loop8 ./shells/_submit.sh shells/scaling_laws/scaling_laws_hyperconnect.sh -- --array=0-3

cd "${REPO_ROOT:-$HOME/looped-lm-scaling}"
uv sync
source .venv/bin/activate

source "${MACHINE_CONFIG:-shells/_machine_config.sh}"
validate_config || exit 1

NPROC_PER_NODE=${NUM_GPUS:-1}

# --- Experiment config ---
TARGET_FLOPS=${TARGET_FLOPS:?ERROR: Set TARGET_FLOPS env var}
NUM_LANES=${NUM_LANES:-2}
MATRIX_LR=0.014

# Model sizes: explicit override or auto from FLOPs budget (matches scaling_laws.sh)
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
if [ -n "$SIZES" ]; then
    IFS=',' read -ra SIZE_ARRAY <<< "$SIZES"
else
    case "$TARGET_FLOPS" in
        4.64e17) SIZE_ARRAY=(6 8 10 12) ;;
        1e18)    SIZE_ARRAY=(6 8 10 12 14) ;;
        2.15e18) SIZE_ARRAY=(8 10 12 14 16 18) ;;
        4.64e18) SIZE_ARRAY=(10 12 14 16 18 20) ;;
        1e19)    SIZE_ARRAY=(12 16 18 20 24) ;;
        2.15e19) SIZE_ARRAY=(18 24 28 34) ;;
        *)       echo "ERROR: No default sizes for TARGET_FLOPS=$TARGET_FLOPS. Set SIZES env var."; exit 1 ;;
    esac
fi
SIZE=${SIZE_ARRAY[$TASK_ID]}

# Device batch size: match scaling_laws.sh heuristic
if [ -z "${DEVICE_BATCH_SIZE:-}" ]; then
    if [ "$SIZE" -ge 30 ]; then
        DEVICE_BATCH_SIZE=8
    elif [ "$SIZE" -ge 18 ]; then
        DEVICE_BATCH_SIZE=16
    else
        DEVICE_BATCH_SIZE=32
    fi
fi

# Architecture selector (loop2 | loop4 | loop8); hyperconnect only makes sense with looping
ARCH=${ARCH:?ERROR: Set ARCH=loop2, loop4, or loop8}
case "$ARCH" in
    loop2)
        N_PRELUDE=2; N_RECUR_BLOCK=8; N_CODA=2; NUM_RECUR=2
        ARCH_TAG="loop2"
        ;;
    loop4)
        N_PRELUDE=2; N_RECUR_BLOCK=4; N_CODA=2; NUM_RECUR=4
        ARCH_TAG="loop"
        ;;
    loop8)
        N_PRELUDE=2; N_RECUR_BLOCK=2; N_CODA=2; NUM_RECUR=8
        ARCH_TAG="loop8"
        ;;
    *)
        echo "ERROR: ARCH=$ARCH not supported (expected loop2, loop4, or loop8)."; exit 1 ;;
esac

TAG="isoflops_hc${NUM_LANES}_${TARGET_FLOPS}_${ARCH_TAG}_s${SIZE}"

echo "=============================================="
echo "Hyperconnect scaling law run: $TAG"
echo "  Arch:        $ARCH (r=$NUM_RECUR, lanes=$NUM_LANES)"
echo "  Size:        s$SIZE"
echo "  Budget:      $TARGET_FLOPS FLOPs"
echo "  Device bs:   $DEVICE_BATCH_SIZE"
echo "=============================================="

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
    --input-injection=hyperconnect \
    --num-lanes=$NUM_LANES \
    --n-prelude=$N_PRELUDE \
    --n-recur-block=$N_RECUR_BLOCK \
    --n-coda=$N_CODA \
    --matrix-lr=$MATRIX_LR \
    --size $SIZE
