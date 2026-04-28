#!/bin/bash
# Flagship model training: s34, isotokens 47B, 512K batch.
#
# Usage (first run):
#   LOOPED=false ./shells/_submit.sh shells/flagship_train/flagship.sh -- --time=48:00:00
#   LOOPED=true  ./shells/_submit.sh shells/flagship_train/flagship.sh -- --time=48:00:00
#
# Usage (resume after timeout):
#   LOOPED=false RESUME=102000 ./shells/_submit.sh shells/flagship_train/flagship.sh -- --time=48:00:00

cd "${REPO_ROOT:-$HOME/looped-lm-scaling}"
uv sync
source .venv/bin/activate

source "${MACHINE_CONFIG:-shells/_machine_config.sh}"
validate_config || exit 1

NPROC_PER_NODE=${NUM_GPUS:-2}

# --- Fixed config ---
SIZE=34
NUM_ITERATIONS=89645            # 47B tokens / 512K batch
TOTAL_BATCH_SIZE=524288         # 512K
DEVICE_BATCH_SIZE=${DEVICE_BATCH_SIZE:-16}
MATRIX_LR=0.014 # MuonH base LR -> scaled by hyperP method
SAVE_EVERY=3000
EVAL_EVERY=-1  # eval offline from checkpoints on a separate GPU
USE_FP8=${USE_FP8:-true}

# --- Architecture ---
LOOPED=${LOOPED:?ERROR: Set LOOPED=true or LOOPED=false}
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

TAG="flagship_${ARCH_TAG}_s${SIZE}"

# --- Resume support ---
RESUME_ARGS=""
if [ -n "$RESUME" ]; then
    RESUME_ARGS="--resume-from-step $RESUME"
fi

EFFICIENCY_ARGS=""
if [ "$USE_FP8" = true ]; then
    EFFICIENCY_ARGS="$EFFICIENCY_ARGS --fp8"
fi

echo "=============================================="
echo "Flagship training: $TAG"
echo "  Size:       s${SIZE}"
echo "  Arch:       ${ARCH_TAG}"
echo "  Tokens:     50B (isotokens)"
echo "  Iterations: ${NUM_ITERATIONS}"
echo "  Batch:      ${TOTAL_BATCH_SIZE} (device: ${DEVICE_BATCH_SIZE})"
echo "  FP8:        ${USE_FP8}"
echo "  Save every: ${SAVE_EVERY}"
if [ -n "$RESUME" ]; then
    echo "  Resume:     ${RESUME}"
fi
echo "=============================================="

torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train \
    --num-iterations $NUM_ITERATIONS \
    --total-batch-size $TOTAL_BATCH_SIZE \
    --model-tag $TAG \
    --run $TAG \
    --device-batch-size $DEVICE_BATCH_SIZE \
    --eval-every=$EVAL_EVERY \
    --core-metric-every=-1 \
    --save-every=$SAVE_EVERY \
    --num-recur=$NUM_RECUR \
    --bptt-k=$NUM_RECUR \
    --input-injection=$INJECTION \
    --n-prelude=$N_PRELUDE \
    --n-recur-block=$N_RECUR_BLOCK \
    --n-coda=$N_CODA \
    --matrix-lr=$MATRIX_LR \
    --size $SIZE \
    $EFFICIENCY_ARGS \
    $RESUME_ARGS

TRAIN_EXIT=$?

# --- Post-training eval (owned + CORE) ---
EVAL_METRICS=${EVAL_METRICS:-}
if [ -n "$EVAL_METRICS" ] && [ $TRAIN_EXIT -eq 0 ]; then
    EVAL_DBS=8
    echo "=============================================="
    echo "Running eval ($EVAL_METRICS) on $TAG"
    echo "=============================================="
    torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_eval \
        --eval $EVAL_METRICS \
        --model-tag $TAG \
        --device-batch-size $EVAL_DBS
elif [ $TRAIN_EXIT -ne 0 ]; then
    echo "Training failed (exit $TRAIN_EXIT), skipping eval"
    exit $TRAIN_EXIT
fi
