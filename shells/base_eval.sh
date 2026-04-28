#!/bin/bash
# Evaluate checkpoints on specified metrics.
# Array job: each task evaluates one checkpoint.
#
# Environment variables:
#   EVAL_TAGS    - comma-separated model tags (required)
#   EVAL_METRICS - comma-separated metrics (default: owned)
#
# Note: val/train loss is already computed at the end of base_train.py on the
# full eval set, so it is not part of the default here.
#
# Usage:
#   # Default owned-taxonomy eval
#   EVAL_TAGS="isoflops_2.15e18_loop_s8,isoflops_2.15e18_loop_s10" \
#     ./shells/_submit.sh shells/base_eval.sh -- --array=0-1
#
#   # With loss re-eval, plus full CORE + Saunshi
#   EVAL_TAGS="isoflops_2.15e18_loop_s8" EVAL_METRICS=core,loss,saunshi \
#     ./shells/_submit.sh shells/base_eval.sh -- --array=0

cd "${REPO_ROOT:-$HOME/looped-lm-scaling}"
uv sync
source .venv/bin/activate

source "${MACHINE_CONFIG:-shells/_machine_config.sh}"
validate_config || exit 1

EVAL_METRICS=${EVAL_METRICS:-owned}

# Parse model tags from environment
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
IFS=',' read -ra TAGS <<< "${EVAL_TAGS:?ERROR: Set EVAL_TAGS env var}"
MODEL_TAG=${TAGS[$TASK_ID]}

if [ -z "$MODEL_TAG" ]; then
    echo "ERROR: No model tag for task $TASK_ID."
    exit 1
fi

echo "Evaluating ($EVAL_METRICS): $MODEL_TAG"

# Determine device batch size based on model size
SIZE=$(echo "$MODEL_TAG" | grep -oP 's\K\d+')
if [ "$SIZE" -ge 26 ]; then
    DEVICE_BATCH_SIZE=8
elif [ "$SIZE" -ge 18 ]; then
    DEVICE_BATCH_SIZE=16
else
    DEVICE_BATCH_SIZE=32
fi

python -m scripts.base_eval \
    --eval $EVAL_METRICS \
    --model-tag $MODEL_TAG \
    --device-batch-size $DEVICE_BATCH_SIZE
