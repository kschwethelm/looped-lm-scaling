#!/bin/bash
set -eo pipefail

# Download pre-packed FineWeb-Edu dataset from HuggingFace Hub.
#
# Usage:
#   ./shells/preprocessing/download_prepacked.sh

cd "${REPO_ROOT:-$HOME/looped-lm-scaling}"

source "${MACHINE_CONFIG:-shells/_machine_config.sh}"
validate_config || exit 1

uv sync
source .venv/bin/activate

python -m scripts.prepack --download KristianS7/prepacked-fineweb-edu-llama2-32K-T2048 \
    --output-dir "$NANOCHAT_BASE_DIR/prepacked_T2048_llama"
