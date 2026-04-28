#!/bin/bash
set -eo pipefail

# Pre-tokenize and pre-pack FineWeb-Edu for training.
# Downloads from HuggingFace, shuffles, splits train/val, tokenizes, and packs
# into ready-to-train Parquet shards — all in one pass.
#
# Usage:
#   ./shells/_submit.sh shells/preprocessing/prepack_cpu.sh

cd "${REPO_ROOT:-$HOME/looped-lm-scaling}"

source "${MACHINE_CONFIG:-shells/_machine_config.sh}"
validate_config || exit 1

uv sync
source .venv/bin/activate

OUTPUT_DIR="$NANOCHAT_BASE_DIR/prepacked_T2048_llama"

echo "=============================================="
echo "Pre-packing FineWeb-Edu (T=2048)"
echo "  Output: $OUTPUT_DIR"
echo "=============================================="

python -u -m scripts.prepack --output-dir="$OUTPUT_DIR"

echo ""
echo "Done! Output: $OUTPUT_DIR"
