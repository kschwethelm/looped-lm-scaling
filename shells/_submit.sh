#!/bin/bash
# Universal SLURM job submission wrapper
# Usage: ./shells/_submit.sh <script.sh> [script_args...] [-- sbatch_overrides...]
#
# Arguments before "--" are passed to the script itself.
# Arguments after "--" are sbatch overrides (e.g. --time=24:00:00).
#
# Examples:
#   ./shells/_submit.sh shells/base_train.sh
#   ./shells/_submit.sh shells/scaling_laws.sh --smoke-test --no-wandb
#   ./shells/_submit.sh shells/scaling_laws.sh --smoke-test -- --time=24:00:00
#   MACHINE_CONFIG=shells/_machine_config_noeagle.sh ./shells/_submit.sh shells/mtp_sweep.sh -- --array=0-5
#
# This script reads SLURM configuration from _machine_config.sh and submits
# the specified script with appropriate resource allocation.
# Override the config file via the MACHINE_CONFIG environment variable.
#
# Convention: Scripts ending with _cpu.sh are submitted as CPU jobs.
# All other scripts are submitted as GPU jobs.

set -e

SCRIPT_PATH="${1:?Usage: $0 <script.sh> [script_args...] [-- sbatch_overrides...]}"
shift  # Remove script path

# Split remaining args: script args go before "--", sbatch overrides after "--".
SCRIPT_ARGS=()
SBATCH_EXTRA_ARGS=()
FOUND_SEPARATOR=false
for arg in "$@"; do
    if [ "$arg" = "--" ]; then
        FOUND_SEPARATOR=true
        continue
    fi
    if [ "$FOUND_SEPARATOR" = true ]; then
        SBATCH_EXTRA_ARGS+=("$arg")
    else
        SCRIPT_ARGS+=("$arg")
    fi
done

# Validate script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "ERROR: Script not found: $SCRIPT_PATH"
    exit 1
fi

# Source machine configuration (override via MACHINE_CONFIG env var)
CONFIG_FILE="${MACHINE_CONFIG:-shells/_machine_config.sh}"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: $CONFIG_FILE not found!"
    echo ""
    echo "Please create it from the template:"
    echo "  cp shells/_machine_config.sh.template shells/_machine_config.sh"
    echo "  # Then edit _machine_config.sh with your cluster settings"
    echo ""
    echo "To use a different config file:"
    echo "  MACHINE_CONFIG=shells/_machine_config_noeagle.sh ./shells/_submit.sh ..."
    exit 1
fi

source "$CONFIG_FILE"

# Extract job name from script filename
JOB_NAME="looped-lm-scaling_$(basename $SCRIPT_PATH .sh)"

# Determine job type based on script name convention
# CPU jobs: scripts ending with _cpu.sh
# GPU jobs: all other scripts
if [[ "$SCRIPT_PATH" =~ _cpu\.sh$ ]]; then
    JOB_TYPE="cpu"
    PARTITION="$SLURM_PARTITION_CPU"
    QOS="$SLURM_QOS_CPU"
    MEM="$SLURM_MEM_CPU"
    GRES=""
    CPUS_PER_TASK="${SLURM_CPUS_CPU:-1}"
else
    JOB_TYPE="gpu"
    PARTITION="$SLURM_PARTITION_GPU"
    QOS="$SLURM_QOS_GPU"
    MEM="$SLURM_MEM_GPU"
    # Build gres string: gpu[:<type>]:<count>[,gpumem:<mem>]
    if [ -n "$SLURM_GPU_TYPE" ]; then
        GRES="--gres=gpu:$SLURM_GPU_TYPE:$NUM_GPUS"
    else
        GRES="--gres=gpu:$NUM_GPUS"
    fi
    if [ -n "$SLURM_GPU_MEM" ]; then
        GRES="$GRES,gpumem:$SLURM_GPU_MEM"
    fi
    CPUS_PER_TASK="${SLURM_CPUS_GPU:-}"
fi

# Create log directory based on script name
LOG_DIR="logs/$(basename $SCRIPT_PATH .sh)"
mkdir -p "$LOG_DIR"

# Detect array job: use %A_%a log pattern so tasks don't overwrite each other
IS_ARRAY=false
for arg in "${SBATCH_EXTRA_ARGS[@]}"; do
    if [[ "$arg" =~ ^--array ]]; then
        IS_ARRAY=true
        break
    fi
done

if [ "$IS_ARRAY" = true ]; then
    LOG_PATTERN="%A_%a"
else
    LOG_PATTERN="%A"
fi

# Build sbatch command
SBATCH_CMD="sbatch"
SBATCH_CMD="$SBATCH_CMD --job-name=$JOB_NAME"
SBATCH_CMD="$SBATCH_CMD --output=$LOG_DIR/$LOG_PATTERN.out"
SBATCH_CMD="$SBATCH_CMD --error=$LOG_DIR/$LOG_PATTERN.err"
SBATCH_CMD="$SBATCH_CMD --nodes=1"
SBATCH_CMD="$SBATCH_CMD --ntasks=1"
SBATCH_CMD="$SBATCH_CMD --mem=$MEM"

# Add partition and QoS
if [ -n "$PARTITION" ]; then
    SBATCH_CMD="$SBATCH_CMD --partition=$PARTITION"
fi
if [ -n "$QOS" ]; then
    SBATCH_CMD="$SBATCH_CMD --qos=$QOS"
fi
if [ -n "$SLURM_ACCOUNT" ]; then
    SBATCH_CMD="$SBATCH_CMD --account=$SLURM_ACCOUNT"
fi

# Add GPU resources for GPU jobs
if [ -n "$GRES" ]; then
    SBATCH_CMD="$SBATCH_CMD $GRES"
fi

# Add CPU count
if [ -n "$CPUS_PER_TASK" ]; then
    SBATCH_CMD="$SBATCH_CMD --cpus-per-task=$CPUS_PER_TASK"
fi

# Add mail notifications for GPU jobs (usually longer running)
if [ "$JOB_TYPE" = "gpu" ] && [ -n "$SLURM_MAIL_USER" ]; then
    SBATCH_CMD="$SBATCH_CMD --mail-type=FAIL,END"
    SBATCH_CMD="$SBATCH_CMD --mail-user=$SLURM_MAIL_USER"
fi

# Add default time limits if not overridden
TIME_SET=false
for arg in "${SBATCH_EXTRA_ARGS[@]}"; do
    if [[ "$arg" =~ ^--time= ]] || [[ "$arg" =~ ^-t ]]; then
        TIME_SET=true
        break
    fi
done

if [ "$TIME_SET" = false ]; then
    SBATCH_CMD="$SBATCH_CMD --time=${SLURM_TIME:-10:00:00}"
fi

# Add any additional user-provided sbatch overrides
for arg in "${SBATCH_EXTRA_ARGS[@]}"; do
    SBATCH_CMD="$SBATCH_CMD $arg"
done

# Add the script path, then script arguments
SBATCH_CMD="$SBATCH_CMD $SCRIPT_PATH"
for arg in "${SCRIPT_ARGS[@]}"; do
    SBATCH_CMD="$SBATCH_CMD $arg"
done

# Show what we're submitting
echo "=============================================="
echo "Submitting SLURM job:"
echo "  Config:    $CONFIG_FILE"
echo "  Type:      $JOB_TYPE"
echo "  Script:    $SCRIPT_PATH"
echo "  Partition: $PARTITION"
echo "  Memory:    $MEM"
if [ -n "$GRES" ]; then
    if [ -n "$SLURM_GPU_TYPE" ]; then
        echo "  GPUs:      $NUM_GPUS x $SLURM_GPU_TYPE"
    else
        echo "  GPUs:      $NUM_GPUS"
    fi
    if [ -n "$SLURM_GPU_MEM" ]; then
        echo "  GPU Mem:   $SLURM_GPU_MEM (each)"
    fi
fi
if [ -n "$CPUS_PER_TASK" ]; then
    echo "  CPUs:      $CPUS_PER_TASK"
fi
if [ "$IS_ARRAY" = true ]; then
    for arg in "${SBATCH_EXTRA_ARGS[@]}"; do
        if [[ "$arg" =~ ^--array=(.+) ]]; then
            echo "  Array:     ${BASH_REMATCH[1]}"
            break
        fi
    done
fi
if [ ${#SCRIPT_ARGS[@]} -gt 0 ]; then
    echo "  Script args: ${SCRIPT_ARGS[*]}"
fi
echo "  Logs:      $LOG_DIR/$LOG_PATTERN.{out,err}"
echo "=============================================="
echo ""

# Submit the job
echo "Running: $SBATCH_CMD"
echo ""
eval "$SBATCH_CMD"
