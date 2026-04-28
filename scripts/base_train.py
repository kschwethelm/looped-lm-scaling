"""
Train base model on FineWeb-Edu
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import gc
import json
import time
import argparse
from dataclasses import asdict
from contextlib import contextmanager, nullcontext

import torch
import torch.nn as nn
import wandb

from nanochat.checkpoint_manager import load_checkpoint, save_checkpoint
from nanochat.common import (
    EVAL_ROWS_FAST,
    EVAL_ROWS_FULL,
    DummyWandb,
    autodetect_device_type,
    compute_cleanup,
    compute_gradient_stats,
    compute_init,
    get_base_dir,
    get_peak_flops,
    print0,
    print_banner,
)
from nanochat.dataloader import prepacked_data_loader, prepacked_eval_loader
from nanochat.engine import Engine
from nanochat.flash_attention import HAS_FA2, HAS_FA3
from nanochat.gpt import GPT, GPTConfig
from nanochat.loss_eval import evaluate_loss
from nanochat.report import get_report
from nanochat.tokenizer import get_tokenizer
from scripts.base_eval import evaluate_core

print_banner()

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Pretrain base model")
# Logging
parser.add_argument(
    "--run",
    type=str,
    default="dummy",
    help="wandb run name ('dummy' disables wandb logging)",
)
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
# Model architecture
parser.add_argument("--size", type=int, default=20, help="model size (model_dim = size * aspect_ratio)")
parser.add_argument("--aspect-ratio", type=int, default=64, help="model_dim = size * aspect_ratio")
parser.add_argument("--head-dim", type=int, default=128, help="target head dimension for attention")
parser.add_argument("--max-seq-len", type=int, default=2048, help="max context length")
# Looped Transformer config
parser.add_argument("--n-prelude", type=int, default=2, help="number of prelude layers")
parser.add_argument("--n-recur-block", type=int, default=4, help="number of layers in the recurrent block")
parser.add_argument("--n-coda", type=int, default=2, help="number of coda layers")
parser.add_argument("--num-recur", type=int, default=4, help="number of recurrences")
parser.add_argument("--bptt-k", type=int, default=4, help="truncate backprop to last k recurrences (limits gradient depth)")
parser.add_argument(
    "--input-injection",
    type=str,
    default="inject_init_prelude",
    choices=["inject_init_prelude", "passthrough", "additive", "hyperconnect"],
    help="input injection mode: inject_init_prelude (default), passthrough (no injection), additive (s + e, parameter-free), or hyperconnect (N lanes with per-step mixing)",
)
parser.add_argument("--num-lanes", type=int, default=2, help="number of hyperconnection lanes (only used when --input-injection=hyperconnect)")
parser.add_argument("--fp8", action=argparse.BooleanOptionalAction, default=False, help="enable FP8 training (requires H100+ GPU)")
# Training horizon (only one used, in order of precedence)
parser.add_argument("--num-iterations", type=int, default=-1, help="explicit number of optimization steps (-1 = disable)")
parser.add_argument("--target-flops", type=float, default=-1.0, help="calculate num_iterations to reach target_flops (-1 = disable)")
parser.add_argument("--target-param-data-ratio", type=int, default=20, help="calculate num_iterations to maintain data:param ratio, -1 = disable)")
# Optimization
parser.add_argument("--device-batch-size", type=int, default=32, help="per-device batch size")
parser.add_argument("--total-batch-size", type=int, default=262144, help="total batch size in tokens (default: 256K)")
parser.add_argument("--embedding-lr", type=float, default=0.3, help="learning rate for embedding parameters (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=0.004, help="learning rate for unembedding parameters (Adam)")
parser.add_argument("--matrix-lr", type=float, default=0.014, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--warmup-ratio", type=float, default=0.0, help="ratio of iterations for LR warmup")
parser.add_argument("--warmdown-ratio", type=float, default=1.0, help="ratio of iterations for LR warmdown (1.0 = full linear decay, HyperP default)")
parser.add_argument("--final-lr-frac", type=float, default=0.1, help="final LR as fraction of initial LR (0.1 = decay to 10%%, HyperP default)")
parser.add_argument("--resume-from-step", type=str, default="-1", help="resume training from this step number or checkpoint alias, e.g. 'pre_warmdown' (-1 = disable)")
parser.add_argument("--resume-checkpoint-dir", type=str, default="", help="override checkpoint dir for resume (for cross-budget reuse); saves still go to the current run's dir")
parser.add_argument("--checkpoint-base-dir", type=str, default="", help="override base dir for checkpoints (default: {NANOCHAT_BASE_DIR}/base_checkpoints)")
# Evaluation
parser.add_argument("--eval-every", type=int, default=250, help="evaluate val loss every N steps (-1 = disable)")
parser.add_argument("--core-metric-every", type=int, default=2000, help="evaluate CORE metric every N steps (-1 = disable)")
parser.add_argument("--core-metric-max-per-task", type=int, default=500, help="examples per task for CORE metric")
parser.add_argument("--sample-every", type=int, default=2000, help="sample from model every N steps (-1 = disable)")
parser.add_argument("--save-every", type=int, default=-1, help="save checkpoints every N steps (-1 = only at end)")
parser.add_argument(
    "--save-at-ratios", type=str, default="",
    help="save checkpoints at these T/P ratios on scaling params (comma-separated, e.g. '5,10,20'). "
         "Automatically adjusts for warmdown: saves at ratio*(1-warmdown_ratio) so warmdown branches "
         "finish at exactly the target ratio."
)
parser.add_argument("--log-every", type=int, default=100, help="log detailed metrics to wandb every N steps")
# Pre-packed data
# Output
parser.add_argument("--model-tag", type=str, default=None, help="override model tag for checkpoint directory name")
# Gradient tracking
parser.add_argument(
    "--track-gradients",
    type=str,
    choices=["none", "basic", "detailed"],
    default="basic",
    help="Gradient tracking level: none (disabled), basic (global norm), detailed (per-component norms)",
)
args = parser.parse_args()
user_config = vars(args).copy()  # for logging
# -----------------------------------------------------------------------------
# Compute init and wandb logging

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
if device_type == "cuda":
    gpu_device_name = torch.cuda.get_device_name(0)
    gpu_peak_flops = get_peak_flops(gpu_device_name)
    print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float("inf")  # MFU not meaningful for CPU/MPS

# wandb logging init
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat", name=args.run, config=user_config)

# Flash Attention status
if HAS_FA3 or HAS_FA2:
    print0("✓ Using Flash Attention, efficient and awesome.")
else:
    print0("!" * 80)
    print0("WARNING: Flash Attention not available, using PyTorch SDPA fallback")
    print0("WARNING: Training will be less efficient without FA")
    print0("!" * 80)

# -----------------------------------------------------------------------------
# Tokenizer will be useful for evaluation and also we need the vocab size to init the model
tokenizer = get_tokenizer()
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# -----------------------------------------------------------------------------
# Initialize the Model

def build_model_meta(size):
    """Build a model on meta device for a given size/width (shapes/dtypes only, no data)."""
    # Model dim is nudged up to nearest multiple of head_dim for clean division
    # (FA3 requires head_dim divisible by 8, and this guarantees head_dim == args.head_dim exactly)
    base_dim = size * args.aspect_ratio
    model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
    num_heads = model_dim // args.head_dim
    config = GPTConfig(
        sequence_len=args.max_seq_len, vocab_size=vocab_size,
        size=size, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        n_prelude=args.n_prelude, n_recur_block=args.n_recur_block, n_coda=args.n_coda,
        num_recur=args.num_recur, bptt_k=args.bptt_k,
        input_injection=args.input_injection, num_lanes=args.num_lanes,
    )
    with torch.device("meta"):
        model_meta = GPT(config)
    return model_meta

# Build the model, move to device, init the weights
model = build_model_meta(args.size) # 1) Build on meta device (only shapes/dtypes, no data)
model_config = model.config
model_config_kwargs = asdict(model_config)
print0(f"Model config:\n{json.dumps(model_config_kwargs, indent=2)}")
model.to_empty(device=device) # 2) All tensors get storage on target device but with uninitialized (garbage) data
model.init_weights() # 3) All tensors get initialized

# If we are resuming, overwrite the model parameters with those of the checkpoint
base_dir = get_base_dir()
output_dirname = args.model_tag if args.model_tag else f"s{args.size}"  # e.g. s12
checkpoint_base = args.checkpoint_base_dir if args.checkpoint_base_dir else os.path.join(base_dir, "base_checkpoints")
checkpoint_dir = os.path.join(checkpoint_base, output_dirname)
# Parse resume_from_step: int step number or string alias (e.g. "pre_warmdown")
resume_from_step: int | str = int(args.resume_from_step) if args.resume_from_step.lstrip("-").isdigit() else args.resume_from_step
resuming = resume_from_step != -1
if resuming:
    resume_dir = args.resume_checkpoint_dir if args.resume_checkpoint_dir else checkpoint_dir
    print0(f"Resuming optimization from step {resume_from_step} (checkpoint dir: {resume_dir})")
    model_data, optimizer_data, meta_data = load_checkpoint(
        resume_dir,
        resume_from_step,
        device,
        load_optimizer=True,
        rank=ddp_rank,
    )
    model.load_state_dict(model_data, strict=True, assign=True)
    # Resolve alias to actual step number (needed for step comparisons later)
    if isinstance(resume_from_step, str):
        resume_from_step = meta_data["step"]
        print0(f"Resolved checkpoint alias to step {resume_from_step}")
    del model_data  # free up this memory after the copy

# -----------------------------------------------------------------------------
# FP8 training initialization (must be done before torch.compile)

if args.fp8:
    if device_type != "cuda":
        print0("Warning: FP8 training requires CUDA, ignoring --fp8 flag")
        args.fp8 = False
    else:
        from nanochat.fp8 import Float8LinearConfig, convert_to_float8_training

        # Filter: dims must be divisible by 16 (FP8 hardware requirement) and large enough
        def fp8_module_filter(mod: nn.Module, fqn: str) -> bool:
            if not isinstance(mod, nn.Linear):
                return False
            if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
                return False
            if min(mod.in_features, mod.out_features) < 128:
                return False
            return True

        fp8_config = Float8LinearConfig.from_recipe_name("tensorwise")
        num_linear = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
        convert_to_float8_training(model, config=fp8_config, module_filter_fn=fp8_module_filter)
        num_fp8 = sum(1 for m in model.modules() if 'Float8' in type(m).__name__)
        num_skipped = num_linear - num_fp8
        print0(f"FP8 training enabled (tensorwise scaling) - converted {num_fp8}/{num_linear} linear layers, skipped {num_skipped} (too small)")

# Context manager to temporarily disable FP8 for BF16 evaluation
@contextmanager
def disable_fp8(model):
    """Temporarily swap Float8Linear modules with nn.Linear for BF16 evaluation."""
    fp8_locations = []
    for name, module in model.named_modules():
        if 'Float8' in type(module).__name__:
            if '.' in name:
                parent_name, attr_name = name.rsplit('.', 1)
                parent = model.get_submodule(parent_name)
            else:
                parent = model
                attr_name = name
            fp8_locations.append((parent, attr_name, module))

    if not fp8_locations:
        yield
        return

    for parent, attr_name, fp8_module in fp8_locations:
        linear = nn.Linear(
            fp8_module.in_features, fp8_module.out_features,
            bias=fp8_module.bias is not None,
            device="meta", dtype=fp8_module.weight.dtype,
        )
        linear.weight = fp8_module.weight
        if fp8_module.bias is not None:
            linear.bias = fp8_module.bias
        setattr(parent, attr_name, linear)

    try:
        yield
    finally:
        for parent, attr_name, fp8_module in fp8_locations:
            setattr(parent, attr_name, fp8_module)

# -----------------------------------------------------------------------------
# Compile the model

orig_model = model  # original, uncompiled model, for saving raw model state_dict and for inference/evaluation (because the shapes may change shape)
model = torch.compile(model)  # Fixed num_recur: compile entire model as a single graph (optimal fusion)

# -----------------------------------------------------------------------------
# Scaling laws and muP extrapolations to determine the optimal training horizon, batch size, and learning rates.

# Get the parameter counts of our model
param_counts = model.num_scaling_params()
print0("Parameter counts:")
for key, value in param_counts.items():
    print0(f"  {key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_token = model.estimate_flops()
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

# 1) Use scaling laws to determine the optimal training horizon in tokens
# The compute-optimal models satisfy the Tokens:Params ratio of --target-param-data-ratio (derived experimentally via scaling laws analysis).
# We've already initialized the model so we have Params. Optimal Tokens is now simply target-param-data-ratio * Params
# We follow Kaplan et al. and exclude embedding parameters
def get_scaling_params(m):
    # Effective parameters accounting for recurrent block reuse
    num_effective_params = m.effective_params(num_recur=model_config.num_recur, exclude_embedding=True)
    print0(f"Effective params (excluding embeddings): {num_effective_params:,}")
    return num_effective_params

num_scaling_params = get_scaling_params(model)
# Calculate target training tokens from the chosen horizon method:
assert args.num_iterations > 0 or args.target_param_data_ratio > 0 or args.target_flops > 0
if args.num_iterations > 0:
    assert args.total_batch_size > 0, "Must specify --total-batch-size when using --num-iterations"
    total_batch_size = args.total_batch_size
    target_tokens = args.num_iterations * total_batch_size
    num_iterations = args.num_iterations
    print0(f"Using user-provided number of iterations: {num_iterations:,}")
elif args.target_flops > 0:
    target_tokens = int(args.target_flops / num_flops_per_token)
    print0(f"Target tokens from target FLOPs: {target_tokens:,}")
else:
    target_tokens = int(args.target_param_data_ratio * num_scaling_params)
    print0(f"Target tokens from data:param ratio {args.target_param_data_ratio}: {target_tokens:,}")

# Reference model for hyperparameter transfer. Architecture-matched to the training model:
# non-looped (r≤1) → 0-20-0 passthrough, looped (r>1) → 2-4-2 inject_init_prelude.
REF_SIZE = 10
REF_DATA_RATIO = 10

def build_reference_model(size: int, looped: bool) -> GPT:
    """Build an architecture-matched reference model on meta device."""
    base_dim = size * args.aspect_ratio
    model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
    num_heads = model_dim // args.head_dim
    if looped:
        config = GPTConfig(sequence_len=2048, vocab_size=vocab_size, size=size,
            n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
            n_prelude=2, n_recur_block=4, n_coda=2,
            num_recur=4, input_injection="inject_init_prelude")
    else:
        config = GPTConfig(sequence_len=2048, vocab_size=vocab_size, size=size,
            n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
            n_prelude=0, n_recur_block=20, n_coda=0,
            num_recur=1, input_injection="passthrough")
    with torch.device("meta"):
        return GPT(config)

is_looped = model_config.num_recur > 1
ref_model = build_reference_model(REF_SIZE, looped=is_looped)
D_REF = REF_DATA_RATIO * get_scaling_params(ref_model)
D_REF_DIM = REF_SIZE * args.aspect_ratio  # reference model dimension (for muP width scaling)
B_REF = 2**18  # reference batch size = 262,144 tokens (256K, validated empirically)
total_batch_size = args.total_batch_size

# 2) Batch size LR correction
batch_lr_scale = 1.0
batch_ratio = total_batch_size / B_REF
if batch_ratio != 1.0:
    # η ∝ B^{0.558} (HyperP, arXiv:2603.28743 §5.4)
    batch_lr_scale = batch_ratio ** 0.558
    print0(f"Scaling LRs by {batch_lr_scale:.4f} for batch size {total_batch_size:,} (reference: {B_REF:,})")

# 3) Data scaling: η ∝ (T/T_ref)^{-0.32} (HyperP, arXiv:2603.28743 §3.4)
# Under ISOFLOPs, wider models train on fewer tokens → they get higher LR.
# At the reference size, data_lr_scale = 1.0 (no correction).
data_lr_scale = (target_tokens / D_REF) ** -0.32
print0(f"Data-scaling LR correction: {data_lr_scale:.4f} (tokens={target_tokens:,}, ref={D_REF:,})")

# -----------------------------------------------------------------------------
# Initialize the Optimizer (MuonH for matrix params, AdamW for rest)
# MuonH constrains weights to a Frobenius sphere, making weight decay a first-order
# no-op (arXiv:2603.28743). No weight decay scaling needed.
# Data scaling (T^{-0.32}) applies only to MuonH matrix params, not AdamW (HyperP Table 1).
optimizer = model.setup_optimizer(
    # AdamW hyperparameters (batch scaling only)
    unembedding_lr=args.unembedding_lr * batch_lr_scale,
    embedding_lr=args.embedding_lr * batch_lr_scale,
    # MuonH hyperparameters (batch + data scaling)
    matrix_lr=args.matrix_lr * batch_lr_scale * data_lr_scale,
    # muP width scaling reference
    d_ref=D_REF_DIM,
)

if resuming:
    optimizer.load_state_dict(optimizer_data)
    del optimizer_data  # free up the memory

# -----------------------------------------------------------------------------
# Initialize the DataLoaders for train/val

dataloader_resume_state_dict = None if not resuming else meta_data["dataloader_state_dict"]
prepacked_dir = os.path.join(base_dir, f"prepacked_T{args.max_seq_len}_llama")
assert os.path.isdir(prepacked_dir), (
    f"Pre-packed data not found at {prepacked_dir}. "
    f"Run 'python -m scripts.prepack' first."
)
train_loader = prepacked_data_loader(
    prepacked_dir,
    args.device_batch_size,
    args.max_seq_len,
    device=device,
    resume_state=dataloader_resume_state_dict,
)
print0(f"Using pre-packed data from {prepacked_dir}")

def build_val_loader():
    return prepacked_eval_loader(prepacked_dir, args.device_batch_size, args.max_seq_len, split="val", device=device)

x, y, dataloader_state_dict = next(train_loader)  # kick off load of the very first batch of data
# For checkpointing: save the state from BEFORE the prefetch that produced x,y.
# dataloader_state_dict is a continuation state (points PAST the yielded batch).
# On resume we need to reproduce the batch in x, so we save the previous state.
dataloader_checkpoint_state = dataloader_resume_state_dict

# -----------------------------------------------------------------------------
# Calculate the number of iterations we will train for and set up the various schedulers

# num_iterations: either it is given, or from target flops, or from target data:param ratio (in that order)
assert args.num_iterations > 0 or args.target_param_data_ratio > 0 or args.target_flops > 0
if args.num_iterations > 0:
    # Override num_iterations to a specific value if given
    num_iterations = args.num_iterations
    print0(f"Using user-provided number of iterations: {num_iterations:,}")
elif args.target_flops > 0:
    num_iterations = round(args.target_flops / (num_flops_per_token * total_batch_size))
    print0(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
elif args.target_param_data_ratio > 0:
    # Calculate the number of iterations from the target param data ratio (the most common use case)
    num_iterations = target_tokens // total_batch_size
    print0(f"Calculated number of iterations from target data:param ratio: {num_iterations:,}")
else:
    raise ValueError("No training horizon specified")
total_tokens = total_batch_size * num_iterations # the actual number of tokens we will train for
print0(f"Total number of training tokens: {total_tokens:,}")
print0(f"Tokens : Scaling params ratio: {total_batch_size * num_iterations / num_scaling_params:.2f}") # e.g. Chinchilla was ~20
print0(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")

# Learning rate schedule (linear warmup, constant, linear warmdown)
def get_lr_multiplier(it):
    warmup_iters = round(args.warmup_ratio * num_iterations)
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * args.final_lr_frac
    
# -----------------------------------------------------------------------------
# Training loop

# Loop state (variables updated by the training loop)
if not resuming:
    step = 0
    val_loss = None  # will be set if eval_every > 0
    min_val_loss = float("inf")
    smooth_train_loss = 0  # EMA of training loss
    total_training_time = 0  # total wall-clock time of training
else:
    step = meta_data["step"]
    loop_state = meta_data["loop_state"]
    val_loss = meta_data["val_loss"]
    min_val_loss = loop_state["min_val_loss"]
    smooth_train_loss = loop_state["smooth_train_loss"]
    total_training_time = loop_state["total_training_time"]

# Gradient accumulation
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len # tokens per iteration for a single rank
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size # total tokens per iteration for all ranks
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

# Compute checkpoint steps from --save-at-ratios
# For each target ratio R, the pre-warmdown step is at R*(1-warmdown_ratio)*scaling_params/batch
# so that a warmdown branch starting from that checkpoint ends at exactly R*scaling_params tokens.
save_at_steps = set()
if args.save_at_ratios:
    for ratio_str in args.save_at_ratios.split(","):
        ratio = float(ratio_str.strip())
        pre_warmdown_tokens = ratio * (1 - args.warmdown_ratio) * num_scaling_params
        save_step = round(pre_warmdown_tokens / total_batch_size)
        if 0 < save_step <= num_iterations:
            save_at_steps.add(save_step)
            warmdown_iters = round(ratio * args.warmdown_ratio * num_scaling_params / total_batch_size)
            print0(f"  ratio={ratio}: save at step {save_step} ({pre_warmdown_tokens/1e9:.2f}B tokens), warmdown branch needs {warmdown_iters} iters")
    print0(f"Save-at-ratios: {len(save_at_steps)} checkpoints scheduled")

# Go!
while True:
    last_step = step == num_iterations  # loop runs num_iterations+1 times so that we can eval/save at the end
    current_flops_per_token = num_flops_per_token
    flops_so_far = num_flops_per_token * total_batch_size * step

    # once in a while: evaluate the val loss (all ranks participate)
    # Final step uses full val set (EVAL_ROWS_FULL), intermediate steps use fast eval (EVAL_ROWS_FAST)
    if last_step or (args.eval_every > 0 and step > 0 and step % args.eval_every == 0):
        model.eval()
        val_loader = build_val_loader()
        eval_rows = EVAL_ROWS_FULL if last_step else EVAL_ROWS_FAST
        eval_steps = eval_rows // (args.device_batch_size * ddp_world_size)
        with autocast_ctx:
            val_metrics = evaluate_loss(model, val_loader, eval_steps)
        val_loss = val_metrics.loss
        val_ppl = val_metrics.ppl
        print0(f"Step {step:05d} | Validation loss: {val_loss:.6f} | ppl: {val_ppl:.4f}")
        if val_loss < min_val_loss:
            min_val_loss = val_loss
        wandb_run.log(
            {
                "step": step,
                "total_training_flops": flops_so_far,
                "total_training_time": total_training_time,
                "val/loss": val_loss,
                "val/ppl": val_ppl,
            }
        )
        model.train()

    # once in a while: estimate the CORE metric (all ranks participate)
    # use the original uncompiled model because the inputs keep changing shape
    results = {}
    if args.core_metric_every > 0 and (last_step or (step > 0 and step % args.core_metric_every == 0)):
        model.eval()
        with disable_fp8(orig_model), autocast_ctx:
            results = evaluate_core(
                orig_model,
                tokenizer,
                device,
                max_per_task=args.core_metric_max_per_task,
            )
        print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
        wandb_run.log(
            {
                "step": step,
                "total_training_flops": flops_so_far,
                "core_metric": results["core_metric"],
                "centered_results": results["centered_results"],
            }
        )
        model.train()

    # once in a while: sample from the model (only on master process)
    # use the original uncompiled model because the inputs keep changing shape
    if args.sample_every > 0 and master_process and (last_step or (step > 0 and step % args.sample_every == 0)):
        model.eval()
        prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "If yesterday was Friday, then tomorrow will be",
            "The opposite of hot is",
            "The planets of the solar system are:",
            "My favorite color is",
            "If 5*x + 3 = 13, then x is",
        ]
        engine = Engine(orig_model, tokenizer)  # use orig_model to avoid recompilation
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<s>")
            with disable_fp8(orig_model), autocast_ctx:
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
            print0(tokenizer.decode(sample[0]))
        model.train()

    # Save checkpoint: at the end, at --save-at-ratios steps, or every --save-every steps
    should_save = (
        last_step
        or step in save_at_steps
        or (step > 0 and step != resume_from_step and args.save_every > 0 and step % args.save_every == 0)
    )
    if should_save:
        tokens_so_far = step * total_batch_size
        print0(f"Saving checkpoint at step {step} ({tokens_so_far:,} tokens)")
        with disable_fp8(orig_model):
            model_state = orig_model.state_dict()
        save_checkpoint(
            checkpoint_dir,
            step,
            model_state,
            optimizer.state_dict(),
            {
                "step": step,
                "tokens": tokens_so_far,
                "val_loss": val_loss,
                "model_config": model_config_kwargs,
                "user_config": user_config,
                "device_batch_size": args.device_batch_size,
                "max_seq_len": args.max_seq_len,
                "dataloader_state_dict": dataloader_checkpoint_state,
                "loop_state": {
                    "min_val_loss": min_val_loss,
                    "smooth_train_loss": smooth_train_loss,
                    "total_training_time": total_training_time,
                },
            },
            rank=ddp_rank,
        )

    # termination conditions (TODO: possibly also add loss explosions etc.)
    if last_step:
        break

    # -------------------------------------------------------------------------
    # single training step
    # evaluate the gradient
    synchronize()
    t0 = time.time()

    num_recur = model_config.num_recur
    for _micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y, num_recur=num_recur)
        train_loss = loss.detach()  # for logging
        loss = loss / grad_accum_steps  # each .backward() is a grad sum => normalize loss here
        loss.backward()
        dataloader_checkpoint_state = dataloader_state_dict  # save state before prefetch (for checkpoint resume)
        x, y, dataloader_state_dict = next(train_loader)  # prefetch the next batch while the GPU is busy with forward/backward

    # Compute model health statistics: gradients and parameters (after all backward passes complete)
    model_health_stats = compute_gradient_stats(orig_model, args.track_gradients)

    # step the optimizer
    lrm = get_lr_multiplier(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
    optimizer.step()
    model.zero_grad(set_to_none=True)

    # Extract effective learning rates from optimizer groups (for logging)
    # Groups order: [lm_head (unembed), embedding, norms, muon_shape1, muon_shape2, ...]
    effective_lr_unembed = optimizer.param_groups[0]["lr"]  # lm_head (AdamW)
    effective_lr_embed = optimizer.param_groups[1]["lr"]    # embedding (AdamW)
    effective_lr_muon = optimizer.param_groups[3]["lr"]     # first Muon group (all Muon groups have same lr)
    train_loss_f = train_loss.item()  # .item() is a CPU-GPU sync point
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    # -------------------------------------------------------------------------

    # logging (CPU action only)
    ema_beta = 0.9  # EMA decay factor for some smoothing just for nicer logging
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f  # EMA the training loss
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))  # debias the EMA
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(total_batch_size / dt)
    flops_per_sec = current_flops_per_token * total_batch_size / dt
    mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)
    if step > 10:
        total_training_time += dt  # only count the time after the first 10 steps
    # Calculate ETA based on average time per step (excluding first 10 steps)
    steps_done = step - 10
    if steps_done > 0:
        avg_time_per_step = total_training_time / steps_done
        remaining_steps = num_iterations - step
        eta_seconds = remaining_steps * avg_time_per_step
        eta_str = f" | eta: {eta_seconds / 60:.1f}m"
    else:
        eta_str = ""
    epoch = dataloader_state_dict["epoch"]
    print0(
        f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f} | epoch: {epoch} | total time: {total_training_time / 60:.2f}m{eta_str}"
    )
    if step % 100 == 0:
        log_data = {
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/loss_raw": train_loss_f,  # raw unsmoothed loss from last microbatch
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
            "train/epoch": epoch,
            "train/num_recur": num_recur,
            "lr/muon": effective_lr_muon,           # effective learning rate for Muon (matrix params)
            "lr/embed": effective_lr_embed,         # effective learning rate for embeddings
            "lr/unembed": effective_lr_unembed,     # effective learning rate for unembedding (lm_head)
            **{f"model_health/{k}": v for k, v in model_health_stats.items()},  # Add model health stats
        }
        wandb_run.log(log_data)

    # state update
    first_step_of_run = (step == 0) or (resuming and step == resume_from_step)
    step += 1

    # The garbage collector is sadly a little bit overactive and for some poorly understood reason,
    # it spends ~500ms scanning for cycles quite frequently, just to end up cleaning up very few tiny objects each time.
    # So we manually manage and help it out here
    if first_step_of_run:
        gc.collect()  # manually collect a lot of garbage from setup
        gc.freeze()  # immediately freeze all currently surviving objects and exclude them from GC
        gc.disable()  # nuclear intervention here: disable GC entirely except:
    elif step % 5000 == 0:  # every 5000 steps...
        gc.collect()  # manually collect, just to be safe for very, very long runs

# print a few more stats
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time / 60:.2f}m")
if val_loss is not None:
    print0(f"Final validation loss: {val_loss:.6f}")

section_name = "Base model training"
if args.model_tag is not None:
    section_name += f" {args.model_tag}"
num_effective_params = model.effective_params(num_recur=model_config.num_recur, exclude_embedding=True)
get_report().log(
    section=section_name,
    data=[
        user_config,  # CLI args
        {  # stats about the training setup
            "Number of parameters": num_params,
            "Effective parameters (w/ recur reuse)": num_effective_params,
            "Number of FLOPs per token": f"{num_flops_per_token:e}",
            "Calculated number of iterations": num_iterations,
            "Number of training tokens": total_tokens,
            "Tokens : Params ratio": total_tokens / num_params,
            "Tokens : Effective Params ratio": total_tokens / num_effective_params,
            "DDP world size": ddp_world_size,
            "warmup_ratio": args.warmup_ratio,
            "warmdown_ratio": args.warmdown_ratio,
            "final_lr_frac": args.final_lr_frac,
        },
        {  # stats about training outcomes
            "Minimum validation loss": min_val_loss if val_loss is not None else None,
            "Final validation loss": val_loss,
            "CORE metric estimate": results.get("core_metric", None),
            "MFU %": f"{mfu:.2f}%",
            "Total training flops": f"{flops_so_far:e}",
            "Total training time": f"{total_training_time / 60:.2f}m",
            "Peak memory usage": f"{get_max_memory() / 1024 / 1024:.2f}MiB",
        },
    ],
)

# cleanup
wandb_run.finish()  # wandb run finish
compute_cleanup()
