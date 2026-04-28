"""
Common utilities for nanochat.
"""

# Validation eval constants (divisible by device_batch_size=32)
# Based on prepacked_T2048_llama val set: 34,608 rows, ~71M tokens
EVAL_ROWS_FAST = 4992      # ~10M tokens, for intermediate training evals
EVAL_ROWS_FULL = 34560     # ~71M tokens, for final eval (~full val set)

import logging
import os
import re
import urllib.request

import torch
import torch.distributed as dist
from filelock import FileLock


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log messages."""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"
    BOLD = "\033[1m"

    def format(self, record):
        # Add color to the level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{self.BOLD}{levelname}{self.RESET}"
        # Format the message
        message = super().format(record)
        # Add color to specific parts of the message
        if levelname == "INFO":
            # Highlight numbers and percentages
            message = re.sub(r"(\d+\.?\d*\s*(?:GB|MB|%|docs))", rf"{self.BOLD}\1{self.RESET}", message)
            message = re.sub(r"(Shard \d+)", rf"{self.COLORS['INFO']}{self.BOLD}\1{self.RESET}", message)
        return message


def setup_default_logging():
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[handler])


setup_default_logging()
logger = logging.getLogger(__name__)


def get_base_dir():
    # co-locate nanochat intermediates with other cached data in ~/.cache (by default)
    if os.environ.get("NANOCHAT_BASE_DIR"):
        nanochat_dir = os.environ.get("NANOCHAT_BASE_DIR")
    else:
        home_dir = os.path.expanduser("~")
        cache_dir = os.path.join(home_dir, ".cache")
        nanochat_dir = os.path.join(cache_dir, "nanochat")
    os.makedirs(nanochat_dir, exist_ok=True)
    return nanochat_dir


def download_file_with_lock(url, filename, postprocess_fn=None):
    """
    Downloads a file from a URL to a local path in the base directory.
    Uses a lock file to prevent concurrent downloads among multiple ranks.
    """
    base_dir = get_base_dir()
    file_path = os.path.join(base_dir, filename)
    lock_path = file_path + ".lock"

    if os.path.exists(file_path):
        return file_path

    with FileLock(lock_path):
        # Only a single rank can acquire this lock
        # All other ranks block until it is released

        # Recheck after acquiring lock
        if os.path.exists(file_path):
            return file_path

        # Download the content as bytes
        print(f"Downloading {url}...")
        with urllib.request.urlopen(url) as response:
            content = response.read()  # bytes

        # Write to local file
        with open(file_path, "wb") as f:
            f.write(content)
        print(f"Downloaded to {file_path}")

        # Run the postprocess function if provided
        if postprocess_fn is not None:
            postprocess_fn(file_path)

    return file_path


def print0(s="", **kwargs):
    ddp_rank = int(os.environ.get("RANK", 0))
    if ddp_rank == 0:
        print(s, **kwargs)


def print_banner():
    # Cool DOS Rebel font ASCII banner made with https://manytools.org/hacker-tools/ascii-banner/
    banner = """
     ████                                           █████                            
    ░░███                                          ░░███                             
     ░███   ██████   ██████  ████████   ██████   ███████                             
     ░███  ███░░███ ███░░███░░███░░███ ███░░███ ███░░███                             
     ░███ ░███ ░███░███ ░███ ░███ ░███░███████ ░███ ░███                             
     ░███ ░███ ░███░███ ░███ ░███ ░███░███░░░  ░███ ░███                             
     █████░░██████ ░░██████  ░███████ ░░██████ ░░████████                            
    ░░░░░  ░░░░░░   ░░░░░░   ░███░░░   ░░░░░░   ░░░░░░░░                             
                             ░███                                                    
                             █████                                                   
                            ░░░░░                                                    
                                                       █████                 █████   
                                                       ░░███                 ░░███    
     ████████    ██████   ████████    ██████   ██████  ░███████    ██████   ███████  
    ░░███░░███  ░░░░░███ ░░███░░███  ███░░███ ███░░███ ░███░░███  ░░░░░███ ░░░███░   
     ░███ ░███   ███████  ░███ ░███ ░███ ░███░███ ░░░  ░███ ░███   ███████   ░███    
     ░███ ░███  ███░░███  ░███ ░███ ░███ ░███░███  ███ ░███ ░███  ███░░███   ░███ ███
     ████ █████░░████████ ████ █████░░██████ ░░██████  ████ █████░░████████  ░░█████ 
    ░░░░ ░░░░░  ░░░░░░░░ ░░░░ ░░░░░  ░░░░░░   ░░░░░░  ░░░░ ░░░░░  ░░░░░░░░    ░░░░░  
    """
    print0(banner)




def is_ddp_requested() -> bool:
    """
    True if launched by torchrun (env present), even before init.
    Used to decide whether we *should* initialize a PG.
    """
    return all(k in os.environ for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))


def is_ddp_initialized() -> bool:
    """
    True if torch.distributed is available and the process group is initialized.
    Used at cleanup to avoid destroying a non-existent PG.
    """
    return dist.is_available() and dist.is_initialized()


def get_dist_info():
    if is_ddp_requested():
        # We rely on torchrun's env to decide if we SHOULD init.
        # (Initialization itself happens in compute init.)
        assert all(var in os.environ for var in ["RANK", "LOCAL_RANK", "WORLD_SIZE"])
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        return True, ddp_rank, ddp_local_rank, ddp_world_size
    else:
        return False, 0, 0, 1


def autodetect_device_type():
    # prefer to use CUDA if available, otherwise use MPS, otherwise fallback on CPU
    if torch.cuda.is_available():
        device_type = "cuda"
    elif torch.backends.mps.is_available():
        device_type = "mps"
    else:
        device_type = "cpu"
    print0(f"Autodetected device type: {device_type}")
    return device_type


def compute_init(device_type="cuda"):  # cuda|cpu|mps
    """Basic initialization that we keep doing over and over, so make common."""

    assert device_type in ["cuda", "mps", "cpu"], "Invalid device type atm"
    if device_type == "cuda":
        assert torch.cuda.is_available(), "Your PyTorch installation is not configured for CUDA but device_type is 'cuda'"
    if device_type == "mps":
        assert torch.backends.mps.is_available(), "Your PyTorch installation is not configured for MPS but device_type is 'mps'"

    # Reproducibility
    # Note that we set the global seeds here, but most of the code uses explicit rng objects.
    # The only place where global rng might be used is nn.Module initialization of the model weights.
    torch.manual_seed(42)
    if device_type == "cuda":
        torch.cuda.manual_seed(42)
    # skipping full reproducibility for now, possibly investigate slowdown later
    # torch.use_deterministic_algorithms(True)

    # Precision
    if device_type == "cuda":
        torch.backends.cuda.matmul.fp32_precision = "tf32"  # uses tf32 instead of fp32 for matmuls

    # Distributed setup: Distributed Data Parallel (DDP), optional, and requires CUDA
    is_ddp_requested, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    if is_ddp_requested and device_type == "cuda":
        device = torch.device("cuda", ddp_local_rank)
        torch.cuda.set_device(device)  # make "cuda" default to this device
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    else:
        device = torch.device(device_type)  # mps|cpu

    if ddp_rank == 0:
        logger.info(f"Distributed world size: {ddp_world_size}")

    return is_ddp_requested, ddp_rank, ddp_local_rank, ddp_world_size, device


def compute_cleanup():
    """Companion function to compute_init, to clean things up before script exit"""
    if is_ddp_initialized():
        dist.destroy_process_group()


class DummyWandb:
    """Useful if we wish to not use wandb but have all the same signatures"""

    def __init__(self):
        pass

    def log(self, *args, **kwargs):
        pass

    def finish(self):
        pass


# hardcoded BF16 peak flops for various GPUs
# inspired by torchtitan: https://github.com/pytorch/torchtitan/blob/main/torchtitan/tools/utils.py
# and PR: https://github.com/karpathy/nanochat/pull/147
def get_peak_flops(device_name: str) -> float:
    name = device_name.lower()

    # Table order matters: more specific patterns first.
    _PEAK_FLOPS_TABLE = (
        # NVIDIA Blackwell
        (["gb200"], 2.5e15),
        (["grace blackwell"], 2.5e15),
        (["b200"], 2.25e15),
        (["b100"], 1.8e15),
        # NVIDIA Hopper
        (["h200", "nvl"], 836e12),
        (["h200", "pcie"], 836e12),
        (["h200"], 989e12),
        (["h100", "nvl"], 835e12),
        (["h100", "pcie"], 756e12),
        (["h100"], 989e12),
        (["h800", "nvl"], 989e12),
        (["h800"], 756e12),
        # NVIDIA Ampere data center
        (["a100"], 312e12),
        (["a800"], 312e12),
        (["a40"], 149.7e12),
        (["a30"], 165e12),
        # NVIDIA Ada data center
        (["l40s"], 362e12),
        (["l40-s"], 362e12),
        (["l40 s"], 362e12),
        (["l4"], 121e12),
        # AMD CDNA accelerators
        (["mi355"], 2.5e15),
        (["mi325"], 1.3074e15),
        (["mi300x"], 1.3074e15),
        (["mi300a"], 980.6e12),
        (["mi250x"], 383e12),
        (["mi250"], 362.1e12),
        # Consumer RTX
        (["5090"], 209.5e12),
        (["4090"], 165.2e12),
        (["3090"], 71e12),
    )
    for patterns, flops in _PEAK_FLOPS_TABLE:
        if all(p in name for p in patterns):
            return flops
    if "data center gpu max 1550" in name:
        # Ponte Vecchio (PVC) - dynamic based on compute units
        max_comp_units = torch.xpu.get_device_properties("xpu").max_compute_units
        return 512 * max_comp_units * 1300 * 10**6

    # Unknown GPU - return inf so MFU shows as 0% rather than a wrong guess
    logger.warning(f"Peak flops undefined for: {device_name}, MFU will show as 0%")
    return float("inf")


def compute_gradient_stats(model: torch.nn.Module, track_level: str) -> dict[str, float]:
    """
    Compute gradient and parameter statistics for tracking optimization dynamics and model health.

    Args:
        model: PyTorch model with computed gradients
        track_level: "none" (disabled), "basic" (global norm), or "detailed" (per-component)

    Returns:
        Dictionary of gradient and parameter statistics for logging
    """
    if track_level == "none":
        return {}

    stats = {}

    # Basic: global gradient and parameter norms (L1 and L2)
    if track_level in ["basic", "detailed"]:
        global_grad_l1 = 0.0
        global_grad_norm_sq = 0.0
        global_param_l1 = 0.0
        global_param_norm_sq = 0.0

        for p in model.parameters():
            # Parameter norms
            global_param_l1 += p.data.abs().sum().item()
            global_param_norm_sq += p.data.pow(2).sum().item()

            # Gradient norms (if available)
            if p.grad is not None:
                global_grad_l1 += p.grad.data.abs().sum().item()
                global_grad_norm_sq += p.grad.data.norm(2).item() ** 2

        stats["grad_norm/l2"] = global_grad_norm_sq**0.5
        stats["grad_norm/l1"] = global_grad_l1
        stats["param_norm/l2"] = global_param_norm_sq**0.5
        stats["param_norm/l1"] = global_param_l1

    # Detailed: per-component and per-optimizer-type breakdown
    if track_level == "detailed":
        # Group by architecture component (prelude/recur_block/coda)
        prelude_grad_l1 = 0.0
        prelude_grad_norm_sq = 0.0
        prelude_param_l1 = 0.0
        prelude_param_norm_sq = 0.0

        recur_grad_l1 = 0.0
        recur_grad_norm_sq = 0.0
        recur_param_l1 = 0.0
        recur_param_norm_sq = 0.0

        coda_grad_l1 = 0.0
        coda_grad_norm_sq = 0.0
        coda_param_l1 = 0.0
        coda_param_norm_sq = 0.0

        other_grad_l1 = 0.0
        other_grad_norm_sq = 0.0
        other_param_l1 = 0.0
        other_param_norm_sq = 0.0

        # Group by optimizer type (muon vs adam)
        muon_grad_l1 = 0.0
        muon_grad_norm_sq = 0.0
        muon_param_l1 = 0.0
        muon_param_norm_sq = 0.0

        adam_grad_l1 = 0.0
        adam_grad_norm_sq = 0.0
        adam_param_l1 = 0.0
        adam_param_norm_sq = 0.0

        for name, p in model.named_parameters():
            # Parameter stats
            param_l1 = p.data.abs().sum().item()
            param_norm_sq = p.data.pow(2).sum().item()

            # Gradient stats (if available)
            if p.grad is not None:
                grad_l1 = p.grad.data.abs().sum().item()
                grad_norm_sq = p.grad.data.norm(2).item() ** 2
            else:
                grad_l1 = 0.0
                grad_norm_sq = 0.0

            # Classify by architecture component
            if "prelude" in name:
                prelude_grad_l1 += grad_l1
                prelude_grad_norm_sq += grad_norm_sq
                prelude_param_l1 += param_l1
                prelude_param_norm_sq += param_norm_sq
            elif ".recur." in name or "transformer.recur" in name:
                recur_grad_l1 += grad_l1
                recur_grad_norm_sq += grad_norm_sq
                recur_param_l1 += param_l1
                recur_param_norm_sq += param_norm_sq
            elif "coda" in name:
                coda_grad_l1 += grad_l1
                coda_grad_norm_sq += grad_norm_sq
                coda_param_l1 += param_l1
                coda_param_norm_sq += param_norm_sq
            else:
                other_grad_l1 += grad_l1
                other_grad_norm_sq += grad_norm_sq
                other_param_l1 += param_l1
                other_param_norm_sq += param_norm_sq

            # Classify by optimizer type
            # Muon: matrix parameters (weights with ndim >= 2)
            # Adam: embeddings and other parameters (biases, norms, etc.)
            if "embed" in name.lower():
                adam_grad_l1 += grad_l1
                adam_grad_norm_sq += grad_norm_sq
                adam_param_l1 += param_l1
                adam_param_norm_sq += param_norm_sq
            elif "weight" in name and p.ndim >= 2:
                muon_grad_l1 += grad_l1
                muon_grad_norm_sq += grad_norm_sq
                muon_param_l1 += param_l1
                muon_param_norm_sq += param_norm_sq
            else:
                adam_grad_l1 += grad_l1
                adam_grad_norm_sq += grad_norm_sq
                adam_param_l1 += param_l1
                adam_param_norm_sq += param_norm_sq

        # Gradient stats
        stats["grad_norm/prelude/l2"] = prelude_grad_norm_sq**0.5
        stats["grad_norm/prelude/l1"] = prelude_grad_l1
        stats["grad_norm/recur_block/l2"] = recur_grad_norm_sq**0.5
        stats["grad_norm/recur_block/l1"] = recur_grad_l1
        stats["grad_norm/coda/l2"] = coda_grad_norm_sq**0.5
        stats["grad_norm/coda/l1"] = coda_grad_l1
        stats["grad_norm/other/l2"] = other_grad_norm_sq**0.5
        stats["grad_norm/other/l1"] = other_grad_l1
        stats["grad_norm/muon_params/l2"] = muon_grad_norm_sq**0.5
        stats["grad_norm/muon_params/l1"] = muon_grad_l1
        stats["grad_norm/adam_params/l2"] = adam_grad_norm_sq**0.5
        stats["grad_norm/adam_params/l1"] = adam_grad_l1

        # Parameter stats
        stats["param_norm/prelude/l2"] = prelude_param_norm_sq**0.5
        stats["param_norm/prelude/l1"] = prelude_param_l1
        stats["param_norm/recur_block/l2"] = recur_param_norm_sq**0.5
        stats["param_norm/recur_block/l1"] = recur_param_l1
        stats["param_norm/coda/l2"] = coda_param_norm_sq**0.5
        stats["param_norm/coda/l1"] = coda_param_l1
        stats["param_norm/other/l2"] = other_param_norm_sq**0.5
        stats["param_norm/other/l1"] = other_param_l1
        stats["param_norm/muon_params/l2"] = muon_param_norm_sq**0.5
        stats["param_norm/muon_params/l1"] = muon_param_l1
        stats["param_norm/adam_params/l2"] = adam_param_norm_sq**0.5
        stats["param_norm/adam_params/l1"] = adam_param_l1

    return stats
