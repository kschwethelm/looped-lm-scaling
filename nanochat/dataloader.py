"""
Pre-packed dataloader for pretraining.

Reads pre-tokenized + pre-packed Parquet files where each row is a
BOS-aligned best-fit packed sequence of T+1 tokens. Zero tokenization
overhead — just reads rows and yields batches.
"""

import json
from pathlib import Path

import torch
import pyarrow.parquet as pq

from nanochat.common import get_dist_info


def _list_prepacked_shards(prepacked_dir: str, split: str = "train") -> list[Path]:
    """List pre-packed Parquet shard files in sorted order."""
    d = Path(prepacked_dir)
    files = sorted(d.glob(f"{split}-*.parquet"))
    assert files, f"No {split}-*.parquet files found in {prepacked_dir}"
    return files


def prepacked_data_loader(
    prepacked_dir: str,
    B: int,
    T: int,
    split: str = "train",
    device: str = "cuda",
    resume_state: dict | None = None,
):
    """
    Dataloader that reads pre-packed Parquet files directly.

    Each row in the Parquet files is a packed sequence of T+1 tokens, already
    BOS-aligned and best-fit packed. This loader just reads rows and yields
    (inputs, targets, state_dict) with zero data processing overhead.

    DDP sharding: rank k reads every Nth row (N = world_size).

    Args:
        prepacked_dir: Directory containing pre-packed Parquet shards.
        B: Batch size (rows per yield).
        T: Sequence length (each row has T+1 tokens).
        device: Target device for tensors.
        resume_state: State dict from a previous yield to resume from.
    """
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    shard_paths = _list_prepacked_shards(prepacked_dir, split=split)

    # Validate that pre-packed data matches expected sequence length
    meta_path = Path(prepacked_dir) / f"meta_{split}.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["row_capacity"] == T + 1, (
            f"Pre-packed data has row_capacity={meta['row_capacity']} but T+1={T + 1}"
        )

    use_cuda = device == "cuda"
    row_capacity = T + 1

    # Pre-allocate buffers (same pattern as the tokenizing loader)
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=use_cuda)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device=device)
    cpu_inputs = cpu_buffer[:B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)

    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)

    # Resume state: skip directly to the right shard/row instead of scanning
    resume_shard_idx = 0
    resume_rg_idx = 0
    resume_row_in_rg = 0
    epoch = 1
    if resume_state and "shard_idx" in resume_state:
        resume_shard_idx = resume_state["shard_idx"]
        resume_rg_idx = resume_state["rg_idx"]
        resume_row_in_rg = resume_state["row_in_rg"]
        epoch = resume_state.get("epoch", 1)

    # Start global_row_idx aligned to batch boundary so batch_pos starts at 0
    resume_row_idx = (resume_state or {}).get("row_idx", 0)
    global_row_idx = (resume_row_idx // (B * ddp_world_size)) * (B * ddp_world_size)

    while True:  # infinite iteration (multi-epoch)
        for shard_idx, shard_path in enumerate(shard_paths):
            if shard_idx < resume_shard_idx:
                continue

            pf = pq.ParquetFile(shard_path)
            for rg_idx in range(pf.num_row_groups):
                if shard_idx == resume_shard_idx and rg_idx < resume_rg_idx:
                    continue

                table = pf.read_row_group(rg_idx)
                all_rows = table.column("tokens").to_pylist()

                assert len(all_rows) >= ddp_world_size, (
                    f"Row group {rg_idx} in {shard_path} has {len(all_rows)} rows "
                    f"but ddp_world_size={ddp_world_size}. All row groups must have "
                    f"at least as many rows as GPUs to prevent DDP hangs."
                )

                start_row = resume_row_in_rg if (shard_idx == resume_shard_idx and rg_idx == resume_rg_idx) else ddp_rank
                for row_in_rg in range(start_row, len(all_rows), ddp_world_size):
                    tokens = all_rows[row_in_rg]
                    batch_pos = (global_row_idx // ddp_world_size) % B

                    row_buffer[batch_pos, :len(tokens)] = torch.tensor(tokens, dtype=torch.long)

                    if batch_pos == B - 1:
                        # Full batch — copy to GPU
                        cpu_inputs.copy_(row_buffer[:, :-1])
                        cpu_targets.copy_(row_buffer[:, 1:])
                        state_dict = {
                            "row_idx": global_row_idx,
                            "shard_idx": shard_idx,
                            "rg_idx": rg_idx,
                            "row_in_rg": row_in_rg + ddp_world_size,
                            "epoch": epoch,
                        }
                        gpu_buffer.copy_(cpu_buffer, non_blocking=use_cuda)
                        yield inputs, targets, state_dict

                    global_row_idx += ddp_world_size

        # Reset skip state for next epoch
        resume_shard_idx = 0
        resume_rg_idx = 0
        resume_row_in_rg = 0
        epoch += 1


def prepacked_eval_loader(prepacked_dir: str, B: int, T: int, split: str = "val", device: str = "cuda"):
    """Prepacked loader for eval — strips state_dict, no resume needed."""
    for inputs, targets, _state in prepacked_data_loader(prepacked_dir, B, T, split=split, device=device):
        yield inputs, targets
