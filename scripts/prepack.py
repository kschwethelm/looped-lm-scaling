"""
Pre-pack the FineWeb-Edu dataset for training.

Downloads pre-shuffled shards from karpathy/fineweb-edu-100b-shuffle,
tokenizes with Llama 2 32k, and packs into ready-to-train Parquet files
where each row is a BOS-aligned best-fit packed sequence of T+1 tokens.

Usage:
    uv run python -m scripts.prepack
    uv run python -m scripts.prepack --max-rows 100  # quick test
"""

import argparse
import json
import os
import random
import time
from collections.abc import Iterator
from multiprocessing import Pool
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import requests

from nanochat.common import get_base_dir
from nanochat.tokenizer import get_tokenizer

# ---------------------------------------------------------------------------
# Karpathy's pre-shuffled FineWeb-Edu shards
SHARD_BASE_URL = "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main"
MAX_SHARD = 1822  # shard_00000.parquet to shard_01822.parquet
NUM_VAL_SHARDS = 2  # last 2 shards reserved for validation


def _download_shard(args: tuple[int, str]) -> bool:
    """Download a single shard with retries."""
    index, data_dir = args
    filename = f"shard_{index:05d}.parquet"
    filepath = os.path.join(data_dir, filename)
    if os.path.exists(filepath):
        return True
    url = f"{SHARD_BASE_URL}/{filename}"
    for attempt in range(1, 6):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            temp_path = f"{filepath}.tmp"
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            os.rename(temp_path, filepath)
            return True
        except (requests.RequestException, IOError) as e:
            print(f"  Attempt {attempt}/5 failed for {filename}: {e}")
            for path in [f"{filepath}.tmp", filepath]:
                if os.path.exists(path):
                    os.remove(path)
            if attempt < 5:
                time.sleep(2 ** attempt)
    return False


def _download_shards(data_dir: str, num_workers: int = 8) -> None:
    """Download all shards (skips existing)."""
    os.makedirs(data_dir, exist_ok=True)
    args = [(i, data_dir) for i in range(MAX_SHARD + 1)]
    with Pool(processes=num_workers) as pool:
        results = pool.map(_download_shard, args)
    ok = sum(results)
    print(f"Downloaded {ok}/{len(results)} shards to {data_dir}")
    assert ok == len(results), f"Failed to download {len(results) - ok} shards"


def _list_shards(data_dir: str) -> list[str]:
    """List all shard parquet files in sorted order."""
    files = sorted(
        f for f in os.listdir(data_dir)
        if f.endswith(".parquet") and not f.endswith(".tmp")
    )
    return [os.path.join(data_dir, f) for f in files]


# ---------------------------------------------------------------------------
# Tokenization and packing


def _tokenized_docs(
    parquet_paths: list[str],
    tokenizer,
    bos_token: int,
    tokenizer_batch_size: int,
    stats: dict,
) -> Iterator[list[int]]:
    """Yield tokenized documents one at a time from parquet shards."""
    for pq_idx, filepath in enumerate(parquet_paths):
        pf = pq.ParquetFile(filepath)
        shard_name = Path(filepath).name
        t_shard = time.time()

        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column("text").to_pylist()

            for i in range(0, len(texts), tokenizer_batch_size):
                batch = texts[i : i + tokenizer_batch_size]
                token_lists = tokenizer.encode(batch, prepend=bos_token)
                stats["total_docs"] += len(token_lists)
                yield from token_lists

        elapsed = time.time() - t_shard
        total_elapsed = time.time() - stats["t_start"]
        print(
            f"  [{pq_idx + 1}/{stats['num_input_shards']}] {shard_name}: "
            f"{elapsed:.1f}s | "
            f"docs: {stats['total_docs']:,} | "
            f"rows: {stats['total_rows']:,} | "
            f"shards written: {stats['shards_written']} | "
            f"total: {total_elapsed:.0f}s"
        )


def _pack_row(doc_buffer: list[list[int]], row_capacity: int) -> list[int] | None:
    """
    Pack a single row using best-fit algorithm.

    For each position: find the largest doc that fits entirely, repeat until
    nothing fits, then crop the shortest doc to fill remaining space.

    Returns None if the buffer runs dry before the row is full.
    """
    row: list[int] = []
    pos = 0
    while pos < row_capacity and doc_buffer:
        remaining = row_capacity - pos

        best_idx = -1
        best_len = 0
        for i, doc in enumerate(doc_buffer):
            doc_len = len(doc)
            if doc_len <= remaining and doc_len > best_len:
                best_idx = i
                best_len = doc_len

        if best_idx >= 0:
            doc = doc_buffer.pop(best_idx)
            row.extend(doc)
            pos += len(doc)
        else:
            shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
            doc = doc_buffer.pop(shortest_idx)
            row.extend(doc[:remaining])
            pos += remaining

    if len(row) != row_capacity:
        return None
    return row


def _pack_rows(
    doc_iter: Iterator[list[int]],
    row_capacity: int,
    buffer_size: int,
) -> Iterator[list[int]]:
    """Yield packed rows, keeping the buffer topped-up before each row."""
    doc_buffer: list[list[int]] = []
    exhausted = False

    def refill():
        nonlocal exhausted
        while len(doc_buffer) < buffer_size and not exhausted:
            try:
                doc_buffer.append(next(doc_iter))
            except StopIteration:
                exhausted = True

    while True:
        refill()
        if not doc_buffer:
            break
        row = _pack_row(doc_buffer, row_capacity)
        if row is None:
            break
        yield row


def _write_shard(output_dir: Path, shard_idx: int, rows: list[list[int]], split: str):
    """Write a single Parquet shard."""
    assert all(
        max(row) <= 65535 for row in rows
    ), "Token values exceed uint16 range (65535). Use a tokenizer with vocab_size <= 65536."
    random.shuffle(rows)
    table = pa.table({"tokens": pa.array(rows, type=pa.list_(pa.uint16()))})
    filename = f"{split}-{shard_idx:05d}.parquet"
    pq.write_table(table, output_dir / filename)


def _pack_split(
    split: str,
    parquet_paths: list[str],
    output_dir: Path,
    tokenizer,
    bos_token: int,
    row_capacity: int,
    buffer_size: int,
    rows_per_shard: int,
    tokenizer_batch_size: int,
    max_rows: int,
) -> dict:
    """Pack a single split (train or val) and write shards + metadata."""
    stats = {
        "total_docs": 0,
        "total_rows": 0,
        "shards_written": 0,
        "num_input_shards": len(parquet_paths),
        "t_start": time.time(),
    }

    doc_iter = _tokenized_docs(parquet_paths, tokenizer, bos_token, tokenizer_batch_size, stats)
    row_iter = _pack_rows(doc_iter, row_capacity, buffer_size)

    shard_idx = 0
    shard_rows: list[list[int]] = []

    for row in row_iter:
        shard_rows.append(row)
        stats["total_rows"] += 1

        if max_rows > 0 and stats["total_rows"] >= max_rows:
            break

        if len(shard_rows) >= rows_per_shard:
            _write_shard(output_dir, shard_idx, shard_rows[:rows_per_shard], split)
            shard_rows = shard_rows[rows_per_shard:]
            shard_idx += 1
            stats["shards_written"] = shard_idx

    if shard_rows:
        _write_shard(output_dir, shard_idx, shard_rows, split)
        shard_idx += 1

    total_rows = stats["total_rows"]
    total_tokens_packed = total_rows * row_capacity

    meta = {
        "dataset": "karpathy/fineweb-edu-100b-shuffle",
        "vocab_size": tokenizer.get_vocab_size(),
        "bos_token_id": bos_token,
        "seq_len": row_capacity - 1,
        "row_capacity": row_capacity,
        "total_rows": total_rows,
        "total_tokens": total_tokens_packed,
        "total_docs": stats["total_docs"],
        "num_shards": shard_idx,
        "buffer_size": buffer_size,
        "split": split,
    }
    meta_path = output_dir / f"meta_{split}.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    total_elapsed = time.time() - stats["t_start"]
    print(f"  {split}: {total_rows:,} rows in {shard_idx} shards ({total_elapsed:.0f}s)")
    print(f"  Total tokens: {total_tokens_packed:,} ({total_tokens_packed / 1e9:.2f}B)")
    return meta


def prepack(
    output_dir: Path,
    seq_len: int = 2048,
    buffer_size: int = 1000,
    rows_per_shard: int = 15_000,
    max_rows: int = -1,
    tokenizer_batch_size: int = 128,
    download_workers: int = 8,
):
    """
    Download, tokenize, and pre-pack the dataset into Parquet shards.

    Uses Karpathy's pre-shuffled FineWeb-Edu shards. Last 2 shards are val.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = get_tokenizer()
    bos_token = tokenizer.get_bos_token_id()
    row_capacity = seq_len + 1

    # Download shards
    data_dir = os.path.join(get_base_dir(), "base_data")
    print(f"Ensuring all {MAX_SHARD + 1} shards are downloaded...")
    _download_shards(data_dir, num_workers=download_workers)

    # Split shards: last NUM_VAL_SHARDS → val, rest → train
    all_shards = _list_shards(data_dir)
    train_shards = all_shards[:-NUM_VAL_SHARDS]
    val_shards = all_shards[-NUM_VAL_SHARDS:]
    print(f"Shards: {len(train_shards)} train, {len(val_shards)} val")
    print(f"Output: {output_dir}\n")

    print("=== Train ===")
    train_meta = _pack_split(
        "train", train_shards, output_dir, tokenizer, bos_token,
        row_capacity, buffer_size, rows_per_shard, tokenizer_batch_size, max_rows,
    )

    print("\n=== Val ===")
    val_meta = _pack_split(
        "val", val_shards, output_dir, tokenizer, bos_token,
        row_capacity, buffer_size, rows_per_shard, tokenizer_batch_size, max_rows,
    )

    print(f"\nDone! Output: {output_dir}")
    return {"train": train_meta, "val": val_meta}


def push_to_hub(output_dir: Path, repo_id: str):
    """Upload the pre-packed dataset to HuggingFace Hub."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    api.upload_large_folder(
        folder_path=str(output_dir),
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"Uploaded to https://huggingface.co/datasets/{repo_id}")


def download_from_hub(repo_id: str, output_dir: Path):
    """Download a pre-packed dataset from HuggingFace Hub."""
    from huggingface_hub import snapshot_download

    if output_dir.exists() and list(output_dir.glob("*.parquet")):
        raise FileExistsError(
            f"{output_dir} already contains parquet files. Delete it first to avoid mixing datasets."
        )
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(output_dir),
    )
    print(f"Downloaded to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-pack FineWeb-Edu for training")
    parser.add_argument("--seq-len", type=int, default=2048, help="sequence length T (default: 2048)")
    parser.add_argument("--buffer-size", type=int, default=1000, help="document buffer size for best-fit packing")
    parser.add_argument("--rows-per-shard", type=int, default=15000, help="rows per output Parquet shard")
    parser.add_argument("--max-rows", type=int, default=-1, help="max packed rows per split (-1 = all)")
    parser.add_argument("--output-dir", type=str, default=None, help="output directory (default: NANOCHAT_BASE_DIR/prepacked_T<seq_len>)")
    parser.add_argument("--push-to-hub", type=str, default=None, help="upload to HuggingFace (e.g. ORG/dataset-name)")
    parser.add_argument("--download", type=str, default=None, help="download from HuggingFace (e.g. ORG/dataset-name)")
    args = parser.parse_args()

    if args.output_dir is None:
        base_dir = get_base_dir()
        output_dir = Path(base_dir) / f"prepacked_T{args.seq_len}_llama"
    else:
        output_dir = Path(args.output_dir)

    if args.download:
        download_from_hub(args.download, output_dir)
    elif args.push_to_hub:
        push_to_hub(output_dir, args.push_to_hub)
    else:
        prepack(
            output_dir=output_dir,
            seq_len=args.seq_len,
            buffer_size=args.buffer_size,
            rows_per_shard=args.rows_per_shard,
            max_rows=args.max_rows,
        )
