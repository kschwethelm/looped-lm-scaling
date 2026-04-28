"""Prepare the owned-taxonomy eval bundle.

Tasks:
- Induction-head (compositional_symbolic) — minimal in-context key/value
  lookup task, inspired by Olsson et al. 2022's induction-head circuit.
"""
import json
import random
import string
from pathlib import Path

import yaml

from nanochat.common import get_base_dir


def get_bundle_dir() -> Path:
    return Path(get_base_dir()) / "owned_bundle"


# --- Induction-head ---

def _random_token(rng: random.Random, length: int = 4) -> str:
    return "".join(rng.choices(string.ascii_lowercase, k=length))


def generate_induction_head_example(rng: random.Random, n_pairs: int = 8) -> dict:
    keys = [_random_token(rng) for _ in range(n_pairs)]
    vals = [_random_token(rng, length=3) for _ in range(n_pairs)]
    lines = [f"{k} -> {v}" for k, v in zip(keys, vals)]
    query_idx = rng.randrange(n_pairs)
    query_key = keys[query_idx]
    gold_val = vals[query_idx]
    context = "\n".join(lines) + f"\n{query_key} ->"
    # No leading space in continuation — YAML continuation_delimiter " " adds it.
    return {"context": context, "continuation": gold_val,
            "choices": vals, "gold": query_idx}


def prepare_induction_head(out_dir: Path, n_examples: int = 1000,
                            seed: int = 1337) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    rows = [generate_induction_head_example(rng, n_pairs=8)
            for _ in range(n_examples)]
    path = out_dir / "induction_head.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"Induction-head: wrote {len(rows)} rows to {path}")


def _task(label: str, uri: str, group: str, task_type: str = "language_modeling",
          num_fewshot: int = 0, continuation_delimiter: str = " ") -> dict:
    return {"label": label, "dataset_uri": uri, "task_type": task_type,
            "num_fewshot": num_fewshot,
            "continuation_delimiter": continuation_delimiter, "group": group}


ALL_TASKS: list[dict] = [
    _task("induction_head", "reasoning_primitives/induction_head.jsonl",
          "reasoning_primitives"),
]


def write_yaml(bundle_dir: Path, tasks: list[dict]) -> None:
    groups = list(dict.fromkeys(t["group"] for t in tasks))
    with open(bundle_dir / "owned.yaml", "w", encoding="utf-8") as f:
        yaml.dump({"groups": groups, "tasks": tasks}, f, sort_keys=False)
    print(f"Config: wrote {bundle_dir / 'owned.yaml'}")


def main() -> None:
    bundle_dir = get_bundle_dir()
    data_dir = bundle_dir / "eval_data"
    print(f"Bundle dir: {bundle_dir}")
    prepare_induction_head(data_dir / "reasoning_primitives")
    write_yaml(bundle_dir, ALL_TASKS)


if __name__ == "__main__":
    main()
