"""
Unified evaluation script for base models.

Supports evaluation modes (comma-separated):
  --eval core      : CORE metric (accuracy on all 22 ICL tasks)
  --eval loss      : Cross-entropy loss (nats) and perplexity on train/val splits
  --eval sample    : Generate samples from the model
  --eval saunshi   : Saunshi downstream benchmark
  --eval owned     : Project-owned taxonomy (5 axes) + optional commonsense (flagship regime)

Use --regime {isoflops,flagship} to control which axes are included in --eval owned
(flagship adds the commonsense axis).

Default is: --eval owned
(val/train loss is already reported by base_train.py at the final step using the
full eval set, so it is not part of the default here.)
"""

import argparse
import csv
import json
import os
import random
import shutil
import tempfile
import time
import zipfile
from contextlib import nullcontext
from pathlib import Path

import torch
import yaml

from nanochat.checkpoint_manager import load_model, load_model_from_dir
from nanochat.common import EVAL_ROWS_FULL, autodetect_device_type, compute_cleanup, compute_init, download_file_with_lock, get_base_dir, print0
from nanochat.core_eval import TaskMetrics, evaluate_task, evaluate_task_full
from nanochat.dataloader import prepacked_eval_loader
from nanochat.engine import Engine
from nanochat.loss_eval import evaluate_loss
from nanochat.tokenizer import Tokenizer

# -----------------------------------------------------------------------------
# HuggingFace loading utilities


class ModelWrapper:
    """Lightweight wrapper to give HuggingFace models a nanochat-compatible interface."""

    def __init__(self, model, max_seq_len=None):
        self.model = model
        self.max_seq_len = max_seq_len

    def __call__(self, input_ids, targets=None, loss_reduction="mean"):
        logits = self.model(input_ids).logits
        if targets is None:
            return logits
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction
        )
        return loss

    def get_device(self):
        return next(self.model.parameters()).device


def load_hf_model(hf_path: str, device):
    """Load a HuggingFace model and tokenizer."""
    print0(f"Loading HuggingFace model from: {hf_path}")
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(hf_path)
    model.to(device)
    model.eval()
    max_seq_len = 1024 if "openai-community/gpt2" in hf_path else None
    model = ModelWrapper(model, max_seq_len=max_seq_len)
    from tokenizers import Tokenizer as HFTokenizer
    tokenizer = Tokenizer(HFTokenizer.from_pretrained(hf_path))
    return model, tokenizer




# -----------------------------------------------------------------------------
# CORE evaluation

EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"

def _filter_tasks(tasks: list[dict], allowlist: set[str] | None) -> list[dict]:
    """Return tasks whose 'label' is in allowlist. None passes through."""
    if allowlist is None:
        return tasks
    return [t for t in tasks if t["label"] in allowlist]


def place_eval_bundle(file_path):
    """Unzip eval_bundle.zip and place it in the base directory."""
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(tmpdir)
        extracted_bundle_dir = os.path.join(tmpdir, "eval_bundle")
        shutil.move(extracted_bundle_dir, eval_bundle_dir)
    print0(f"Placed eval_bundle directory at {eval_bundle_dir}")


def evaluate_core(model, tokenizer, device, max_per_task=-1, num_recur=None, task_allowlist: set[str] | None = None):
    """
    Evaluate a base model on the CORE benchmark.
    Returns dict with results, centered_results, and core_metric.
    """
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    # Download the eval bundle if needed
    if not os.path.exists(eval_bundle_dir):
        download_file_with_lock(EVAL_BUNDLE_URL, "eval_bundle.zip", postprocess_fn=place_eval_bundle)

    config_path = os.path.join(eval_bundle_dir, "core.yaml")
    data_base_path = os.path.join(eval_bundle_dir, "eval_data")
    eval_meta_data = os.path.join(eval_bundle_dir, "eval_meta_data.csv")

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    tasks = config["icl_tasks"]
    tasks = _filter_tasks(tasks, task_allowlist)

    # Load random baseline values
    random_baselines = {}
    with open(eval_meta_data, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_name = row["Eval Task"]
            random_baseline = row["Random baseline"]
            random_baselines[task_name] = float(random_baseline)

    # Evaluate each task
    results = {}
    losses: dict[str, float] = {}
    centered_results = {}
    task_groups = {}  # label -> group name (derived from dataset_uri directory)
    for task in tasks:
        start_time = time.time()
        label = task["label"]
        group = task["dataset_uri"].split("/")[0]
        task_groups[label] = group
        task_meta = {
            "task_type": task["icl_task_type"],
            "dataset_uri": task["dataset_uri"],
            "num_fewshot": task["num_fewshot"][0],
            "continuation_delimiter": task.get("continuation_delimiter", " "),
        }
        print0(f"Evaluating: {label} ({task_meta['num_fewshot']}-shot, type: {task_meta['task_type']})... ", end="")

        data_path = os.path.join(data_base_path, task_meta["dataset_uri"])
        with open(data_path, encoding="utf-8") as f:
            data = [json.loads(line.strip()) for line in f]

        # Shuffle for consistent subsampling when using max_per_task
        shuffle_rng = random.Random(1337)
        shuffle_rng.shuffle(data)
        if max_per_task > 0:
            data = data[:max_per_task]

        metrics = evaluate_task(model, tokenizer, data, device, task_meta, num_recur=num_recur)
        accuracy = metrics.accuracy
        # Keep raw accuracy as the main per-task result for back-compat with CSV writer;
        # expose continuation loss alongside so it threads into axis aggregation.
        results[label] = accuracy
        losses[label] = metrics.loss
        random_baseline = random_baselines[label]
        centered_result = (accuracy - 0.01 * random_baseline) / (1.0 - 0.01 * random_baseline)
        centered_results[label] = centered_result
        elapsed = time.time() - start_time
        print0(f"accuracy: {accuracy:.4f} | loss: {metrics.loss:.4f} | centered: {centered_result:.4f} | time: {elapsed:.2f}s")

    # Group averages (centered scores)
    group_labels: dict[str, list[str]] = {}
    for label, group in task_groups.items():
        group_labels.setdefault(group, []).append(label)
    group_results = {g: sum(centered_results[l] for l in labels) / len(labels)
                     for g, labels in group_labels.items()}

    core_metric = sum(centered_results.values()) / len(centered_results)
    out = {"results": results, "losses": losses, "centered_results": centered_results,
           "group_results": group_results, "core_metric": core_metric}
    return out


# -----------------------------------------------------------------------------
# Saunshi evaluation


def evaluate_saunshi(model, tokenizer, device,
                     max_per_task: int = -1, num_recur: int | None = None,
                     task_allowlist: set[str] | None = None):
    """
    Evaluate a base model on the Saunshi downstream benchmark.
    Returns accuracy, continuation loss (nats), and perplexity per task and per group.
    Group/overall loss and PPL are macro-averaged (equal weight per task), not
    pooled across examples — consistent with how CORE and Saunshi average accuracy.
    """
    bundle_dir = Path(get_base_dir()) / "saunshi_bundle"
    config_path = bundle_dir / "saunshi.yaml"
    data_base_path = bundle_dir / "eval_data"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Saunshi bundle not found at {bundle_dir}. "
            "Run: uv run python dev/eval_bundles/saunshi/prepare_saunshi_bundle.py"
        )

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    tasks = config["tasks"]
    tasks = _filter_tasks(tasks, task_allowlist)

    results: dict[str, TaskMetrics] = {}
    for task in tasks:
        start_time = time.time()
        label = task["label"]
        task_meta = {
            "task_type": task["task_type"],
            "dataset_uri": task["dataset_uri"],
            "num_fewshot": task["num_fewshot"],
            "continuation_delimiter": task.get("continuation_delimiter", " "),
        }
        print0(f"Evaluating: {label} ({task_meta['num_fewshot']}-shot)... ", end="")

        data_path = data_base_path / task_meta["dataset_uri"]
        with open(data_path, encoding="utf-8") as f:
            data = [json.loads(line.strip()) for line in f]

        shuffle_rng = random.Random(1337)
        shuffle_rng.shuffle(data)
        if max_per_task > 0:
            data = data[:max_per_task]

        metrics = evaluate_task_full(model, tokenizer, data, device, task_meta,
                                     num_recur=num_recur)
        results[label] = metrics
        elapsed = time.time() - start_time
        print0(f"acc: {metrics.accuracy:.4f} | loss: {metrics.loss:.4f} | ppl: {metrics.ppl:.2f} | time: {elapsed:.2f}s")

    # Compute per-group averages
    group_tasks: dict[str, list[str]] = {}
    for task in tasks:
        g = task.get("group", "ungrouped")
        group_tasks.setdefault(g, []).append(task["label"])

    group_results: dict[str, TaskMetrics] = {}
    for group, labels in group_tasks.items():
        group_acc = sum(results[l].accuracy for l in labels) / len(labels)
        group_loss = sum(results[l].loss for l in labels) / len(labels)
        group_ppl = sum(results[l].ppl for l in labels) / len(labels)
        group_results[group] = TaskMetrics(accuracy=group_acc, loss=group_loss, ppl=group_ppl)

    n_groups = len(group_results)
    saunshi_metric = sum(g.accuracy for g in group_results.values()) / n_groups
    saunshi_loss = sum(g.loss for g in group_results.values()) / n_groups
    saunshi_ppl = sum(g.ppl for g in group_results.values()) / n_groups
    return {
        "results": results,
        "group_results": group_results,
        "saunshi_metric": saunshi_metric,
        "saunshi_loss": saunshi_loss,
        "saunshi_ppl": saunshi_ppl,
    }


# -----------------------------------------------------------------------------
# Owned taxonomy evaluation
# -----------------------------------------------------------------------------

OWNED_AXES: dict[str, dict[str, list[str]]] = {
    # axis -> {"core": [...], "saunshi": [...], "owned": [...]}
    "parametric_knowledge": {
        "core": [], "saunshi": ["triviaqa", "nq", "webq"], "owned": [],
    },
    "reading_comp": {
        "core": ["lambada_openai"],
        "saunshi": ["squadv2", "tydiqa_goldp", "drop", "coqa"],
        "owned": [],
    },
    "math_word_problems": {
        "core": [],
        "saunshi": ["svamp", "asdiv", "mawps"],
        "owned": [],
    },
    "reasoning_primitives": {
        "core": [],
        "saunshi": ["var_assign_d0_math", "var_assign_d0_code",
                    "var_assign_d1_math", "var_assign_d1_code"],
        "owned": ["induction_head"],
    },
    "compositional_symbolic": {
        "core": ["bigbench_dyck_languages", "bigbench_qa_wikidata", "arc_easy",
                 "bigbench_cs_algorithms"],
        "saunshi": [], "owned": [],
    },
}

FLAGSHIP_EXTRA_AXES: dict[str, dict[str, list[str]]] = {
    "commonsense": {
        "core": ["piqa", "hellaswag_zeroshot", "openbook_qa", "winograd"],
        "saunshi": [], "owned": [],
    },
}


def _collect_owned(axes: dict[str, dict[str, list[str]]]) -> tuple[set, set, set]:
    core: set[str] = set()
    saunshi: set[str] = set()
    owned: set[str] = set()
    for a in axes.values():
        core.update(a["core"]); saunshi.update(a["saunshi"]); owned.update(a["owned"])
    return core, saunshi, owned


def evaluate_owned_bundle(model, tokenizer, device, task_allowlist: set[str],
                           max_per_task: int = -1, num_recur: int | None = None) -> dict:
    """Evaluate the owned-bundle tasks (induction_head).
    Returns {label: TaskMetrics}."""
    bundle_dir = Path(get_base_dir()) / "owned_bundle"
    config_path = bundle_dir / "owned.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Owned bundle not found at {bundle_dir}. "
            "Run: uv run python dev/eval_bundles/owned/prepare_owned_bundle.py"
        )
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    tasks = _filter_tasks(config["tasks"], task_allowlist)
    results: dict[str, TaskMetrics] = {}
    for task in tasks:
        start_time = time.time()
        label = task["label"]
        task_meta = {
            "task_type": task["task_type"],
            "dataset_uri": task["dataset_uri"],
            "num_fewshot": task["num_fewshot"],
            "continuation_delimiter": task.get("continuation_delimiter", " "),
        }
        print0(f"Evaluating: {label} ({task_meta['num_fewshot']}-shot)... ", end="")
        data_path = bundle_dir / "eval_data" / task_meta["dataset_uri"]
        with open(data_path, encoding="utf-8") as f:
            data = [json.loads(l) for l in f]
        random.Random(1337).shuffle(data)
        if max_per_task > 0:
            data = data[:max_per_task]
        metrics = evaluate_task_full(model, tokenizer, data, device, task_meta,
                                     num_recur=num_recur)
        results[label] = metrics
        elapsed = time.time() - start_time
        print0(f"acc: {metrics.accuracy:.4f} | loss: {metrics.loss:.4f} | ppl: {metrics.ppl:.2f} | time: {elapsed:.2f}s")
    return results


def evaluate_owned(model, tokenizer, device, axes: dict[str, dict[str, list[str]]],
                    max_per_task: int = -1, num_recur: int | None = None) -> dict:
    """Run filtered CORE + filtered Saunshi + owned bundle, aggregate per axis.

    Per-axis accuracy is a macro-average of *raw* task accuracies (not CORE-
    centered). Raw CORE accuracy mixed with raw Saunshi accuracy is internally
    consistent for arch-vs-arch comparisons (same tasks, same scoring across
    archs) but not comparable to the top-line CORE centered metric.
    """
    core_labels, saunshi_labels, owned_labels = _collect_owned(axes)

    core_out = evaluate_core(model, tokenizer, device, max_per_task=max_per_task,
                              num_recur=num_recur,
                              task_allowlist=core_labels) if core_labels else None
    saunshi_out = evaluate_saunshi(model, tokenizer, device, max_per_task=max_per_task,
                                    num_recur=num_recur,
                                    task_allowlist=saunshi_labels) if saunshi_labels else None
    owned_out = evaluate_owned_bundle(model, tokenizer, device,
                                       task_allowlist=owned_labels,
                                       max_per_task=max_per_task,
                                       num_recur=num_recur) if owned_labels else None

    axis_results: dict[str, dict] = {}
    for axis_name, parts in axes.items():
        accs: list[float] = []
        losses: list[float] = []
        for label in parts["core"]:
            if core_out and label in core_out["results"]:
                accs.append(core_out["results"][label])
                if "losses" in core_out and label in core_out["losses"]:
                    losses.append(core_out["losses"][label])
        for label in parts["saunshi"]:
            if saunshi_out and label in saunshi_out["results"]:
                m = saunshi_out["results"][label]
                accs.append(m.accuracy); losses.append(m.loss)
        for label in parts["owned"]:
            if owned_out and label in owned_out:
                m = owned_out[label]
                accs.append(m.accuracy); losses.append(m.loss)
        axis_results[axis_name] = {
            "accuracy": sum(accs) / len(accs) if accs else float("nan"),
            "loss": sum(losses) / len(losses) if losses else float("nan"),
            "n_acc": len(accs), "n_loss": len(losses),
        }
    valid = [r["accuracy"] for r in axis_results.values() if r["n_acc"] > 0]
    owned_metric = sum(valid) / len(valid) if valid else float("nan")
    return {"axis_results": axis_results, "owned_metric": owned_metric,
            "raw_core": core_out, "raw_saunshi": saunshi_out, "raw_owned": owned_out}


# -----------------------------------------------------------------------------
# Main


def main():
    parser = argparse.ArgumentParser(description="Base model evaluation")
    parser.add_argument(
        "--eval", type=str, default="owned",
        help="Comma-separated evaluations: core, loss, sample, saunshi, owned "
             "(default: owned)"
    )
    parser.add_argument("--hf-path", type=str, default=None, help="HuggingFace model path (e.g. openai-community/gpt2)")
    parser.add_argument("--model-tag", type=str, default=None, help="nanochat model tag to identify the checkpoint directory")
    parser.add_argument("--checkpoints-dir", type=str, default=None, help="Custom checkpoints directory (default: base_checkpoints)")
    parser.add_argument("--step", type=int, default=None, help="Model step to load (default = last)")
    parser.add_argument("--max-per-task", type=int, default=-1, help="Max examples per CORE task (-1 = all)")
    parser.add_argument("--device-batch-size", type=int, default=32, help="Per-device batch size for loss evaluation")
    parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
    parser.add_argument("--num-recur", type=str, default=None, help="Comma-separated recurrence depths to evaluate, e.g. '2,4,6' (default = model's num_recur)")
    parser.add_argument("--kv-budget", type=int, default=None,
                        help="KV cache budget: number of recurrence slots cached per layer. "
                             "Default = num_recur (cache all recurrences, allowing attention across all loop states). "
                             "Set to 1 to cache only the final recurrence (memory-efficient).")
    parser.add_argument("--regime", type=str, default="isoflops",
                        choices=["isoflops", "flagship"],
                        help="isoflops=Tier A axes only; flagship=adds commonsense axis")

    args = parser.parse_args()

    # Parse evaluation modes
    eval_modes = set(mode.strip() for mode in args.eval.split(","))
    valid_modes = {"core", "loss", "sample", "saunshi", "owned"}
    invalid = eval_modes - valid_modes
    if invalid:
        parser.error(f"Invalid eval modes: {invalid}. Valid: {valid_modes}")

    # Distributed / precision setup
    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()

    # Load model and tokenizer
    is_hf_model = args.hf_path is not None
    if is_hf_model:
        model, tokenizer = load_hf_model(args.hf_path, device)
        sequence_len = model.max_seq_len or 1024
    else:
        if args.checkpoints_dir:
            model, tokenizer, meta = load_model_from_dir(args.checkpoints_dir, device, phase="eval",
                                                          model_tag=args.model_tag, step=args.step)
        else:
            model, tokenizer, meta = load_model("base", device, phase="eval", model_tag=args.model_tag, step=args.step)
        sequence_len = meta["model_config"]["sequence_len"]

    # Prepacked data directory
    prepacked_dir = os.path.join(get_base_dir(), f"prepacked_T{sequence_len}_llama")
    assert os.path.isdir(prepacked_dir), (
        f"Pre-packed data not found at {prepacked_dir}. "
        f"Run 'python -m scripts.prepack' first."
    )

    # Parse num_recur into a list of values to sweep
    if args.num_recur is not None:
        num_recur_values = [int(x) for x in args.num_recur.split(",")]
    elif is_hf_model:
        num_recur_values = [None]
    else:
        raw = meta["model_config"].get("num_recur")
        num_recur_values = [int(raw) if raw is not None else None]

    print0(f"Eval modes: {', '.join(sorted(eval_modes))}")
    print0(f"Recursion depths to evaluate: {num_recur_values}")
    print0(f"Regime: {args.regime}")

    # Base slug for combined CSV output (without recur suffix)
    if is_hf_model:
        base_slug = args.hf_path.replace("/", "-")
    else:
        base_slug = f"{args.model_tag}_step{meta['step']:06d}"

    # Summary CSV: one row per num_recur with aggregate metrics
    base_dir = get_base_dir()
    eval_dir = os.path.join(base_dir, "base_eval")
    os.makedirs(eval_dir, exist_ok=True)
    summary_csv_path = os.path.join(eval_dir, f"{base_slug}.csv")

    def read_completed_recur(csv_path: str) -> set[str]:
        """Read a CSV and return the set of num_recur values already present."""
        if not os.path.exists(csv_path):
            return set()
        completed = set()
        with open(csv_path, encoding="utf-8") as f:
            for line in f:
                if line.startswith("#") or line.startswith("num_recur"):
                    continue
                parts = line.split(",", 1)
                if parts:
                    completed.add(parts[0].strip())
        return completed

    completed_recur = read_completed_recur(summary_csv_path)

    # Write comment header with hyperparameters (once, when creating the file)
    if ddp_rank == 0 and not os.path.exists(summary_csv_path):
        model_id = args.hf_path if is_hf_model else f"{args.model_tag}_step{meta['step']}"
        with open(summary_csv_path, "w", encoding="utf-8", newline="") as f:
            f.write(f"# model={model_id}, eval={args.eval}"
                    f", device_batch_size={args.device_batch_size}"
                    f", eval_rows={EVAL_ROWS_FULL}"
                    f", max_per_task={args.max_per_task}\n")

    for num_recur in num_recur_values:
        nr_str = str(num_recur)

        if nr_str in completed_recur:
            print0(f"\nSkipping num_recur={num_recur} (already in CSV)")
            continue

        # Build model name for this recursion depth
        if is_hf_model:
            model_name = args.hf_path
        else:
            model_name = f"{args.model_tag} (step {meta['step']}, num_recur={num_recur})"

        print0(f"\n{'#' * 80}")
        print0(f"Evaluating model: {model_name}")
        print0(f"{'#' * 80}")

        # Results to log
        core_results = None
        saunshi_results = None
        owned_results = None
        loss_results = {}

        # --- CORE evaluation ---
        if "core" in eval_modes:
            print0("\n" + "=" * 80)
            print0("CORE Evaluation")
            print0("=" * 80)
            with autocast_ctx:
                core_results = evaluate_core(model, tokenizer, device, max_per_task=args.max_per_task,
                                             num_recur=num_recur)
            print0("-" * 80)
            for group, acc in core_results["group_results"].items():
                print0(f"  {group:<30s} {acc:.4f}")
            print0("-" * 80)
            print0(f"  {'core_metric':<30s} {core_results['core_metric']:.4f}")

        # --- Saunshi evaluation ---
        if "saunshi" in eval_modes:
            print0("\n" + "=" * 80)
            print0("Saunshi Evaluation")
            print0("=" * 80)
            with autocast_ctx:
                saunshi_results = evaluate_saunshi(model, tokenizer, device,
                                                   max_per_task=args.max_per_task, num_recur=num_recur)
            print0("-" * 80)
            for group, m in saunshi_results["group_results"].items():
                print0(f"  {group:<30s} acc: {m.accuracy:.4f} | loss: {m.loss:.4f} | ppl: {m.ppl:.2f}")
            print0("-" * 80)
            print0(f"  {'saunshi_metric':<30s} acc: {saunshi_results['saunshi_metric']:.4f}"
                   f" | loss: {saunshi_results['saunshi_loss']:.4f}"
                   f" | ppl: {saunshi_results['saunshi_ppl']:.2f}")

        # --- Owned taxonomy evaluation ---
        if "owned" in eval_modes:
            print0("\n" + "=" * 80)
            print0("Owned Taxonomy Evaluation")
            print0("=" * 80)
            axes = {**OWNED_AXES}
            if args.regime == "flagship":
                axes.update(FLAGSHIP_EXTRA_AXES)
            # Default per-task cap of 10k examples for owned mode — keeps wall-clock
            # bounded on big tasks (TriviaQA, BBH-QA-Wikidata) while leaving signal
            # intact. Overridable via --max-per-task.
            owned_max = args.max_per_task if args.max_per_task > 0 else 10000
            print0(f"Owned mode max-per-task cap: {owned_max}")
            with autocast_ctx:
                owned_results = evaluate_owned(model, tokenizer, device, axes,
                                                max_per_task=owned_max,
                                                num_recur=num_recur)
            print0("-" * 80)
            for axis, r in owned_results["axis_results"].items():
                print0(f"  {axis:<28s} acc={r['accuracy']:.4f} loss={r['loss']:.4f}"
                       f" (n_acc={r['n_acc']} n_loss={r['n_loss']})")
            print0(f"  {'owned_metric':<28s} {owned_results['owned_metric']:.4f}")

        # --- Loss evaluation ---
        if "loss" in eval_modes:
            print0("\n" + "=" * 80)
            print0("Loss Evaluation")
            print0("=" * 80)
            eval_rows = EVAL_ROWS_FULL
            rows_per_step = args.device_batch_size * ddp_world_size
            if eval_rows % rows_per_step != 0:
                eval_rows = (eval_rows // rows_per_step) * rows_per_step
                print0(f"Adjusted eval_rows to {eval_rows} (must be divisible by {rows_per_step})")
            steps = eval_rows // rows_per_step

            for split_name in ["train", "val"]:
                loader = prepacked_eval_loader(prepacked_dir, args.device_batch_size, sequence_len, split_name, device=device)
                t0 = time.time()
                with autocast_ctx:
                    metrics = evaluate_loss(model, loader, steps, num_recur=num_recur)
                elapsed = time.time() - t0
                loss_results[split_name] = metrics
                print0(f"{split_name} loss: {metrics.loss:.6f} | ppl: {metrics.ppl:.4f} | time: {elapsed:.1f}s")

        # --- Append summary row to CSV ---
        if ddp_rank == 0:
            # Build header and row dynamically based on what was evaluated
            columns = ["num_recur"]
            values = [nr_str]
            if core_results:
                columns.append("core_metric")
                values.append(f"{core_results['core_metric']:.6f}")
                for label in core_results["centered_results"]:
                    columns.append(label)
                    values.append(f"{core_results['centered_results'][label]:.6f}")
                if "losses" in core_results:
                    for label, loss_val in core_results["losses"].items():
                        columns.append(f"{label}_loss")
                        values.append(f"{loss_val:.6f}")
            if saunshi_results:
                columns.extend(["saunshi_metric", "saunshi_loss", "saunshi_ppl"])
                values.append(f"{saunshi_results['saunshi_metric']:.6f}")
                values.append(f"{saunshi_results['saunshi_loss']:.6f}")
                values.append(f"{saunshi_results['saunshi_ppl']:.4f}")
                for group, m in saunshi_results["group_results"].items():
                    columns.extend([f"saunshi_{group}", f"saunshi_{group}_loss", f"saunshi_{group}_ppl"])
                    values.append(f"{m.accuracy:.6f}")
                    values.append(f"{m.loss:.6f}")
                    values.append(f"{m.ppl:.4f}")
                for label, m in saunshi_results["results"].items():
                    columns.extend([f"saunshi_{label}", f"saunshi_{label}_loss", f"saunshi_{label}_ppl"])
                    values.append(f"{m.accuracy:.6f}")
                    values.append(f"{m.loss:.6f}")
                    values.append(f"{m.ppl:.4f}")
            if owned_results:
                columns.append("owned_metric")
                values.append(f"{owned_results['owned_metric']:.6f}")
                for axis, r in owned_results["axis_results"].items():
                    columns.extend([f"owned_{axis}_acc", f"owned_{axis}_loss"])
                    values.append(f"{r['accuracy']:.6f}")
                    values.append(f"{r['loss']:.6f}")
                # Per-task values — essential for re-aggregating under
                # later taxonomy changes without re-running the eval.
                raw_core = owned_results.get("raw_core")
                if raw_core:
                    cent = raw_core.get("centered_results", {})
                    raw_acc = raw_core.get("results", {})
                    losses = raw_core.get("losses", {})
                    for label in sorted(set(cent) | set(raw_acc) | set(losses)):
                        if label in cent:
                            columns.append(f"owned_task_{label}_core_centered")
                            values.append(f"{cent[label]:.6f}")
                        if label in raw_acc:
                            columns.append(f"owned_task_{label}_acc")
                            values.append(f"{raw_acc[label]:.6f}")
                        if label in losses:
                            columns.append(f"owned_task_{label}_loss")
                            values.append(f"{losses[label]:.6f}")
                raw_saunshi = owned_results.get("raw_saunshi")
                if raw_saunshi and "results" in raw_saunshi:
                    for label, m in raw_saunshi["results"].items():
                        columns.extend([f"owned_task_{label}_acc",
                                        f"owned_task_{label}_loss",
                                        f"owned_task_{label}_ppl"])
                        values.append(f"{m.accuracy:.6f}")
                        values.append(f"{m.loss:.6f}")
                        values.append(f"{m.ppl:.4f}")
                raw_owned = owned_results.get("raw_owned")
                if raw_owned:
                    for label, m in raw_owned.items():
                        columns.extend([f"owned_task_{label}_acc",
                                        f"owned_task_{label}_loss"])
                        values.append(f"{m.accuracy:.6f}")
                        values.append(f"{m.loss:.6f}")
            if loss_results:
                columns.extend(["train_loss", "val_loss", "train_ppl", "val_ppl"])
                values.append(f"{loss_results['train'].loss:.6f}")
                values.append(f"{loss_results['val'].loss:.6f}")
                values.append(f"{loss_results['train'].ppl:.4f}")
                values.append(f"{loss_results['val'].ppl:.4f}")

            # Guard against mixed-mode append: if an existing data-row header
            # in the CSV differs from the columns we're about to append,
            # the owned_* columns would end up headerless. Fail loudly.
            if ddp_rank == 0 and os.path.exists(summary_csv_path):
                with open(summary_csv_path, encoding="utf-8") as f:
                    existing_cols = None
                    for line in f:
                        if line.startswith("#") or not line.strip():
                            continue
                        existing_cols = line.strip().split(",")
                        break
                if existing_cols and set(existing_cols) != set(columns):
                    new_cols = set(columns) - set(existing_cols)
                    missing_cols = set(existing_cols) - set(columns)
                    raise RuntimeError(
                        f"CSV schema mismatch for {summary_csv_path}: "
                        f"existing header has {len(existing_cols)} cols, "
                        f"new row has {len(columns)} cols. "
                        f"New columns: {new_cols}. Missing: {missing_cols}. "
                        f"Delete the CSV and re-run to avoid headerless columns."
                    )

            # Write header row if this is the first data row (no non-comment
            # non-blank line exists yet).
            write_header = True
            if os.path.exists(summary_csv_path):
                with open(summary_csv_path, encoding="utf-8") as f:
                    for line in f:
                        if line.strip() and not line.startswith("#"):
                            write_header = False
                            break
            with open(summary_csv_path, "a", encoding="utf-8", newline="") as f:
                if write_header:
                    f.write(",".join(columns) + "\n")
                f.write(",".join(values) + "\n")
            print0(f"\nSummary appended to: {summary_csv_path}")

        # --- Sampling ---
        if "sample" in eval_modes and not is_hf_model:
            print0("\n" + "=" * 80)
            print0("Model Samples")
            print0("=" * 80)
            if ddp_rank == 0:
                prompts = [
                    "The capital of France is",
                    "The chemical symbol of gold is",
                    "If yesterday was Friday, then tomorrow will be",
                    "The opposite of hot is",
                    "The planets of the solar system are:",
                    "My favorite color is",
                    "If 5*x + 3 = 13, then x is",
                ]
                engine = Engine(model, tokenizer)
                print0("\nConditioned samples:")
                for prompt in prompts:
                    tokens = tokenizer(prompt, prepend="<s>")
                    with autocast_ctx:
                        sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0, num_recur=num_recur, kv_budget=args.kv_budget)
                    sample_str = tokenizer.decode(sample[0])
                    print0("-" * 80)
                    print0(sample_str)

                print0("\nUnconditioned samples:")
                tokens = tokenizer("", prepend="<s>")
                with autocast_ctx:
                    uncond, _ = engine.generate_batch(tokens, num_samples=8, max_tokens=128, temperature=1.0, num_recur=num_recur, kv_budget=args.kv_budget)
                for sample in uncond:
                    sample_str = tokenizer.decode(sample)
                    print0("-" * 80)
                    print0(sample_str)
        elif "sample" in eval_modes and is_hf_model:
            print0("\nSkipping sampling for HuggingFace models (not supported)")

    compute_cleanup()


if __name__ == "__main__":
    main()
