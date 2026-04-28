"""
Prepare the Saunshi downstream eval bundle.

Downloads datasets from HuggingFace and writes JSONL files to
~/.cache/nanochat/saunshi_bundle/ along with a saunshi.yaml config.

Based on the benchmark from Saunshi et al. (2502.17416), §5.1.
15 of 16 tasks implemented — QuAC is excluded due to a broken legacy
HuggingFace datasets loader (allenai/quac requires parquet workaround
and has inconsistent schema across versions).

JSONL format follows the CORE eval bundle convention:
- context includes the answer delimiter with trailing space (e.g. "...\\nAnswer: ")
- continuation has no leading whitespace (e.g. "Denver Broncos")
- continuation_delimiter in YAML is " " (default), except for reasoning primitives

Usage:
    uv run python dev/eval_bundles/saunshi/prepare_saunshi_bundle.py
"""

import json
import random
import re
import sys
from pathlib import Path

import yaml

from nanochat.common import get_base_dir


GROUPS = ["math_word_problems", "closed_book_qa", "open_book_qa", "reasoning_primitives"]


def get_bundle_dir() -> Path:
    bundle_dir = Path(get_base_dir()) / "saunshi_bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    eval_data = bundle_dir / "eval_data"
    eval_data.mkdir(exist_ok=True)
    for group in GROUPS:
        (eval_data / group).mkdir(exist_ok=True)
    return bundle_dir


def _write_jsonl(records: list[dict], out_path: Path, label: str, skipped: int = 0) -> int:
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    extra = f" (skipped {skipped})" if skipped else ""
    print(f"{label}: wrote {len(records)} records → {out_path}{extra}")
    return len(records)


# ==============================================================================
# Math word problems
# Format: "Question: {q}\nAnswer: " + "{numeric_answer}"
# Matches CORE TriviaQA-style prompt format.
# ==============================================================================

def prepare_svamp(out_dir: Path) -> int:
    from datasets import load_dataset

    ds = load_dataset("ChilleD/SVAMP", split="test")
    records = []
    for item in ds:
        question = item["Body"].strip().rstrip(".") + ". " + item["Question"].strip()
        fval = float(item["Answer"])
        answer = str(int(fval)) if fval == int(fval) else str(fval)
        records.append({"context": f"Question: {question}\nAnswer: ", "continuation": answer})
    return _write_jsonl(records, out_dir / "svamp.jsonl", "SVAMP")


def prepare_asdiv(out_dir: Path) -> int:
    from datasets import load_dataset

    ds = load_dataset("EleutherAI/asdiv", split="validation")
    records = []
    for item in ds:
        question = item["body"].strip().rstrip(".") + " " + item["question"].strip()
        raw_answer = item["answer"].strip()
        # Strip units in parentheses e.g. "9 (apples)" → "9"
        answer = re.sub(r"\s*\(.*?\)", "", raw_answer).strip()
        records.append({"context": f"Question: {question}\nAnswer: ", "continuation": answer})
    return _write_jsonl(records, out_dir / "asdiv.jsonl", "ASDiv")


def prepare_mawps(out_dir: Path) -> int:
    from datasets import load_dataset

    ds = load_dataset("mwpt5/MAWPS", split="train")
    records = []
    skipped = 0
    for item in ds:
        question_raw = item["Question"].strip()
        numbers_str = item.get("Numbers", "") or ""
        answer_raw = str(item["Answer"]).strip()

        # Substitute N_00, N_01, ... placeholders with actual numbers (space-separated).
        def fmt_num(s: str) -> str:
            try:
                f = float(s)
                return str(int(f)) if f == int(f) else s
            except ValueError:
                return s

        if numbers_str.strip():
            numbers = [fmt_num(n) for n in numbers_str.split()]
            question = question_raw
            for i, num in enumerate(numbers):
                placeholder = f"N_{i:02d}"
                question = question.replace(placeholder, num)
        else:
            question = question_raw

        if re.search(r"N_\d{2}", question):
            skipped += 1
            continue

        try:
            fval = float(answer_raw)
            answer = str(int(fval)) if fval == int(fval) else answer_raw
        except ValueError:
            answer = answer_raw
        records.append({"context": f"Question: {question}\nAnswer: ", "continuation": answer})
    return _write_jsonl(records, out_dir / "mawps.jsonl", "MAWPS", skipped=skipped)


# ==============================================================================
# Closed book QA
# Format: "Question: {q}\nAnswer: " + "{answer}"
# Matches CORE TriviaQA format exactly.
# ==============================================================================

def prepare_triviaqa(out_dir: Path) -> int:
    from datasets import load_dataset
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
    records = []
    for item in ds:
        answer = item["answer"]["value"].strip()
        if not answer:
            continue
        records.append({"context": f"Question: {item['question'].strip()}\nAnswer: ", "continuation": answer})
    return _write_jsonl(records, out_dir / "triviaqa.jsonl", "TriviaQA")


def prepare_nq(out_dir: Path) -> int:
    from datasets import load_dataset
    ds = load_dataset("nq_open", split="validation")
    records = []
    for item in ds:
        answers = item["answer"]
        if not answers:
            continue
        records.append({"context": f"Question: {item['question'].strip()}\nAnswer: ", "continuation": answers[0].strip()})
    return _write_jsonl(records, out_dir / "nq.jsonl", "NQ")


def prepare_webq(out_dir: Path) -> int:
    from datasets import load_dataset
    ds = load_dataset("web_questions", split="test")
    records = []
    for item in ds:
        answers = item["answers"]
        if not answers:
            continue
        records.append({"context": f"Question: {item['question'].strip()}\nAnswer: ", "continuation": answers[0].strip()})
    return _write_jsonl(records, out_dir / "webq.jsonl", "WebQ")


def prepare_tydiqa_nocontext(out_dir: Path) -> int:
    from datasets import load_dataset
    ds = load_dataset("google-research-datasets/tydiqa", "secondary_task", split="validation")
    records = []
    for item in ds:
        if item["id"].split("-")[0] != "english":
            continue
        answers = item["answers"]["text"]
        if not answers:
            continue
        records.append({"context": f"Question: {item['question'].strip()}\nAnswer: ", "continuation": answers[0].strip()})
    return _write_jsonl(records, out_dir / "tydiqa_nocontext.jsonl", "TydiQA-NoContext")


# ==============================================================================
# Open book QA
# Format: "Context: {passage}\nQuestion: {q}\nAnswer: " + "{answer}"
# Matches CORE SQuAD format (uses "Context:" prefix, not "Passage:").
# CoQA uses CORE's multi-turn format with "Story:" prefix and QA history.
# ==============================================================================

def prepare_tydiqa_goldp(out_dir: Path) -> int:
    from datasets import load_dataset
    ds = load_dataset("google-research-datasets/tydiqa", "secondary_task", split="validation")
    records = []
    for item in ds:
        if item["id"].split("-")[0] != "english":
            continue
        answers = item["answers"]["text"]
        if not answers:
            continue
        records.append({
            "context": f"Context: {item['context'].strip()}\nQuestion: {item['question'].strip()}\nAnswer: ",
            "continuation": answers[0].strip(),
        })
    return _write_jsonl(records, out_dir / "tydiqa_goldp.jsonl", "TydiQA-GoldP")


def prepare_squadv2(out_dir: Path) -> int:
    from datasets import load_dataset
    ds = load_dataset("rajpurkar/squad_v2", split="validation")
    records = []
    for item in ds:
        answers = item["answers"]["text"]
        if not answers:  # unanswerable
            continue
        records.append({
            "context": f"Context: {item['context'].strip()}\nQuestion: {item['question'].strip()}\nAnswer: ",
            "continuation": answers[0].strip(),
        })
    return _write_jsonl(records, out_dir / "squadv2.jsonl", "SquadV2")


def prepare_drop(out_dir: Path) -> int:
    from datasets import load_dataset
    ds = load_dataset("ucinlp/drop", split="validation")
    records = []
    for item in ds:
        spans = item["answers_spans"]["spans"]
        if not spans:
            continue
        records.append({
            "context": f"Context: {item['passage'].strip()}\nQuestion: {item['question'].strip()}\nAnswer: ",
            "continuation": spans[0].strip(),
        })
    return _write_jsonl(records, out_dir / "drop.jsonl", "Drop")


def prepare_coqa(out_dir: Path) -> int:
    """Prepare CoQA with multi-turn context, matching the CORE eval bundle format.

    CORE format:
        Below is a story followed by a series of related questions. ...
        Story: {passage}
        Preceding questions:
        Question: {q1}
        Answer: {a1}
        ...
        Final question:
        Question: {current_q}
        Answer:
    """
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/coqa", split="validation")
    preamble = ("Below is a story followed by a series of related questions. "
                "Please answer the final question by referring to the story "
                "and the previous questions.")
    records = []
    for item in ds:
        passage = item["story"].strip()
        questions = item["questions"]
        answers = item["answers"]["input_text"]
        for turn_idx, (q, a) in enumerate(zip(questions, answers)):
            if not a.strip():
                continue
            # Build context with QA history (matching CORE format)
            parts = [f"{preamble}\nStory: {passage}"]
            if turn_idx > 0:
                parts.append("Preceding questions:")
                for prev_q, prev_a in zip(questions[:turn_idx], answers[:turn_idx]):
                    parts.append(f"Question: {prev_q.strip()}\nAnswer: {prev_a.strip()}")
            parts.append(f"\nFinal question:\nQuestion: {q.strip()}\nAnswer: ")
            records.append({
                "context": "\n".join(parts),
                "continuation": a.strip(),
            })
    return _write_jsonl(records, out_dir / "coqa.jsonl", "CoQA")


# NOTE: QuAC (allenai/quac) is excluded from the benchmark. The HuggingFace
# datasets loader is broken for this dataset (requires parquet workaround with
# inconsistent schema across versions). This is the only task from Saunshi et al.
# that we drop. Open-book QA group has 4 tasks instead of the paper's 5.


# ==============================================================================
# Reasoning primitives — synthetic variable assignment
#
# Reimplements the depth-0 / depth-1 variable-assignment probes from Saunshi
# et al. 2024 (arXiv:2409.19044, Appendix B); reused verbatim in Saunshi et al.
# 2025 (arXiv:2502.17416). The task format is:
#
#   math:   "The following is a set of simple mathematical equations."\n\n
#           n=22\nr=16\n...\n
#           What is the numerical value of n?\nAnswer:
#   code:   "The following is a very short Python program. ..."\nProgram:\n
#           n = 22\nr = 16\n...\n
#           \nQuestion:\nWhat is the value of n?\nAnswer:
#
# Depth-0: 5 base vars, each assigned a direct integer in [1, 25];
#          query targets one of the 5 base vars.
# Depth-1: 5 base vars + 5 aliases, with a strict 1-to-1 base <-> alias map
#          (each alias references a distinct base). Bases listed first, then
#          aliases (no forward references). Query targets one of the 5 aliases.
#
# Random-guess baseline is 1/5 = 20% (the model can choose among the 5 base
# values that appear in the prompt). Continuation is the answer token
# (e.g. "22"); continuation_delimiter=" " is baked into the scaffolding.
# ==============================================================================

_VAR_LETTERS = list("abcdefghijklmnopqrstuvwxyz")
_VAL_LO, _VAL_HI = 1, 25
_N_VARS = 5

_MATH_HEADER = "The following is a set of simple mathematical equations."
_MATH_QUERY_TMPL = "What is the numerical value of {var}?\nAnswer:"
_CODE_HEADER = ("The following is a very short Python program. "
                "Use the program to resolve the value of the variable in the question.")
_CODE_QUERY_TMPL = "Question:\nWhat is the value of {var}?\nAnswer:"


def _make_var_assign_d0(rng: random.Random) -> dict:
    """5 base vars, direct integer assignments, query one base var."""
    bases = rng.sample(_VAR_LETTERS, _N_VARS)
    base_vals = {v: str(rng.randint(_VAL_LO, _VAL_HI)) for v in bases}
    query_var = rng.choice(bases)
    assignments = [(v, base_vals[v]) for v in bases]
    return {"assignments": assignments, "query": query_var,
            "answer": base_vals[query_var]}


def _make_var_assign_d1(rng: random.Random) -> dict:
    """5 base vars + 5 aliases (strict 1-to-1). Bases first then aliases."""
    all_vars = rng.sample(_VAR_LETTERS, 2 * _N_VARS)
    bases = all_vars[:_N_VARS]
    aliases = all_vars[_N_VARS:]
    base_vals = {v: str(rng.randint(_VAL_LO, _VAL_HI)) for v in bases}
    shuffled_bases = bases[:]
    rng.shuffle(shuffled_bases)
    alias_map = dict(zip(aliases, shuffled_bases))
    query_var = rng.choice(aliases)
    answer = base_vals[alias_map[query_var]]
    assignments = (
        [(v, base_vals[v]) for v in bases]          # bases first
        + [(a, alias_map[a]) for a in aliases]      # then aliases
    )
    return {"assignments": assignments, "query": query_var, "answer": answer}


def _to_math(ex: dict) -> dict:
    body = "\n".join(f"{v}={val}" for v, val in ex["assignments"])
    ctx = (f"{_MATH_HEADER}\n\n{body}\n"
           f"{_MATH_QUERY_TMPL.format(var=ex['query'])}")
    return {"context": ctx, "continuation": ex["answer"]}


def _to_code(ex: dict) -> dict:
    body = "\n".join(f"{v} = {val}" for v, val in ex["assignments"])
    ctx = (f"{_CODE_HEADER}\nProgram:\n{body}\n\n"
           f"{_CODE_QUERY_TMPL.format(var=ex['query'])}")
    return {"context": ctx, "continuation": ex["answer"]}


def prepare_reasoning_primitives(out_dir: Path, n_examples: int = 1000, seed: int = 42) -> None:
    rng = random.Random(seed)
    tasks = [
        ("var_assign_d0_math", _make_var_assign_d0, _to_math),
        ("var_assign_d0_code", _make_var_assign_d0, _to_code),
        ("var_assign_d1_math", _make_var_assign_d1, _to_math),
        ("var_assign_d1_code", _make_var_assign_d1, _to_code),
    ]
    for label, maker, formatter in tasks:
        records = [formatter(maker(rng)) for _ in range(n_examples)]
        _write_jsonl(records, out_dir / f"{label}.jsonl", label)


# -----------------------------------------------------------------------------
# YAML config
# -----------------------------------------------------------------------------

def _task(label: str, uri: str, group: str, num_fewshot: int = 5,
          continuation_delimiter: str = " ") -> dict:
    """Build a task config entry. Default delimiter is " " (CORE convention)."""
    return {"label": label, "dataset_uri": uri, "task_type": "language_modeling",
            "num_fewshot": num_fewshot, "continuation_delimiter": continuation_delimiter,
            "group": group}


ALL_TASKS = [
    # Math word problems — delimiter baked into context as "\nAnswer: "
    _task("svamp", "math_word_problems/svamp.jsonl", "math_word_problems"),
    _task("asdiv", "math_word_problems/asdiv.jsonl", "math_word_problems"),
    _task("mawps", "math_word_problems/mawps.jsonl", "math_word_problems"),
    # Closed book QA — delimiter baked into context as "\nAnswer: "
    _task("triviaqa",         "closed_book_qa/triviaqa.jsonl",         "closed_book_qa"),
    _task("nq",               "closed_book_qa/nq.jsonl",               "closed_book_qa"),
    _task("webq",             "closed_book_qa/webq.jsonl",             "closed_book_qa"),
    _task("tydiqa_nocontext", "closed_book_qa/tydiqa_nocontext.jsonl", "closed_book_qa"),
    # Open book QA — delimiter baked into context; 3-shot (passages make prompts long)
    _task("tydiqa_goldp", "open_book_qa/tydiqa_goldp.jsonl", "open_book_qa", num_fewshot=3),
    _task("squadv2",      "open_book_qa/squadv2.jsonl",      "open_book_qa", num_fewshot=3),
    _task("drop",         "open_book_qa/drop.jsonl",         "open_book_qa", num_fewshot=3),
    _task("coqa",         "open_book_qa/coqa.jsonl",         "open_book_qa", num_fewshot=1),
    # Reasoning primitives — scaffolding ends in "Answer:" so delimiter is " "
    _task("var_assign_d0_math", "reasoning_primitives/var_assign_d0_math.jsonl",
          "reasoning_primitives"),
    _task("var_assign_d0_code", "reasoning_primitives/var_assign_d0_code.jsonl",
          "reasoning_primitives"),
    _task("var_assign_d1_math", "reasoning_primitives/var_assign_d1_math.jsonl",
          "reasoning_primitives"),
    _task("var_assign_d1_code", "reasoning_primitives/var_assign_d1_code.jsonl",
          "reasoning_primitives"),
]


def write_yaml(bundle_dir: Path, tasks: list[dict]) -> None:
    groups = list(dict.fromkeys(t["group"] for t in tasks))
    config = {"groups": groups, "tasks": tasks}
    out_path = bundle_dir / "saunshi.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"Config: wrote {out_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    bundle_dir = get_bundle_dir()
    data_dir = bundle_dir / "eval_data"

    print(f"Bundle dir: {bundle_dir}\n")

    errors = []

    datasets = [
        ("SVAMP",            prepare_svamp,            "math_word_problems"),
        ("ASDiv",            prepare_asdiv,            "math_word_problems"),
        ("MAWPS",            prepare_mawps,            "math_word_problems"),
        ("TriviaQA",         prepare_triviaqa,         "closed_book_qa"),
        ("NQ",               prepare_nq,               "closed_book_qa"),
        ("WebQ",             prepare_webq,             "closed_book_qa"),
        ("TydiQA-NoContext", prepare_tydiqa_nocontext, "closed_book_qa"),
        ("TydiQA-GoldP",     prepare_tydiqa_goldp,     "open_book_qa"),
        ("SquadV2",          prepare_squadv2,          "open_book_qa"),
        ("Drop",             prepare_drop,             "open_book_qa"),
        ("CoQA",             prepare_coqa,             "open_book_qa"),
    ]
    for name, fn, group in datasets:
        try:
            fn(data_dir / group)
        except Exception as e:
            print(f"ERROR {name}: {e}", file=sys.stderr)
            errors.append(name)

    try:
        prepare_reasoning_primitives(data_dir / "reasoning_primitives")
    except Exception as e:
        print(f"ERROR ReasoningPrimitives: {e}", file=sys.stderr)
        errors.append("ReasoningPrimitives")

    if errors:
        print(f"\nFailed: {errors}. Fix errors above and re-run.", file=sys.stderr)
        sys.exit(1)

    write_yaml(bundle_dir, ALL_TASKS)


if __name__ == "__main__":
    main()
