"""
Functions for evaluating the CORE metric, as described in the DCLM paper.
https://arxiv.org/abs/2406.11794

TODOs:
- All tasks ~match except for squad. We get 31% reference is 37%. Figure out why.
"""

import math
import random
from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.distributed as dist
from jinja2 import Template


@dataclass
class TaskMetrics:
    accuracy: float
    loss: float  # mean cross-entropy in nats per token
    ppl: float


class ExampleLossInfo(NamedTuple):
    is_correct: bool
    loss_sum: float        # sum of per-token cross-entropy (nats) over continuation
    num_tokens: int        # number of continuation tokens

# -----------------------------------------------------------------------------
# Prompt rendering utilities


def render_prompts_mc(item, continuation_delimiter, fewshot_examples=None):
    """Render complete prompts for a multiple choice question"""
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.query }}{{ continuation_delimiter }}{{ example.choices[example.gold] }}

{% endfor -%}
{{ item.query }}{{ continuation_delimiter }}{{ choice }}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        "fewshot_examples": fewshot_examples,
        "continuation_delimiter": continuation_delimiter,
        "item": item,
    }
    prompts = [template.render(choice=choice, **context) for choice in item["choices"]]
    return prompts


def render_prompts_schema(item, continuation_delimiter, fewshot_examples=None):
    """Render complete prompts for a schema question"""
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.context_options[example.gold] }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ context }}{{ continuation_delimiter }}{{ item.continuation }}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        "fewshot_examples": fewshot_examples,
        "continuation_delimiter": continuation_delimiter,
        "item": item,
    }
    prompts = [template.render(context=context_option, **context) for context_option in item["context_options"]]
    return prompts


def render_prompts_lm(item, continuation_delimiter, fewshot_examples=None):
    """
    Render complete prompt for a language modeling task.
    Notice that we manually trim the context in the template,
    which in some datasets seems to have trailing whitespace (which we don't want).
    """
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.context | trim }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ item.context | trim }}{{ continuation_delimiter }}{% if include_continuation %}{{ item.continuation }}{% endif %}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        "fewshot_examples": fewshot_examples,
        "continuation_delimiter": continuation_delimiter,
        "item": item,
    }
    # Return two prompts: without and with the continuation
    prompt_without = template.render(include_continuation=False, **context)
    prompt_with = template.render(include_continuation=True, **context)
    # Due to the way the data seems to be stored, I think I need to strip in the case of LM here.
    # Otherwise we may get trailing whitespaces in prompt_without (which get absorbed into the next
    # token in prompt_with), meaning we don't get a nice and clean prefix in the token space
    # to detect the final continuation. Tokenizers...
    prompt_without = prompt_without.strip()
    return [prompt_without, prompt_with]


def find_common_length(token_sequences, direction="left"):
    """
    Find the length of the common prefix or suffix across token sequences
    - direction: 'left' for prefix, 'right' for suffix
    """
    min_len = min(len(seq) for seq in token_sequences)
    indices = {"left": range(min_len), "right": range(-1, -min_len - 1, -1)}[direction]
    # Find the first position where the token sequences differ
    for i, idx in enumerate(indices):
        token = token_sequences[0][idx]
        if not all(seq[idx] == token for seq in token_sequences):
            return i
    return min_len


def stack_sequences(tokens, pad_token_id):
    """Stack up a list of token sequences, pad to longest on the right"""
    bsz, seq_len = len(tokens), max(len(x) for x in tokens)
    input_ids = torch.full((bsz, seq_len), pad_token_id, dtype=torch.long)
    for i, x in enumerate(tokens):
        input_ids[i, : len(x)] = torch.tensor(x, dtype=torch.long)
    return input_ids


def batch_sequences_mc(tokenizer, prompts):
    # In multiple choice, contexts are the same but the continuation is different (common prefix)
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    # figure out the start and end of each continuation
    answer_start_idx = find_common_length(tokens, direction="left")
    start_indices = [answer_start_idx] * len(prompts)
    end_indices = [len(x) for x in tokens]
    return tokens, start_indices, end_indices


def batch_sequences_schema(tokenizer, prompts):
    # In schema tasks, contexts vary but continuation is the same (common suffix)
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    # figure out the start and end of each context
    suffix_length = find_common_length(tokens, direction="right")
    end_indices = [len(x) for x in tokens]
    start_indices = [ei - suffix_length for ei in end_indices]
    return tokens, start_indices, end_indices


def batch_sequences_lm(tokenizer, prompts):
    # In LM tasks, we have two prompts: without and with continuation
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    tokens_without, tokens_with = tokens
    start_idx, end_idx = len(tokens_without), len(tokens_with)
    assert start_idx < end_idx, "prompt without is supposed to be a prefix of prompt with"
    assert tokens_without == tokens_with[:start_idx], "prompt without is supposed to be a prefix of prompt with"
    # we only need the with continuation prompt in the LM task, i.e. batch size of 1
    return [tokens_with], [start_idx], [end_idx]


@torch.no_grad()
def forward_model(model, input_ids, num_recur=None):
    """
    Take BxT tensor of token ids, return BxT tensor of losses and argmax predictions.
    The last column of losses is set to nan because we don't have autoregressive targets there.
    """
    batch_size, seq_len = input_ids.size()
    logits = model(input_ids, num_recur=num_recur)
    # Roll the tensor to the left by one position to get the (autoregressive) target ids
    target_ids = torch.roll(input_ids, shifts=-1, dims=1)
    # Calculate cross entropy at all positions
    losses = torch.nn.functional.cross_entropy(
        logits.view(batch_size * seq_len, -1),
        target_ids.view(batch_size * seq_len),
        reduction="none",
    ).view(batch_size, seq_len)
    # Set the last column to be nan because there is no autoregressive loss there
    losses[:, -1] = float("nan")
    # Get the argmax predictions at each position
    predictions = logits.argmax(dim=-1)
    return losses, predictions


@torch.no_grad()
def evaluate_example(idx, model, tokenizer, data, device, task_meta, num_recur=None,
                     return_losses: bool = False):
    """Evaluate a single example. Returns bool (is_correct) by default.
    If return_losses=True, returns ExampleLossInfo across all task types:
      - language_modeling: sum of per-token CE on the continuation span
      - multiple_choice / schema: sum of per-token CE on the *gold* continuation span
        (the span the task scores against — same span used to pick the argmax).
    """
    item = data[idx]
    task_type = task_meta["task_type"]
    num_fewshot = task_meta["num_fewshot"]
    continuation_delimiter = task_meta["continuation_delimiter"]

    # Sample few-shot examples (excluding current item)
    fewshot_examples = []
    if num_fewshot > 0:
        rng = random.Random(1234 + idx)
        available_indices = [i for i in range(len(data)) if i != idx]
        fewshot_indices = rng.sample(available_indices, num_fewshot)
        fewshot_examples = [data[i] for i in fewshot_indices]

    # Render prompts and batch sequences based on task type
    if task_type == "multiple_choice":
        prompts = render_prompts_mc(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_mc(tokenizer, prompts)
    elif task_type == "schema":
        prompts = render_prompts_schema(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_schema(tokenizer, prompts)
    elif task_type == "language_modeling":
        prompts = render_prompts_lm(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_lm(tokenizer, prompts)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    # Some models can't forward sequences beyond a certain length (e.g. GPT-2)
    # In these cases, we have to truncate sequences to max length and adjust the indices
    if hasattr(model, "max_seq_len") and model.max_seq_len is not None:
        max_tokens = model.max_seq_len
        new_tokens, new_start_idxs, new_end_idxs = [], [], []
        for t, s, e in zip(tokens, start_idxs, end_idxs):
            if len(t) > max_tokens:
                num_to_crop = len(t) - max_tokens
                new_tokens.append(t[-max_tokens:])  # take the last max_tokens tokens
                new_start_idxs.append(s - num_to_crop)  # shift the indices down
                new_end_idxs.append(e - num_to_crop)
                assert s - num_to_crop >= 0, "this should never happen right?"
                assert e - num_to_crop >= 0, "this should never happen right?"
            else:
                new_tokens.append(t)  # keep unchanged
                new_start_idxs.append(s)
                new_end_idxs.append(e)
        tokens, start_idxs, end_idxs = new_tokens, new_start_idxs, new_end_idxs

    # Stack up all the sequences into a batch
    pad_token_id = tokenizer.get_bos_token_id()  # use BOS as pad token is ok
    input_ids = stack_sequences(tokens, pad_token_id)
    input_ids = input_ids.to(device)

    # Forward the model, get the autoregressive loss and argmax prediction at each token
    losses, predictions = forward_model(model, input_ids, num_recur=num_recur)

    # See if the losses/predictions come out correctly
    if task_type == "language_modeling":
        # language modeling task is currently always batch size 1
        si = start_idxs[0]
        ei = end_idxs[0]
        # predictions[i] predict input_ids[i+1] autoregressively
        predicted_tokens = predictions[0, si - 1 : ei - 1]
        actual_tokens = input_ids[0, si:ei]
        is_correct = torch.all(predicted_tokens == actual_tokens).item()
        if return_losses:
            continuation_losses = losses[0, si - 1 : ei - 1]
            return ExampleLossInfo(
                is_correct=is_correct,
                loss_sum=continuation_losses.sum().item(),
                num_tokens=ei - si,
            )
    elif task_type in ["multiple_choice", "schema"]:
        # For MC/schema: find the option with lowest average loss
        mean_losses = [losses[i, si - 1 : ei - 1].mean().item() for i, (si, ei) in enumerate(zip(start_idxs, end_idxs))]
        pred_idx = mean_losses.index(min(mean_losses))
        gold_idx = item["gold"]
        is_correct = pred_idx == gold_idx
        if return_losses:
            si, ei = start_idxs[gold_idx], end_idxs[gold_idx]
            gold_losses = losses[gold_idx, si - 1 : ei - 1]
            return ExampleLossInfo(
                is_correct=is_correct,
                loss_sum=gold_losses.sum().item(),
                num_tokens=ei - si,
            )
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    return is_correct


def evaluate_task(model, tokenizer, data, device, task_meta,
                  num_recur: int | None = None) -> TaskMetrics:
    """Evaluate a task across many examples.

    Returns a TaskMetrics with accuracy, mean continuation cross-entropy (nats/token),
    and perplexity. Works for all task types (language_modeling, multiple_choice, schema);
    for MC/schema the loss is the CE of the *gold* continuation span.

    Handles distributed dispatch (torchrun).
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    n = len(data)
    correct = torch.zeros(n, dtype=torch.float32, device=device)
    loss_nats = torch.zeros(n, dtype=torch.float64, device=device)
    num_tokens = torch.zeros(n, dtype=torch.float64, device=device)

    for idx in range(rank, n, world_size):
        result = evaluate_example(idx, model, tokenizer, data, device, task_meta,
                                  num_recur=num_recur, return_losses=True)
        correct[idx] = float(result.is_correct)
        loss_nats[idx] = result.loss_sum
        num_tokens[idx] = result.num_tokens

    if world_size > 1:
        dist.barrier()
        for t in (correct, loss_nats, num_tokens):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

    accuracy = correct.mean().item()
    total_nats = loss_nats.sum().item()
    total_tokens = num_tokens.sum().item()
    # Token-weighted mean: each gold-span token contributes equally. For MC
    # tasks with variable-length gold continuations this weights the aggregate
    # toward longer-gold examples. Bias is symmetric across architectures, so
    # arch-vs-arch gaps are unaffected; only absolute values are length-biased.
    loss = total_nats / total_tokens if total_tokens > 0 else float("nan")
    ppl = math.exp(loss) if total_tokens > 0 else float("nan")
    return TaskMetrics(accuracy=accuracy, loss=loss, ppl=ppl)


# Backwards-compatible alias: evaluate_task_full used to be the loss-aware variant,
# restricted to language_modeling. evaluate_task now returns TaskMetrics for all
# task types, so they are the same function.
evaluate_task_full = evaluate_task
