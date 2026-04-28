"""
A number of functions that help with evaluating a base model.
"""

import math
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class LossMetrics:
    loss: float  # mean cross-entropy in nats per token
    ppl: float   # perplexity = exp(loss)


@torch.no_grad()
def evaluate_loss(model, batches, steps, num_recur=None) -> LossMetrics:
    """
    Compute mean cross-entropy loss (nats per token) and perplexity.
    Ignores target tokens with index < 0 (e.g. padding with ignore_index=-1).
    """
    total_nats = torch.tensor(0.0, dtype=torch.float64, device=model.get_device())
    total_tokens = torch.tensor(0, dtype=torch.int64, device=model.get_device())
    batch_iter = iter(batches)
    for _ in range(steps):
        x, y = next(batch_iter)
        loss2d = model(x, y, loss_reduction="none", num_recur=num_recur)  # (B, T)
        loss2d = loss2d.view(-1)
        y = y.view(-1)
        if (y.int() < 0).any():
            valid = y >= 0
            total_nats += loss2d[valid].sum()
            total_tokens += valid.sum()
        else:
            total_nats += loss2d.sum()
            total_tokens += y.numel()
    # sum reduce across all ranks
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size > 1:
        dist.all_reduce(total_nats, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)
    total_nats = total_nats.item()
    total_tokens = total_tokens.item()
    if total_tokens == 0:
        return LossMetrics(loss=float("inf"), ppl=float("inf"))
    loss = total_nats / total_tokens
    ppl = math.exp(loss)
    return LossMetrics(loss=loss, ppl=ppl)
