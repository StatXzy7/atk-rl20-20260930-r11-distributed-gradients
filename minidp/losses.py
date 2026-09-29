"""Summed token-weighted cross entropy.

Objective for a logical update:

    loss = sum_{ranks, microbatches, valid positions} CE /
           total number of valid positions

so local per-microbatch losses are *summed*, and the divide happens once with
the globally all-reduced valid-token count.
"""
import torch

from .data import IGNORE


def token_loss_sum(model, x, y):
    """Return (scalar sum of CE over valid positions, valid token count)."""
    logits = model(x.long())  # [S, V]
    per_token = torch.nn.functional.cross_entropy(
        logits, y, ignore_index=IGNORE, reduction="none"
    )
    mask = y != IGNORE
    ntok = int(mask.sum().item())
    # Masked sum: when the microbatch has zero valid tokens this is a scalar 0
    # still attached to the graph through logits, so backward() zeros grads
    # deterministically instead of detaching.
    total = per_token.masked_fill(~mask, 0.0).sum()
    return total, ntok


def compute_loss_scale(local_token_total, global_token_total):
    if global_token_total <= 0:
        return None
    return local_token_total / float(global_token_total)
