"""Single-process full-batch oracle.

Consumes the ENTIRE dataset (one sample per microbatch, same per-microbatch
loss scaling mb_tokens/T) so its accumulated gradient is exactly

    sum over every global valid token of dCE/dtheta / T

which is the same quantity the two-rank SUM all-reduce produces. No data is
dropped to make shards equal.
"""
import torch

from .core import global_grad_norm
from .losses import token_loss_sum


def run_oracle(model, optimizer, plans, *, max_norm=1.0, faulted_updates=()):
    """``plans`` is a list (one entry per logical update) of Plan objects whose
    concatenated samples are the full dataset.

    ``faulted_updates`` mirrors the distributed scenario: on a faulted update
    the oracle computes grads (including the injected non-finite value on the
    faulting rank's first parameter shard) but skips clip/step, just like the
    synchronized skip in the distributed run.
    """
    history = []
    for step, plan in enumerate(plans):
        params = [p for _, p in model.named_parameters()]
        optimizer.zero_grad(set_to_none=False)

        total_tokens = 0
        total_loss = torch.zeros((), dtype=torch.float64)
        for mb in plan.microbatches:
            mb_loss = torch.zeros((), dtype=torch.float64)
            mb_tokens = 0
            for x, y in mb:
                s, n = token_loss_sum(model, x, y)
                mb_loss = mb_loss + s
                mb_tokens += n
            total_tokens += mb_tokens
            total_loss = total_loss + mb_loss

        assert total_tokens > 0
        (total_loss / float(total_tokens)).backward()

        fault = step in set(faulted_updates)
        if fault:
            with torch.no_grad():
                params[0].grad.add_(float("nan"))

        applied = False
        norm_value = None
        if all(torch.isfinite(p.grad).all().item() for p in params):
            gnorm = global_grad_norm(params)
            norm_value = float(gnorm.item())
            clip = min(1.0, max_norm / norm_value) if norm_value > 0 else 1.0
            if clip != 1.0:
                with torch.no_grad():
                    for p in params:
                        p.grad.mul_(clip)
            optimizer.step()
            applied = True

        history.append({
            "applied": applied,
            "global_tokens": total_tokens,
            "global_grad_norm_pre_clip": norm_value,
            "global_loss": float(total_loss.item()) / total_tokens,
        })
    return history
