"""Training core: local accumulation, SUM all-reduce, skip, global clip.

Correct order for one logical update
------------------------------------
1. Every rank independently runs all its microbatches, summing *raw* token
   loss and valid-token counts (no rank-local mean, no data discarded).
2. All-reduce SUM of the scalar valid-token count -> global token total T.
3. Each rank scales its summed token loss by 1/T. NOTE the factor is 1/T,
   not mb_tokens/T: mb_loss is already a *sum* of per-token losses, so
   multiplying by mb_tokens/T would weight every token an extra mb_tokens
   times. Backward then accumulates local_sum_grad/T.
4. Gradients are all-reduced with SUM. A MEAN all-reduce would be WRONG:
   grads are token-weighted, not rank-averaged. Empty ranks simply contribute
   zeros and still must call the collective in the same order.
5. Non-finite detection: every rank checks its local grads; an all-reduce MIN
   of 0.0/1.0 flags propagates any bad rank to ALL ranks.
6. Only if the global finite flag is true do all ranks clip by the SAME global
   grad norm (every rank holds identical reduced grads, so no extra collective
   is needed) and step AdamW. Otherwise params AND optimizer state are skipped
   on every rank.

Collective-order invariant: the set/order of collectives on every update is
identical on every rank INDEPENDENT of how much data it owns. Local compute and
optimizer.step() are gated; collectives never are. That is why an all-empty
world still runs the gradient and finite collectives on zero tensors instead
of skipping them.
"""
import torch
import torch.distributed as dist

from .losses import token_loss_sum

BUCKET_CAP_BYTES = 640  # demo model: 7 grad tensors pack into 2 buckets


class Stats:
    def __init__(self):
        self.collective_calls = 0
        self.tensor_bytes = 0
        self.by_kind = {}

    def record(self, kind, nbytes):
        self.collective_calls += 1
        self.tensor_bytes += int(nbytes)
        self.by_kind[kind] = self.by_kind.get(kind, 0) + 1

    def as_dict(self):
        return {
            "collective_calls": self.collective_calls,
            "tensor_bytes": self.tensor_bytes,
            "by_kind": dict(self.by_kind),
        }


class Comm:
    """Counts collective calls and communicated payload bytes (one way)."""

    def __init__(self, pg):
        self.pg = pg
        self.stats = Stats()

    def allreduce_sum(self, t):
        self.stats.record("allreduce_sum", t.numel() * t.element_size())
        dist.all_reduce(t, op=dist.ReduceOp.SUM, group=self.pg)
        return t

    def allreduce_min(self, t):
        self.stats.record("allreduce_min", t.numel() * t.element_size())
        dist.all_reduce(t, op=dist.ReduceOp.MIN, group=self.pg)
        return t


def plan_buckets(params, cap_bytes=BUCKET_CAP_BYTES, dtype=torch.float64):
    """Greedily pack small grad tensors into flat contiguous buckets.

    All ranks build buckets from parameters in the identical order.
    """
    es = torch.empty((), dtype=dtype).element_size()
    buckets, cur, cur_bytes = [], [], 0
    for p in params:
        nbytes = p.numel() * es
        if cur and cur_bytes + nbytes > cap_bytes:
            buckets.append(cur)
            cur, cur_bytes = [], 0
        cur.append(p)
        cur_bytes += nbytes
    if cur:
        buckets.append(cur)
    return buckets


def allreduce_grads_per_param(comm, params):
    for p in params:
        comm.allreduce_sum(p.grad)


def allreduce_grads_bucketed(comm, params):
    for bucket in plan_buckets(params):
        flat = torch.cat([p.grad.reshape(-1) for p in bucket])
        comm.allreduce_sum(flat)
        offset = 0
        for p in bucket:
            n = p.numel()
            p.grad.copy_(flat[offset:offset + n].view_as(p.grad))
            offset += n


def local_accumulate(model, plan, optimizer, global_token_total):
    """Run local microbatches; grads end up as local_sum_grad / T.

    Returns (raw local token-loss sum, local valid token count). An empty plan
    yields loss 0 and zero gradients but touches every param via the zeroing.
    """
    optimizer.zero_grad(set_to_none=False)
    raw_loss = 0.0
    local_tokens = 0
    inv_t = 1.0 / float(global_token_total)
    for mb in plan.microbatches:
        mb_loss = torch.zeros((), dtype=torch.float64)
        mb_tokens = 0
        for x, y in mb:
            s, n = token_loss_sum(model, x, y)
            mb_loss = mb_loss + s
            mb_tokens += n
        if mb_tokens > 0:
            (mb_loss * inv_t).backward()
        else:
            # Graph-connected zero contribution: deterministic zero grads.
            (mb_loss * 0.0).backward()
        raw_loss += float(mb_loss.detach())
        local_tokens += mb_tokens
    return raw_loss, local_tokens


def grads_are_finite(params):
    return all(torch.isfinite(p.grad).all().item() for p in params)


def global_grad_norm(params):
    return torch.sqrt(sum(torch.sum(p.grad.detach() ** 2) for p in params))


def logical_update(comm, model, optimizer, plan, *, bucketed, max_norm=1.0,
                   fault_this_update=False):
    """One global logical update. All ranks issue identical collectives.

    Returns a result dict. ``applied`` is identical on every rank.
    """
    params = [p for _, p in model.named_parameters()]

    # ---- collective 1: global valid-token count (SUM) ----
    t_local = 0
    for mb in plan.microbatches:
        for _, y in mb:
            t_local += int((y != -100).sum().item())
    t_tensor = torch.tensor(float(t_local), dtype=torch.float64)
    comm.allreduce_sum(t_tensor)
    global_tokens = int(round(float(t_tensor.item())))

    # ---- local forward/backward (gated; no collectives inside) ----
    if global_tokens > 0:
        raw_loss, local_tokens = local_accumulate(
            model, plan, optimizer, global_tokens)
        if fault_this_update:
            with torch.no_grad():
                params[0].grad.add_(float("nan"))
    else:
        # Keep grads zero on every rank so the unconditional collectives below
        # remain well-defined even when the whole world has no valid token.
        optimizer.zero_grad(set_to_none=False)
        raw_loss = 0.0

    # ---- collectives 2..: gradient SUM all-reduce, UNCONDITIONAL ----
    if bucketed:
        allreduce_grads_bucketed(comm, params)
    else:
        allreduce_grads_per_param(comm, params)

    # ---- collective: global finite flag (MIN), UNCONDITIONAL ----
    # A rank with no data always reports 1 (its zeros cannot be non-finite).
    # A faulting rank's NaN survives the SUM all-reduce above, so its own
    # finite check fails; MIN propagates the 0 to every rank.
    if global_tokens > 0:
        ok = 1.0 if grads_are_finite(params) else 0.0
    else:
        ok = 1.0
    flag = torch.tensor(ok, dtype=torch.float64)
    comm.allreduce_min(flag)
    global_ok = bool(flag.item() >= 1.0)

    applied = False
    global_norm_value = None
    if global_tokens > 0 and global_ok:
        gnorm = global_grad_norm(params)
        global_norm_value = float(gnorm.item())
        if global_norm_value > 0:
            clip = min(1.0, max_norm / global_norm_value)
            if clip != 1.0:
                with torch.no_grad():
                    for p in params:
                        p.grad.mul_(clip)
        optimizer.step()
        applied = True

    return {
        "applied": applied,
        "global_tokens": global_tokens,
        "global_grad_norm_pre_clip": global_norm_value,
        "raw_local_loss": raw_loss,
    }
