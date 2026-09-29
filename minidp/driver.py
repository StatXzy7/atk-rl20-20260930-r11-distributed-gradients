"""Launch two Gloo ranks as subprocesses, run the oracle, compare everything."""
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field

import torch

from .data import make_dataset, shard_dataset, make_plan, oracle_plan
from .model import init_model
from .oracle import run_oracle
from .losses import token_loss_sum
from .core import global_grad_norm


def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@dataclass
class Scenario:
    name: str
    split_idx: int
    # mb_specs[step][rank] = list of microbatch sizes; [] = empty rank
    mb_specs: list
    updates: int = 3
    bucketed: bool = False
    fault_rank: int = -1
    fault_step: int = -1


def run_workers(scn, timeout=60, bench_warmup=0, bench_iters=0,
                hang_rank=-1, hang_step=-1):
    port = free_port()
    init = f"127.0.0.1:{port}"
    tmp = tempfile.mkdtemp(prefix="minidp_")
    procs, outs = [], []
    cmd_base = [sys.executable, "-m", "minidp.worker",
                "--world-size", "2", "--init", init,
                "--split-idx", str(scn.split_idx),
                "--updates", str(scn.updates),
                "--fault-rank", str(scn.fault_rank),
                "--fault-step", str(scn.fault_step),
                "--bench-warmup", str(bench_warmup),
                "--bench-iters", str(bench_iters),
                "--hang-rank", str(hang_rank),
                "--hang-step", str(hang_step)]
    if scn.bucketed:
        cmd_base.append("--bucketed")
    for rank in range(2):
        spec = ":".join(",".join(str(z) for z in scn.mb_specs[step][rank])
                        for step in range(len(scn.mb_specs)))
        out = os.path.join(tmp, f"rank{rank}.json")
        cmd = cmd_base + ["--rank", str(rank),
                          "--mb-sizes", spec, "--out", out]
        procs.append(subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE))
        outs.append(out)
    timed_out = []
    deadline = time.monotonic() + timeout
    results = [None, None]
    for rank, p in enumerate(procs):
        rem = max(1.0, deadline - time.monotonic())
        try:
            out, err = p.communicate(timeout=rem)
        except subprocess.TimeoutExpired:
            p.kill()
            p.communicate()
            timed_out.append(rank)
            continue
        if p.returncode != 0:
            raise RuntimeError(f"rank {rank} failed:\n{err.decode()}")
        with open(outs[rank]) as f:
            results[rank] = json.load(f)
    return results, timed_out, tmp


def build_oracle(scn):
    model = init_model(2024)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2,
                                  betas=(0.9, 0.999), weight_decay=0.1)
    dataset = make_dataset()
    plans, faulted = [], set()
    for step in range(scn.updates):
        spec_step = scn.mb_specs[step % len(scn.mb_specs)]
        shard0 = dataset[:scn.split_idx]
        shard1 = dataset[scn.split_idx:]
        p0 = make_plan(shard0, spec_step[0])
        p1 = make_plan(shard1, spec_step[1])
        global_samples = p0.samples() + p1.samples()
        plans.append(oracle_plan(global_samples))
        if scn.fault_rank >= 0 and step == scn.fault_step:
            faulted.add(step)
    history = run_oracle(model, optimizer, plans,
                         faulted_updates=faulted)
    return history


def _to_tensors(snap):
    return ({k: torch.tensor(v, dtype=torch.float64)
             for k, v in snap["params"].items()},
            {k: {kk: (torch.tensor(vv, dtype=torch.float64)
                      if isinstance(vv, list) else vv)
                 for kk, vv in v.items()}
             for k, v in snap["state"].items()})


def compare(scn, results):
    """Compare rank0 vs rank1 and each vs oracle; also recompute grads."""
    oracle_hist = build_oracle(scn)
    r0, r1 = results
    report = {"per_step": [], "max_abs_diff": 0.0}

    # independently recompute the full global gradient each applied step
    dataset = make_dataset()
    gmodel = init_model(2024)
    gopt = torch.optim.AdamW(gmodel.parameters(), lr=1e-2,
                             betas=(0.9, 0.999), weight_decay=0.1)

    for step in range(scn.updates):
        spec_step = scn.mb_specs[step % len(scn.mb_specs)]
        shard0 = dataset[:scn.split_idx]
        shard1 = dataset[scn.split_idx:]
        samples = (make_plan(shard0, spec_step[0]).samples() +
                   make_plan(shard1, spec_step[1]).samples())
        gopt.zero_grad(set_to_none=False)
        total_loss = torch.zeros((), dtype=torch.float64)
        toks = 0
        for x, y in samples:
            s, n = token_loss_sum(gmodel, x, y)
            total_loss = total_loss + s
            toks += n
        (total_loss / toks).backward()
        recomputed_norm = float(global_grad_norm(list(gmodel.parameters())).item())
        gnorm_pre = oracle_hist[step]["global_grad_norm_pre_clip"]
        if gnorm_pre is not None:
            assert abs(recomputed_norm - gnorm_pre) < 1e-9, (
                step, recomputed_norm, gnorm_pre)

        applied = r0["history"][step]["applied"]
        assert applied == r1["history"][step]["applied"]
        assert applied == oracle_hist[step]["applied"]
        assert r0["history"][step]["global_tokens"] == toks

        if applied:
            gnorm = global_grad_norm(list(gmodel.parameters())).item()
            clip = min(1.0, 1.0 / gnorm) if gnorm > 0 else 1.0
            if clip != 1.0:
                for p in gmodel.parameters():
                    p.grad.mul_(clip)
            gopt.step()

        p0, s0 = _to_tensors(r0["snapshots"][step])
        p1, s1 = _to_tensors(r1["snapshots"][step])
        step_max = 0.0
        for name in p0:
            step_max = max(step_max,
                           float((p0[name] - p1[name]).abs().max()))
        # compare rank0 to independently re-stepped oracle model
        for (name, p) in gmodel.named_parameters():
            step_max = max(step_max, float((p0[name] - p.detach()).abs().max()))
        # optimizer state agreement
        for name in s0:
            for kk in ("exp_avg", "exp_avg_sq"):
                step_max = max(
                    step_max,
                    float((s0[name][kk] - s1[name][kk]).abs().max()))
        report["per_step"].append({
            "step": step, "applied": applied,
            "global_tokens": toks,
            "grad_norm": gnorm_pre,
            "max_abs_param_or_state_diff": step_max,
        })
        report["max_abs_diff"] = max(report["max_abs_diff"], step_max)
    report["stats"] = r0["stats"]
    report["bench"] = [r0.get("bench"), r1.get("bench")]
    report["oracle_history"] = oracle_hist
    return report
