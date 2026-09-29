"""One Gloo rank. Invoked as a subprocess by the driver."""
import argparse
import json
import os
import time

import torch
import torch.distributed as dist

from .core import Comm, logical_update
from .data import make_dataset, shard_dataset, make_plan, count_valid
from .model import init_model


def snapshot(model, optimizer):
    params = {n: p.detach().clone() for n, p in model.named_parameters()}
    state = {}
    for n, p in model.named_parameters():
        st = optimizer.state.get(p, {})
        state[n] = {k: (v.clone() if torch.is_tensor(v) else v)
                    for k, v in st.items()}
    return {"params": params, "state": state}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--world-size", type=int, default=2)
    ap.add_argument("--init", required=True)  # host:port
    ap.add_argument("--mb-sizes", required=True,
                    help="colon-separated updates, each comma list of mb sizes")
    ap.add_argument("--split-idx", type=int, required=True)
    ap.add_argument("--updates", type=int, required=True)
    ap.add_argument("--bucketed", action="store_true")
    ap.add_argument("--fault-rank", type=int, default=-1)
    ap.add_argument("--fault-step", type=int, default=-1)
    ap.add_argument("--hang-rank", type=int, default=-1)
    ap.add_argument("--hang-step", type=int, default=-1)
    ap.add_argument("--hang-before-collective", type=int, default=0)
    ap.add_argument("--bench-warmup", type=int, default=0)
    ap.add_argument("--bench-iters", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    host, port = args.init.split(":")
    os.environ["MASTER_ADDR"] = host
    os.environ["MASTER_PORT"] = port
    os.environ["RANK"] = str(args.rank)
    os.environ["WORLD_SIZE"] = str(args.world_size)
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"

    dist.init_process_group(
        backend="gloo", rank=args.rank, world_size=args.world_size,
    )
    comm = Comm(dist.group.WORLD)

    model = init_model(2024)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2,
                                  betas=(0.9, 0.999), weight_decay=0.1)

    dataset = make_dataset()
    shards = shard_dataset(dataset, args.split_idx)
    shard = shards[args.rank]

    update_specs = [
        [int(x) for x in s.split(",") if x != ""]
        for s in args.mb_sizes.split(":")
    ]
    plans = [make_plan(shard, spec) for spec in update_specs]

    # Data is reused across logical updates (toy streaming source); the point
    # is multi-update optimizer-state agreement, not generalization.
    plans = (plans * ((args.updates + len(plans) - 1) // len(plans)))[:args.updates]

    history = []
    snapshots = []
    dist.barrier()
    for step, plan in enumerate(plans):
        fault = (args.rank == args.fault_rank and step == args.fault_step)
        if args.rank == args.hang_rank and step == args.hang_step:
            # Simulate a rank dropping out before the first collective:
            # the other rank must detect the timeout instead of hanging tests.
            time.sleep(60)
        res = logical_update(comm, model, optimizer, plan,
                             bucketed=args.bucketed,
                             fault_this_update=fault)
        history.append(res)
        snapshots.append(snapshot(model, optimizer))
        dist.barrier()

    bench = None
    train_stats = dict(comm.stats.as_dict())
    if args.bench_iters > 0:
        bench = run_bench(comm, args)
        bench["train_calls_before_bench"] = train_stats["collective_calls"]

    dist.barrier()
    result = {
        "rank": args.rank,
        "history": history,
        "snapshots": [{
            "params": {k: v.tolist() for k, v in s["params"].items()},
            "state": {
                k: {kk: (vv.tolist() if torch.is_tensor(vv) else vv)
                    for kk, vv in v.items()}
                for k, v in s["state"].items()
            },
        } for s in snapshots],
        "stats": train_stats,
        "bench": bench,
        "local_tokens_per_update": [
            count_valid(plan.samples()) for plan in plans
        ],
    }
    with open(args.out, "w") as f:
        json.dump(result, f)
    dist.barrier()
    dist.destroy_process_group()


def run_bench(comm, args):
    """Timed measured updates with warmup; includes synchronization.

    Timing deliberately brackets blocking collectives plus a final barrier, so
    it is completion time of the slowest rank, never async-launch time.
    """
    from .model import build_model

    model = build_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    g = torch.Generator().manual_seed(7 + args.rank)
    x = torch.randn(32, 5, generator=g, dtype=torch.float64)
    y = torch.randint(0, 4, (32,), generator=g)

    def step_once():
        opt.zero_grad(set_to_none=False)
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        loss.backward()
        params = list(model.parameters())
        if args.bucketed:
            from .core import allreduce_grads_bucketed
            allreduce_grads_bucketed(comm, params)
        else:
            from .core import allreduce_grads_per_param
            allreduce_grads_per_param(comm, params)
        opt.step()

    for _ in range(args.bench_warmup):
        step_once()
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(args.bench_iters):
        step_once()
    dist.barrier()
    elapsed = time.perf_counter() - t0
    return {
        "mode": "bucketed" if args.bucketed else "per_param",
        "elapsed_ms": elapsed * 1000.0,
        "iters": args.bench_iters,
    }


if __name__ == "__main__":
    main()
