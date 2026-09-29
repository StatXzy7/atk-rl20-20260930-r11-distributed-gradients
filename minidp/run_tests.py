"""End-to-end scenarios. No GPU, no external data, no DDP.

Each distributed scenario launches two real Gloo subprocesses with a timeout
so a collective-ordering bug manifests as a detected timeout, not an infinite
hang.
"""
import json
import sys
import time

from .driver import Scenario, run_workers, compare


SCENARIOS = [
    # 1. extremely uneven shards + tail microbatch + 3 consecutive updates
    Scenario(
        name="uneven_tail",
        split_idx=20,                 # rank0: 20 samples, rank1: 2
        mb_specs=[
            [[8, 7, 5], [1, 0, 1]],  # rank1 has an empty mb and a size-1 tail
            [[8, 7, 5], [2]],
            [[8, 7, 5], [1, 0, 0, 1]],
        ],
        updates=3,
    ),
    # 2. a rank with zero valid data for a whole logical update
    Scenario(
        name="empty_rank",
        split_idx=22,                 # rank1 owns nothing at all
        mb_specs=[
            [[9, 8, 5], []],
            [[9, 8, 5], []],
        ],
        updates=2,
    ),
    # 3. fault on rank0 on the middle update; both ranks must skip
    Scenario(
        name="fault_rank0_step1",
        split_idx=15,
        mb_specs=[
            [[8, 7], [7]],
            [[8, 7], [7]],
            [[8, 7], [7]],
        ],
        updates=3,
        fault_rank=0,
        fault_step=1,
    ),
    # 4. fault on the data-heavy rank after an empty-mb phase
    Scenario(
        name="fault_heavy_after_empty",
        split_idx=20,
        mb_specs=[
            [[8, 7, 5], []],
            [[8, 7, 5], [1, 0, 1]],
        ],
        updates=2,
        fault_rank=0,
        fault_step=1,
    ),
    # 5. same uneven workload as #1 but gradient all-reduce via buckets
    Scenario(
        name="uneven_bucketed",
        split_idx=20,
        mb_specs=[
            [[8, 7, 5], [1, 0, 1]],
            [[8, 7, 5], [2]],
            [[8, 7, 5], [1, 0, 0, 1]],
        ],
        updates=3,
        bucketed=True,
    ),
]


def main():
    summary = []
    for scn in SCENARIOS:
        t0 = time.perf_counter()
        results, timed_out, tmp = run_workers(scn, timeout=60)
        assert not timed_out, f"{scn.name}: unexpected worker timeout"
        report = compare(scn, results)
        report["name"] = scn.name
        report["wall_s"] = round(time.perf_counter() - t0, 2)
        summary.append(report)
        print(f"[ok] {scn.name}  wall={report['wall_s']}s  "
              f"max_abs_diff={report['max_abs_diff']:.2e}  "
              f"collectives={report['stats']['collective_calls']}  "
              f"tensor_bytes={report['stats']['tensor_bytes']}")
        for st in report["per_step"]:
            print(f"     step{st['step']} applied={st['applied']} "
                  f"tokens={st['global_tokens']} "
                  f"gnorm={None if st['grad_norm'] is None else round(st['grad_norm'], 6)} "
                  f"diff={st['max_abs_param_or_state_diff']:.2e}")

    # Deadlock detection: one rank sleeps instead of joining collectives.
    hang = Scenario(name="hang_detection", split_idx=15,
                    mb_specs=[[[8, 7], [7]]], updates=1)
    t0 = time.perf_counter()
    results, timed_out, _ = run_workers(hang, timeout=12, hang_rank=1,
                                        hang_step=0)
    dt = time.perf_counter() - t0
    assert 1 in timed_out, "deadlocking worker was not killed by timeout"
    print(f"[ok] hang_detection  rank1 killed after ~{dt:.1f}s timeout")

    # Timed bucket-vs-perparam comparison, with warmup and full sync.
    bench_scn = Scenario(name="bench", split_idx=15,
                         mb_specs=[[[8, 7], [7]]], updates=1)
    bench = {}
    for mode, bucketed in (("per_param", False), ("bucketed", True)):
        b = Scenario(name=f"bench_{mode}", split_idx=15,
                     mb_specs=[[[8, 7], [7]]], updates=1, bucketed=bucketed)
        res, to, _ = run_workers(b, timeout=90, bench_warmup=10,
                                 bench_iters=100)
        assert not to
        # slowest rank is the end-to-end time
        ms = max(r["bench"]["elapsed_ms"] for r in res)
        calls = res[0]["bench"]["train_calls_before_bench"]
        bench[mode] = {"slowest_rank_ms": round(ms, 2),
                       "train_sum_collectives": calls}
        print(f"[bench] {mode}: {ms:.1f} ms/100 iters (slowest rank, synced)")
    print(f"[bench] message counts per-param={bench['per_param']} "
          f"bucketed={bench['bucketed']}  "
          "(fewer messages != faster end-to-end)")

    clean = []
    for r in summary:
        clean.append({k: v for k, v in r.items()
                      if k not in ("oracle_history",)})
    out = {"scenarios": clean, "hang_timeout_s": round(dt, 2),
           "bench": bench}
    with open("test_report.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("\nwrote test_report.json")


if __name__ == "__main__":
    main()
