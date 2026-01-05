#!/usr/bin/env python
"""
Run each benchmark for the PyTorch baseline and ActorPool worker counts to produce a table row.
"""
from __future__ import annotations

import argparse
import dataclasses
import math
import re
import subprocess
from typing import Dict, List, Optional

WORKERS = [2, 4, 6]


@dataclasses.dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    command_template: str  # must contain {workers}


SPECS: List[BenchmarkSpec] = [
    BenchmarkSpec(
        "Minimum Diameter Averaging (MDA)",
        "python benchmarks/pytorch/mda_actor_pool.py --num-grads 30 --grad-dim 2048 --f 10 --chunk-size 256 "
        "--pool-backend process --repeat 2 --warmup 0",
    ),
    BenchmarkSpec(
        "Coordinate-wise Trimmed Mean (CwTM)",
        "python benchmarks/pytorch/cwtm_actor_pool.py --num-grads 64 --grad-dim 65536 --f 8 --chunk-size 8192 "
        "--pool-backend process --repeat 2 --warmup 0",
    ),
    BenchmarkSpec(
        "Multi-Krum",
        "python benchmarks/pytorch/multikrum_actor_pool.py --num-grads 80 --grad-dim 65536 --f 20 --q 12 --chunk-size 20 "
        "--pool-backend process --repeat 3 --warmup 0",
    ),
    BenchmarkSpec(
        "Centered Clipping",
        "python benchmarks/pytorch/centered_clipping_actor_pool.py --num-grads 64 --grad-dim 65536 --c-tau 0.1 --iters 10 "
        "--chunk-size 32 --pool-backend process --repeat 3 --warmup 0",
    ),
    BenchmarkSpec(
        "Comparative Gradient Elimination",
        "python benchmarks/pytorch/cge_actor_pool.py --num-grads 64 --grad-dim 65536 --f 8 --chunk-size 16384 "
        "--pool-backend process --repeat 3 --warmup 0",
    ),
    BenchmarkSpec(
        "Empire Attack",
        "python benchmarks/pytorch/empire_actor_pool.py --num-grads 64 --grad-dim 65536 --chunk-size 32 "
        "--pool-backend process --repeat 3 --warmup 0",
    ),
    BenchmarkSpec(
        "Little Attack",
        "python benchmarks/pytorch/little_actor_pool.py --num-grads 96 --grad-dim 65536 --f 12 --chunk-size 8192 "
        "--pool-backend process --repeat 3 --warmup 0",
    ),
    BenchmarkSpec(
        "Nearest Neighbor Mixing",
        "python benchmarks/pytorch/nnm_actor_pool.py --num-vectors 196 --dim 4096 --f 32 --chunk-size 2048 "
        "--pool-backend process --repeat 3 --warmup 0",
    ),
    BenchmarkSpec(
        "Mean-of-Medians (MeaMed)",
        "python benchmarks/pytorch/meamed_actor_pool.py --num-grads 64 --grad-dim 65536 --f 8 --chunk-size 8192 "
        "--pool-backend process --repeat 2 --warmup 0",
    ),
    BenchmarkSpec(
        "Bucketing (pre-agg)",
        "python benchmarks/pytorch/bucketing_actor_pool.py --num-vectors 512 --dim 16384 --bucket-size 32 "
        "--pool-backend process --repeat 3 --warmup 0",
    ),
]


def _run_cmd(cmd: str) -> str:
    print(f"Running: {cmd}")
    result = subprocess.run(
        cmd,
        shell=True,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _parse_output(out: str) -> tuple[float, Dict[int, float]]:
    direct_match = re.search(r"Direct.*?([0-9]+\.[0-9]+)", out)
    if not direct_match:
        raise ValueError("Unable to parse direct runtime.")
    direct_ms = float(direct_match.group(1))

    actor_ms: Dict[int, float] = {}
    for match in re.finditer(r"ActorPool x(\d+).*?([0-9]+\.[0-9]+)", out):
        workers = int(match.group(1))
        actor_ms[workers] = float(match.group(2))
    return direct_ms, actor_ms


def _format_speedup(direct: float, runtime: float) -> str:
    speedup = direct / runtime if runtime > 0 else math.inf
    return f"{runtime:.2f} ({speedup:.2f}×)"


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile benchmarks for selected worker counts.")
    parser.add_argument(
        "--bench",
        action="append",
        help="Benchmark name (substring) to run. Repeatable.",
    )
    parser.add_argument(
        "--workers",
        default="2,4,6",
        help="Comma-separated worker counts (default: 2,4,6)",
    )
    args = parser.parse_args()

    selected_workers = [int(w.strip()) for w in args.workers.split(",") if w.strip()]

    worker_arg = ",".join(str(w) for w in selected_workers)
    rows = []
    for spec in SPECS:
        if args.bench and not any(term.lower() in spec.name.lower() for term in args.bench):
            continue
        cmd = spec.command_template
        if worker_arg:
            cmd = f"{cmd} --pool-workers {worker_arg}"
        output = _run_cmd(cmd)
        direct_ms, actor_ms = _parse_output(output)
        baseline_ms: Optional[float] = direct_ms
        actor_results: Dict[int, float] = actor_ms
        row = [
            spec.name,
            f"{baseline_ms:.2f}" if baseline_ms is not None else "n/a",
        ]
        for workers in selected_workers:
            runtime = actor_results.get(workers)
            row.append(_format_speedup(baseline_ms, runtime) if runtime else "n/a")
        rows.append(row)

    worker_headers = [f"x{w}" for w in selected_workers]
    header = ["Workload", "PyTorch (ms)", *worker_headers]
    print("\nSummary Table:\n")
    print(" | ".join(header))
    print(" | ".join("---" for _ in header))
    for row in rows:
        print(" | ".join(row))


if __name__ == "__main__":
    main()
