#!/usr/bin/env python
"""Benchmark the CAF aggregator with and without ActorPool execution."""
from __future__ import annotations

import argparse
import asyncio
import time
from dataclasses import dataclass
from typing import Sequence

import torch

from byzpy.aggregators.norm_wise.caf import CAF
from byzpy.engine.graph.ops import make_single_operator_graph
from byzpy.engine.graph.pool import ActorPool, ActorPoolConfig
from byzpy.engine.graph.scheduler import NodeScheduler

try:
    from ._worker_args import DEFAULT_WORKER_COUNTS, coerce_worker_counts, parse_worker_counts
except ImportError:
    from _worker_args import DEFAULT_WORKER_COUNTS  # type: ignore
    from _worker_args import coerce_worker_counts, parse_worker_counts


@dataclass(frozen=True)
class BenchmarkRun:
    mode: str
    avg_seconds: float

    @property
    def avg_ms(self) -> float:
        return self.avg_seconds * 1_000.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark CAF with ActorPool vs PyTorch.")
    parser.add_argument("--num-grads", type=int, default=64, help="Number of gradients (n).")
    parser.add_argument("--grad-dim", type=int, default=65536, help="Gradient dimension.")
    parser.add_argument("--f", type=int, default=8, help="Fault count parameter (f).")
    parser.add_argument("--chunk-size", type=int, default=32, help="Gradients per subtask.")
    parser.add_argument(
        "--power-iters",
        type=int,
        default=3,
        help="Power iterations for eigenvector estimate.",
    )
    default_workers = ",".join(str(w) for w in DEFAULT_WORKER_COUNTS)
    parser.add_argument(
        "--pool-workers",
        type=str,
        default=default_workers,
        help=f"Comma/space separated worker counts (default: {default_workers}).",
    )
    parser.add_argument("--pool-backend", type=str, default="process", help="Actor backend.")
    parser.add_argument("--warmup", type=int, default=1, help="Warm-up iterations per mode.")
    parser.add_argument("--repeat", type=int, default=3, help="Timed iterations per mode.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for gradients.")
    args = parser.parse_args()
    args.pool_workers = parse_worker_counts(args.pool_workers)
    return args


def _make_gradients(n: int, dim: int, seed: int, device: torch.device) -> list[torch.Tensor]:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return [torch.randn(dim, generator=gen, device=device, dtype=torch.float32) for _ in range(n)]


def _time_direct(
    aggregator: CAF,
    grads: Sequence[torch.Tensor],
    *,
    iterations: int,
    warmup: int,
) -> float:
    for _ in range(warmup):
        aggregator.aggregate(grads)
    start = time.perf_counter()
    for _ in range(iterations):
        aggregator.aggregate(grads)
    return (time.perf_counter() - start) / max(1, iterations)


async def _time_scheduler(
    scheduler: NodeScheduler,
    grads: Sequence[torch.Tensor],
    *,
    iterations: int,
    warmup: int,
) -> float:
    async def _run_once():
        await scheduler.run({"gradients": grads})

    for _ in range(warmup):
        await _run_once()

    start = time.perf_counter()
    for _ in range(iterations):
        await _run_once()
    return (time.perf_counter() - start) / max(1, iterations)


async def _benchmark(args: argparse.Namespace) -> list[BenchmarkRun]:
    device = torch.device("cpu")
    grads = _make_gradients(args.num_grads, args.grad_dim, args.seed, device)
    worker_counts = coerce_worker_counts(getattr(args, "pool_workers", DEFAULT_WORKER_COUNTS))

    aggregator = CAF(f=args.f, chunk_size=args.chunk_size, power_iters=args.power_iters)
    direct_time = _time_direct(aggregator, grads, iterations=args.repeat, warmup=args.warmup)

    graph = make_single_operator_graph(
        node_name="caf",
        operator=CAF(f=args.f, chunk_size=args.chunk_size, power_iters=args.power_iters),
        input_keys=("gradients",),
    )
    scheduler_single = NodeScheduler(graph, pool=None)
    single_time = await _time_scheduler(
        scheduler_single,
        grads,
        iterations=args.repeat,
        warmup=args.warmup,
    )

    runs = [
        BenchmarkRun("Direct aggregate (PyTorch)", direct_time),
        BenchmarkRun("Single-thread (NodeScheduler)", single_time),
    ]

    for workers in worker_counts:
        pool = ActorPool([ActorPoolConfig(backend=args.pool_backend, count=workers)])
        await pool.start()
        try:
            scheduler_pool = NodeScheduler(graph, pool=pool)
            pool_time = await _time_scheduler(
                scheduler_pool,
                grads,
                iterations=args.repeat,
                warmup=args.warmup,
            )
        finally:
            await pool.shutdown()
        runs.append(BenchmarkRun(f"ActorPool x{workers} ({args.pool_backend})", pool_time))

    return runs


def _print_results(runs: Sequence[BenchmarkRun]) -> None:
    baseline = runs[0].avg_seconds
    print("\nCAF Benchmark")
    print("-------------")
    print(f"{'Mode':40s} {'Avg ms':>12s} {'Speedup vs direct':>20s}")
    for run in runs:
        speedup = baseline / run.avg_seconds if run.avg_seconds else float("inf")
        print(f"{run.mode:40s} {run.avg_ms:12.2f} {speedup:20.2f}x")


def main() -> None:
    args = _parse_args()
    runs = asyncio.run(_benchmark(args))
    _print_results(runs)


if __name__ == "__main__":
    main()
