#!/usr/bin/env python
"""
Benchmark Comparative Gradient Elimination with and without ActorPool acceleration.
"""
from __future__ import annotations

import argparse
import asyncio
import time
from dataclasses import dataclass
from typing import Sequence

import torch

from byzpy.aggregators.norm_wise.comparative_gradient_elimination import (
    ComparativeGradientElimination,
)
from byzpy.engine.graph.ops import make_single_operator_graph
from byzpy.engine.graph.pool import ActorPool, ActorPoolConfig
from byzpy.engine.graph.scheduler import NodeScheduler

try:
    from ._worker_args import DEFAULT_WORKER_COUNTS, coerce_worker_counts, parse_worker_counts
except ImportError:
    from _worker_args import DEFAULT_WORKER_COUNTS, coerce_worker_counts, parse_worker_counts


@dataclass(frozen=True)
class BenchmarkRun:
    mode: str
    avg_seconds: float

    @property
    def avg_ms(self) -> float:
        return self.avg_seconds * 1_000.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Comparative Gradient Elimination.")
    parser.add_argument("--num-grads", type=int, default=128, help="Number of gradients.")
    parser.add_argument("--grad-dim", type=int, default=131072, help="Gradient dimension.")
    parser.add_argument("--f", type=int, default=16, help="Number of gradients to drop.")
    parser.add_argument("--chunk-size", type=int, default=32768, help="Feature chunk size.")
    default_workers = ",".join(str(count) for count in DEFAULT_WORKER_COUNTS)
    parser.add_argument(
        "--pool-workers",
        type=str,
        default=default_workers,
        help=f"Comma/space separated worker counts for ActorPool runs (default: {default_workers}).",
    )
    parser.add_argument("--pool-backend", type=str, default="process", help="Actor backend.")
    parser.add_argument("--warmup", type=int, default=1, help="Warm-up iterations per mode.")
    parser.add_argument("--repeat", type=int, default=3, help="Timed iterations per mode.")
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed.")
    args = parser.parse_args()
    args.pool_workers = parse_worker_counts(args.pool_workers)
    return args


def _make_gradients(n: int, dim: int, seed: int) -> list[torch.Tensor]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    return [torch.randn(dim, generator=gen) for _ in range(n)]


def _time_direct(
    agg: ComparativeGradientElimination,
    grads: Sequence[torch.Tensor],
    *,
    iterations: int,
    warmup: int,
) -> float:
    for _ in range(warmup):
        agg.aggregate(grads)
    start = time.perf_counter()
    for _ in range(iterations):
        agg.aggregate(grads)
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
    worker_counts = coerce_worker_counts(getattr(args, "pool_workers", DEFAULT_WORKER_COUNTS))
    grads = _make_gradients(args.num_grads, args.grad_dim, args.seed)
    aggregator = ComparativeGradientElimination(f=args.f, chunk_size=args.chunk_size)

    direct = _time_direct(aggregator, grads, iterations=args.repeat, warmup=args.warmup)

    graph = make_single_operator_graph(
        node_name="cge",
        operator=ComparativeGradientElimination(f=args.f, chunk_size=args.chunk_size),
        input_keys=("gradients",),
    )

    scheduler_single = NodeScheduler(graph, pool=None)
    single = await _time_scheduler(
        scheduler_single,
        grads,
        iterations=args.repeat,
        warmup=args.warmup,
    )

    runs = [
        BenchmarkRun("Direct aggregate (PyTorch)", direct),
        BenchmarkRun("Single-thread (NodeScheduler)", single),
    ]
    for workers in worker_counts:
        pool = ActorPool([ActorPoolConfig(backend=args.pool_backend, count=workers)])
        await pool.start()
        try:
            scheduler_pool = NodeScheduler(graph, pool=pool)
            pooled = await _time_scheduler(
                scheduler_pool,
                grads,
                iterations=args.repeat,
                warmup=args.warmup,
            )
        finally:
            await pool.shutdown()
        runs.append(BenchmarkRun(f"ActorPool x{workers} ({args.pool_backend})", pooled))
    return runs


def _print(runs: Sequence[BenchmarkRun]) -> None:
    baseline = runs[0].avg_seconds
    print("\nComparative Gradient Elimination Benchmark")
    print("------------------------------------------")
    print(f"{'Mode':40s} {'Avg ms':>10s} {'Speedup vs direct':>20s}")
    for run in runs:
        speedup = baseline / run.avg_seconds if run.avg_seconds else float("inf")
        print(f"{run.mode:40s} {run.avg_ms:10.2f} {speedup:20.2f}x")


def main() -> None:
    args = _parse_args()
    runs = asyncio.run(_benchmark(args))
    _print(runs)


if __name__ == "__main__":
    main()
