#!/usr/bin/env python
"""
Benchmark the Smallest Maximum Eigenvalue Averaging (SMEA) aggregator with and
without distributed execution through the ActorPool.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from dataclasses import dataclass
from typing import Sequence

import torch

from byzpy.aggregators.geometric_wise import SMEA
from byzpy.engine.graph.ops import make_single_operator_graph
from byzpy.engine.graph.pool import ActorPool, ActorPoolConfig
from byzpy.engine.graph.scheduler import NodeScheduler

try:
    from ._worker_args import DEFAULT_WORKER_COUNTS, coerce_worker_counts, parse_worker_counts
except ImportError:  # pragma: no cover - when invoked directly
    from _worker_args import DEFAULT_WORKER_COUNTS, coerce_worker_counts, parse_worker_counts


@dataclass(frozen=True)
class BenchmarkRun:
    mode: str
    avg_seconds: float

    @property
    def avg_ms(self) -> float:
        return self.avg_seconds * 1_000.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark SMEA with ActorPool vs single-thread.")
    parser.add_argument("--num-grads", type=int, default=12, help="Number of gradients (n).")
    parser.add_argument("--grad-dim", type=int, default=1024, help="Gradient dimension.")
    parser.add_argument(
        "--f", type=int, default=3, help="Number of vectors to drop (SMEA parameter)."
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=128,
        help="Combinations evaluated per subtask.",
    )
    default_workers = ",".join(str(count) for count in DEFAULT_WORKER_COUNTS)
    parser.add_argument(
        "--pool-workers",
        type=str,
        default=default_workers,
        help=f"Comma/space separated worker counts for ActorPool runs (default: {default_workers}).",
    )
    parser.add_argument(
        "--pool-backend",
        type=str,
        default="process",
        help="Actor backend (thread/process/...).",
    )
    parser.add_argument("--warmup", type=int, default=1, help="Warm-up iterations per mode.")
    parser.add_argument("--repeat", type=int, default=3, help="Timed iterations per mode.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for synthetic gradients.")
    args = parser.parse_args()
    args.pool_workers = parse_worker_counts(args.pool_workers)
    return args


def _maybe_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _make_gradients(n: int, dim: int, seed: int, device: torch.device) -> list[torch.Tensor]:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return [torch.randn(dim, generator=gen, device=device, dtype=torch.float32) for _ in range(n)]


async def _time_scheduler(
    scheduler: NodeScheduler,
    grads: Sequence[torch.Tensor],
    *,
    iterations: int,
    warmup: int,
    sync_device: torch.device,
) -> float:
    async def _run_once() -> None:
        await scheduler.run({"gradients": grads})
        _maybe_sync(sync_device)

    for _ in range(warmup):
        await _run_once()

    start = time.perf_counter()
    for _ in range(iterations):
        await _run_once()
    elapsed = time.perf_counter() - start
    return elapsed / max(1, iterations)


def _time_direct(
    aggregator: SMEA,
    grads: Sequence[torch.Tensor],
    *,
    iterations: int,
    warmup: int,
    sync_device: torch.device,
) -> float:
    def _run_once() -> None:
        aggregator.aggregate(grads)
        _maybe_sync(sync_device)

    for _ in range(warmup):
        _run_once()

    start = time.perf_counter()
    for _ in range(iterations):
        _run_once()
    elapsed = time.perf_counter() - start
    return elapsed / max(1, iterations)


async def _benchmark(args: argparse.Namespace) -> list[BenchmarkRun]:
    worker_counts = coerce_worker_counts(getattr(args, "pool_workers", DEFAULT_WORKER_COUNTS))
    device = torch.device("cpu")
    grads = _make_gradients(args.num_grads, args.grad_dim, args.seed, device)

    direct_aggregator = SMEA(f=args.f, chunk_size=args.chunk_size)
    direct_time = _time_direct(
        direct_aggregator,
        grads,
        iterations=args.repeat,
        warmup=args.warmup,
        sync_device=device,
    )

    aggregator = SMEA(f=args.f, chunk_size=args.chunk_size)
    graph = make_single_operator_graph(
        node_name="smea",
        operator=aggregator,
        input_keys=("gradients",),
    )

    scheduler_single = NodeScheduler(graph, pool=None)
    single_time = await _time_scheduler(
        scheduler_single,
        grads,
        iterations=args.repeat,
        warmup=args.warmup,
        sync_device=device,
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
                sync_device=device,
            )
        finally:
            await pool.shutdown()
        runs.append(BenchmarkRun(f"ActorPool x{workers} ({args.pool_backend})", pool_time))
    return runs


def _print_results(runs: Sequence[BenchmarkRun]) -> None:
    baseline_run = next((run for run in runs if "Direct aggregate" in run.mode), runs[0])
    baseline = baseline_run.avg_seconds
    print("\nSmallest Maximum Eigenvalue Averaging Benchmark")
    print("----------------------------------------------")
    print(f"{'Mode':40s} {'Avg ms':>10s} {'Speedup vs direct':>20s}")
    for run in runs:
        speedup = baseline / run.avg_seconds if run.avg_seconds else float("inf")
        print(f"{run.mode:40s} {run.avg_ms:10.2f} {speedup:20.2f}x")


def main() -> None:
    args = _parse_args()
    runs = asyncio.run(_benchmark(args))
    _print_results(runs)


if __name__ == "__main__":
    main()
