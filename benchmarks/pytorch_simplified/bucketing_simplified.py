#!/usr/bin/env python
"""
Benchmark the Bucketing pre-aggregator using the simplified API.
"""
from __future__ import annotations

import argparse
import asyncio
import time
from dataclasses import dataclass
from typing import Sequence

import torch

from byzpy import OperatorExecutor, run_operator
from byzpy.engine.graph.pool import ActorPoolConfig
from byzpy.pre_aggregators.bucketing import Bucketing

try:
    from benchmarks.pytorch._worker_args import (
        DEFAULT_WORKER_COUNTS,
        coerce_worker_counts,
        parse_worker_counts,
    )
except ImportError:
    try:
        from ..pytorch._worker_args import (
            DEFAULT_WORKER_COUNTS,
            coerce_worker_counts,
            parse_worker_counts,
        )
    except ImportError:
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).parent.parent / "pytorch"))
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
    parser = argparse.ArgumentParser(
        description="Benchmark Bucketing pre-aggregation using simplified API."
    )
    parser.add_argument("--num-vectors", type=int, default=512, help="Number of vectors.")
    parser.add_argument("--dim", type=int, default=16384, help="Vector dimension.")
    parser.add_argument("--bucket-size", type=int, default=32, help="Bucket size.")
    parser.add_argument("--feature-chunk", type=int, default=8192, help="Features per subtask.")
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


def _make_vectors(n: int, dim: int, seed: int) -> list[torch.Tensor]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    return [torch.randn(dim, generator=gen) for _ in range(n)]


def _time_direct(
    agg: Bucketing, vecs: Sequence[torch.Tensor], *, iterations: int, warmup: int
) -> float:
    for _ in range(warmup):
        agg.pre_aggregate(vecs)
    start = time.perf_counter()
    for _ in range(iterations):
        agg.pre_aggregate(vecs)
    return (time.perf_counter() - start) / max(1, iterations)


async def _time_run_operator(
    operator: Bucketing,
    vecs: Sequence[torch.Tensor],
    *,
    pool_config: ActorPoolConfig | None = None,
    iterations: int,
    warmup: int,
) -> float:
    """Time run_operator() for single-threaded case (no pool overhead)."""

    async def _run_once():
        await run_operator(operator=operator, inputs={"vectors": vecs}, pool_config=pool_config)

    for _ in range(warmup):
        await _run_once()

    start = time.perf_counter()
    for _ in range(iterations):
        await _run_once()
    return (time.perf_counter() - start) / max(1, iterations)


async def _time_executor(
    operator: Bucketing,
    vecs: Sequence[torch.Tensor],
    *,
    pool_config: ActorPoolConfig,
    iterations: int,
    warmup: int,
) -> float:
    """Time OperatorExecutor with pool (reuses pool across iterations)."""
    executor = OperatorExecutor(operator, pool_config=pool_config)
    async with executor:

        for _ in range(warmup):
            await executor.run({"vectors": vecs})

        start = time.perf_counter()
        for _ in range(iterations):
            await executor.run({"vectors": vecs})
        return (time.perf_counter() - start) / max(1, iterations)


async def _benchmark(args: argparse.Namespace) -> list[BenchmarkRun]:
    worker_counts = coerce_worker_counts(getattr(args, "pool_workers", DEFAULT_WORKER_COUNTS))
    vecs = _make_vectors(args.num_vectors, args.dim, args.seed)

    agg = Bucketing(
        bucket_size=args.bucket_size,
        feature_chunk_size=args.feature_chunk,
    )

    direct = _time_direct(agg, vecs, iterations=args.repeat, warmup=args.warmup)

    single = await _time_run_operator(
        agg,
        vecs,
        pool_config=None,
        iterations=args.repeat,
        warmup=args.warmup,
    )

    runs = [
        BenchmarkRun("Direct aggregate (PyTorch)", direct),
        BenchmarkRun("Single-thread (run_operator)", single),
    ]

    for workers in worker_counts:
        pool_config = ActorPoolConfig(backend=args.pool_backend, count=workers)
        pooled = await _time_executor(
            agg,
            vecs,
            pool_config=pool_config,
            iterations=args.repeat,
            warmup=args.warmup,
        )
        runs.append(BenchmarkRun(f"ActorPool x{workers} ({args.pool_backend})", pooled))
    return runs


def _print(runs: Sequence[BenchmarkRun]) -> None:
    baseline = runs[0].avg_seconds
    print("\nBucketing Pre-Aggregator Benchmark (Simplified API)")
    print("---------------------------------------------------")
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
