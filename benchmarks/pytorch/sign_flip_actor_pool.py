#!/usr/bin/env python
"""
Benchmark the SignFlip attack with and without ActorPool chunking.
"""
from __future__ import annotations

import argparse
import asyncio
import time
from dataclasses import dataclass

import torch

from byzpy.attacks.sign_flip import SignFlipAttack
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
    parser = argparse.ArgumentParser(description="Benchmark the SignFlip attack.")
    parser.add_argument("--grad-dim", type=int, default=262144, help="Gradient dimension.")
    parser.add_argument("--chunk-size", type=int, default=8192, help="Coordinates per subtask.")
    parser.add_argument("--scale", type=float, default=-1.0, help="Scaling factor.")
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


def _make_gradient(dim: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    return torch.randn(dim, generator=gen)


def _time_direct(
    attack: SignFlipAttack, grad: torch.Tensor, *, iterations: int, warmup: int
) -> float:
    for _ in range(warmup):
        attack.apply(base_grad=grad)
    start = time.perf_counter()
    for _ in range(iterations):
        attack.apply(base_grad=grad)
    return (time.perf_counter() - start) / max(1, iterations)


async def _time_scheduler(
    scheduler: NodeScheduler,
    grad: torch.Tensor,
    *,
    iterations: int,
    warmup: int,
) -> float:
    async def _run_once():
        await scheduler.run({"base_grad": grad})

    for _ in range(warmup):
        await _run_once()

    start = time.perf_counter()
    for _ in range(iterations):
        await _run_once()
    return (time.perf_counter() - start) / max(1, iterations)


async def _benchmark(args: argparse.Namespace) -> list[BenchmarkRun]:
    worker_counts = coerce_worker_counts(getattr(args, "pool_workers", DEFAULT_WORKER_COUNTS))
    grad = _make_gradient(args.grad_dim, args.seed)
    attack = SignFlipAttack(scale=args.scale, chunk_size=args.chunk_size)

    direct = _time_direct(attack, grad, iterations=args.repeat, warmup=args.warmup)

    graph = make_single_operator_graph(
        node_name="signflip",
        operator=SignFlipAttack(scale=args.scale, chunk_size=args.chunk_size),
        input_keys=("base_grad",),
    )

    scheduler_single = NodeScheduler(graph, pool=None)
    single = await _time_scheduler(
        scheduler_single,
        grad,
        iterations=args.repeat,
        warmup=args.warmup,
    )

    runs = [
        BenchmarkRun("Direct attack (PyTorch)", direct),
        BenchmarkRun("Single-thread (NodeScheduler)", single),
    ]
    for workers in worker_counts:
        pool = ActorPool([ActorPoolConfig(backend=args.pool_backend, count=workers)])
        await pool.start()
        try:
            scheduler_pool = NodeScheduler(graph, pool=pool)
            pooled = await _time_scheduler(
                scheduler_pool,
                grad,
                iterations=args.repeat,
                warmup=args.warmup,
            )
        finally:
            await pool.shutdown()
        runs.append(BenchmarkRun(f"ActorPool x{workers} ({args.pool_backend})", pooled))
    return runs


def _print(runs):
    baseline = runs[0].avg_seconds
    print("\nSignFlip Attack Benchmark")
    print("------------------------")
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
