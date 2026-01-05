#!/usr/bin/env python
"""
Benchmark a CPU-bound pure Python workload with and without the ActorPool.

EmpireAttack is dominated by a single highly optimized ``torch.mean`` call, so
splitting it across workers does not beat the baseline unless Torch threading is
manually constrained. This script instead targets a workload that spends most of
its time in Python loops (which Torch cannot parallelize internally), making it
ideal for distribution across a ``process`` ActorPool.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import random
import time
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from byzpy.engine.graph.graph import ComputationGraph, GraphNode, graph_input
from byzpy.engine.graph.operator import OpContext, Operator
from byzpy.engine.graph.pool import ActorPool, ActorPoolConfig
from byzpy.engine.graph.scheduler import NodeScheduler
from byzpy.engine.graph.subtask import SubTask


@dataclass(frozen=True)
class BenchmarkRun:
    mode: str
    avg_seconds: float

    @property
    def avg_ms(self) -> float:
        return self.avg_seconds * 1_000.0


class HeavyPythonOp(Operator):
    """
    Synthetic operator that spends its time inside Python control flow and math
    module calls. Each subtask receives a slice of the payload, runs an expensive
    loop, and returns a single float so the reduction step only needs to sum
    partials.
    """

    name = "heavy_python"
    supports_subtasks = True

    def __init__(self, *, chunk_size: int, inner_iters: int) -> None:
        self.chunk_size = max(1, chunk_size)
        self.inner_iters = max(1, inner_iters)

    def compute(self, inputs: Mapping[str, Any], *, context: OpContext) -> float:  # type: ignore[override]
        payload = inputs["payload"]
        return _run_chunk(payload, self.inner_iters)

    def create_subtasks(self, inputs: Mapping[str, Any], *, context: OpContext) -> Iterable[SubTask]:  # type: ignore[override]
        payload = inputs["payload"]
        subtasks = []
        for idx in range(0, len(payload), self.chunk_size):
            part = payload[idx : idx + self.chunk_size]
            subtasks.append(
                SubTask(
                    fn=_run_chunk,
                    args=(part, self.inner_iters),
                    kwargs={},
                    name=f"chunk-{idx}",
                )
            )
        return subtasks

    def reduce_subtasks(self, partials: Sequence[float], inputs, *, context: OpContext) -> float:  # type: ignore[override]
        return sum(float(x) for x in partials)


def _run_chunk(values: Sequence[float], inner_iters: int) -> float:
    total = 0.0
    for base in values:
        x = float(base)
        acc = 0.0
        for i in range(1, inner_iters + 1):
            angle = x / (i + 0.5)
            acc += math.sin(angle) * math.cos(angle + x) + math.sqrt(1.0 + angle * angle)
        total += acc
    return total


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark a pure-Python workload with ActorPool vs single-thread."
    )
    parser.add_argument("--tasks", type=int, default=5000, help="Number of work items.")
    parser.add_argument(
        "--inner-iters", type=int, default=2000, help="Loop iterations per work item."
    )
    parser.add_argument("--chunk-size", type=int, default=250, help="Tasks per subtask.")
    parser.add_argument("--pool-backend", type=str, default="process", help="Actor backend to use.")
    parser.add_argument(
        "--pool-workers",
        type=int,
        default=8,
        help="Number of workers in the ActorPool.",
    )
    parser.add_argument("--warmup", type=int, default=1, help="Warm-up iterations per mode.")
    parser.add_argument("--repeat", type=int, default=3, help="Timed iterations per mode.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for task generation.")
    return parser.parse_args()


def _make_payload(tasks: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    return [rng.uniform(1.0, 10.0) for _ in range(tasks)]


def _make_graph(op: Operator) -> ComputationGraph:
    node = GraphNode(
        name="heavy",
        op=op,
        inputs={"payload": graph_input("payload")},
    )
    return ComputationGraph(nodes=[node], outputs=["heavy"])


async def _time_scheduler(
    scheduler: NodeScheduler,
    payload: Sequence[float],
    *,
    iterations: int,
    warmup: int,
) -> float:
    async def _run_once() -> None:
        await scheduler.run({"payload": payload})

    for _ in range(warmup):
        await _run_once()

    start = time.perf_counter()
    for _ in range(iterations):
        await _run_once()
    elapsed = time.perf_counter() - start
    return elapsed / max(1, iterations)


async def _benchmark(args: argparse.Namespace) -> list[BenchmarkRun]:
    payload = _make_payload(args.tasks, args.seed)
    op = HeavyPythonOp(chunk_size=args.chunk_size, inner_iters=args.inner_iters)
    graph = _make_graph(op)

    scheduler_direct = NodeScheduler(graph, pool=None)
    direct_time = await _time_scheduler(
        scheduler_direct,
        payload,
        iterations=args.repeat,
        warmup=args.warmup,
    )

    pool = ActorPool([ActorPoolConfig(backend=args.pool_backend, count=args.pool_workers)])
    await pool.start()
    try:
        scheduler_pool = NodeScheduler(graph, pool=pool)
        pool_time = await _time_scheduler(
            scheduler_pool,
            payload,
            iterations=args.repeat,
            warmup=args.warmup,
        )
    finally:
        await pool.shutdown()

    return [
        BenchmarkRun("Single-thread (Python loops)", direct_time),
        BenchmarkRun(f"ActorPool x{args.pool_workers} ({args.pool_backend})", pool_time),
    ]


def _print_results(runs: Sequence[BenchmarkRun]) -> None:
    baseline = runs[0].avg_seconds
    print("\nHeavy Python Workload Benchmark")
    print("--------------------------------")
    print(f"{'Mode':40s} {'Avg ms':>10s} {'Speedup vs single':>20s}")
    for run in runs:
        speedup = baseline / run.avg_seconds if run.avg_seconds else float("inf")
        print(f"{run.mode:40s} {run.avg_ms:10.2f} {speedup:20.2f}x")


def main() -> None:
    args = _parse_args()
    runs = asyncio.run(_benchmark(args))
    _print_results(runs)


if __name__ == "__main__":
    main()
