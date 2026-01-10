"""
Benchmark comparing NodeScheduler vs ParallelScheduler on multi-stage pipelines.

This benchmark tests the performance difference between sequential execution
(NodeScheduler) and parallel execution (ParallelScheduler) using ByzPy's ActorPool.

Pipeline structure:
    gradients -> preprocess (local, numpy) -> aggregate (pool)
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from byzpy.aggregators.coordinate_wise.median import CoordinateWiseMedian
from byzpy.engine.graph.graph import ComputationGraph, GraphInput, GraphNode
from byzpy.engine.graph.operator import OpContext, Operator
from byzpy.engine.graph.parallel_scheduler import ParallelScheduler
from byzpy.engine.graph.pool import ActorPool, ActorPoolConfig
from byzpy.engine.graph.scheduler import NodeScheduler


class _PreprocessingOperator(Operator):
    name = "preprocessing"
    supports_subtasks = False

    def __init__(self, work_iterations: int = 20):
        self.work_iterations = work_iterations

    def _preprocess(self, gradients: list) -> list:
        import numpy as np

        result = []
        for grad in gradients:
            arr = grad.numpy() if hasattr(grad, "numpy") else np.array(grad)
            for _ in range(self.work_iterations):
                arr = arr - np.mean(arr)  # Center
                arr = arr / (np.std(arr) + 1e-8)  # Normalize
                arr = np.clip(arr, -3, 3)  # Clip outliers
            result.append(torch.from_numpy(arr.copy()))
        return result

    async def run(
        self,
        inputs: Mapping[str, Any],
        *,
        context: OpContext,
        pool: Any,
    ) -> Any:
        gradients = inputs["gradients"]
        return await asyncio.to_thread(self._preprocess, gradients)

    def compute(self, inputs: Mapping[str, Any], *, context: OpContext) -> Any:
        return self._preprocess(inputs["gradients"])


_benchmarks_dir = Path(__file__).parent.parent
if str(_benchmarks_dir) not in sys.path:
    sys.path.insert(0, str(_benchmarks_dir))

try:
    from pytorch._worker_args import (
        DEFAULT_WORKER_COUNTS,
        coerce_worker_counts,
        parse_worker_counts,
    )
except ImportError:
    try:
        from benchmarks.pytorch._worker_args import (
            DEFAULT_WORKER_COUNTS,
            coerce_worker_counts,
            parse_worker_counts,
        )
    except ImportError:
        from ..pytorch._worker_args import (
            DEFAULT_WORKER_COUNTS,
            coerce_worker_counts,
            parse_worker_counts,
        )


@dataclass(frozen=True)
class BenchmarkRun:
    mode: str
    avg_seconds: float

    @property
    def avg_ms(self) -> float:
        return self.avg_seconds * 1_000.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark NodeScheduler vs ParallelScheduler on pipelines with ActorPool."
    )
    parser.add_argument("--num-grads", type=int, default=64, help="Number of gradients.")
    parser.add_argument("--grad-dim", type=int, default=200_000, help="Gradient dimension.")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=8192,
        help="Chunk size for parallel operators.",
    )
    default_workers = ",".join(str(count) for count in DEFAULT_WORKER_COUNTS)
    parser.add_argument(
        "--pool-workers",
        type=str,
        default=default_workers,
        help=f"Comma/space separated worker counts (default: {default_workers}).",
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
    parser.add_argument(
        "--branches",
        type=int,
        default=4,
        help="Number of parallel branches (default: 4).",
    )
    parser.add_argument(
        "--preprocess-iterations",
        type=int,
        default=30,
        help="Preprocessing iterations per branch (controls local work time).",
    )
    parser.add_argument(
        "--max-pending-subtasks",
        type=int,
        default=None,
        help="Max pending subtasks across all concurrent operators (default: pool_size * 8).",
    )

    args = parser.parse_args()
    args.pool_workers = parse_worker_counts(args.pool_workers)
    return args


def _make_gradients(n: int, dim: int, seed: int, device: torch.device) -> list[torch.Tensor]:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return [torch.randn(dim, generator=gen, device=device, dtype=torch.float32) for _ in range(n)]


def _build_pipeline_graph(
    num_branches: int,
    preprocess_iterations: int,
    chunk_size: int,
) -> ComputationGraph:
    """
    Build a computation graph where:
    - Preprocessing runs locally
    - Aggregation uses the ActorPool

    Graph structure (N branches):
        gradients (input)
            |
            +---> preprocess_1 (local, numpy) -> median_1 (pool)
            +---> preprocess_2 (local, numpy) -> median_2 (pool)
            +---> preprocess_3 (local, numpy) -> median_3 (pool)
            ...
    """
    gradients_input = GraphInput("gradients")

    nodes = []
    for i in range(num_branches):
        preprocess_node = GraphNode(
            name=f"preprocess_{i}",
            op=_PreprocessingOperator(work_iterations=preprocess_iterations),
            inputs={"gradients": gradients_input},
        )

        agg_node = GraphNode(
            name=f"median_{i}",
            op=CoordinateWiseMedian(chunk_size=chunk_size),
            inputs={"gradients": f"preprocess_{i}"},
        )

        nodes.extend([preprocess_node, agg_node])

    return ComputationGraph(
        nodes=nodes,
        outputs=[f"median_{i}" for i in range(num_branches)],
    )


def _print_results(runs: Sequence[BenchmarkRun], title: str) -> None:
    baseline_run = next((run for run in runs if "NodeScheduler" in run.mode), runs[0])
    baseline = baseline_run.avg_seconds
    print(f"\n{title}")
    print("=" * 80)
    print(f"{'Mode':50s} {'Avg ms':>12s} {'Speedup':>15s}")
    print("-" * 80)
    for run in runs:
        speedup = baseline / run.avg_seconds if run.avg_seconds else float("inf")
        print(f"{run.mode:50s} {run.avg_ms:12.2f} {speedup:15.2f}x")

    print("\n" + "=" * 80)
    print("Summary: ParallelScheduler vs NodeScheduler")
    print("-" * 80)

    sequential_runs = [r for r in runs if "NodeScheduler" in r.mode]
    parallel_runs = [r for r in runs if "ParallelScheduler" in r.mode]

    for seq_run in sequential_runs:
        pool_config = (
            seq_run.mode.split("pool")[-1].strip() if "pool" in seq_run.mode else "no pool"
        )
        par_run = next(
            (r for r in parallel_runs if pool_config in r.mode),
            None,
        )
        if par_run:
            speedup = seq_run.avg_seconds / par_run.avg_seconds
            print(
                f"pool {pool_config:10s}: {seq_run.avg_ms:8.2f}ms -> {par_run.avg_ms:8.2f}ms "
                f"({speedup:.2f}x speedup)"
            )


async def _benchmark(args: argparse.Namespace) -> list[BenchmarkRun]:
    num_branches = args.branches
    preprocess_iterations = args.preprocess_iterations
    chunk_size = args.chunk_size
    num_grads = args.num_grads
    grad_dim = args.grad_dim

    worker_counts = coerce_worker_counts(args.pool_workers)
    device = torch.device("cpu")

    print(f"\n[Pipeline Benchmark: {num_branches} branches]")
    print("Each branch: preprocess (local, numpy) -> aggregate (pool)")
    print(f"Gradients: {num_grads} x {grad_dim:,} dims = {num_grads * grad_dim * 4 / 1e6:.1f} MB")

    grads = _make_gradients(num_grads, grad_dim, args.seed, device)
    graph = _build_pipeline_graph(num_branches, preprocess_iterations, chunk_size)

    runs: list[BenchmarkRun] = []

    async def _time_pipeline(scheduler, iterations: int, warmup: int) -> float:
        async def _run_once():
            await scheduler.run({"gradients": grads})

        for _ in range(warmup):
            await _run_once()

        start = time.perf_counter()
        for _ in range(iterations):
            await _run_once()
        elapsed = time.perf_counter() - start
        return elapsed / max(1, iterations)

    for workers in worker_counts:
        pool = ActorPool([ActorPoolConfig(backend=args.pool_backend, count=workers)])
        await pool.start()
        try:
            print(f"\nTesting with pool x{workers}...")

            # NodeScheduler with pool
            scheduler_seq = NodeScheduler(graph, pool=pool)
            seq_time = await _time_pipeline(
                scheduler_seq, iterations=args.repeat, warmup=args.warmup
            )
            runs.append(BenchmarkRun(f"NodeScheduler (pool x{workers})", seq_time))

            # ParallelScheduler with pool
            scheduler_par = ParallelScheduler(
                graph, pool=pool, max_pending_subtasks=args.max_pending_subtasks
            )
            par_time = await _time_pipeline(
                scheduler_par, iterations=args.repeat, warmup=args.warmup
            )
            runs.append(BenchmarkRun(f"ParallelScheduler (pool x{workers})", par_time))

            # Calculate and print speedup
            speedup = seq_time / par_time if par_time > 0 else 0
            print(
                f"  NodeScheduler: {seq_time*1000:.1f}ms, ParallelScheduler: {par_time*1000:.1f}ms ({speedup:.2f}x)"
            )

        finally:
            await pool.shutdown()

    return runs


async def main_async() -> None:
    args = _parse_args()
    runs = await _benchmark(args)
    _print_results(runs, "Pipeline Benchmark: Preprocessing (local) + Aggregation (pool)")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
