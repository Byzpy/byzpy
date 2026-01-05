#!/usr/bin/env python
"""
Utility script to regenerate benchmark plots for the ActorPool demos.

Generates PNGs under benchmarks/plots/ summarizing the speedups measured
by `actor_pool_python.py` and `mda_actor_pool.py`.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.pytorch.actor_pool_python import BenchmarkRun as ActorRun
from benchmarks.pytorch.actor_pool_python import _benchmark as actor_benchmark
from benchmarks.pytorch.cwtm_actor_pool import BenchmarkRun as CwRun
from benchmarks.pytorch.cwtm_actor_pool import _benchmark as cwtm_benchmark
from benchmarks.pytorch.mda_actor_pool import BenchmarkRun as MDARun
from benchmarks.pytorch.mda_actor_pool import _benchmark as mda_benchmark
from benchmarks.pytorch.meamed_actor_pool import BenchmarkRun as MeamedRun
from benchmarks.pytorch.meamed_actor_pool import _benchmark as meamed_benchmark
from benchmarks.pytorch.mnist_training_actor_pool import BenchmarkRun as TrainRun
from benchmarks.pytorch.mnist_training_actor_pool import _benchmark as train_benchmark


async def _run_actor_pool_cases(
    workers: Sequence[int],
) -> list[tuple[int, float, float]]:
    results: list[tuple[int, float, float]] = []
    for count in workers:
        args = SimpleNamespace(
            tasks=8000,
            inner_iters=3000,
            chunk_size=250,
            pool_backend="process",
            pool_workers=count,
            warmup=0,
            repeat=2,
            seed=0,
        )
        runs: list[ActorRun] = await actor_benchmark(args)
        single = next(run for run in runs if "Single-thread" in run.mode)
        pool = next(run for run in runs if "ActorPool" in run.mode)
        results.append((count, single.avg_seconds, pool.avg_seconds))
    return results


async def _run_mda_case(worker_count: int, *, chunk_size: int) -> tuple[float, float, float, float]:
    args = SimpleNamespace(
        num_grads=18,
        grad_dim=2048,
        f=6,
        chunk_size=chunk_size,
        pool_workers=worker_count,
        pool_backend="process",
        warmup=0,
        repeat=2,
        seed=0,
    )
    runs: list[MDARun] = await mda_benchmark(args)
    direct = next(run for run in runs if "Direct aggregate" in run.mode)
    single = next(run for run in runs if "Single-thread" in run.mode)
    pool = next(run for run in runs if "ActorPool" in run.mode)
    return worker_count, direct.avg_seconds, single.avg_seconds, pool.avg_seconds


async def _run_cwtm_case(
    worker_count: int, *, chunk_size: int
) -> tuple[float, float, float, float]:
    args = SimpleNamespace(
        num_grads=64,
        grad_dim=65536,
        f=8,
        chunk_size=chunk_size,
        pool_workers=worker_count,
        pool_backend="process",
        warmup=0,
        repeat=2,
        seed=0,
    )
    runs: list[CwRun] = await cwtm_benchmark(args)
    direct = next(run for run in runs if "Direct aggregate" in run.mode)
    single = next(run for run in runs if "Single-thread" in run.mode)
    pool = next(run for run in runs if "ActorPool" in run.mode)
    return worker_count, direct.avg_seconds, single.avg_seconds, pool.avg_seconds


async def _run_meamed_case(
    worker_count: int, *, chunk_size: int, f: int
) -> tuple[float, float, float, float]:
    args = SimpleNamespace(
        num_grads=64,
        grad_dim=65536,
        f=f,
        chunk_size=chunk_size,
        pool_workers=worker_count,
        pool_backend="process",
        warmup=0,
        repeat=2,
        seed=0,
    )
    runs: list[MeamedRun] = await meamed_benchmark(args)
    direct = next(run for run in runs if "Direct aggregate" in run.mode)
    single = next(run for run in runs if "Single-thread" in run.mode)
    pool = next(run for run in runs if "ActorPool" in run.mode)
    return worker_count, direct.avg_seconds, single.avg_seconds, pool.avg_seconds


def _plot_bars(
    labels: Sequence[str],
    series: Sequence[Sequence[float]],
    *,
    series_labels: Sequence[str],
    ylabel: str,
    title: str,
    outfile: Path,
) -> None:
    x = range(len(labels))
    num_series = len(series)
    bar_width = 0.8 / max(1, num_series)
    plt.figure(figsize=(6, 4))
    offsets = [((idx - (num_series - 1) / 2) * bar_width) for idx in range(num_series)]
    colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]

    for idx, data in enumerate(series):
        positions = [i + offsets[idx] for i in x]
        plt.bar(
            positions,
            [t * 1_000 for t in data],
            width=bar_width,
            color=colors[idx % len(colors)],
            label=series_labels[idx],
        )

    plt.xlabel("Configuration")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(list(x), labels)
    plt.legend()
    plt.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile)
    plt.close()


async def main() -> None:
    parser = argparse.ArgumentParser(description="Generate benchmark plots.")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "benchmarks" / "plots")
    args = parser.parse_args()

    actor_workers = [8]
    actor_data = await _run_actor_pool_cases(actor_workers)
    mda_workers = [4]
    mda_results = [await _run_mda_case(count, chunk_size=512) for count in mda_workers]
    cwtm_workers = [4]
    cwtm_results = [await _run_cwtm_case(count, chunk_size=8192) for count in cwtm_workers]
    meamed_workers = [4]
    meamed_results = [
        await _run_meamed_case(count, chunk_size=8192, f=8) for count in meamed_workers
    ]
    mnist_args = SimpleNamespace(
        num_workers=14,
        byz_workers=4,
        f=4,
        chunk_size=128,
        rounds=3,
        batch_size=64,
        lr=0.05,
        pool_workers=4,
        pool_backend="process",
        seed=0,
        data_root="./data",
    )
    mnist_runs: list[TrainRun] = await train_benchmark(mnist_args)

    _plot_bars(
        labels=[f"{w} workers" for w in mda_workers],
        series=[
            [item[1] for item in mda_results],
            [item[2] for item in mda_results],
            [item[3] for item in mda_results],
        ],
        series_labels=["PyTorch direct", "Single-thread scheduler", "ActorPool"],
        ylabel="Average milliseconds",
        title="MDA Aggregator runtime (lower is better)",
        outfile=args.output_dir / "mda_runtimes.png",
    )

    _plot_bars(
        labels=[f"{w} workers" for w in cwtm_workers],
        series=[
            [item[1] for item in cwtm_results],
            [item[2] for item in cwtm_results],
            [item[3] for item in cwtm_results],
        ],
        series_labels=["PyTorch direct", "Single-thread scheduler", "ActorPool"],
        ylabel="Average milliseconds",
        title="CwTM Aggregator runtime (lower is better)",
        outfile=args.output_dir / "cwtm_runtimes.png",
    )

    _plot_bars(
        labels=[f"{w} workers" for w in meamed_workers],
        series=[
            [item[1] for item in meamed_results],
            [item[2] for item in meamed_results],
            [item[3] for item in meamed_results],
        ],
        series_labels=["PyTorch direct", "Single-thread scheduler", "ActorPool"],
        ylabel="Average milliseconds",
        title="Mean-of-Medians runtime (lower is better)",
        outfile=args.output_dir / "meamed_runtimes.png",
    )

    _plot_bars(
        labels=[f"{w} workers" for w in actor_workers],
        series=[
            [item[1] for item in actor_data],
            [item[2] for item in actor_data],
        ],
        series_labels=["Single-thread (Python loops)", "ActorPool"],
        ylabel="Average milliseconds",
        title="ActorPool Python workload runtime",
        outfile=args.output_dir / "actor_pool_python.png",
    )

    _plot_bars(
        labels=["MNIST training"],
        series=[
            [mnist_runs[0].avg_seconds],
            [mnist_runs[1].avg_seconds],
        ],
        series_labels=["PyTorch baseline", "ActorPool x4"],
        ylabel="Average milliseconds",
        title="MNIST training with MDA aggregation",
        outfile=args.output_dir / "mnist_training.png",
    )


if __name__ == "__main__":
    asyncio.run(main())
