#!/usr/bin/env python
"""
Benchmark ByzPy ParameterServer training with MultiKrum aggregator.

This benchmark demonstrates ByzPy's ParameterServer architecture where:
- Each node (honest/Byzantine) has its own aggregator/preaggregator/attack behavior
- Each node has its own actor pool configuration
- The ParameterServer aggregates gradients using MultiKrum (with optional ActorPool)

The benchmark runs training with and without ActorPool to show performance benefits.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import torch
import torch.utils.data as data
from torchvision import datasets, transforms

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.ps.nodes import DistributedPSByzNode, DistributedPSHonestNode, select_pool_backend

from byzpy.aggregators.geometric_wise.krum import MultiKrum
from byzpy.configs.actor import set_actor
from byzpy.engine.graph.pool import ActorPool, ActorPoolConfig
from byzpy.engine.node.actors import ByzantineNodeActor, HonestNodeActor
from byzpy.engine.parameter_server.ps import ParameterServer

try:
    from benchmarks.pytorch._worker_args import (
        DEFAULT_WORKER_COUNTS,
        coerce_worker_counts,
        parse_worker_counts,
    )
except ImportError:
    from _worker_args import DEFAULT_WORKER_COUNTS  # type: ignore
    from _worker_args import coerce_worker_counts, parse_worker_counts


@dataclass(frozen=True)
class BenchmarkRun:
    label: str
    total_seconds: float
    iterations: int

    @property
    def total_ms(self) -> float:
        return self.total_seconds * 1_000.0

    @property
    def avg_ms(self) -> float:
        return self.total_ms / max(1, self.iterations)


def _shard_indices(n_items: int, n_shards: int) -> List[List[int]]:
    """Split dataset indices into shards."""
    return [list(range(i, n_items, n_shards)) for i in range(n_shards)]


async def _train_byzpy(
    *,
    num_honest: int,
    num_byz: int,
    rounds: int,
    batch_size: int,
    lr: float,
    f: int,
    q: int,
    chunk_size: int,
    pool_workers: int | None,
    pool_backend: str,
    actor_backend: str,
    seed: int,
    data_root: str,
) -> BenchmarkRun:
    """Train using ByzPy ParameterServer with MultiKrum."""
    torch.manual_seed(seed)

    tfm = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root=data_root, train=True, download=True, transform=tfm)
    shards = _shard_indices(len(train_dataset), num_honest)

    node_pool_backend = select_pool_backend(actor_backend)

    honest_actors: List[HonestNodeActor] = []
    for i in range(num_honest):
        h = await HonestNodeActor.spawn(
            DistributedPSHonestNode,
            backend=set_actor(actor_backend),
            kwargs=dict(
                indices=shards[i],
                batch_size=batch_size,
                shuffle=True,
                lr=lr,
                momentum=0.9,
                device="cpu",
                data_root=data_root,
                pool_backend=node_pool_backend,
            ),
        )
        honest_actors.append(h)

    byz_actors: List[ByzantineNodeActor] = []
    for _ in range(num_byz):
        b = await ByzantineNodeActor.spawn(
            DistributedPSByzNode,
            backend=set_actor(actor_backend),
            kwargs=dict(
                device="cpu",
                scale=-1.0,
                pool_backend=node_pool_backend,
            ),
        )
        byz_actors.append(b)

    aggregator = MultiKrum(f=f, q=q, chunk_size=chunk_size)
    pool = None
    if pool_workers is not None and pool_workers > 1:
        pool = ActorPool(
            [ActorPoolConfig(backend=pool_backend, count=pool_workers, name="aggregator-pool")]
        )
        await pool.start()

    ps = ParameterServer(
        honest_nodes=honest_actors,
        byzantine_nodes=byz_actors,
        aggregator=aggregator,
        update_byzantines=False,
        actor_pool=pool,
    )

    start = time.perf_counter()
    for _ in range(rounds):
        await ps.round()
    elapsed = time.perf_counter() - start

    await ps.shutdown()
    if pool is not None:
        await pool.shutdown()

    label = "ByzPy ParameterServer (MultiKrum)"
    if pool_workers is not None and pool_workers > 1:
        label += f" + ActorPool x{pool_workers}"
    return BenchmarkRun(label, elapsed, rounds)


async def _benchmark(args: argparse.Namespace) -> List[BenchmarkRun]:
    """Run ByzPy benchmarks with different ActorPool configurations."""
    runs: List[BenchmarkRun] = []
    worker_counts = coerce_worker_counts(getattr(args, "pool_workers", DEFAULT_WORKER_COUNTS))

    # ByzPy without pool
    byzpy_direct = await _train_byzpy(
        num_honest=args.num_honest,
        num_byz=args.num_byz,
        rounds=args.rounds,
        batch_size=args.batch_size,
        lr=args.lr,
        f=args.f,
        q=args.q,
        chunk_size=args.chunk_size,
        pool_workers=None,
        pool_backend=args.pool_backend,
        actor_backend=args.actor_backend,
        seed=args.seed,
        data_root=args.data_root,
    )
    runs.append(byzpy_direct)

    for workers in worker_counts:
        byzpy_pool = await _train_byzpy(
            num_honest=args.num_honest,
            num_byz=args.num_byz,
            rounds=args.rounds,
            batch_size=args.batch_size,
            lr=args.lr,
            f=args.f,
            q=args.q,
            chunk_size=args.chunk_size,
            pool_workers=workers,
            pool_backend=args.pool_backend,
            actor_backend=args.actor_backend,
            seed=args.seed,
            data_root=args.data_root,
        )
        runs.append(byzpy_pool)

    return runs


def _print_results(runs: Sequence[BenchmarkRun]) -> None:
    """Print benchmark results."""
    baseline = runs[0].total_seconds

    print("\nByzPy ParameterServer Training Benchmark (MultiKrum)")
    print("=" * 70)
    print(f"{'Mode':50s} {'Total ms':>12s} {'Avg ms/round':>15s} {'Speedup':>12s}")
    print("-" * 70)

    for run in runs:
        speedup = baseline / run.total_seconds if run.total_seconds > 0 else float("inf")
        speedup_label = f"{speedup:.2f}x" if speedup != float("inf") else "N/A"
        if run == runs[0]:
            speedup_label = "baseline"
        print(f"{run.label:50s} {run.total_ms:12.2f} {run.avg_ms:15.2f} {speedup_label:>12s}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark ByzPy ParameterServer training with MultiKrum aggregator."
    )
    parser.add_argument("--num-honest", type=int, default=10, help="Number of honest nodes.")
    parser.add_argument("--num-byz", type=int, default=3, help="Number of Byzantine nodes.")
    parser.add_argument("--rounds", type=int, default=50, help="Number of training rounds.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size per node.")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate.")
    parser.add_argument("--f", type=int, default=3, help="MultiKrum fault tolerance parameter.")
    parser.add_argument(
        "--q",
        type=int,
        default=None,
        help="MultiKrum parameter q (defaults to n - f - 1).",
    )
    parser.add_argument("--chunk-size", type=int, default=32, help="MultiKrum chunk size.")
    default_workers = ",".join(str(count) for count in DEFAULT_WORKER_COUNTS)
    parser.add_argument(
        "--pool-workers",
        type=str,
        default=default_workers,
        help=f"Comma/space separated worker counts for ActorPool (default: {default_workers}).",
    )
    parser.add_argument("--pool-backend", type=str, default="process", help="ActorPool backend.")
    parser.add_argument("--actor-backend", type=str, default="process", help="Node actor backend.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--data-root", type=str, default="./data", help="MNIST data directory.")
    args = parser.parse_args()
    args.pool_workers = parse_worker_counts(args.pool_workers)
    if args.q is None:
        args.q = max(1, args.num_honest - args.f - 1)
    return args


def main() -> None:
    args = _parse_args()
    runs = asyncio.run(_benchmark(args))
    _print_results(runs)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Benchmark cancelled.", file=sys.stderr)
        raise
