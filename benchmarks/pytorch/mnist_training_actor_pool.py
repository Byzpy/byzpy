#!/usr/bin/env python
"""
Benchmark a complete MNIST training loop where gradient aggregation uses
Minimum Diameter Averaging (MDA). We compare a pure PyTorch baseline (no actor
pool) against an ActorPool-backed run that parallelizes the combinatorial MDA
search across multiple workers.
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
import torch.nn as nn
import torch.utils.data as data
from torchvision import datasets, transforms

from byzpy.aggregators.geometric_wise.minimum_diameter_average import MinimumDiameterAveraging
from byzpy.engine.graph.ops import make_single_operator_graph
from byzpy.engine.graph.pool import ActorPool, ActorPoolConfig
from byzpy.engine.graph.scheduler import NodeScheduler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.p2p.nodes import SmallCNN


@dataclass(frozen=True)
class BenchmarkRun:
    mode: str
    avg_seconds: float

    @property
    def avg_ms(self) -> float:
        return self.avg_seconds * 1_000.0


def _flatten_grads(model: nn.Module) -> torch.Tensor:
    parts = []
    for p in model.parameters():
        grad = torch.zeros_like(p) if p.grad is None else p.grad
        parts.append(grad.view(-1))
    return torch.cat(parts)


def _apply_gradient(model: nn.Module, grad_vec: torch.Tensor, lr: float) -> None:
    offset = 0
    for p in model.parameters():
        numel = p.numel()
        chunk = grad_vec[offset : offset + numel].view_as(p).to(p.device)
        p.data = p.data - lr * chunk
        offset += numel


def _shard_indices(n_items: int, n_shards: int) -> List[List[int]]:
    return [list(range(i, n_items, n_shards)) for i in range(n_shards)]


def _build_worker_loaders(
    *,
    num_honest_workers: int,
    batch_size: int,
    data_root: str,
    seed: int,
) -> list[data.DataLoader]:
    torch.manual_seed(seed)
    tfm = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root=data_root, train=True, download=True, transform=tfm)
    shards = _shard_indices(len(dataset), num_honest_workers)
    loaders = []
    for shard in shards:
        subset = data.Subset(dataset, shard)
        loader = data.DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,
            pin_memory=False,
        )
        loaders.append(loader)
    return loaders


def _next_batch(loader: data.DataLoader, iters: list[object], idx: int):
    try:
        batch = next(iters[idx])
    except StopIteration:
        iters[idx] = iter(loader)
        batch = next(iters[idx])
    return batch


async def _train_once(
    *,
    num_workers: int,
    num_byzantine: int,
    batch_size: int,
    rounds: int,
    lr: float,
    f: int,
    chunk_size: int,
    pool_workers: int,
    pool_backend: str,
    seed: int,
    data_root: str,
) -> float:
    if num_byzantine >= num_workers:
        raise ValueError("num_byzantine must be < num_workers")
    num_honest = num_workers - num_byzantine

    device = torch.device("cpu")
    model = SmallCNN().to(device)
    loaders = _build_worker_loaders(
        num_honest_workers=num_honest,
        batch_size=batch_size,
        data_root=data_root,
        seed=seed,
    )
    iters = [iter(loader) for loader in loaders]
    criterion = nn.CrossEntropyLoss()

    aggregator = MinimumDiameterAveraging(f=f, chunk_size=chunk_size)
    graph = make_single_operator_graph(
        node_name="mda",
        operator=aggregator,
        input_keys=("gradients",),
    )
    pool = None
    if pool_workers > 1:
        pool = ActorPool(
            [ActorPoolConfig(backend=pool_backend, count=pool_workers, name="mda-worker")]
        )
        await pool.start()
    scheduler = NodeScheduler(graph, pool=pool)

    start = time.perf_counter()
    for _ in range(rounds):
        grads: List[torch.Tensor] = []
        for idx in range(num_honest):
            batch = _next_batch(loaders[idx], iters, idx)
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            model.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            grads.append(_flatten_grads(model).detach().cpu())

        if num_byzantine > 0:
            honest_stack = torch.stack(grads, dim=0)
            mean_grad = honest_stack.mean(dim=0)
            for _ in range(num_byzantine):
                grads.append((-mean_grad).clone())
        aggregated = await scheduler.run({"gradients": grads})
        agg_grad = aggregated["mda"]
        _apply_gradient(model, agg_grad.to(device), lr=lr)
    elapsed = time.perf_counter() - start

    if pool is not None:
        await pool.shutdown()
    return elapsed


async def _benchmark(args: argparse.Namespace) -> list[BenchmarkRun]:
    baseline = await _train_once(
        num_workers=args.num_workers,
        num_byzantine=args.byz_workers,
        batch_size=args.batch_size,
        rounds=args.rounds,
        lr=args.lr,
        f=args.f,
        chunk_size=args.chunk_size,
        pool_workers=1,
        pool_backend=args.pool_backend,
        seed=args.seed,
        data_root=args.data_root,
    )

    runs = [BenchmarkRun("PyTorch baseline (no ActorPool)", baseline)]

    if args.pool_workers > 1:
        actor_time = await _train_once(
            num_workers=args.num_workers,
            num_byzantine=args.byz_workers,
            batch_size=args.batch_size,
            rounds=args.rounds,
            lr=args.lr,
            f=args.f,
            chunk_size=args.chunk_size,
            pool_workers=args.pool_workers,
            pool_backend=args.pool_backend,
            seed=args.seed,
            data_root=args.data_root,
        )
        runs.append(
            BenchmarkRun(f"ActorPool x{args.pool_workers} ({args.pool_backend})", actor_time)
        )

    return runs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark MNIST training with MDA aggregator and ActorPool."
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=14,
        help="Number of workers providing gradients.",
    )
    parser.add_argument("--byz-workers", type=int, default=4, help="Number of Byzantine workers.")
    parser.add_argument("--f", type=int, default=4, help="MDA parameter f (vectors dropped).")
    parser.add_argument(
        "--chunk-size", type=int, default=128, help="Subsets evaluated per subtask."
    )
    parser.add_argument("--rounds", type=int, default=3, help="Training rounds (global steps).")
    parser.add_argument("--batch-size", type=int, default=64, help="Per-worker batch size.")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate for SGD updates.")
    parser.add_argument("--pool-workers", type=int, default=4, help="ActorPool worker processes.")
    parser.add_argument(
        "--pool-backend",
        type=str,
        default="process",
        help="Actor backend (process/thread/...).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--data-root", type=str, default="./data", help="MNIST data directory.")
    return parser.parse_args()


def _print_runs(runs: Sequence[BenchmarkRun]) -> None:
    baseline = runs[0].avg_seconds
    print("\nMNIST Training Benchmark (MDA Aggregator)")
    print("----------------------------------------")
    print(f"{'Mode':45s} {'Avg ms':>10s} {'Speedup vs baseline':>22s}")
    for run in runs:
        speedup = baseline / run.avg_seconds if run.avg_seconds else float("inf")
        print(f"{run.mode:45s} {run.avg_ms:10.2f} {speedup:22.2f}x")


def main() -> None:
    args = _parse_args()
    runs = asyncio.run(_benchmark(args))
    _print_runs(runs)


if __name__ == "__main__":
    main()
