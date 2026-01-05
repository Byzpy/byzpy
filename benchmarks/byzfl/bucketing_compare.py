#!/usr/bin/env python
"""
Benchmark the ByzFL implementation of the Bucketing pre-aggregator.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class BenchmarkRun:
    mode: str
    avg_seconds: float

    @property
    def avg_ms(self) -> float:
        return self.avg_seconds * 1_000.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark ByzFL's Bucketing pre-aggregation.")
    parser.add_argument("--num-vectors", type=int, default=512, help="Number of vectors (n).")
    parser.add_argument("--dim", type=int, default=16384, help="Vector dimension.")
    parser.add_argument("--bucket-size", type=int, default=32, help="Bucket size.")
    parser.add_argument("--warmup", type=int, default=0, help="Warm-up iterations.")
    parser.add_argument("--repeat", type=int, default=2, help="Timed iterations.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for synthetic vectors.")
    parser.add_argument(
        "--timeout",
        type=float,
        default=0.0,
        help="Maximum seconds to spend timing ByzFL (0 disables the limit).",
    )
    return parser.parse_args()


def _maybe_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _make_vectors(n: int, dim: int, seed: int, device: torch.device) -> list[torch.Tensor]:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return [torch.randn(dim, generator=gen, device=device, dtype=torch.float32) for _ in range(n)]


def _require_byzfl() -> "type[object]":
    try:
        from byzfl.aggregators.preaggregators import Bucketing as ByzflBucketing
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise SystemExit(
            "The byzfl package is required for this benchmark (pip install byzfl or add it to your environment)."
        ) from exc
    return ByzflBucketing


def _time_byzfl(
    preagg: object,
    vectors: Sequence[torch.Tensor],
    *,
    iterations: int,
    warmup: int,
    sync_device: torch.device,
    timeout: float | None = None,
) -> float:
    def _run_once() -> None:
        preagg(vectors)
        _maybe_sync(sync_device)

    for _ in range(warmup):
        _run_once()

    start = time.perf_counter()
    for _ in range(iterations):
        _run_once()
        if timeout is not None and (time.perf_counter() - start) > timeout:
            raise TimeoutError
    elapsed = time.perf_counter() - start
    return elapsed / max(1, iterations)


def main() -> None:
    args = _parse_args()
    device = torch.device("cpu")
    vectors = _make_vectors(args.num_vectors, args.dim, args.seed, device)
    byzfl_cls = _require_byzfl()
    preagg = byzfl_cls(s=args.bucket_size)
    limit = args.timeout if args.timeout > 0 else None
    try:
        avg_seconds = _time_byzfl(
            preagg,
            vectors,
            iterations=args.repeat,
            warmup=args.warmup,
            sync_device=device,
            timeout=limit,
        )
        runs = [BenchmarkRun("ByzFL Bucketing", avg_seconds)]
    except TimeoutError:
        label = "ByzFL Bucketing (timeout)"
        placeholder = limit if limit is not None else float("inf")
        runs = [BenchmarkRun(label, placeholder)]

    print("\nBucketing (ByzFL)")
    print("-----------------")
    for run in runs:
        print(f"{run.mode:35s} {run.avg_ms:10.2f} ms")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:  # pragma: no cover - manual interruption
        print("Benchmark cancelled.", file=sys.stderr)
        raise
