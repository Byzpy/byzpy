#!/usr/bin/env python
"""
Benchmark the ByzFL implementation of the Gaussian attack.
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
    parser = argparse.ArgumentParser(description="Benchmark ByzFL's Gaussian attack.")
    parser.add_argument("--num-grads", type=int, default=64, help="Honest gradients")
    parser.add_argument("--grad-dim", type=int, default=65536, help="Gradient dimension")
    parser.add_argument("--mu", type=float, default=0.0, help="Gaussian mean")
    parser.add_argument("--sigma", type=float, default=1.0, help="Gaussian stddev")
    parser.add_argument("--warmup", type=int, default=0, help="Warm-up iterations.")
    parser.add_argument("--repeat", type=int, default=2, help="Timed iterations.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for honest gradients.")
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


def _make_gradients(n: int, dim: int, seed: int, device: torch.device) -> list[torch.Tensor]:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return [torch.randn(dim, generator=gen, device=device, dtype=torch.float32) for _ in range(n)]


def _require_byzfl() -> "type[object]":
    try:
        from byzfl.attacks.attacks import Gaussian as ByzflGaussian
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise SystemExit(
            "The byzfl package is required for this benchmark (pip install byzfl or add it to your environment)."
        ) from exc
    return ByzflGaussian


def _time_byzfl(
    attack: object,
    grads: Sequence[torch.Tensor],
    *,
    iterations: int,
    warmup: int,
    sync_device: torch.device,
    timeout: float | None = None,
) -> float:
    def _run_once() -> None:
        attack(grads)
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
    grads = _make_gradients(args.num_grads, args.grad_dim, args.seed, device)
    byzfl_cls = _require_byzfl()
    attack = byzfl_cls(mu=float(args.mu), sigma=float(args.sigma))
    limit = args.timeout if args.timeout > 0 else None
    try:
        avg_seconds = _time_byzfl(
            attack,
            grads,
            iterations=args.repeat,
            warmup=args.warmup,
            sync_device=device,
            timeout=limit,
        )
        runs = [BenchmarkRun("ByzFL Direct attack", avg_seconds)]
    except TimeoutError:
        label = "ByzFL Direct attack (timeout)"
        placeholder = limit if limit is not None else float("inf")
        runs = [BenchmarkRun(label, placeholder)]

    print("\nGaussian Attack (ByzFL)")
    print("-----------------------")
    for run in runs:
        print(f"{run.mode:35s} {run.avg_ms:10.2f} ms")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:  # pragma: no cover - manual interruption
        print("Benchmark cancelled.", file=sys.stderr)
        raise
