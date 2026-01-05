#!/usr/bin/env python
"""
Minimal decentralized ParameterServer demo using NodeRunner-based orchestration.
"""
from __future__ import annotations

import argparse

import torch

from byzpy.engine.parameter_server.runner import ParameterServerRunner
from byzpy.engine.transport.local import LocalTransport
from byzpy.engine.transport.tcp import TcpTransport


def main() -> None:
    parser = argparse.ArgumentParser(description="Decentralized PS demo (runner-based).")
    parser.add_argument(
        "--transport",
        choices=["local", "tcp"],
        default="local",
        help="Transport backend for messaging.",
    )
    args = parser.parse_args()

    transport = LocalTransport() if args.transport == "local" else TcpTransport()
    grads = [
        torch.tensor([1.0, 0.0]),
        torch.tensor([0.0, 1.0]),
        torch.tensor([1.0, 1.0]),
    ]
    runner = ParameterServerRunner(
        worker_grad_fns=[lambda g=g: g for g in grads], transport=transport
    )
    runner.start()
    try:
        out = runner.run_round()
        print("Aggregated:", out)
    finally:
        runner.stop()
        if hasattr(transport, "close"):
            transport.close()


if __name__ == "__main__":
    main()
