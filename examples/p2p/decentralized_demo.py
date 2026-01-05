#!/usr/bin/env python
"""
Minimal decentralized P2P demo using NodeRunner-based orchestration.

This avoids the actor backend entirely: each node runs in its own process
with its own scheduler loop, exchanging messages via in-process queues.
"""
from __future__ import annotations

import argparse
from typing import List

import torch

from byzpy.engine.peer_to_peer.runner import DecentralizedPeerToPeer
from byzpy.engine.peer_to_peer.topology import Topology
from byzpy.engine.transport.local import LocalTransport
from byzpy.engine.transport.tcp import TcpTransport


class HonestStub:
    def __init__(self, vec: torch.Tensor, lr: float = 0.1) -> None:
        self.vec = vec
        self.lr = lr

    async def p2p_half_step(self, lr: float):
        return self.vec * lr

    async def p2p_aggregate(self, neighbor_vectors):
        return sum(neighbor_vectors) / len(neighbor_vectors)

    async def apply_server_gradient(self, g):
        # no-op for demo
        return None


class ByzantineStub:
    async def p2p_broadcast_vector(self, neighbor_vectors, like):
        # Broadcast zeros to simulate an attack.
        return like * 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Decentralized P2P demo (runner-based).")
    parser.add_argument(
        "--transport",
        choices=["local", "tcp"],
        default="local",
        help="Transport backend for messaging.",
    )
    args = parser.parse_args()

    transport = LocalTransport() if args.transport == "local" else TcpTransport()
    topo = Topology.complete(3)
    honest: List[HonestStub] = [
        HonestStub(torch.tensor([1.0, 0.0])),
        HonestStub(torch.tensor([0.0, 1.0])),
        HonestStub(torch.tensor([1.0, 1.0])),
    ]
    byz: List[ByzantineStub] = []
    runner = DecentralizedPeerToPeer(honest, byz, topo, lr=0.1, transport=transport)
    runner.start()
    try:
        for r in range(3):
            runner.step_once()
            outs = [runner.cluster.state(str(i)).get("out") for i in range(len(honest))]
            print(f"Round {r+1}: outputs={outs}")
    finally:
        runner.stop()
        if hasattr(transport, "close"):
            transport.close()


if __name__ == "__main__":
    main()
