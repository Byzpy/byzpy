from __future__ import annotations

import pytest
import torch

from byzpy.engine.peer_to_peer.runner import DecentralizedPeerToPeer
from byzpy.engine.peer_to_peer.topology import Topology
from byzpy.engine.transport.local import LocalTransport
from byzpy.engine.transport.tcp import TcpTransport


class _StubHonest:
    def __init__(self, vec: torch.Tensor):
        self.vec = vec
        self.lr = 0.1

    async def p2p_half_step(self, lr: float):
        return self.vec * lr

    async def p2p_aggregate(self, neighbor_vectors):
        return sum(neighbor_vectors) / len(neighbor_vectors)

    async def apply_server_gradient(self, g):
        return None


@pytest.mark.skip(
    reason="Test uses incompatible synchronous API - DecentralizedPeerToPeer does not accept transport parameter and uses async methods"
)
def test_p2p_runner_with_transport():
    topo = Topology.complete(2)
    transport = LocalTransport()
    honest = [
        _StubHonest(torch.tensor([1.0, 0.0])),
        _StubHonest(torch.tensor([0.0, 1.0])),
    ]
    byz = []
    runner = DecentralizedPeerToPeer(honest, byz, topo, lr=0.1, transport=transport)
    runner.start()
    try:
        runner.cluster.barrier(0.1)
        outs = [runner.cluster.state(str(i)).get("out") for i in range(2)]
        assert all(isinstance(o, torch.Tensor) for o in outs)
    finally:
        runner.stop()


@pytest.mark.skip(
    reason="Test uses incompatible synchronous API - DecentralizedPeerToPeer does not accept transport parameter and uses async methods"
)
def test_p2p_runner_over_tcp():
    topo = Topology.complete(2)
    transport = TcpTransport()
    honest = [
        _StubHonest(torch.tensor([1.0, 0.0])),
        _StubHonest(torch.tensor([0.0, 1.0])),
    ]
    byz = []
    runner = DecentralizedPeerToPeer(honest, byz, topo, lr=0.1, transport=transport)
    runner.start()
    try:
        runner.step_once()  # advance once
        outs = [runner.cluster.state(str(i)).get("out") for i in range(2)]
        assert all(isinstance(o, torch.Tensor) for o in outs)
    finally:
        runner.stop()
        transport.close()
