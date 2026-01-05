from __future__ import annotations

import pytest
import torch

from byzpy.engine.peer_to_peer.runner import DecentralizedPeerToPeer
from byzpy.engine.peer_to_peer.topology import Topology


class _StubHonest:
    def __init__(self, grad: torch.Tensor):
        self._grad = grad
        self.last = None
        self.lr = 0.1

    async def p2p_half_step(self, lr: float):
        return self._grad * lr

    async def p2p_aggregate(self, neighbor_vectors):
        return sum(neighbor_vectors) / len(neighbor_vectors)

    async def apply_server_gradient(self, g):
        self.last = g


class _StubByz:
    async def p2p_broadcast_vector(self, neighbor_vectors, like):
        return like * 0.0


@pytest.mark.skip(
    reason="Test uses incompatible synchronous API - DecentralizedPeerToPeer uses async methods and DecentralizedCluster lacks barrier/state methods"
)
def test_decentralized_p2p_round():
    topo = Topology.complete(3)
    honest = [
        _StubHonest(torch.tensor([1.0, 0.0])),
        _StubHonest(torch.tensor([0.0, 1.0])),
        _StubHonest(torch.tensor([1.0, 1.0])),
    ]
    byz: list[_StubByz] = []

    runner = DecentralizedPeerToPeer(honest, byz, topo, lr=0.1)
    runner.start()
    try:
        runner.cluster.barrier(0.1)
        # Each honest broadcasted half-step; aggregation averages neighbors (including self in this stub)
        outs = [runner.cluster.state(str(i)).get("out") for i in range(3)]
        assert all(isinstance(o, torch.Tensor) for o in outs)
    finally:
        runner.stop()


@pytest.mark.skip(
    reason="Test uses incompatible synchronous API - DecentralizedPeerToPeer uses async methods and DecentralizedCluster lacks barrier/state methods"
)
def test_decentralized_p2p_with_byzantine():
    topo = Topology.complete(2)
    honest = [_StubHonest(torch.tensor([1.0, 0.0]))]
    byz = [_StubByz()]
    runner = DecentralizedPeerToPeer(honest, byz, topo, lr=0.1)
    runner.start()
    try:
        runner.cluster.barrier(0.1)
        # Honest node should have produced an output tensor even with byzantine present.
        out = runner.cluster.state("0").get("out")
        assert isinstance(out, torch.Tensor)
    finally:
        runner.stop()
