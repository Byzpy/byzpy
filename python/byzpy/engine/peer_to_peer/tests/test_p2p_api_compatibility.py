"""
Category 7: API Compatibility Tests

Tests for backwards compatibility with existing PeerToPeer API.
"""

from __future__ import annotations

import asyncio

import pytest
import torch

from byzpy.aggregators.coordinate_wise import CoordinateWiseMedian
from byzpy.engine.actor.backends.thread import ThreadActorBackend
from byzpy.engine.graph.pool import ActorPoolConfig
from byzpy.engine.node.actors import ByzantineNodeActor, HonestNodeActor
from byzpy.engine.peer_to_peer.topology import Topology
from byzpy.engine.peer_to_peer.train import PeerToPeer


class _StubHonestNode:
    """Stub honest node for testing."""

    def __init__(self, grad: torch.Tensor):
        self._grad = grad
        self.lr = 0.1

    async def p2p_half_step(self, lr: float):
        return self._grad * lr

    async def p2p_aggregate(self, neighbor_vectors):
        if not neighbor_vectors:
            return self._grad
        return sum(neighbor_vectors) / len(neighbor_vectors)


class _StubByzantineNode:
    """Stub byzantine node for testing."""

    async def p2p_broadcast_vector(self, neighbor_vectors=None, like=None):
        if like is not None:
            return torch.zeros_like(like)
        return torch.tensor([0.0, 0.0])


async def create_honest_actor(grad: torch.Tensor) -> HonestNodeActor:
    """Helper to create an honest node actor."""
    backend = ThreadActorBackend()
    actor = await HonestNodeActor.spawn(
        _StubHonestNode,
        backend=backend,
        kwargs={"grad": grad},
    )
    return actor


async def create_byzantine_actor() -> ByzantineNodeActor:
    """Helper to create a byzantine node actor."""
    backend = ThreadActorBackend()
    actor = await ByzantineNodeActor.spawn(
        _StubByzantineNode,
        backend=backend,
    )
    return actor


@pytest.mark.asyncio
async def test_peertopeer_api_compatibility():
    """Verify PeerToPeer maintains backwards-compatible API."""
    topology = Topology.ring(3, k=1)
    honest_nodes = [
        await create_honest_actor(torch.tensor([1.0, 0.0])),
        await create_honest_actor(torch.tensor([0.0, 1.0])),
        await create_honest_actor(torch.tensor([1.0, 1.0])),
    ]

    p2p = PeerToPeer(
        honest_nodes=honest_nodes,
        byzantine_nodes=[],
        topology=topology,
        lr=0.01,
        channel_name="p2p",  # Should be accepted for compatibility
    )

    # API should match original
    await p2p.bootstrap()
    await p2p.round()
    await p2p.shutdown()


@pytest.mark.asyncio
async def test_peertopeer_multiple_rounds():
    """Verify PeerToPeer can run multiple training rounds."""
    topology = Topology.ring(3, k=1)
    honest_nodes = [
        await create_honest_actor(torch.tensor([1.0, 0.0])),
        await create_honest_actor(torch.tensor([0.0, 1.0])),
        await create_honest_actor(torch.tensor([1.0, 1.0])),
    ]

    p2p = PeerToPeer(
        honest_nodes=honest_nodes,
        byzantine_nodes=[],
        topology=topology,
        lr=0.01,
    )

    await p2p.bootstrap()

    # Run multiple rounds
    for round_num in range(5):
        await p2p.round()
        await asyncio.sleep(0.1)

    await p2p.shutdown()
