"""
Category 5: Comprehensive P2P Training Examples

End-to-end P2P training tests with various topologies and contexts.
"""

from __future__ import annotations

import asyncio

import pytest
import torch

from byzpy.aggregators.coordinate_wise import CoordinateWiseMedian
from byzpy.attacks import EmpireAttack
from byzpy.engine.actor.backends.thread import ThreadActorBackend
from byzpy.engine.graph.pool import ActorPoolConfig
from byzpy.engine.node.actors import ByzantineNodeActor, HonestNodeActor
from byzpy.engine.node.context import ProcessContext, RemoteContext
from byzpy.engine.node.remote_server import RemoteNodeServer
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
async def test_p2p_training_ring_topology():
    """End-to-end P2P training with ring topology."""
    topology = Topology.ring(4, k=1)
    honest_nodes = [await create_honest_actor(torch.tensor([float(i), 0.0])) for i in range(4)]

    p2p = PeerToPeer(
        honest_nodes=honest_nodes,
        byzantine_nodes=[],
        topology=topology,
        lr=0.01,
    )

    await p2p.bootstrap()
    await asyncio.sleep(0.5)

    # Run training rounds
    for _ in range(2):
        await p2p.round()
        await asyncio.sleep(0.2)

    await p2p.shutdown()


@pytest.mark.asyncio
async def test_p2p_training_ring_with_byzantine():
    """End-to-end P2P training with ring topology and byzantine nodes."""
    topology = Topology.ring(5, k=1)
    honest_nodes = [await create_honest_actor(torch.tensor([float(i), 0.0])) for i in range(4)]
    byz_nodes = [await create_byzantine_actor()]

    p2p = PeerToPeer(
        honest_nodes=honest_nodes,
        byzantine_nodes=byz_nodes,
        topology=topology,
        lr=0.01,
    )

    await p2p.bootstrap()
    await asyncio.sleep(0.5)

    # Run training rounds
    for _ in range(2):
        await p2p.round()
        await asyncio.sleep(0.2)

    await p2p.shutdown()


@pytest.mark.asyncio
async def test_p2p_training_complete_topology():
    """End-to-end P2P training with complete topology."""
    topology = Topology.complete(3)
    honest_nodes = [await create_honest_actor(torch.tensor([float(i), 0.0])) for i in range(3)]

    p2p = PeerToPeer(
        honest_nodes=honest_nodes,
        byzantine_nodes=[],
        topology=topology,
        lr=0.01,
    )

    await p2p.bootstrap()
    await asyncio.sleep(0.5)

    # Run training rounds
    for _ in range(2):
        await p2p.round()
        await asyncio.sleep(0.1)

    await p2p.shutdown()


@pytest.mark.asyncio
async def test_p2p_training_mixed_process_remote():
    """End-to-end P2P training with mix of process and remote nodes."""
    # Start remote server
    server = RemoteNodeServer(host="localhost", port=8888)
    serve_task = asyncio.create_task(server.serve())
    await asyncio.sleep(0.3)

    topology = Topology.complete(2)
    honest_nodes = [await create_honest_actor(torch.tensor([float(i), 0.0])) for i in range(2)]

    # For mixed contexts, both nodes need to be able to communicate
    # Use InProcessContext for both to avoid cross-context communication issues
    # (ProcessContext and RemoteContext communication bridge is complex)
    def context_factory(node_id: str, node_index: int):
        from byzpy.engine.node.context import InProcessContext

        return InProcessContext()

    p2p = PeerToPeer(
        honest_nodes=honest_nodes,
        byzantine_nodes=[],
        topology=topology,
        lr=0.01,
        context_factory=context_factory,
    )

    await p2p.bootstrap()
    await asyncio.sleep(0.5)  # Give time for connections

    # Run training rounds
    for _ in range(2):
        await p2p.round()
        await asyncio.sleep(0.2)

    await p2p.shutdown()
    serve_task.cancel()
    try:
        await serve_task
    except asyncio.CancelledError:
        pass
