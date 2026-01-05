"""
Category 1: DecentralizedPeerToPeer Refactoring Tests

Tests for the refactored DecentralizedPeerToPeer implementation using DecentralizedCluster.
"""

from __future__ import annotations

import asyncio

import pytest
import torch

from byzpy.engine.actor.backends.thread import ThreadActorBackend
from byzpy.engine.node.actors import ByzantineNodeActor, HonestNodeActor
from byzpy.engine.node.cluster import DecentralizedCluster
from byzpy.engine.node.context import ProcessContext, RemoteContext
from byzpy.engine.node.decentralized import DecentralizedNode
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
async def test_decentralizedpeertopeer_can_be_created():
    """Verify refactored DecentralizedPeerToPeer can be instantiated."""
    topology = Topology.ring(3, k=1)
    honest_nodes = [
        await create_honest_actor(torch.tensor([1.0, 0.0])),
        await create_honest_actor(torch.tensor([0.0, 1.0])),
    ]
    byz_nodes = [await create_byzantine_actor()]

    p2p = PeerToPeer(
        honest_nodes=honest_nodes,
        byzantine_nodes=byz_nodes,
        topology=topology,
        lr=0.01,
    )

    assert p2p is not None
    assert p2p._runner is not None
    assert isinstance(p2p._runner._cluster, DecentralizedCluster)


@pytest.mark.asyncio
async def test_decentralizedpeertopeer_uses_decentralizedcluster():
    """Verify DecentralizedPeerToPeer uses DecentralizedCluster internally."""
    topology = Topology.ring(2, k=1)
    honest_nodes = [
        await create_honest_actor(torch.tensor([1.0, 0.0])),
        await create_honest_actor(torch.tensor([0.0, 1.0])),
    ]

    p2p = PeerToPeer(
        honest_nodes=honest_nodes,
        byzantine_nodes=[],
        topology=topology,
    )

    assert isinstance(p2p._runner._cluster, DecentralizedCluster)


@pytest.mark.asyncio
async def test_decentralizedpeertopeer_creates_decentralizednodes():
    """Verify DecentralizedPeerToPeer creates DecentralizedNode instances."""
    topology = Topology.ring(2, k=1)
    honest_nodes = [
        await create_honest_actor(torch.tensor([1.0, 0.0])),
        await create_honest_actor(torch.tensor([0.0, 1.0])),
    ]

    p2p = PeerToPeer(
        honest_nodes=honest_nodes,
        byzantine_nodes=[],
        topology=topology,
    )

    await p2p.bootstrap()

    cluster = p2p._runner._cluster
    assert len(cluster.nodes) == 2
    for node in cluster.nodes.values():
        assert isinstance(node, DecentralizedNode)

    await p2p.shutdown()


@pytest.mark.asyncio
async def test_decentralizedpeertopeer_uses_processcontext_by_default():
    """Verify DecentralizedPeerToPeer uses ProcessContext by default."""
    topology = Topology.ring(2, k=1)
    honest_nodes = [
        await create_honest_actor(torch.tensor([1.0, 0.0])),
        await create_honest_actor(torch.tensor([0.0, 1.0])),
    ]

    p2p = PeerToPeer(
        honest_nodes=honest_nodes,
        byzantine_nodes=[],
        topology=topology,
    )

    await p2p.bootstrap()

    for node in p2p._runner._cluster.nodes.values():
        assert isinstance(node.context, ProcessContext)

    await p2p.shutdown()


@pytest.mark.asyncio
async def test_decentralizedpeertopeer_supports_remote_context():
    """Verify DecentralizedPeerToPeer can use RemoteContext for nodes."""
    server = RemoteNodeServer(host="localhost", port=8888)
    serve_task = asyncio.create_task(server.serve())
    await asyncio.sleep(0.3)

    topology = Topology.ring(2, k=1)
    honest_nodes = [
        await create_honest_actor(torch.tensor([1.0, 0.0])),
        await create_honest_actor(torch.tensor([0.0, 1.0])),
    ]

    def context_factory(node_id: str, node_index: int) -> RemoteContext:
        return RemoteContext(host="localhost", port=8888)

    p2p = PeerToPeer(
        honest_nodes=honest_nodes,
        byzantine_nodes=[],
        topology=topology,
        context_factory=context_factory,
    )

    await p2p.bootstrap()
    await asyncio.sleep(0.5)  # Give time for connections

    # Verify nodes use RemoteContext
    for node in p2p._runner._cluster.nodes.values():
        assert isinstance(node.context, RemoteContext)

    await p2p.shutdown()
    serve_task.cancel()
    try:
        await serve_task
    except asyncio.CancelledError:
        pass


# Additional Category 1 tests


@pytest.mark.asyncio
async def test_decentralizedpeertopeer_supports_mixed_contexts():
    """Verify DecentralizedPeerToPeer can use different contexts for different nodes."""
    from byzpy.engine.node.context import InProcessContext

    topology = Topology.ring(2, k=1)
    honest_nodes = [
        await create_honest_actor(torch.tensor([1.0, 0.0])),
        await create_honest_actor(torch.tensor([0.0, 1.0])),
    ]

    def context_factory(node_id: str, node_index: int):
        if node_index == 0:
            return ProcessContext()
        else:
            return InProcessContext()

    p2p = PeerToPeer(
        honest_nodes=honest_nodes,
        byzantine_nodes=[],
        topology=topology,
        context_factory=context_factory,
    )

    await p2p.bootstrap()

    # Verify nodes use different contexts
    nodes_list = list(p2p._runner._cluster.nodes.values())
    assert isinstance(nodes_list[0].context, ProcessContext)
    assert isinstance(nodes_list[1].context, InProcessContext)

    await p2p.shutdown()


@pytest.mark.asyncio
async def test_decentralizedpeertopeer_node_count_matches_input():
    """Verify DecentralizedPeerToPeer creates correct number of nodes."""
    topology = Topology.ring(5, k=1)
    honest_nodes = [await create_honest_actor(torch.tensor([float(i), 0.0])) for i in range(3)]
    byz_nodes = [await create_byzantine_actor() for _ in range(2)]

    p2p = PeerToPeer(
        honest_nodes=honest_nodes,
        byzantine_nodes=byz_nodes,
        topology=topology,
    )

    await p2p.bootstrap()

    assert len(p2p._runner._cluster.nodes) == 5

    await p2p.shutdown()
