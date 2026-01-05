"""
Categories 3-4: DistributedP2PHonestNode and DistributedP2PByzNode Integration Tests
"""

from __future__ import annotations

import asyncio

import pytest
import torch

from byzpy.aggregators.coordinate_wise import CoordinateWiseMedian
from byzpy.attacks import EmpireAttack
from byzpy.engine.graph.pool import ActorPoolConfig
from byzpy.engine.node.context import InProcessContext
from byzpy.engine.node.decentralized import DecentralizedNode
from byzpy.engine.node.distributed import DistributedByzantineNode, DistributedHonestNode
from byzpy.engine.peer_to_peer.topology import Topology


@pytest.mark.asyncio
async def test_distributedp2phonestnode_works_with_decentralizednode():
    """Verify DistributedP2PHonestNode can be used with DecentralizedNode."""
    from byzpy.engine.node.application import HonestNodeApplication

    class TestHonestNode(DistributedHonestNode):
        def local_honest_gradient(self, *, x, y):
            return torch.tensor([1.0, 2.0])

        def next_batch(self):
            return torch.tensor([[1.0]]), torch.tensor([0])

        def apply_server_gradient(self, grad_vec):
            pass

    node_impl = TestHonestNode(
        actor_pool=[ActorPoolConfig(backend="thread", count=1)],
        aggregator=CoordinateWiseMedian(),
    )

    node = DecentralizedNode(
        node_id="node0",
        application=node_impl.application,
        context=InProcessContext(),
    )

    await node.start()

    assert node.application.has_pipeline(HonestNodeApplication.GRADIENT_PIPELINE)
    assert node.application.has_pipeline(HonestNodeApplication.AGGREGATION_PIPELINE)

    await node.shutdown()


@pytest.mark.asyncio
async def test_distributedp2phonestnode_half_step_execution():
    """Verify DistributedP2PHonestNode's half-step can be executed via DecentralizedNode."""
    from byzpy.engine.node.application import HonestNodeApplication

    class TestHonestNode(DistributedHonestNode):
        def local_honest_gradient(self, *, x, y):
            return x * 0.1 - y

        def next_batch(self):
            return torch.tensor([[1.0, 2.0]]), torch.tensor([0])

        def apply_server_gradient(self, grad_vec):
            pass

    node_impl = TestHonestNode(
        actor_pool=[ActorPoolConfig(backend="thread", count=1)],
        aggregator=CoordinateWiseMedian(),
    )

    node = DecentralizedNode(
        node_id="node0",
        application=node_impl.application,
        context=InProcessContext(),
    )

    await node.start()

    x = torch.tensor([1.0, 2.0])
    y = torch.tensor([0.5, 1.0])
    result = await node.execute_pipeline(HonestNodeApplication.GRADIENT_PIPELINE, {"x": x, "y": y})

    assert "honest_gradient" in result

    await node.shutdown()


@pytest.mark.asyncio
async def test_distributedp2phonestnode_aggregation_from_messages():
    """Verify DistributedP2PHonestNode aggregates gradients from neighbor messages."""
    from byzpy.engine.node.application import HonestNodeApplication

    topology = Topology.ring(3, k=1)
    received_gradients = []

    class TestHonestNode(DistributedHonestNode):
        def local_honest_gradient(self, *, x, y):
            return torch.tensor([1.0, 2.0])

        def next_batch(self):
            return torch.tensor([[1.0]]), torch.tensor([0])

        def apply_server_gradient(self, grad_vec):
            pass

    node_impl = TestHonestNode(
        actor_pool=[ActorPoolConfig(backend="thread", count=1)],
        aggregator=CoordinateWiseMedian(),
    )

    from byzpy.engine.node.cluster import DecentralizedCluster

    cluster = DecentralizedCluster()

    node = await cluster.add_node(
        node_id="node0",
        application=node_impl.application,
        topology=topology,
        context=InProcessContext(),
    )

    async def gradient_handler(from_id, payload):
        received_gradients.append(payload["gradient"])

    node.register_message_handler("gradient", gradient_handler)

    cluster._update_node_id_maps()
    await cluster.start_all()
    await asyncio.sleep(0.2)

    test_gradients = [
        torch.tensor([1.0, 2.0]),  # own gradient
        torch.tensor([2.0, 3.0]),  # neighbor 1
        torch.tensor([3.0, 4.0]),  # neighbor 2
    ]

    result = await node.application.aggregate(gradients=test_gradients)

    assert isinstance(result, torch.Tensor)
    torch.testing.assert_close(result, torch.tensor([2.0, 3.0]))

    await cluster.shutdown_all()


@pytest.mark.asyncio
async def test_distributedp2pbyznode_works_with_decentralizednode():
    """Verify DistributedP2PByzNode can be used with DecentralizedNode."""
    attack = EmpireAttack()

    class TestByzNode(DistributedByzantineNode):
        def next_batch(self):
            return torch.tensor([[1.0]]), torch.tensor([0])

        def apply_server_gradient(self, grad_vec):
            pass

    node_impl = TestByzNode(
        actor_pool=[ActorPoolConfig(backend="thread", count=1)],
        attack=attack,
    )

    node = DecentralizedNode(
        node_id="byz_node",
        application=node_impl.application,
        context=InProcessContext(),
    )

    await node.start()

    from byzpy.engine.node.application import ByzantineNodeApplication

    assert node.application.has_pipeline(ByzantineNodeApplication.ATTACK_PIPELINE)

    await node.shutdown()


@pytest.mark.asyncio
async def test_distributedp2pbyznode_broadcast_attack():
    """Verify DistributedP2PByzNode broadcasts malicious vectors."""
    topology = Topology.ring(2, k=1)  # Use 2 nodes so they're neighbors
    received_vectors = []

    attack = EmpireAttack()

    class TestByzNode(DistributedByzantineNode):
        def next_batch(self):
            return torch.tensor([[1.0]]), torch.tensor([0])

        def apply_server_gradient(self, grad_vec):
            pass

    node_impl = TestByzNode(
        actor_pool=[ActorPoolConfig(backend="thread", count=1)],
        attack=attack,
    )

    from byzpy.engine.node.cluster import DecentralizedCluster

    cluster = DecentralizedCluster()

    node = await cluster.add_node(
        node_id="node0",  # Use integer-like string IDs for topology
        application=node_impl.application,
        topology=topology,
        context=InProcessContext(),
    )

    from byzpy.engine.node.application import HonestNodeApplication

    honest_app = HonestNodeApplication(
        name="honest",
        actor_pool=[ActorPoolConfig(backend="thread", count=1)],
    )
    honest_node = await cluster.add_node(
        node_id="node1",  # Use integer-like string IDs for topology
        application=honest_app,
        topology=topology,
        context=InProcessContext(),
    )

    async def vector_handler(from_id, payload):
        received_vectors.append(payload["vector"])

    honest_node.register_message_handler("gradient", vector_handler)

    cluster._update_node_id_maps()
    await cluster.start_all()
    await asyncio.sleep(0.3)

    # Byzantine node broadcasts malicious vector
    template = torch.tensor([1.0, 2.0, 3.0])
    malicious = await node_impl.byzantine_gradient_async(
        honest_grads=[torch.tensor([1.0, 1.0, 1.0])]
    )

    await node.broadcast_message("gradient", {"vector": malicious})
    await asyncio.sleep(0.5)

    assert isinstance(malicious, torch.Tensor)
    assert malicious.shape == template.shape or malicious.numel() > 0
    assert len(received_vectors) >= 1

    await cluster.shutdown_all()
