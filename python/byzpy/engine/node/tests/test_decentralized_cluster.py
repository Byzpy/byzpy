from __future__ import annotations

import asyncio

import pytest

from byzpy.engine.graph.pool import ActorPoolConfig
from byzpy.engine.node.application import NodeApplication
from byzpy.engine.node.cluster import DecentralizedCluster
from byzpy.engine.node.context import ProcessContext


def create_test_application():
    """Create a test NodeApplication."""
    return NodeApplication(
        name="test-app",
        actor_pool=[ActorPoolConfig(backend="thread", count=1)],
    )


# Category 3: DecentralizedCluster


def test_decentralizedcluster_can_be_created():
    """Verify DecentralizedCluster can be instantiated."""
    cluster = DecentralizedCluster()
    assert cluster is not None
    assert len(cluster.nodes) == 0


@pytest.mark.asyncio
async def test_decentralizedcluster_add_node():
    """Verify add_node() creates process-based node."""
    cluster = DecentralizedCluster()

    node = await cluster.add_node(
        node_id="node1",
        application=create_test_application(),
    )

    assert node.node_id == "node1"
    assert isinstance(node.context, ProcessContext)
    assert "node1" in cluster.nodes


@pytest.mark.asyncio
async def test_decentralizedcluster_start_all():
    """Verify start_all() starts all nodes."""
    cluster = DecentralizedCluster()

    await cluster.add_node("node1", create_test_application())
    await cluster.add_node("node2", create_test_application())

    await cluster.start_all()

    for node in cluster.nodes.values():
        assert node._running
        assert node.context._process.is_alive()

    await cluster.shutdown_all()


@pytest.mark.asyncio
async def test_decentralizedcluster_shutdown_all():
    """Verify shutdown_all() stops all nodes."""
    cluster = DecentralizedCluster()

    await cluster.add_node("node1", create_test_application())
    await cluster.add_node("node2", create_test_application())
    await cluster.start_all()

    # Save process references before shutdown
    processes = [node.context._process for node in cluster.nodes.values()]

    await cluster.shutdown_all()

    for node in cluster.nodes.values():
        assert not node._running
    for proc in processes:
        assert not proc.is_alive()


@pytest.mark.asyncio
async def test_decentralizedcluster_get_node():
    """Verify get_node() retrieves node by ID."""
    cluster = DecentralizedCluster()

    node1 = await cluster.add_node("node1", create_test_application())

    retrieved = cluster.get_node("node1")
    assert retrieved is node1

    assert cluster.get_node("nonexistent") is None


@pytest.mark.asyncio
async def test_decentralizedcluster_remove_node():
    """Verify remove_node() removes and shuts down node."""
    cluster = DecentralizedCluster()

    node1 = await cluster.add_node("node1", create_test_application())
    await cluster.start_all()

    process = node1.context._process

    await cluster.remove_node("node1")

    assert "node1" not in cluster.nodes
    assert not node1._running
    assert not process.is_alive()


@pytest.mark.asyncio
async def test_decentralizedcluster_duplicate_node_id():
    """Verify cluster rejects duplicate node IDs."""
    cluster = DecentralizedCluster()

    await cluster.add_node("node1", create_test_application())

    with pytest.raises(ValueError, match="already exists"):
        await cluster.add_node("node1", create_test_application())

    await cluster.shutdown_all()
