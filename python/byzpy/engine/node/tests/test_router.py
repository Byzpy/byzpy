"""Tests for MessageRouter - Categories 1 and 2 from Milestone 4 Test Plan."""

import asyncio

import pytest

# =============================================================================
# Category 1: MessageRouter Core (8 tests)
# =============================================================================

# 1.1 MessageRouter Creation and Configuration


def test_messagerouter_can_be_created():
    """Verify MessageRouter can be instantiated with topology and node_id."""
    from byzpy.engine.node.router import MessageRouter
    from byzpy.engine.peer_to_peer.topology import Topology

    topology = Topology.ring(5, k=1)  # 5-node ring
    router = MessageRouter(topology=topology, node_id=0)

    assert router is not None
    assert router.node_id == 0
    assert router.topology is topology


def test_messagerouter_without_topology():
    """Verify MessageRouter can work without topology (allows all sends)."""
    from byzpy.engine.node.router import MessageRouter

    router = MessageRouter(topology=None, node_id="node1")

    # Without topology, all sends should be allowed
    assert router.can_send_to("node2")
    assert router.can_send_to("any_node")


def test_messagerouter_with_string_node_ids():
    """Verify MessageRouter works with string node IDs via node_id_map."""
    from byzpy.engine.node.router import MessageRouter
    from byzpy.engine.peer_to_peer.topology import Topology

    topology = Topology.ring(3, k=1)
    node_id_map = {0: "alice", 1: "bob", 2: "charlie"}

    router = MessageRouter(topology=topology, node_id="alice", node_id_map=node_id_map)

    # Alice (0) neighbors in ring: bob (1), charlie (2)
    assert router.can_send_to("bob")
    assert router.can_send_to("charlie")


# 1.2 Neighbor Computation


def test_messagerouter_computes_out_neighbors():
    """Verify get_out_neighbors() returns correct outgoing neighbors."""
    from byzpy.engine.node.router import MessageRouter
    from byzpy.engine.peer_to_peer.topology import Topology

    # Ring topology: 0 -> 1, 0 -> 4 (k=1, n=5)
    topology = Topology.ring(5, k=1)
    router = MessageRouter(topology=topology, node_id=0)

    neighbors = router.get_out_neighbors()

    # Node 0 should have neighbors 1 and 4 (prev and next in ring)
    assert set(neighbors) == {1, 4}


def test_messagerouter_computes_in_neighbors():
    """Verify get_in_neighbors() returns correct incoming neighbors."""
    from byzpy.engine.node.router import MessageRouter
    from byzpy.engine.peer_to_peer.topology import Topology

    topology = Topology.ring(5, k=1)
    router = MessageRouter(topology=topology, node_id=0)

    in_neighbors = router.get_in_neighbors()

    # Node 0 receives from 1 and 4 in bidirectional ring
    assert set(in_neighbors) == {1, 4}


def test_messagerouter_complete_topology_all_neighbors():
    """Verify complete topology has all other nodes as neighbors."""
    from byzpy.engine.node.router import MessageRouter
    from byzpy.engine.peer_to_peer.topology import Topology

    topology = Topology.complete(4)
    router = MessageRouter(topology=topology, node_id=0)

    neighbors = router.get_out_neighbors()

    # Complete graph: every node connects to all others
    assert set(neighbors) == {1, 2, 3}


# 1.3 Send Permission Checks


def test_messagerouter_can_send_to_neighbor():
    """Verify can_send_to() returns True for valid neighbors."""
    from byzpy.engine.node.router import MessageRouter
    from byzpy.engine.peer_to_peer.topology import Topology

    topology = Topology.ring(5, k=1)
    router = MessageRouter(topology=topology, node_id=0)

    # Node 0's neighbors are 1 and 4
    assert router.can_send_to(1) is True
    assert router.can_send_to(4) is True


def test_messagerouter_cannot_send_to_non_neighbor():
    """Verify can_send_to() returns False for non-neighbors."""
    from byzpy.engine.node.router import MessageRouter
    from byzpy.engine.peer_to_peer.topology import Topology

    topology = Topology.ring(5, k=1)
    router = MessageRouter(topology=topology, node_id=0)

    # Node 0 cannot send to 2 or 3 in k=1 ring
    assert router.can_send_to(2) is False
    assert router.can_send_to(3) is False


# =============================================================================
# Category 2: Routing Patterns (9 tests)
# =============================================================================


# Mock node for testing routing patterns
class MockDecentralizedNode:
    """Minimal mock node for context registration."""

    def __init__(self, node_id, context):
        self.node_id = node_id
        self.context = context


# 2.1 Direct Routing


@pytest.mark.asyncio
async def test_messagerouter_route_direct():
    """Verify route_direct() sends message to specific neighbor."""
    from byzpy.engine.node.context import InProcessContext
    from byzpy.engine.node.router import MessageRouter
    from byzpy.engine.peer_to_peer.topology import Topology

    topology = Topology.ring(3, k=1)

    # Setup nodes with contexts
    ctx0 = InProcessContext()
    ctx1 = InProcessContext()

    # Create mock nodes
    node0 = MockDecentralizedNode(node_id=0, context=ctx0)
    node1 = MockDecentralizedNode(node_id=1, context=ctx1)

    await ctx0.start(node0)
    await ctx1.start(node1)

    router = MessageRouter(topology=topology, node_id=0)

    await router.route_direct(
        target_node_id=1,
        message_type="gradient",
        payload={"data": [1.0, 2.0]},
        context=ctx0,
    )

    # Verify message was delivered to node 1
    received = []
    async for msg in ctx1.receive_messages():
        received.append(msg)
        break

    assert len(received) == 1
    assert received[0]["type"] == "gradient"
    assert received[0]["from"] == 0


@pytest.mark.asyncio
async def test_messagerouter_route_direct_to_non_neighbor_raises():
    """Verify route_direct() raises error for non-neighbor."""
    from byzpy.engine.node.router import MessageRouter
    from byzpy.engine.peer_to_peer.topology import Topology

    topology = Topology.ring(5, k=1)
    router = MessageRouter(topology=topology, node_id=0)

    # Node 0 cannot send to node 2 (not a neighbor in k=1 ring)
    with pytest.raises(ValueError, match="not a neighbor"):
        await router.route_direct(
            target_node_id=2,
            message_type="test",
            payload={},
            context=None,
        )


# 2.2 Broadcast Routing


@pytest.mark.asyncio
async def test_messagerouter_route_broadcast():
    """Verify route_broadcast() sends message to all neighbors."""
    from byzpy.engine.node.context import InProcessContext
    from byzpy.engine.node.router import MessageRouter
    from byzpy.engine.peer_to_peer.topology import Topology

    topology = Topology.ring(4, k=1)

    # Setup 4 nodes
    contexts = [InProcessContext() for _ in range(4)]
    nodes = [MockDecentralizedNode(node_id=i, context=contexts[i]) for i in range(4)]

    for ctx, node in zip(contexts, nodes):
        await ctx.start(node)

    router = MessageRouter(topology=topology, node_id=0)

    # Broadcast from node 0 to its neighbors (1 and 3)
    await router.route_broadcast(
        message_type="sync",
        payload={"round": 1},
        context=contexts[0],
    )

    # Nodes 1 and 3 should receive (neighbors of 0)
    # Node 2 should NOT receive (not a neighbor of 0 in k=1 ring)
    msg1 = await asyncio.wait_for(contexts[1]._inbox.get(), timeout=1.0)
    msg3 = await asyncio.wait_for(contexts[3]._inbox.get(), timeout=1.0)

    assert msg1["type"] == "sync"
    assert msg3["type"] == "sync"

    # Node 2's inbox should be empty
    assert contexts[2]._inbox.empty()


@pytest.mark.asyncio
async def test_messagerouter_route_broadcast_complete_topology():
    """Verify broadcast in complete topology reaches all other nodes."""
    from byzpy.engine.node.context import InProcessContext
    from byzpy.engine.node.router import MessageRouter
    from byzpy.engine.peer_to_peer.topology import Topology

    topology = Topology.complete(4)

    contexts = [InProcessContext() for _ in range(4)]
    nodes = [MockDecentralizedNode(node_id=i, context=contexts[i]) for i in range(4)]

    for ctx, node in zip(contexts, nodes):
        await ctx.start(node)

    router = MessageRouter(topology=topology, node_id=0)

    await router.route_broadcast(
        message_type="global_update",
        payload={"params": [0.1]},
        context=contexts[0],
    )

    # All other nodes (1, 2, 3) should receive
    for i in [1, 2, 3]:
        msg = await asyncio.wait_for(contexts[i]._inbox.get(), timeout=1.0)
        assert msg["type"] == "global_update"


@pytest.mark.asyncio
async def test_messagerouter_route_broadcast_resilient_to_failures():
    """Verify route_broadcast() continues sending to remaining neighbors even if some fail."""
    from byzpy.engine.node.context import NodeContext
    from byzpy.engine.node.router import MessageRouter
    from byzpy.engine.peer_to_peer.topology import Topology

    topology = Topology.ring(3, k=1)  # Node 0 neighbors: 1, 2

    # Create a mock context that fails for node 1 but succeeds for node 2
    class FailingContext(NodeContext):
        """Context that fails for specific node IDs."""

        def __init__(self, fail_for_nodes):
            self.fail_for_nodes = fail_for_nodes
            self.successful_sends = []
            self.failed_sends = []

        async def start(self, node):
            pass

        async def send_message(self, to_node_id: str, message_type: str, payload):
            to_id_str = str(to_node_id)
            if to_id_str in self.fail_for_nodes:
                self.failed_sends.append(to_id_str)
                raise RuntimeError(f"Simulated failure for node {to_id_str}")
            self.successful_sends.append(to_id_str)

        async def receive_messages(self):
            while False:
                yield  # Never yield

        async def shutdown(self):
            pass

    failing_context = FailingContext(fail_for_nodes=["1"])

    router = MessageRouter(topology=topology, node_id=0)

    # Broadcast should continue even if node 1 fails
    # Should not raise exception
    await router.route_broadcast(
        message_type="test",
        payload={"data": 42},
        context=failing_context,
    )

    # Verify that node 2 was attempted (and succeeded)
    assert "2" in failing_context.successful_sends

    # Verify that node 1 failed but didn't stop the broadcast
    assert "1" in failing_context.failed_sends

    # Both nodes should have been attempted
    assert len(failing_context.successful_sends) == 1
    assert len(failing_context.failed_sends) == 1


# 2.3 Multicast Routing


@pytest.mark.asyncio
async def test_messagerouter_route_multicast():
    """Verify route_multicast() sends to subset of neighbors."""
    from byzpy.engine.node.context import InProcessContext
    from byzpy.engine.node.router import MessageRouter
    from byzpy.engine.peer_to_peer.topology import Topology

    topology = Topology.complete(4)  # All nodes are neighbors

    contexts = [InProcessContext() for _ in range(4)]
    nodes = [MockDecentralizedNode(node_id=i, context=contexts[i]) for i in range(4)]

    for ctx, node in zip(contexts, nodes):
        await ctx.start(node)

    router = MessageRouter(topology=topology, node_id=0)

    # Multicast to only nodes 1 and 2 (subset of neighbors)
    await router.route_multicast(
        target_node_ids=[1, 2],
        message_type="partial_update",
        payload={"subset": True},
        context=contexts[0],
    )

    # Nodes 1 and 2 receive
    msg1 = await asyncio.wait_for(contexts[1]._inbox.get(), timeout=1.0)
    msg2 = await asyncio.wait_for(contexts[2]._inbox.get(), timeout=1.0)

    assert msg1["type"] == "partial_update"
    assert msg2["type"] == "partial_update"

    # Node 3 should NOT receive (not in multicast list)
    assert contexts[3]._inbox.empty()


@pytest.mark.asyncio
async def test_messagerouter_route_multicast_validates_neighbors():
    """Verify multicast fails if any target is not a neighbor."""
    from byzpy.engine.node.router import MessageRouter
    from byzpy.engine.peer_to_peer.topology import Topology

    topology = Topology.ring(5, k=1)  # Node 0 neighbors: 1, 4
    router = MessageRouter(topology=topology, node_id=0)

    # Try to multicast to 1 (valid) and 2 (not a neighbor)
    with pytest.raises(ValueError, match="not a neighbor"):
        await router.route_multicast(
            target_node_ids=[1, 2],
            message_type="test",
            payload={},
            context=None,
        )


@pytest.mark.asyncio
async def test_messagerouter_route_multicast_empty_list():
    """Verify multicast with empty list does nothing."""
    from byzpy.engine.node.router import MessageRouter
    from byzpy.engine.peer_to_peer.topology import Topology

    topology = Topology.ring(3, k=1)
    router = MessageRouter(topology=topology, node_id=0)

    # Should not raise, just do nothing
    await router.route_multicast(
        target_node_ids=[],
        message_type="test",
        payload={},
        context=None,
    )


# 2.4 Reply Routing


@pytest.mark.asyncio
async def test_messagerouter_route_reply():
    """Verify route_reply() sends back to message sender if valid neighbor."""
    from byzpy.engine.node.context import InProcessContext
    from byzpy.engine.node.router import MessageRouter
    from byzpy.engine.peer_to_peer.topology import Topology

    topology = Topology.ring(3, k=1)  # Bidirectional

    contexts = [InProcessContext() for _ in range(3)]
    nodes = [MockDecentralizedNode(node_id=i, context=contexts[i]) for i in range(3)]

    for ctx, node in zip(contexts, nodes):
        await ctx.start(node)

    # Node 1 receives message from node 0
    original_msg = {"from": 0, "type": "request", "payload": {"id": 123}}

    router = MessageRouter(topology=topology, node_id=1)

    # Node 1 replies to sender (node 0)
    await router.route_reply(
        original_message=original_msg,
        message_type="response",
        payload={"result": "ok"},
        context=contexts[1],
    )

    # Node 0 receives the reply
    reply = await asyncio.wait_for(contexts[0]._inbox.get(), timeout=1.0)

    assert reply["from"] == 1
    assert reply["type"] == "response"
    assert reply["payload"]["result"] == "ok"
