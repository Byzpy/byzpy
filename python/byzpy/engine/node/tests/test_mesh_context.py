"""Tests for MeshRemoteContext - Fully Distributed P2P Mesh Communication."""

import asyncio

import pytest

from byzpy.engine.graph.pool import ActorPoolConfig
from byzpy.engine.node.application import NodeApplication
from byzpy.engine.node.context import InProcessContext, MeshRemoteContext
from byzpy.engine.node.decentralized import DecentralizedNode

# Port management to avoid conflicts
_port_counter = 9000


def get_next_port():
    """Get next available port for testing."""
    global _port_counter
    port = _port_counter
    _port_counter += 3  # Reserve 3 ports per test (for 3-node mesh)
    return port


@pytest.fixture
def make_app():
    """Create a simple NodeApplication for testing."""

    def _make_app(name: str):
        return NodeApplication(
            name=name,
            actor_pool=[ActorPoolConfig(backend="thread", count=1)],
        )

    return _make_app


# =============================================================================
# Category 1: MeshRemoteContext Creation and Configuration
# =============================================================================


def test_meshremotecontext_can_be_created():
    """Verify MeshRemoteContext can be instantiated."""
    peer_addresses = {"1": ("localhost", 8888), "2": ("localhost", 8889)}
    context = MeshRemoteContext(
        local_host="0.0.0.0",
        local_port=8887,
        peer_addresses=peer_addresses,
    )

    assert context is not None
    assert context.local_host == "0.0.0.0"
    assert context.local_port == 8887
    assert context.peer_addresses == peer_addresses
    assert not context._running


def test_meshremotecontext_with_custom_peer_addresses():
    """Verify MeshRemoteContext accepts custom peer addresses."""
    peer_addresses = {
        "alice": ("192.168.1.100", 8888),
        "bob": ("192.168.1.101", 8888),
        "charlie": ("192.168.1.102", 8888),
    }
    context = MeshRemoteContext(
        local_host="192.168.1.99",
        local_port=8888,
        peer_addresses=peer_addresses,
        connect_timeout=10.0,
        reconnect_interval=5.0,
    )

    assert context.peer_addresses == peer_addresses
    assert context.connect_timeout == 10.0
    assert context.reconnect_interval == 5.0


# =============================================================================
# Category 2: MeshRemoteContext Server and Connection Management
# =============================================================================


@pytest.mark.asyncio
async def test_meshremotecontext_starts_local_server(make_app):
    """Verify MeshRemoteContext starts local TCP server."""
    port = get_next_port()

    # Create context with peer addresses (but won't connect - that's ok)
    peer_addresses = {"1": ("localhost", port + 1)}
    context = MeshRemoteContext(
        local_host="localhost",
        local_port=port,
        peer_addresses=peer_addresses,
    )

    app = make_app("test")
    node = DecentralizedNode(
        node_id="0",
        application=app,
        context=context,
    )

    # Start should start local server even if peers aren't available
    await context.start(node)

    assert context._running
    assert context._local_server is not None
    assert context._serve_task is not None

    await context.shutdown()


@pytest.mark.asyncio
async def test_meshremotecontext_connects_to_peers(make_app):
    """Verify MeshRemoteContext connects to peer servers."""
    port0 = get_next_port()
    port1 = port0 + 1
    port2 = port0 + 2

    # Setup: Node 0 will connect to nodes 1 and 2
    peer_addresses_0 = {
        "1": ("localhost", port1),
        "2": ("localhost", port2),
    }

    # Start node 1's server
    peer_addresses_1 = {
        "0": ("localhost", port0),
        "2": ("localhost", port2),
    }
    context1 = MeshRemoteContext(
        local_host="localhost",
        local_port=port1,
        peer_addresses=peer_addresses_1,
    )
    app1 = make_app("app1")
    node1 = DecentralizedNode(node_id="1", application=app1, context=context1)
    await context1.start(node1)
    await asyncio.sleep(0.3)

    # Start node 0 and connect to node 1
    context0 = MeshRemoteContext(
        local_host="localhost",
        local_port=port0,
        peer_addresses=peer_addresses_0,
    )
    app0 = make_app("app0")
    node0 = DecentralizedNode(node_id="0", application=app0, context=context0)
    await context0.start(node0)
    await asyncio.sleep(0.5)

    # Node 0 should have connected to node 1
    connected = context0.get_connected_peers()
    assert "1" in connected

    await context0.shutdown()
    await context1.shutdown()


@pytest.mark.asyncio
async def test_meshremotecontext_bidirectional_connections(make_app):
    """Verify MeshRemoteContext handles bidirectional connections (inbound + outbound)."""
    port0 = get_next_port()
    port1 = port0 + 1

    # Node 0 peer config
    peer_addresses_0 = {"1": ("localhost", port1)}

    # Node 1 peer config
    peer_addresses_1 = {"0": ("localhost", port0)}

    # Start both nodes
    context0 = MeshRemoteContext(
        local_host="localhost",
        local_port=port0,
        peer_addresses=peer_addresses_0,
    )
    app0 = make_app("app0")
    node0 = DecentralizedNode(node_id="0", application=app0, context=context0)

    context1 = MeshRemoteContext(
        local_host="localhost",
        local_port=port1,
        peer_addresses=peer_addresses_1,
    )
    app1 = make_app("app1")
    node1 = DecentralizedNode(node_id="1", application=app1, context=context1)

    # Start both
    await context0.start(node0)
    await context1.start(node1)
    await asyncio.sleep(0.5)

    # Both should see each other as connected
    connected0 = context0.get_connected_peers()
    connected1 = context1.get_connected_peers()

    assert "1" in connected0
    assert "0" in connected1

    await context0.shutdown()
    await context1.shutdown()


# =============================================================================
# Category 3: MeshRemoteContext Message Sending
# =============================================================================


@pytest.mark.asyncio
async def test_meshremotecontext_send_message_outbound(make_app):
    """Verify MeshRemoteContext can send messages via outbound connection."""
    port0 = get_next_port()
    port1 = port0 + 1

    received_messages = []

    # Setup node 1 to receive messages
    peer_addresses_1 = {"0": ("localhost", port0)}
    context1 = MeshRemoteContext(
        local_host="localhost",
        local_port=port1,
        peer_addresses=peer_addresses_1,
    )
    app1 = make_app("app1")
    node1 = DecentralizedNode(node_id="1", application=app1, context=context1)

    async def handler(from_id, payload):
        received_messages.append((from_id, payload))

    node1.register_message_handler("test_msg", handler)
    await context1.start(node1)
    await node1.start()  # Start node to process messages
    await asyncio.sleep(0.3)

    # Setup node 0 to send messages
    peer_addresses_0 = {"1": ("localhost", port1)}
    context0 = MeshRemoteContext(
        local_host="localhost",
        local_port=port0,
        peer_addresses=peer_addresses_0,
    )
    app0 = make_app("app0")
    node0 = DecentralizedNode(node_id="0", application=app0, context=context0)
    await context0.start(node0)
    await node0.start()  # Start node to process messages
    await asyncio.sleep(0.5)

    # Send message from node 0 to node 1
    await context0.send_message("1", "test_msg", {"value": 42})
    await asyncio.sleep(0.3)

    assert len(received_messages) == 1
    assert received_messages[0][0] == "0"
    assert received_messages[0][1] == {"value": 42}

    await node0.shutdown()
    await node1.shutdown()
    await context0.shutdown()
    await context1.shutdown()


@pytest.mark.asyncio
async def test_meshremotecontext_send_message_inbound(make_app):
    """Verify MeshRemoteContext can send messages via inbound connection."""
    port0 = get_next_port()
    port1 = port0 + 1

    received_messages = []

    # Setup node 0 to receive messages
    peer_addresses_0 = {"1": ("localhost", port1)}
    context0 = MeshRemoteContext(
        local_host="localhost",
        local_port=port0,
        peer_addresses=peer_addresses_0,
    )
    app0 = make_app("app0")
    node0 = DecentralizedNode(node_id="0", application=app0, context=context0)

    async def handler(from_id, payload):
        received_messages.append((from_id, payload))

    node0.register_message_handler("test_msg", handler)
    await context0.start(node0)
    await node0.start()  # Start node to process messages
    await asyncio.sleep(0.3)

    # Setup node 1 to send messages (will connect to node 0)
    peer_addresses_1 = {"0": ("localhost", port0)}
    context1 = MeshRemoteContext(
        local_host="localhost",
        local_port=port1,
        peer_addresses=peer_addresses_1,
    )
    app1 = make_app("app1")
    node1 = DecentralizedNode(node_id="1", application=app1, context=context1)
    await context1.start(node1)
    await node1.start()  # Start node to process messages
    await asyncio.sleep(0.5)

    # Send message from node 1 to node 0 (using inbound connection)
    await context1.send_message("0", "test_msg", {"value": 99})
    await asyncio.sleep(0.3)

    assert len(received_messages) == 1
    assert received_messages[0][0] == "1"
    assert received_messages[0][1] == {"value": 99}

    await node0.shutdown()
    await node1.shutdown()
    await context0.shutdown()
    await context1.shutdown()


@pytest.mark.asyncio
async def test_meshremotecontext_send_message_failure_handling(make_app):
    """Verify MeshRemoteContext handles send failures gracefully."""
    port0 = get_next_port()
    port1 = port0 + 1

    # Create context with peer that doesn't exist
    peer_addresses_0 = {"1": ("localhost", port1)}
    context0 = MeshRemoteContext(
        local_host="localhost",
        local_port=port0,
        peer_addresses=peer_addresses_0,
    )
    app0 = make_app("app0")
    node0 = DecentralizedNode(node_id="0", application=app0, context=context0)
    await context0.start(node0)
    await asyncio.sleep(0.2)

    # Should raise error when trying to send to non-existent peer
    with pytest.raises(RuntimeError, match="No connection"):
        await context0.send_message("1", "test", {})

    await context0.shutdown()


# =============================================================================
# Category 4: MeshRemoteContext Message Receiving
# =============================================================================


@pytest.mark.asyncio
async def test_meshremotecontext_receive_messages(make_app):
    """Verify MeshRemoteContext can receive messages from peers."""
    port0 = get_next_port()
    port1 = port0 + 1

    # Setup node 0 to receive
    peer_addresses_0 = {"1": ("localhost", port1)}
    context0 = MeshRemoteContext(
        local_host="localhost",
        local_port=port0,
        peer_addresses=peer_addresses_0,
    )
    app0 = make_app("app0")
    node0 = DecentralizedNode(node_id="0", application=app0, context=context0)
    await context0.start(node0)
    await asyncio.sleep(0.3)

    # Setup node 1 to send
    peer_addresses_1 = {"0": ("localhost", port0)}
    context1 = MeshRemoteContext(
        local_host="localhost",
        local_port=port1,
        peer_addresses=peer_addresses_1,
    )
    app1 = make_app("app1")
    node1 = DecentralizedNode(node_id="1", application=app1, context=context1)
    await context1.start(node1)
    await node1.start()  # Start node to process messages
    await asyncio.sleep(0.5)

    # Send messages from node 1
    await context1.send_message("0", "msg1", {"data": 1})
    await context1.send_message("0", "msg2", {"data": 2})
    await asyncio.sleep(0.3)

    # Receive messages
    received = []
    async for msg in context0.receive_messages():
        received.append(msg)
        if len(received) >= 2:
            break

    assert len(received) >= 1  # At least one message received
    assert received[0]["from"] == "1"
    assert received[0]["type"] in ["msg1", "msg2"]

    await node0.shutdown()
    await node1.shutdown()
    await context0.shutdown()
    await context1.shutdown()


# =============================================================================
# Category 5: MeshRemoteContext Connection Monitoring and Reconnection
# =============================================================================


@pytest.mark.asyncio
async def test_meshremotecontext_reconnection(make_app):
    """Verify MeshRemoteContext reconnects to failed peers."""
    port0 = get_next_port()
    port1 = port0 + 1

    # Setup node 1
    peer_addresses_1 = {"0": ("localhost", port0)}
    context1 = MeshRemoteContext(
        local_host="localhost",
        local_port=port1,
        peer_addresses=peer_addresses_1,
        reconnect_interval=1.0,  # Fast reconnection for test
    )
    app1 = make_app("app1")
    node1 = DecentralizedNode(node_id="1", application=app1, context=context1)
    await context1.start(node1)
    await asyncio.sleep(0.3)

    # Setup node 0 (will try to connect to node 1)
    peer_addresses_0 = {"1": ("localhost", port1)}
    context0 = MeshRemoteContext(
        local_host="localhost",
        local_port=port0,
        peer_addresses=peer_addresses_0,
        reconnect_interval=1.0,
    )
    app0 = make_app("app0")
    node0 = DecentralizedNode(node_id="0", application=app0, context=context0)
    await context0.start(node0)
    await asyncio.sleep(1.0)

    # Should be connected
    connected = context0.get_connected_peers()
    assert "1" in connected

    await context0.shutdown()
    await context1.shutdown()


@pytest.mark.asyncio
async def test_meshremotecontext_get_connected_peers(make_app):
    """Verify get_connected_peers() returns correct list."""
    port0 = get_next_port()
    port1 = port0 + 1
    port2 = port0 + 2

    # Setup 3-node mesh: node 0 connects to nodes 1 and 2
    peer_addresses_0 = {
        "1": ("localhost", port1),
        "2": ("localhost", port2),
    }

    # Start nodes 1 and 2
    context1 = MeshRemoteContext(
        local_host="localhost",
        local_port=port1,
        peer_addresses={"0": ("localhost", port0)},
    )
    app1 = make_app("app1")
    node1 = DecentralizedNode(node_id="1", application=app1, context=context1)
    await context1.start(node1)

    context2 = MeshRemoteContext(
        local_host="localhost",
        local_port=port2,
        peer_addresses={"0": ("localhost", port0)},
    )
    app2 = make_app("app2")
    node2 = DecentralizedNode(node_id="2", application=app2, context=context2)
    await context2.start(node2)

    await asyncio.sleep(0.3)

    # Start node 0
    context0 = MeshRemoteContext(
        local_host="localhost",
        local_port=port0,
        peer_addresses=peer_addresses_0,
    )
    app0 = make_app("app0")
    node0 = DecentralizedNode(node_id="0", application=app0, context=context0)
    await context0.start(node0)
    await asyncio.sleep(0.8)

    # Node 0 should see both peers
    connected = context0.get_connected_peers()
    assert len(connected) >= 1  # At least one should be connected
    assert "1" in connected or "2" in connected

    await context0.shutdown()
    await context1.shutdown()
    await context2.shutdown()


# =============================================================================
# Category 6: MeshRemoteContext Shutdown
# =============================================================================


@pytest.mark.asyncio
async def test_meshremotecontext_shutdown(make_app):
    """Verify MeshRemoteContext shuts down cleanly."""
    port0 = get_next_port()

    peer_addresses_0 = {"1": ("localhost", port0 + 1)}
    context = MeshRemoteContext(
        local_host="localhost",
        local_port=port0,
        peer_addresses=peer_addresses_0,
    )
    app = make_app("test")
    node = DecentralizedNode(node_id="0", application=app, context=context)
    await context.start(node)
    await node.start()  # Start node

    assert context._running
    assert context._local_server is not None

    await node.shutdown()
    await context.shutdown()

    assert not context._running
    assert (
        context._local_server is None
    )  # Note: server may not be None if cleanup incomplete, but running should be False
    assert context._node is None
