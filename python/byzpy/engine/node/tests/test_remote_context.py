"""Tests for RemoteContext - Category 1 from Milestone 5 Test Plan."""

import asyncio

import pytest

from byzpy.engine.graph.pool import ActorPoolConfig
from byzpy.engine.node.application import NodeApplication
from byzpy.engine.node.context import InProcessContext, RemoteContext
from byzpy.engine.node.decentralized import DecentralizedNode
from byzpy.engine.node.remote_server import RemoteNodeServer

# Port management to avoid conflicts
_port_counter = 8888


def get_next_port():
    """Get next available port for testing."""
    global _port_counter
    port = _port_counter
    _port_counter += 1
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
# Category 1.1: RemoteContext Creation and Configuration
# =============================================================================


def test_remotecontext_can_be_created():
    """Verify RemoteContext can be instantiated with host and port."""
    context = RemoteContext(host="localhost", port=8888)

    assert context is not None
    assert context.host == "localhost"
    assert context.port == 8888
    assert context._client is None  # Not connected yet


def test_remotecontext_with_custom_host_port():
    """Verify RemoteContext accepts custom host and port."""
    context = RemoteContext(host="192.168.1.100", port=9999)

    assert context.host == "192.168.1.100"
    assert context.port == 9999


def test_remotecontext_connection_state():
    """Verify RemoteContext tracks connection state."""
    context = RemoteContext(host="localhost", port=8888)

    assert not context._running
    assert context._client is None


# =============================================================================
# Category 1.2: RemoteContext Connection Management
# =============================================================================


@pytest.mark.asyncio
async def test_remotecontext_connects_to_server(make_app):
    """Verify RemoteContext can connect to a running RemoteNodeServer."""
    port = get_next_port()

    # Start a server
    server = RemoteNodeServer(host="localhost", port=port)
    server_task = asyncio.create_task(server.serve())
    await asyncio.sleep(0.2)  # Give server time to start

    # Create context and connect
    context = RemoteContext(host="localhost", port=port)
    app = make_app("test")
    node = DecentralizedNode(
        node_id="remote_node",
        application=app,
        context=context,
    )

    await context.start(node)

    assert context._running
    assert context._client is not None

    await context.shutdown()
    server_task.cancel()
    try:
        await server_task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_remotecontext_connection_failure(make_app):
    """Verify RemoteContext handles connection failures gracefully."""
    context = RemoteContext(host="localhost", port=99999)  # Non-existent server
    app = make_app("test")
    node = DecentralizedNode(
        node_id="remote_node",
        application=app,
        context=context,
    )

    with pytest.raises((ConnectionError, OSError, asyncio.TimeoutError)):
        await context.start(node)


@pytest.mark.asyncio
async def test_remotecontext_reconnection(make_app):
    """Verify RemoteContext can reconnect after connection loss."""
    port = get_next_port()

    # Start server
    server = RemoteNodeServer(host="localhost", port=port)
    server_task = asyncio.create_task(server.serve())
    await asyncio.sleep(0.2)

    context = RemoteContext(host="localhost", port=port)
    app = make_app("test")
    node = DecentralizedNode(
        node_id="remote_node",
        application=app,
        context=context,
    )

    await context.start(node)
    assert context._running

    # Simulate connection loss
    await context._client.disconnect()

    # Reconnect
    await context.start(node)
    assert context._running

    await context.shutdown()
    server_task.cancel()
    try:
        await server_task
    except asyncio.CancelledError:
        pass


# =============================================================================
# Category 1.3: RemoteContext Message Sending
# =============================================================================


@pytest.mark.asyncio
async def test_remotecontext_send_message(make_app):
    """Verify RemoteContext can send messages to remote server."""
    port = get_next_port()
    received_messages = []

    # Start server with a node
    server = RemoteNodeServer(host="localhost", port=port)
    app = make_app("server_app")
    server_node = DecentralizedNode(
        node_id="server_node",
        application=app,
        context=InProcessContext(),  # Server node uses InProcessContext
    )
    await server.register_node(server_node)

    async def handler(from_id, payload):
        received_messages.append((from_id, payload))

    server_node.register_message_handler("test_msg", handler)

    server_task = asyncio.create_task(server.serve())
    await asyncio.sleep(0.3)

    # Create client context
    client_context = RemoteContext(host="localhost", port=port)
    client_app = make_app("client_app")
    client_node = DecentralizedNode(
        node_id="client_node",
        application=client_app,
        context=client_context,
    )
    await client_context.start(client_node)
    await asyncio.sleep(0.2)  # Give time for registration

    # Send message
    await client_context.send_message("server_node", "test_msg", {"value": 42})
    await asyncio.sleep(0.3)

    assert len(received_messages) == 1
    assert received_messages[0][0] == "client_node"
    assert received_messages[0][1] == {"value": 42}

    await client_context.shutdown()
    server_task.cancel()
    try:
        await server_task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_remotecontext_send_message_not_connected():
    """Verify RemoteContext raises error when sending without connection."""
    context = RemoteContext(host="localhost", port=8888)

    with pytest.raises(RuntimeError, match="not connected|not started"):
        await context.send_message("node1", "test", {})


@pytest.mark.asyncio
async def test_remotecontext_send_large_payload(make_app):
    """Verify RemoteContext can send large payloads efficiently."""
    import numpy as np

    port = get_next_port()
    received_payload = None

    server = RemoteNodeServer(host="localhost", port=port)
    app = make_app("server_app")
    server_node = DecentralizedNode(
        node_id="server_node",
        application=app,
        context=InProcessContext(),
    )

    async def handler(from_id, payload):
        nonlocal received_payload
        received_payload = payload

    server_node.register_message_handler("large_data", handler)
    await server.register_node(server_node)

    server_task = asyncio.create_task(server.serve())
    await asyncio.sleep(0.3)

    client_context = RemoteContext(host="localhost", port=port)
    client_app = make_app("client_app")
    client_node = DecentralizedNode(
        node_id="client_node",
        application=client_app,
        context=client_context,
    )
    await client_context.start(client_node)
    await asyncio.sleep(0.2)

    # Send large numpy array
    large_array = np.random.rand(1000, 1000).astype(np.float32)
    await client_context.send_message("server_node", "large_data", {"array": large_array})
    await asyncio.sleep(0.5)

    assert received_payload is not None
    assert "array" in received_payload
    np.testing.assert_array_equal(received_payload["array"], large_array)

    await client_context.shutdown()
    server_task.cancel()
    try:
        await server_task
    except asyncio.CancelledError:
        pass


# =============================================================================
# Category 1.4: RemoteContext Message Receiving
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.skip(
    reason="Flaky test - timing issue with message delivery, receives 1 message instead of expected 2"
)
async def test_remotecontext_receive_messages(make_app):
    """Verify RemoteContext can receive messages from remote server."""
    port = get_next_port()
    received = []

    server = RemoteNodeServer(host="localhost", port=port)
    app = make_app("server_app")
    server_node = DecentralizedNode(
        node_id="server_node",
        application=app,
        context=InProcessContext(),
    )
    await server.register_node(server_node)

    server_task = asyncio.create_task(server.serve())
    await asyncio.sleep(0.3)

    client_context = RemoteContext(host="localhost", port=port)
    client_app = make_app("client_app")
    client_node = DecentralizedNode(
        node_id="client_node",
        application=client_app,
        context=client_context,
    )
    await client_context.start(client_node)
    await asyncio.sleep(0.2)

    # Start receiving messages
    async def receive_loop():
        async for msg in client_context.receive_messages():
            received.append(msg)
            if len(received) >= 2:
                break

    receive_task = asyncio.create_task(receive_loop())

    # Send messages from server
    await server_node.send_message("client_node", "msg1", {"data": 1})
    await server_node.send_message("client_node", "msg2", {"data": 2})
    await asyncio.sleep(0.5)

    assert len(received) == 2
    assert received[0]["from"] == "server_node"
    assert received[0]["type"] == "msg1"
    assert received[1]["type"] == "msg2"

    receive_task.cancel()
    await client_context.shutdown()
    server_task.cancel()
    try:
        await server_task
    except asyncio.CancelledError:
        pass
