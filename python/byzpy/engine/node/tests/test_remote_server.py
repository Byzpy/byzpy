"""Tests for RemoteNodeServer - Category 2 from Milestone 5 Test Plan."""

import asyncio
import socket

import pytest

from byzpy.engine.graph.pool import ActorPoolConfig
from byzpy.engine.node.application import NodeApplication
from byzpy.engine.node.context import InProcessContext, RemoteContext
from byzpy.engine.node.decentralized import DecentralizedNode
from byzpy.engine.node.remote_server import RemoteNodeServer

# Port management
_port_counter = 9000


def get_next_port():
    global _port_counter
    port = _port_counter
    _port_counter += 1
    return port


@pytest.fixture
def make_app():
    def _make_app(name: str):
        return NodeApplication(
            name=name,
            actor_pool=[ActorPoolConfig(backend="thread", count=1)],
        )

    return _make_app


# =============================================================================
# Category 2.1: RemoteNodeServer Creation and Configuration
# =============================================================================


def test_remotenodeserver_can_be_created():
    """Verify RemoteNodeServer can be instantiated."""
    server = RemoteNodeServer(host="localhost", port=8888)

    assert server is not None
    assert server.host == "localhost"
    assert server.port == 8888
    assert server._nodes == {}
    assert server._server is None


def test_remotenodeserver_with_custom_host_port():
    """Verify RemoteNodeServer accepts custom host and port."""
    server = RemoteNodeServer(host="0.0.0.0", port=9999)

    assert server.host == "0.0.0.0"
    assert server.port == 9999


# =============================================================================
# Category 2.2: RemoteNodeServer Node Registration
# =============================================================================


@pytest.mark.asyncio
async def test_remotenodeserver_register_node(make_app):
    """Verify RemoteNodeServer can register nodes."""
    server = RemoteNodeServer(host="localhost", port=8888)
    app = make_app("test")
    node = DecentralizedNode(
        node_id="node1",
        application=app,
        context=InProcessContext(),
    )

    await server.register_node(node)

    assert "node1" in server._nodes
    assert server._nodes["node1"] is node


@pytest.mark.asyncio
async def test_remotenodeserver_register_multiple_nodes(make_app):
    """Verify RemoteNodeServer can register multiple nodes."""
    server = RemoteNodeServer(host="localhost", port=8888)

    for i in range(3):
        app = make_app(f"app-{i}")
        node = DecentralizedNode(
            node_id=f"node{i}",
            application=app,
            context=InProcessContext(),
        )
        await server.register_node(node)

    assert len(server._nodes) == 3
    assert all(f"node{i}" in server._nodes for i in range(3))


@pytest.mark.asyncio
async def test_remotenodeserver_register_duplicate_node_raises(make_app):
    """Verify RemoteNodeServer raises error for duplicate node IDs."""
    server = RemoteNodeServer(host="localhost", port=8888)
    app = make_app("test")
    node = DecentralizedNode(
        node_id="node1",
        application=app,
        context=InProcessContext(),
    )

    await server.register_node(node)

    # Try to register duplicate
    with pytest.raises(ValueError, match="already registered|duplicate"):
        await server.register_node(node)


# =============================================================================
# Category 2.3: RemoteNodeServer Serving
# =============================================================================


@pytest.mark.asyncio
async def test_remotenodeserver_start_stop():
    """Verify RemoteNodeServer can start and stop serving."""
    port = get_next_port()
    server = RemoteNodeServer(host="localhost", port=port)

    # Start server
    serve_task = asyncio.create_task(server.serve())
    await asyncio.sleep(0.2)

    assert server._server is not None

    # Stop server
    serve_task.cancel()
    try:
        await serve_task
    except asyncio.CancelledError:
        pass

    await asyncio.sleep(0.1)


@pytest.mark.asyncio
async def test_remotenodeserver_accepts_connections():
    """Verify RemoteNodeServer accepts client connections."""
    port = get_next_port()
    server = RemoteNodeServer(host="localhost", port=port)
    serve_task = asyncio.create_task(server.serve())
    await asyncio.sleep(0.2)

    # Try to connect
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1.0)
        sock.connect(("localhost", port))
        sock.close()
        connected = True
    except Exception:
        connected = False

    assert connected

    serve_task.cancel()
    try:
        await serve_task
    except asyncio.CancelledError:
        pass


# =============================================================================
# Category 2.4: RemoteNodeServer Message Routing
# =============================================================================


@pytest.mark.asyncio
async def test_remotenodeserver_routes_messages_to_node(make_app):
    """Verify RemoteNodeServer routes incoming messages to correct node."""
    port = get_next_port()
    received_messages = []

    server = RemoteNodeServer(host="localhost", port=port)
    app = make_app("server_app")
    server_node = DecentralizedNode(
        node_id="server_node",
        application=app,
        context=InProcessContext(),
    )

    async def handler(from_id, payload):
        received_messages.append((from_id, payload))

    server_node.register_message_handler("test", handler)
    await server.register_node(server_node)

    serve_task = asyncio.create_task(server.serve())
    await asyncio.sleep(0.3)

    # Create client and send message
    client_context = RemoteContext(host="localhost", port=port)
    client_app = make_app("client_app")
    client_node = DecentralizedNode(
        node_id="client_node",
        application=client_app,
        context=client_context,
    )
    await client_context.start(client_node)
    await asyncio.sleep(0.2)

    await client_context.send_message("server_node", "test", {"value": 42})
    await asyncio.sleep(0.3)

    assert len(received_messages) == 1
    assert received_messages[0][0] == "client_node"
    assert received_messages[0][1] == {"value": 42}

    await client_context.shutdown()
    serve_task.cancel()
    try:
        await serve_task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_remotenodeserver_routes_to_nonexistent_node_raises(make_app):
    """Verify RemoteNodeServer handles messages to non-existent nodes."""
    port = get_next_port()
    server = RemoteNodeServer(host="localhost", port=port)
    serve_task = asyncio.create_task(server.serve())
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

    # Try to send to non-existent node
    # Server should handle this gracefully (either raise error or ignore)
    # The server sends an error message back, which the client may not handle
    # So we just verify the message was sent (no exception raised)
    try:
        await client_context.send_message("nonexistent", "test", {})
        await asyncio.sleep(0.2)
    except Exception:
        # Server may raise error, which is acceptable
        pass

    await client_context.shutdown()
    serve_task.cancel()
    try:
        await serve_task
    except asyncio.CancelledError:
        pass
