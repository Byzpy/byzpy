"""Tests for RemoteNodeClient - Category 3 from Milestone 5 Test Plan."""

import asyncio

import pytest

from byzpy.engine.graph.pool import ActorPoolConfig
from byzpy.engine.node.application import NodeApplication
from byzpy.engine.node.context import InProcessContext
from byzpy.engine.node.decentralized import DecentralizedNode
from byzpy.engine.node.remote_client import RemoteNodeClient
from byzpy.engine.node.remote_server import RemoteNodeServer

_port_counter = 9100


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
# Category 3.1: RemoteNodeClient Connection
# =============================================================================


@pytest.mark.asyncio
async def test_remotenodeclient_connects():
    """Verify RemoteNodeClient can connect to server."""
    port = get_next_port()
    server = RemoteNodeServer(host="localhost", port=port)
    serve_task = asyncio.create_task(server.serve())
    await asyncio.sleep(0.3)

    client = RemoteNodeClient(host="localhost", port=port)
    await client.connect()

    assert client.is_connected()

    await client.disconnect()
    serve_task.cancel()
    try:
        await serve_task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_remotenodeclient_connection_failure():
    """Verify RemoteNodeClient handles connection failures."""
    client = RemoteNodeClient(host="localhost", port=99999)  # Non-existent

    with pytest.raises((ConnectionError, OSError, asyncio.TimeoutError)):
        await client.connect()


# =============================================================================
# Category 3.2: RemoteNodeClient Message Transport
# =============================================================================


@pytest.mark.asyncio
async def test_remotenodeclient_send_receive(make_app):
    """Verify RemoteNodeClient can send and receive messages."""
    port = get_next_port()
    received = []

    server = RemoteNodeServer(host="localhost", port=port)
    app = make_app("server_app")
    server_node = DecentralizedNode(
        node_id="server_node",
        application=app,
        context=InProcessContext(),
    )

    async def handler(from_id, payload):
        received.append((from_id, payload))

    server_node.register_message_handler("test", handler)
    await server.register_node(server_node)

    serve_task = asyncio.create_task(server.serve())
    await asyncio.sleep(0.3)

    client = RemoteNodeClient(host="localhost", port=port)
    await client.connect()
    await client.register_node("client_node")

    # Send message
    await client.send_message("server_node", "test", {"value": 42}, from_node_id="client_node")
    await asyncio.sleep(0.3)

    assert len(received) == 1

    await client.disconnect()
    serve_task.cancel()
    try:
        await serve_task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_remotenodeclient_handles_disconnection():
    """Verify RemoteNodeClient handles server disconnection gracefully."""
    port = get_next_port()
    server = RemoteNodeServer(host="localhost", port=port)
    serve_task = asyncio.create_task(server.serve())
    await asyncio.sleep(0.3)

    client = RemoteNodeClient(host="localhost", port=port)
    await client.connect()

    # Stop server properly
    server._running = False
    # Close all client connections
    for writer in list(server._client_connections.values()):
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass
    serve_task.cancel()
    try:
        await serve_task
    except asyncio.CancelledError:
        pass
    await asyncio.sleep(0.4)  # Give time for connections to close

    # Client should detect disconnection
    # The receive loop should detect it when trying to read
    # Wait for receive loop to detect
    for _ in range(20):  # Check multiple times with short intervals
        if not client.is_connected():
            break
        await asyncio.sleep(0.05)

    assert not client.is_connected()

    await client.disconnect()
