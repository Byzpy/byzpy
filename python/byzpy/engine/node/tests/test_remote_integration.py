"""Tests for Remote Node Integration - Categories 5-7 from Milestone 5 Test Plan."""

import asyncio

import pytest
import torch

from byzpy.engine.graph.pool import ActorPoolConfig
from byzpy.engine.node.application import NodeApplication
from byzpy.engine.node.cluster import DecentralizedCluster
from byzpy.engine.node.context import InProcessContext, ProcessContext, RemoteContext
from byzpy.engine.node.decentralized import DecentralizedNode
from byzpy.engine.node.remote_server import RemoteNodeServer
from byzpy.engine.peer_to_peer.topology import Topology

_port_counter = 9200


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
# Category 5: DecentralizedNode with RemoteContext
# =============================================================================


@pytest.mark.asyncio
async def test_decentralizednode_with_remotecontext(make_app):
    """Verify DecentralizedNode works with RemoteContext."""
    port = get_next_port()

    # Start server
    server = RemoteNodeServer(host="localhost", port=port)
    serve_task = asyncio.create_task(server.serve())
    await asyncio.sleep(0.3)

    # Create remote node
    context = RemoteContext(host="localhost", port=port)
    app = make_app("remote_app")
    node = DecentralizedNode(
        node_id="remote_node",
        application=app,
        context=context,
    )

    await node.start()
    assert node._running

    await node.shutdown()
    serve_task.cancel()
    try:
        await serve_task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_decentralizednode_remote_send_message(make_app):
    """Verify DecentralizedNode can send messages via RemoteContext."""
    port = get_next_port()
    received = []

    # Start server with a node
    server = RemoteNodeServer(host="localhost", port=port)
    server_app = make_app("server_app")
    server_node = DecentralizedNode(
        node_id="server_node",
        application=server_app,
        context=InProcessContext(),
    )

    async def handler(from_id, payload):
        received.append((from_id, payload))

    server_node.register_message_handler("test", handler)
    await server.register_node(server_node)

    serve_task = asyncio.create_task(server.serve())
    await asyncio.sleep(0.3)

    # Create remote client node
    client_context = RemoteContext(host="localhost", port=port)
    client_app = make_app("client_app")
    client_node = DecentralizedNode(
        node_id="client_node",
        application=client_app,
        context=client_context,
    )

    await client_node.start()
    await asyncio.sleep(0.2)  # Give time for registration
    await client_node.send_message("server_node", "test", {"value": 42})
    await asyncio.sleep(0.3)

    assert len(received) == 1
    assert received[0][0] == "client_node"

    await client_node.shutdown()
    serve_task.cancel()
    try:
        await serve_task
    except asyncio.CancelledError:
        pass


# =============================================================================
# Category 6: DecentralizedCluster with Remote Nodes
# =============================================================================


@pytest.mark.asyncio
async def test_decentralizedcluster_mixed_contexts(make_app):
    """Verify DecentralizedCluster can manage nodes with different contexts."""
    port = get_next_port()

    # Start remote server
    server = RemoteNodeServer(host="localhost", port=port)
    serve_task = asyncio.create_task(server.serve())
    await asyncio.sleep(0.3)

    cluster = DecentralizedCluster()

    # Add local process node
    local_app = make_app("local")
    await cluster.add_node("local_node", local_app, context=ProcessContext())

    # Add remote node
    remote_app = make_app("remote")
    remote_context = RemoteContext(host="localhost", port=port)
    await cluster.add_node("remote_node", remote_app, context=remote_context)

    await cluster.start_all()
    await asyncio.sleep(0.3)  # Give time for connections

    # Verify both nodes are running
    assert cluster.get_node("local_node")._running
    assert cluster.get_node("remote_node")._running

    await cluster.shutdown_all()
    serve_task.cancel()
    try:
        await serve_task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_decentralizedcluster_remote_node_communication(make_app):
    """Verify nodes in cluster can communicate across remote contexts."""
    port = get_next_port()
    received = []

    # Start remote server
    server = RemoteNodeServer(host="localhost", port=port)
    serve_task = asyncio.create_task(server.serve())
    await asyncio.sleep(0.3)

    cluster = DecentralizedCluster()

    # Add local node - register with server so it can receive from remote nodes
    # Note: Cross-context communication (ProcessContext <-> RemoteContext) requires
    # the local node to be accessible via the server
    from byzpy.engine.node.context import InProcessContext

    local_app = make_app("local")
    local_node = DecentralizedNode(
        node_id="local_node",
        application=local_app,
        context=InProcessContext(),
    )
    await server.register_node(local_node)
    # Manually add to cluster (bypassing add_node to avoid ProcessContext creation)
    cluster.nodes["local_node"] = local_node
    cluster._node_id_map[0] = "local_node"

    async def handler(from_id, payload):
        received.append((from_id, payload))

    local_node.register_message_handler("test", handler)

    # Add remote node
    remote_app = make_app("remote")
    remote_context = RemoteContext(host="localhost", port=port)
    remote_node = await cluster.add_node("remote_node", remote_app, context=remote_context)
    cluster._node_id_map[1] = "remote_node"

    # Start local node manually (it's already registered with server)
    await local_node.start()

    await cluster.start_all()
    await asyncio.sleep(0.6)  # Give time for connections

    # Send from remote to local
    await remote_node.send_message("local_node", "test", {"value": 42})
    await asyncio.sleep(0.5)

    assert len(received) == 1
    assert received[0][0] == "remote_node"

    await cluster.shutdown_all()
    serve_task.cancel()
    try:
        await serve_task
    except asyncio.CancelledError:
        pass


# =============================================================================
# Category 7: Integration Tests - Distributed P2P Training
# =============================================================================


@pytest.mark.asyncio
async def test_distributed_p2p_training_remote_nodes(make_app):
    """End-to-end test: P2P training with nodes on remote servers."""
    port = get_next_port()

    # Start remote server (both nodes connect to same server for communication)
    # Note: For nodes on different servers to communicate, we'd need server-to-server
    # communication, which is beyond current scope
    server = RemoteNodeServer(host="localhost", port=port)
    serve_task = asyncio.create_task(server.serve())
    await asyncio.sleep(0.4)

    topology = Topology.ring(2, k=1)
    cluster = DecentralizedCluster()

    gradients_received = {"node0": [], "node1": []}

    # Add remote nodes (both connect to same server)
    for i in range(2):
        app = make_app(f"app-{i}")
        context = RemoteContext(host="localhost", port=port)
        node = await cluster.add_node(f"node{i}", app, topology=topology, context=context)
        cluster._node_id_map[i] = f"node{i}"

        def make_handler(nid):
            async def on_gradient(from_id, payload):
                gradients_received[nid].append(torch.tensor(payload["gradient"]))

            return on_gradient

        node.register_message_handler("gradient", make_handler(f"node{i}"))

    # Update node_id_maps before starting
    cluster._update_node_id_maps()
    await cluster.start_all()
    await asyncio.sleep(0.6)  # Give time for connections

    # Each node broadcasts gradient
    for node_id, node in cluster.nodes.items():
        local_grad = torch.randn(5)
        await node.broadcast_message("gradient", {"gradient": local_grad.tolist()})

    await asyncio.sleep(0.6)

    # Verify each node received from neighbor
    for node_id in gradients_received:
        assert len(gradients_received[node_id]) == 1  # Ring k=1 has 1 neighbor

    await cluster.shutdown_all()
    serve_task.cancel()
    try:
        await serve_task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_distributed_p2p_training_mixed_local_remote(make_app):
    """End-to-end test: P2P training with mix of local and remote nodes."""
    port = get_next_port()

    # Start remote server
    server = RemoteNodeServer(host="localhost", port=port)
    serve_task = asyncio.create_task(server.serve())
    await asyncio.sleep(0.3)

    topology = Topology.complete(2)
    cluster = DecentralizedCluster()

    gradients_received = {"local": [], "remote": []}

    # Add local node - register with server so it can communicate with remote nodes
    from byzpy.engine.node.context import InProcessContext

    local_app = make_app("local")
    local_node = DecentralizedNode(
        node_id="local",
        application=local_app,
        context=InProcessContext(),
        topology=topology,
    )
    await server.register_node(local_node)
    cluster.nodes["local"] = local_node
    cluster._node_id_map[0] = "local"

    def make_handler(nid):
        async def on_gradient(from_id, payload):
            gradients_received[nid].append(torch.tensor(payload["gradient"]))

        return on_gradient

    local_node.register_message_handler("gradient", make_handler("local"))
    await local_node.start()

    # Add remote node
    remote_app = make_app("remote")
    remote_context = RemoteContext(host="localhost", port=port)
    remote_node = await cluster.add_node(
        "remote", remote_app, topology=topology, context=remote_context
    )
    cluster._node_id_map[1] = "remote"
    remote_node.register_message_handler("gradient", make_handler("remote"))

    await cluster.start_all()
    await asyncio.sleep(0.6)

    # Exchange gradients
    local_grad = torch.randn(5)
    remote_grad = torch.randn(5)

    await local_node.broadcast_message("gradient", {"gradient": local_grad.tolist()})
    await remote_node.broadcast_message("gradient", {"gradient": remote_grad.tolist()})

    await asyncio.sleep(0.6)

    # Both should receive from each other (complete topology)
    assert len(gradients_received["local"]) == 1
    assert len(gradients_received["remote"]) == 1

    await cluster.shutdown_all()
    serve_task.cancel()
    try:
        await serve_task
    except asyncio.CancelledError:
        pass
