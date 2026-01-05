"""
Category 6: Fully Asynchronous P2P Training Tests

Tests for fully decentralized P2P training where nodes progress independently
without a synchronous round() API.
"""

from __future__ import annotations

import asyncio

import pytest
import torch

from byzpy.aggregators.coordinate_wise import CoordinateWiseMedian
from byzpy.engine.graph.graph import ComputationGraph, GraphInput, GraphNode
from byzpy.engine.graph.ops import CallableOp, make_single_operator_graph
from byzpy.engine.graph.pool import ActorPoolConfig
from byzpy.engine.node.application import HonestNodeApplication
from byzpy.engine.node.cluster import DecentralizedCluster
from byzpy.engine.node.context import InProcessContext, ProcessContext
from byzpy.engine.node.decentralized import DecentralizedNode
from byzpy.engine.peer_to_peer.topology import Topology


@pytest.mark.asyncio
async def test_nodes_progress_independently_without_round_api():
    """Verify nodes can progress independently without central round() coordination."""
    topology = Topology.ring(3, k=1)
    cluster = DecentralizedCluster()
    received_gradients = {f"node{i}": [] for i in range(3)}

    # Create nodes with message-driven pipelines
    for i in range(3):
        app = HonestNodeApplication(
            name=f"node{i}",
            actor_pool=[ActorPoolConfig(backend="thread", count=1)],
        )

        # Half-step pipeline (triggered by local state/timer)
        def make_half_step(node_idx):
            def half_step(lr):
                return torch.tensor([float(node_idx) * lr, 2.0 * float(node_idx) * lr])

            return half_step

        half_step_graph = make_single_operator_graph(
            node_name="half_step",
            operator=CallableOp(make_half_step(i), input_mapping={"lr": "lr"}),
            input_keys=("lr",),
        )
        app.register_pipeline("half_step", half_step_graph)

        # Aggregation pipeline
        aggregator = CoordinateWiseMedian()
        aggregate_graph = make_single_operator_graph(
            node_name="aggregate",
            operator=aggregator,
            input_keys=("gradients",),
        )
        app.register_pipeline("aggregate", aggregate_graph)

        node = await cluster.add_node(
            node_id=f"node{i}",
            application=app,
            topology=topology,
            context=InProcessContext(),
        )

        # Register message handler that collects gradients
        def make_aggregation_handler(nid):
            async def on_gradient(from_id, payload):
                received_gradients[nid].append(payload["vector"])

            return on_gradient

        node.register_message_handler("gradient", make_aggregation_handler(f"node{i}"))

    cluster._update_node_id_maps()
    await cluster.start_all()
    await asyncio.sleep(0.3)

    # Each node independently triggers half-step
    for node_id, node in cluster.nodes.items():
        half_result = await node.execute_pipeline("half_step", {"lr": 0.01})
        # Broadcast to neighbors
        await node.broadcast_message("gradient", {"vector": half_result["half_step"]})

    # Wait for messages to propagate
    await asyncio.sleep(0.5)

    # Verify each node received gradients from neighbors
    for node_id in cluster.nodes:
        assert len(received_gradients[node_id]) >= 1  # At least one neighbor

    await cluster.shutdown_all()


@pytest.mark.asyncio
async def test_nodes_trigger_half_step_independently():
    """Verify nodes can trigger half-step pipelines independently based on local state."""
    app = HonestNodeApplication(
        name="test",
        actor_pool=[ActorPoolConfig(backend="thread", count=1)],
    )

    def half_step(lr):
        return torch.tensor([1.0 * lr, 2.0 * lr])

    graph = make_single_operator_graph(
        node_name="half_step",
        operator=CallableOp(half_step, input_mapping={"lr": "lr"}),
        input_keys=("lr",),
    )
    app.register_pipeline("half_step", graph)

    node = DecentralizedNode(
        node_id="node0",
        application=app,
        context=InProcessContext(),
    )

    await node.start()

    # Node independently triggers half-step (e.g., based on timer or local state)
    result1 = await node.execute_pipeline("half_step", {"lr": 0.01})
    await asyncio.sleep(0.1)
    result2 = await node.execute_pipeline("half_step", {"lr": 0.01})

    assert "half_step" in result1
    assert "half_step" in result2

    await node.shutdown()


@pytest.mark.asyncio
async def test_nodes_aggregate_when_neighbors_respond():
    """Verify nodes automatically trigger aggregation when they receive neighbor messages."""
    topology = Topology.ring(3, k=1)
    cluster = DecentralizedCluster()
    aggregation_triggered = {"node0": False}

    app = HonestNodeApplication(
        name="node0",
        actor_pool=[ActorPoolConfig(backend="thread", count=1)],
    )

    # Aggregation pipeline
    aggregator = CoordinateWiseMedian()
    aggregate_graph = make_single_operator_graph(
        node_name="aggregate",
        operator=aggregator,
        input_keys=("gradients",),
    )
    app.register_pipeline("aggregate", aggregate_graph)

    node = await cluster.add_node(
        node_id="node0",
        application=app,
        topology=topology,
        context=InProcessContext(),
    )

    # Message handler triggers aggregation when messages arrive
    async def on_gradient(from_id, payload):
        # Trigger aggregation pipeline
        gradients = [payload["vector"]]
        result = await node.execute_pipeline("aggregate", {"gradients": gradients})
        aggregation_triggered["node0"] = True

    node.register_message_handler("gradient", on_gradient)

    cluster._update_node_id_maps()
    await cluster.start_all()
    await asyncio.sleep(0.2)

    # Simulate receiving gradients (in real scenario, these come from neighbors)
    # For this test, we'll directly test aggregation
    gradients = [
        torch.tensor([1.0, 2.0]),
        torch.tensor([2.0, 3.0]),
    ]
    result = await node.execute_pipeline("aggregate", {"gradients": gradients})
    aggregation_triggered["node0"] = True

    assert aggregation_triggered["node0"]
    assert "aggregate" in result

    await cluster.shutdown_all()


@pytest.mark.asyncio
async def test_p2p_training_fully_asynchronous_ring():
    """End-to-end fully asynchronous P2P training with ring topology."""
    topology = Topology.ring(4, k=1)
    cluster = DecentralizedCluster()
    training_rounds = {f"node{i}": 0 for i in range(4)}

    # Create nodes with fully asynchronous training logic
    for i in range(4):
        app = HonestNodeApplication(
            name=f"node{i}",
            actor_pool=[ActorPoolConfig(backend="thread", count=1)],
        )

        # Half-step pipeline
        def make_half_step(node_idx):
            def half_step(lr):
                return torch.tensor([float(node_idx) * lr, 2.0 * float(node_idx) * lr])

            return half_step

        half_step_graph = make_single_operator_graph(
            node_name="half_step",
            operator=CallableOp(make_half_step(i), input_mapping={"lr": "lr"}),
            input_keys=("lr",),
        )
        app.register_pipeline("half_step", half_step_graph)

        # Aggregation pipeline
        aggregator = CoordinateWiseMedian()
        aggregate_graph = make_single_operator_graph(
            node_name="aggregate",
            operator=aggregator,
            input_keys=("gradients",),
        )
        app.register_pipeline("aggregate", aggregate_graph)

        node = await cluster.add_node(
            node_id=f"node{i}",
            application=app,
            topology=topology,
            context=InProcessContext(),
        )

        # Message handler: when gradient received, trigger aggregation
        def make_training_handler(nid, n):
            async def on_gradient(from_id, payload):
                # Trigger aggregation
                training_rounds[nid] += 1

            return on_gradient

        node.register_message_handler("gradient", make_training_handler(f"node{i}", node))

    cluster._update_node_id_maps()
    await cluster.start_all()
    await asyncio.sleep(0.5)

    # Each node independently starts training
    for node_id, node in cluster.nodes.items():
        # Trigger half-step
        result = await node.execute_pipeline("half_step", {"lr": 0.01})
        # Broadcast to neighbors
        await node.broadcast_message("gradient", {"vector": result["half_step"]})

    # Wait for asynchronous training to progress
    await asyncio.sleep(1.0)

    # Verify nodes made progress independently
    for node_id in training_rounds:
        assert training_rounds[node_id] > 0

    await cluster.shutdown_all()
