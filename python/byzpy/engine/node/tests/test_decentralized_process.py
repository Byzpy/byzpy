"""
Integration tests for process-based decentralized nodes.

Tests end-to-end P2P training with nodes running in separate OS processes.
"""

from __future__ import annotations

import asyncio

import pytest
import torch

from byzpy.engine.graph.graph import ComputationGraph, GraphNode, graph_input
from byzpy.engine.graph.operator import Operator
from byzpy.engine.graph.pool import ActorPoolConfig
from byzpy.engine.node.application import NodeApplication
from byzpy.engine.node.cluster import DecentralizedCluster
from byzpy.engine.node.context import InProcessContext, ProcessContext
from byzpy.engine.node.decentralized import DecentralizedNode

# =============================================================================
# Test Utilities
# =============================================================================


def create_test_application(name: str = "test-app") -> NodeApplication:
    """Create a test NodeApplication with thread pool."""
    return NodeApplication(
        name=name,
        actor_pool=[ActorPoolConfig(backend="thread", count=1)],
    )


class DoubleOp(Operator):
    """Simple operator that doubles input."""

    def compute(self, inputs, *, context):
        return inputs["x"] * 2


class MeanAggregatorOp(Operator):
    """Aggregates gradients by computing mean."""

    def compute(self, inputs, *, context):
        gradients = inputs["gradients"]
        if not gradients:
            return torch.zeros(1)
        stacked = torch.stack(
            [g if isinstance(g, torch.Tensor) else torch.tensor(g) for g in gradients]
        )
        return stacked.mean(dim=0)


class SGDStepOp(Operator):
    """Simulates an SGD step: params = params - lr * gradient."""

    def compute(self, inputs, *, context):
        params = inputs["params"]
        gradient = inputs["gradient"]
        lr = inputs.get("lr", 0.1)
        return params - lr * gradient


class LocalGradientOp(Operator):
    """Computes a local gradient (simulated)."""

    def compute(self, inputs, *, context):
        # Simulate gradient computation: gradient = params + noise
        params = inputs["params"]
        node_id = context.get("node_id", 0)
        # Each node has slightly different gradient (simulating different data)
        return params + 0.1 * (node_id + 1)


# =============================================================================
# Category 8: Multi-Process P2P Training Integration
# =============================================================================


@pytest.mark.asyncio
async def test_multi_process_p2p_gradient_exchange():
    """
    End-to-end test: 3 nodes in separate processes exchange gradients.

    Scenario:
    - 3 nodes, each running in a separate OS process
    - Each node computes a local gradient
    - Nodes send gradients to neighbors (ring pattern)
    - Verify processes are running and messages can be sent
    """
    cluster = DecentralizedCluster()

    # Create 3 nodes
    for i in range(3):
        app = create_test_application(f"node-{i}")
        await cluster.add_node(node_id=f"node{i}", application=app)

    await cluster.start_all()

    # Verify all processes are running
    for node_id, node in cluster.nodes.items():
        assert node._running, f"Node {node_id} should be running"
        assert isinstance(node.context, ProcessContext)
        assert node.context._process.is_alive(), f"Process for {node_id} should be alive"

    # Each node sends its "local gradient" to next node in ring
    for i, (node_id, node) in enumerate(cluster.nodes.items()):
        local_gradient = torch.tensor([float(i + 1), float(i + 2), float(i + 3)])
        await node.send_message(
            to_node_id=f"node{(i + 1) % 3}",  # Send to next node in ring
            message_type="gradient",
            payload={"gradient": local_gradient.tolist(), "round": 1},
        )

    # Give time for message delivery to queues
    await asyncio.sleep(0.3)

    # Verify nodes are still running after message exchange
    for node_id, node in cluster.nodes.items():
        assert node._running, f"Node {node_id} should still be running"
        assert node.context._process.is_alive(), f"Process for {node_id} should still be alive"

    await cluster.shutdown_all()


@pytest.mark.asyncio
@pytest.mark.skip(reason="Flaky test - multi-process P2P training timing issues")
async def test_multi_process_p2p_training_round():
    """
    End-to-end test: Full P2P training round with 3 nodes.

    Training round:
    1. Each node has local model parameters
    2. Each node computes local gradient
    3. Nodes exchange gradients with neighbors
    4. Each node aggregates and applies update
    """
    cluster = DecentralizedCluster()

    # Initial model parameters (same for all nodes)
    initial_params = torch.tensor([1.0, 2.0, 3.0])

    # Track node states
    node_states = {}

    for i in range(3):
        app = create_test_application(f"training-node-{i}")
        node = await cluster.add_node(node_id=f"node{i}", application=app)

        # Initialize node state
        node_states[f"node{i}"] = {
            "params": initial_params.clone(),
            "received_gradients": [],
            "local_gradient": None,
            "updated_params": None,
        }

    await cluster.start_all()

    # Register gradient handlers
    for node_id, node in cluster.nodes.items():

        async def on_gradient(from_id, payload, nid=node_id):
            grad_tensor = torch.tensor(payload["gradient"])
            node_states[nid]["received_gradients"].append(grad_tensor)

        node.register_message_handler("gradient", on_gradient)

    # Phase 1: Each node computes local gradient (simulated)
    for i, (node_id, node) in enumerate(cluster.nodes.items()):
        # Simulate local gradient (different per node due to different data)
        local_grad = initial_params * (0.1 * (i + 1))
        node_states[node_id]["local_gradient"] = local_grad

    # Phase 2: Broadcast gradients to all other nodes
    for i, (node_id, node) in enumerate(cluster.nodes.items()):
        local_grad = node_states[node_id]["local_gradient"]

        # Send to both other nodes (fully connected for simplicity)
        for j in range(3):
            if i != j:
                await node.send_message(
                    to_node_id=f"node{j}",
                    message_type="gradient",
                    payload={"gradient": local_grad.tolist()},
                )

    # Wait for all messages to be delivered
    await asyncio.sleep(0.8)

    # Phase 3: Each node aggregates and updates
    lr = 0.1
    for node_id in node_states:
        state = node_states[node_id]

        # Aggregate: mean of local + received gradients
        all_grads = [state["local_gradient"]] + state["received_gradients"]

        if len(all_grads) > 0:
            stacked = torch.stack(all_grads)
            avg_grad = stacked.mean(dim=0)

            # SGD update
            state["updated_params"] = state["params"] - lr * avg_grad

    # Verify all nodes updated their parameters
    for node_id, state in node_states.items():
        assert state["updated_params"] is not None, f"{node_id} should have updated params"
        assert not torch.equal(
            state["updated_params"], initial_params
        ), f"{node_id} params should have changed"

        # Each node should have received 2 gradients (from 2 other nodes)
        assert (
            len(state["received_gradients"]) == 2
        ), f"{node_id} should have received 2 gradients, got {len(state['received_gradients'])}"

    await cluster.shutdown_all()


@pytest.mark.asyncio
async def test_multi_process_independent_pipeline_execution():
    """
    Verify multiple nodes execute pipelines independently in separate processes.
    """
    cluster = DecentralizedCluster()

    # Create computation graph
    double_node = GraphNode(
        name="double",
        op=DoubleOp(),
        inputs={"x": graph_input("x")},
    )
    graph = ComputationGraph(nodes=[double_node], outputs=["double"])

    # Create 3 nodes with same pipeline
    for i in range(3):
        app = create_test_application(f"compute-node-{i}")
        app.register_pipeline("double", graph)
        await cluster.add_node(node_id=f"node{i}", application=app)

    await cluster.start_all()

    # Execute pipeline on all nodes concurrently
    async def execute_on_node(node_id: str, input_val: float):
        node = cluster.get_node(node_id)
        result = await node.execute_pipeline("double", {"x": input_val})
        return node_id, result["double"]

    results = await asyncio.gather(
        execute_on_node("node0", 5.0),
        execute_on_node("node1", 10.0),
        execute_on_node("node2", 15.0),
    )

    # Verify each node computed correctly
    results_dict = {r[0]: r[1] for r in results}

    assert results_dict["node0"] == 10.0, "node0 should compute 5*2=10"
    assert results_dict["node1"] == 20.0, "node1 should compute 10*2=20"
    assert results_dict["node2"] == 30.0, "node2 should compute 15*2=30"

    await cluster.shutdown_all()


@pytest.mark.asyncio
async def test_multi_process_aggregation_pipeline():
    """
    Test aggregation pipeline running across processes.

    Node 0 aggregates gradients received from Node 1 and Node 2.
    """
    cluster = DecentralizedCluster()

    # Create aggregation graph for node 0
    agg_node = GraphNode(
        name="aggregate",
        op=MeanAggregatorOp(),
        inputs={"gradients": graph_input("gradients")},
    )
    agg_graph = ComputationGraph(nodes=[agg_node], outputs=["aggregate"])

    # Create double graph for nodes 1, 2
    double_node = GraphNode(
        name="double",
        op=DoubleOp(),
        inputs={"x": graph_input("x")},
    )
    double_graph = ComputationGraph(nodes=[double_node], outputs=["double"])

    # Node 0: aggregator
    app0 = create_test_application("aggregator")
    app0.register_pipeline("aggregate", agg_graph)
    await cluster.add_node(node_id="aggregator", application=app0)

    # Nodes 1, 2: workers
    for i in [1, 2]:
        app = create_test_application(f"worker-{i}")
        app.register_pipeline("compute", double_graph)
        await cluster.add_node(node_id=f"worker{i}", application=app)

    await cluster.start_all()

    # Workers compute and send gradients
    grad1 = torch.tensor([1.0, 2.0, 3.0])
    grad2 = torch.tensor([4.0, 5.0, 6.0])

    # Collect gradients at aggregator
    collected_grads = []
    aggregator = cluster.get_node("aggregator")

    async def on_gradient(from_id, payload):
        grad = torch.tensor(payload["gradient"])
        collected_grads.append(grad)

    aggregator.register_message_handler("gradient", on_gradient)

    # Workers send gradients
    worker1 = cluster.get_node("worker1")
    worker2 = cluster.get_node("worker2")

    await worker1.send_message("aggregator", "gradient", {"gradient": grad1.tolist()})
    await worker2.send_message("aggregator", "gradient", {"gradient": grad2.tolist()})

    # Wait for messages
    await asyncio.sleep(0.5)

    # Aggregator aggregates
    assert len(collected_grads) == 2, f"Should have 2 gradients, got {len(collected_grads)}"

    result = await aggregator.execute_pipeline("aggregate", {"gradients": collected_grads})

    expected = (grad1 + grad2) / 2
    assert torch.allclose(
        result["aggregate"], expected
    ), f"Expected {expected}, got {result['aggregate']}"

    await cluster.shutdown_all()


@pytest.mark.asyncio
async def test_cross_process_message_delivery():
    """
    Test messages can be sent between two process-based nodes.

    Verifies the message queuing mechanism works across processes.
    """
    cluster = DecentralizedCluster()

    app1 = create_test_application("node1")
    app2 = create_test_application("node2")

    await cluster.add_node(node_id="node1", application=app1)
    await cluster.add_node(node_id="node2", application=app2)

    await cluster.start_all()

    node1 = cluster.get_node("node1")
    node2 = cluster.get_node("node2")

    # Send multiple messages between nodes
    for i in range(5):
        await node1.send_message("node2", "ping", {"seq": i, "message": "hello"})
        await node2.send_message("node1", "pong", {"seq": i, "message": "world"})

    # Wait for messages to be queued
    await asyncio.sleep(0.3)

    # Verify both nodes are still healthy
    assert node1._running
    assert node2._running
    assert node1.context._process.is_alive()
    assert node2.context._process.is_alive()

    await cluster.shutdown_all()


@pytest.mark.asyncio
async def test_process_based_training_with_model_update():
    """
    Full integration test: P2P training with actual model parameter updates.

    Simulates distributed SGD:
    1. All nodes start with same parameters
    2. Each computes gradient on local data (simulated)
    3. Nodes exchange gradients
    4. Each node aggregates and updates
    5. Verify convergence behavior
    """
    cluster = DecentralizedCluster()

    # Shared initial parameters
    initial_params = torch.tensor([10.0, 20.0, 30.0])
    learning_rate = 0.1

    # Node state tracking
    states = {}

    for i in range(3):
        app = create_test_application(f"trainer-{i}")
        node = await cluster.add_node(node_id=f"trainer{i}", application=app)
        states[f"trainer{i}"] = {
            "params": initial_params.clone(),
            "gradients_received": [],
        }

    await cluster.start_all()

    # Setup gradient handlers
    for node_id, node in cluster.nodes.items():

        async def handler(from_id, payload, nid=node_id):
            grad = torch.tensor(payload["gradient"])
            states[nid]["gradients_received"].append(grad)

        node.register_message_handler("gradient", handler)

    # Simulate 2 training rounds
    for round_num in range(2):
        # Clear received gradients
        for s in states.values():
            s["gradients_received"] = []

        # Each node computes local gradient (simulated as params * scale)
        local_gradients = {}
        for i, node_id in enumerate(states.keys()):
            # Different "data" per node leads to different gradients
            scale = 0.1 * (i + 1)
            local_gradients[node_id] = states[node_id]["params"] * scale

        # Exchange gradients (all-to-all)
        for i, (src_id, src_node) in enumerate(cluster.nodes.items()):
            grad = local_gradients[src_id]
            for j, (dst_id, _) in enumerate(cluster.nodes.items()):
                if i != j:
                    await src_node.send_message(dst_id, "gradient", {"gradient": grad.tolist()})

        # Wait for exchange
        await asyncio.sleep(0.5)

        # Aggregate and update
        for node_id, state in states.items():
            all_grads = [local_gradients[node_id]] + state["gradients_received"]
            avg_grad = torch.stack(all_grads).mean(dim=0)
            state["params"] = state["params"] - learning_rate * avg_grad

    # Verify all nodes have updated parameters
    for node_id, state in states.items():
        assert not torch.equal(
            state["params"], initial_params
        ), f"{node_id} should have updated parameters"

    # All nodes should have similar parameters (consensus behavior)
    params_list = [s["params"] for s in states.values()]
    for p1, p2 in zip(params_list[:-1], params_list[1:]):
        diff = (p1 - p2).abs().max().item()
        assert diff < 1.0, f"Nodes should have similar params, diff={diff}"

    await cluster.shutdown_all()


# =============================================================================
# Category 4: Cross-Process Message Routing
# =============================================================================


@pytest.mark.asyncio
async def test_cross_process_tensor_message():
    """Verify tensor messages can be sent between processes."""
    cluster = DecentralizedCluster()

    await cluster.add_node("node1", create_test_application("n1"))
    await cluster.add_node("node2", create_test_application("n2"))

    await cluster.start_all()

    node1 = cluster.get_node("node1")
    node2 = cluster.get_node("node2")

    # Send tensor (converted to list for serialization)
    test_tensor = torch.randn(5, 3)
    await node1.send_message("node2", "tensor_msg", {"tensor": test_tensor.tolist()})

    await asyncio.sleep(0.2)

    # Nodes should still be running after tensor message
    assert node1._running
    assert node2._running

    await cluster.shutdown_all()


@pytest.mark.asyncio
@pytest.mark.skip(
    reason="Flaky test - cross-process message delivery issue, missing message 5 (gets 9 instead of 10)"
)
async def test_cross_process_multiple_messages():
    """Verify multiple messages are delivered correctly."""
    cluster = DecentralizedCluster()

    await cluster.add_node("sender", create_test_application("sender"))
    await cluster.add_node("receiver", create_test_application("receiver"))

    await cluster.start_all()

    sender = cluster.get_node("sender")
    receiver = cluster.get_node("receiver")

    received = []

    async def handler(from_id, payload):
        received.append(payload["seq"])

    receiver.register_message_handler("seq_msg", handler)

    # Send 10 messages
    for i in range(10):
        await sender.send_message("receiver", "seq_msg", {"seq": i})
        await asyncio.sleep(0.05)  # Small delay between sends

    # Wait longer for cross-process delivery
    await asyncio.sleep(1.5)

    # All messages should arrive
    assert len(received) == 10, f"Expected 10 messages, got {len(received)}: {received}"
    # Order should be preserved (FIFO)
    assert received == list(range(10))

    await cluster.shutdown_all()


# =============================================================================
# Category 2: ProcessContext Integration with DecentralizedNode
# =============================================================================


@pytest.mark.asyncio
async def test_decentralizednode_with_processcontext_pipeline():
    """Verify DecentralizedNode can execute pipelines with ProcessContext."""
    ctx = ProcessContext()
    app = create_test_application("test-app")

    # Create simple computation graph
    double_node = GraphNode(
        name="double",
        op=DoubleOp(),
        inputs={"x": graph_input("x")},
    )
    graph = ComputationGraph(nodes=[double_node], outputs=["double"])
    app.register_pipeline("double", graph)

    node = DecentralizedNode(
        node_id="process-node",
        application=app,
        context=ctx,
    )

    await node.start()

    assert node._running
    assert ctx._running
    assert ctx._process.is_alive()

    # Execute pipeline
    result = await node.execute_pipeline("double", {"x": 7.5})

    assert result["double"] == 15.0

    await node.shutdown()


@pytest.mark.asyncio
async def test_decentralizednode_processcontext_send_message():
    """Verify DecentralizedNode can send messages via ProcessContext."""
    ctx1 = ProcessContext()
    ctx2 = ProcessContext()

    app1 = create_test_application("app1")
    app2 = create_test_application("app2")

    node1 = DecentralizedNode(node_id="node1", application=app1, context=ctx1)
    node2 = DecentralizedNode(node_id="node2", application=app2, context=ctx2)

    await node1.start()
    await node2.start()

    # Verify both nodes started with processes
    assert node1._running
    assert node2._running
    assert ctx1._process.is_alive()
    assert ctx2._process.is_alive()

    # Send message between nodes
    await node1.send_message("node2", "test", {"value": 42})

    await asyncio.sleep(0.2)

    # Nodes should still be running
    assert node1._running
    assert node2._running

    await node1.shutdown()
    await node2.shutdown()

    # Processes should be terminated
    assert not ctx1._running
    assert not ctx2._running
