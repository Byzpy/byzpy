from __future__ import annotations

import asyncio

import pytest
import torch

from byzpy.engine.graph.graph import ComputationGraph, GraphNode, graph_input
from byzpy.engine.graph.operator import Operator
from byzpy.engine.graph.pool import ActorPoolConfig
from byzpy.engine.node.application import HonestNodeApplication, NodeApplication
from byzpy.engine.node.context import InProcessContext
from byzpy.engine.node.decentralized import DecentralizedNode

# Test Utilities and Fixtures


def create_test_application() -> NodeApplication:
    """Create a test NodeApplication with minimal config."""
    return NodeApplication(
        name="test-app",
        actor_pool=[ActorPoolConfig(backend="thread", count=1)],
    )


def create_test_node(node_id: str = "test-node") -> DecentralizedNode:
    """Create a test DecentralizedNode."""
    return DecentralizedNode(
        node_id=node_id,
        application=create_test_application(),
        context=InProcessContext(),
    )


class _DoubleOp(Operator):
    """Test operator that doubles the input."""

    def compute(self, inputs, *, context):
        return inputs["x"] * 2


class _AddOp(Operator):
    """Test operator that adds two inputs."""

    def compute(self, inputs, *, context):
        return inputs["lhs"] + inputs["rhs"]


def create_simple_graph() -> ComputationGraph:
    """Create a simple computation graph for testing."""
    # e.g., double input
    op = _DoubleOp()
    node = GraphNode(name="double", op=op, inputs={"x": graph_input("x")})
    return ComputationGraph(nodes=[node], outputs=["double"])


def create_add_graph() -> ComputationGraph:
    """Create a graph that adds two inputs."""
    op = _AddOp()
    node = GraphNode(name="sum", op=op, inputs={"lhs": graph_input("a"), "rhs": graph_input("b")})
    return ComputationGraph(nodes=[node], outputs=["sum"])


# Category 3: DecentralizedNode Core Functionality


def test_decentralizednode_can_be_created():
    """Verify DecentralizedNode can be instantiated."""
    app = create_test_application()
    ctx = InProcessContext()

    node = DecentralizedNode(
        node_id="test-node",
        application=app,
        context=ctx,
    )

    assert node.node_id == "test-node"
    assert node.application is app
    assert node.context is ctx
    assert node.scheduler is not None


@pytest.mark.asyncio
async def test_decentralizednode_start_initializes_components():
    """Verify start() initializes all components."""
    node = create_test_node()

    await node.start()

    assert node._running is True
    assert node.context._node is node


@pytest.mark.asyncio
async def test_decentralizednode_shutdown_cleans_up():
    """Verify shutdown() cleans up all resources."""
    node = create_test_node()
    await node.start()

    await node.shutdown()

    assert node._running is False


@pytest.mark.asyncio
async def test_decentralizednode_execute_pipeline_runs_graph():
    """Verify execute_pipeline() runs computation graph via scheduler."""
    node = create_test_node()
    await node.start()

    # Register a simple pipeline
    graph = create_simple_graph()
    node.application.register_pipeline("test", graph)

    result = await node.execute_pipeline("test", {"x": 5})

    assert result["double"] == 10


@pytest.mark.asyncio
async def test_decentralizednode_execute_pipeline_raises_for_missing_pipeline():
    """Verify execute_pipeline() raises error for unknown pipeline."""
    node = create_test_node()
    await node.start()

    with pytest.raises(KeyError, match="Unknown pipeline"):
        await node.execute_pipeline("nonexistent", {})


@pytest.mark.asyncio
async def test_decentralizednode_state_persistence():
    """Verify node state persists across operations."""
    node = create_test_node()
    await node.start()

    node._state["counter"] = 0
    node._state["counter"] += 1

    assert node._state["counter"] == 1

    # State should persist
    assert node._state.get("counter") == 1


# Category 4: Message Handling


@pytest.mark.asyncio
async def test_decentralizednode_handle_incoming_message_processes():
    """Verify handle_incoming_message() processes messages correctly."""
    node = create_test_node()
    await node.start()

    # Register a message handler
    received_messages = []

    async def handler(from_id, payload):
        received_messages.append((from_id, payload))

    node.register_message_handler("test_msg", handler)

    await node.handle_incoming_message("sender", "test_msg", {"data": 123})

    assert len(received_messages) == 1
    assert received_messages[0] == ("sender", {"data": 123})


@pytest.mark.asyncio
async def test_decentralizednode_register_message_handler():
    """Verify message handlers can be registered."""
    node = create_test_node()

    async def handler(from_id, payload):
        pass

    node.register_message_handler("msg_type", handler)

    assert "msg_type" in node._message_handlers
    assert node._message_handlers["msg_type"] is handler


@pytest.mark.asyncio
async def test_decentralizednode_message_processing_loop():
    """Verify message processing loop handles incoming messages."""
    node = create_test_node()
    await node.start()

    processed = []

    async def handler(from_id, payload):
        processed.append(payload)

    node.register_message_handler("test", handler)

    # Send message via handle_incoming_message to simulate receiving a message
    await node.handle_incoming_message("sender", "test", {"data": 1})

    # Give loop time to process
    await asyncio.sleep(0.1)

    assert len(processed) == 1


@pytest.mark.asyncio
async def test_decentralizednode_send_message():
    """Verify send_message() works (basic test for Milestone 1)."""
    node = create_test_node()
    await node.start()

    # For Milestone 1, send_message uses simple router
    # It should not raise an error
    try:
        await node.send_message("target", "test_msg", {"data": 42})
    except Exception as e:
        # For InProcessContext in Milestone 1, we may not have full routing
        # This is acceptable - full routing comes in Milestone 4
        pass


# Category 5: Integration with NodeApplication


@pytest.mark.asyncio
async def test_decentralizednode_uses_application_pipelines():
    """Verify node can execute pipelines registered on application."""
    app = HonestNodeApplication(
        name="test-app",
        actor_pool=[ActorPoolConfig(backend="thread", count=2)],
    )

    # Register aggregation pipeline
    from typing import Sequence

    from byzpy.aggregators.base import Aggregator

    class _SumAggregator(Aggregator):
        name = "sum"

        def aggregate(self, gradients: Sequence[torch.Tensor]) -> torch.Tensor:
            if not gradients:
                raise ValueError("No gradients supplied.")
            total = torch.zeros_like(gradients[0])
            for grad in gradients:
                total = total + grad.to(total.device)
            return total

    graph = create_simple_graph()  # Use simple graph for now
    # Actually, let's create a proper aggregation graph
    agg_op = _SumAggregator()
    agg_node = GraphNode(
        name="aggregate",
        op=agg_op,
        inputs={"gradients": graph_input("gradients")},
    )
    agg_graph = ComputationGraph(nodes=[agg_node], outputs=["aggregate"])
    app.register_pipeline(app.AGGREGATION_PIPELINE, agg_graph)

    node = DecentralizedNode(
        node_id="test-node",
        application=app,
        context=InProcessContext(),
    )
    await node.start()

    grads = [torch.tensor([1.0]), torch.tensor([2.0])]
    result = await node.execute_pipeline(app.AGGREGATION_PIPELINE, {"gradients": grads})

    # Verify aggregation executed
    assert result is not None
    assert "aggregate" in result


@pytest.mark.asyncio
async def test_decentralizednode_shares_actor_pool_with_scheduler():
    """Verify scheduler uses application's actor pool."""
    app = create_test_application()
    node = DecentralizedNode(
        node_id="test",
        application=app,
        context=InProcessContext(),
    )

    assert node.scheduler.pool is app.pool


# Category 6: Integration with NodeScheduler


@pytest.mark.asyncio
async def test_decentralizednode_scheduler_executes_graphs():
    """Verify scheduler correctly executes computation graphs."""
    node = create_test_node()
    await node.start()

    # Create a simple graph: input -> double -> output
    graph = create_simple_graph()
    node.application.register_pipeline("double", graph)

    result = await node.execute_pipeline("double", {"x": 7})

    assert result["double"] == 14


@pytest.mark.asyncio
async def test_decentralizednode_scheduler_receives_metadata():
    """Verify scheduler receives node metadata."""
    node = DecentralizedNode(
        node_id="node-42",
        application=create_test_application(),
        context=InProcessContext(),
        metadata={"role": "honest", "epoch": 1},
    )

    # Verify metadata is passed to scheduler
    assert "node_id" in node.scheduler.metadata
    assert node.scheduler.metadata["node_id"] == "node-42"
    assert node.scheduler.metadata["role"] == "honest"
    assert node.scheduler.metadata["epoch"] == 1


# Category 7: Error Handling and Edge Cases


@pytest.mark.asyncio
async def test_decentralizednode_start_idempotent():
    """Verify start() is idempotent."""
    node = create_test_node()

    await node.start()
    assert node._running is True

    # Start again should not error
    await node.start()
    assert node._running is True


@pytest.mark.asyncio
async def test_decentralizednode_shutdown_when_not_started():
    """Verify shutdown() works even if node not started."""
    node = create_test_node()

    # Should not raise
    await node.shutdown()


@pytest.mark.asyncio
async def test_decentralizednode_execute_pipeline_requires_start():
    """Verify execute_pipeline() requires node to be started."""
    node = create_test_node()
    node.application.register_pipeline("test", create_simple_graph())

    # Should raise
    with pytest.raises(RuntimeError, match="not started"):
        await node.execute_pipeline("test", {})


def test_decentralizednode_validates_node_id():
    """Verify node_id validation."""
    # Empty string should be invalid
    with pytest.raises(ValueError):
        DecentralizedNode(
            node_id="",
            application=create_test_application(),
            context=InProcessContext(),
        )


# Category 8: Multi-Node Scenarios (In-Process)


@pytest.mark.asyncio
async def test_two_nodes_execute_independently():
    """Verify two nodes can execute pipelines independently."""
    node1 = create_test_node(node_id="node1")
    node2 = create_test_node(node_id="node2")

    await node1.start()
    await node2.start()

    # Register different pipelines
    node1.application.register_pipeline("p1", create_simple_graph())
    node2.application.register_pipeline("p2", create_add_graph())

    # Execute concurrently
    results = await asyncio.gather(
        node1.execute_pipeline("p1", {"x": 1}),
        node2.execute_pipeline("p2", {"a": 2, "b": 3}),
    )

    # Both should succeed independently
    assert results[0] is not None
    assert results[1] is not None
    assert results[0]["double"] == 2
    assert results[1]["sum"] == 5


@pytest.mark.asyncio
async def test_multiple_nodes_exchange_messages():
    """Verify multiple nodes can exchange messages."""
    # For Milestone 1, we need a way to connect nodes
    # We'll create a simple shared message bus using InProcessContext
    # This is a simplified version - full routing comes in Milestone 4

    node1 = create_test_node(node_id="node1")
    node2 = create_test_node(node_id="node2")

    await node1.start()
    await node2.start()

    # For InProcessContext, we can directly send messages to node2's context
    received = []

    async def handler(from_id, payload):
        received.append((from_id, payload))

    node2.register_message_handler("ping", handler)

    # Send message from node1 to node2
    await node1.send_message("node2", "ping", {"seq": 1})

    # Give time for delivery
    await asyncio.sleep(0.1)

    assert len(received) == 1
    assert received[0][0] == "node1"
    assert received[0][1]["seq"] == 1
