from __future__ import annotations

import asyncio

import pytest
import torch

from byzpy.engine.graph.graph import ComputationGraph, GraphInput, GraphNode
from byzpy.engine.graph.operator import MessageTriggerOp
from byzpy.engine.graph.pool import ActorPoolConfig
from byzpy.engine.graph.scheduler import MessageAwareNodeScheduler
from byzpy.engine.node.application import HonestNodeApplication, NodeApplication
from byzpy.engine.node.context import InProcessContext
from byzpy.engine.node.decentralized import DecentralizedNode

# Test Utilities


def create_test_application():
    """Create a test NodeApplication with minimal config."""
    return NodeApplication(
        name="test-app",
        actor_pool=[ActorPoolConfig(backend="thread", count=1)],
    )


def create_test_node(node_id: str = "test-node"):
    """Create a test DecentralizedNode."""
    return DecentralizedNode(
        node_id=node_id,
        application=create_test_application(),
        context=InProcessContext(),
    )


def create_test_node_with_pool(node_id: str = "test-node"):
    """Create a test node with actor pool."""
    app = NodeApplication(
        name="test-app",
        actor_pool=[ActorPoolConfig(backend="thread", count=2)],
    )
    return DecentralizedNode(
        node_id=node_id,
        application=app,
        context=InProcessContext(),
    )


# Category 4: MessageAwareNodeScheduler Integration with DecentralizedNode


@pytest.mark.asyncio
async def test_decentralizednode_uses_messageaware_scheduler():
    """Verify DecentralizedNode uses MessageAwareNodeScheduler."""
    node = create_test_node()
    await node.start()

    assert isinstance(node.scheduler, MessageAwareNodeScheduler)


@pytest.mark.asyncio
async def test_decentralizednode_delivers_messages_to_scheduler():
    """Verify incoming messages are delivered to scheduler."""
    node = create_test_node()
    await node.start()

    # Wait for message in scheduler
    wait_task = asyncio.create_task(node.scheduler.wait_for_message("test_msg"))
    await asyncio.sleep(0.01)

    # Send message to node
    await node.handle_incoming_message("sender", "test_msg", {"data": 123})

    # Scheduler should receive it
    result = await wait_task
    assert result == {"data": 123}


@pytest.mark.asyncio
async def test_decentralizednode_message_driven_pipeline():
    """Verify pipeline can be triggered by incoming message."""
    # Create pipeline that waits for message
    trigger_op = MessageTriggerOp("start_msg")
    node_op = GraphNode(name="trigger", op=trigger_op, inputs={})
    graph = ComputationGraph(nodes=[node_op], outputs=["trigger"])

    node = create_test_node()
    await node.start()
    node.application.register_pipeline("message_pipeline", graph)

    # Start pipeline execution (will wait)
    exec_task = asyncio.create_task(node.execute_pipeline("message_pipeline", {}))
    await asyncio.sleep(0.01)

    # Send message
    await node.handle_incoming_message("sender", "start_msg", {"action": "go"})

    # Pipeline should complete
    result = await exec_task
    assert result["trigger"] == {"action": "go"}


# Category 5: Complex Message-Driven Scenarios


@pytest.mark.asyncio
async def test_multiple_graphs_wait_for_same_message():
    """Verify multiple graph executions can wait for same message type."""
    trigger_op = MessageTriggerOp("broadcast_msg")
    node_op = GraphNode(name="trigger", op=trigger_op, inputs={})
    graph = ComputationGraph(nodes=[node_op], outputs=["trigger"])

    node = create_test_node()
    await node.start()
    node.application.register_pipeline("pipeline", graph)

    # Start multiple executions
    exec1 = asyncio.create_task(node.execute_pipeline("pipeline", {}))
    exec2 = asyncio.create_task(node.execute_pipeline("pipeline", {}))
    await asyncio.sleep(0.01)

    # Deliver one message - should wake both
    await node.handle_incoming_message("sender", "broadcast_msg", {"data": 1})

    # Both should get the message (from cache)
    result1 = await exec1
    result2 = await exec2
    assert result1["trigger"] == {"data": 1}
    assert result2["trigger"] == {"data": 1}


# Category 7: Integration with Existing Components


@pytest.mark.asyncio
async def test_message_driven_aggregation():
    """Verify aggregation can be triggered by message."""
    from typing import Sequence

    from byzpy.aggregators.base import Aggregator

    class _SumAggregator(Aggregator):
        name = "sum"

        def aggregate(self, gradients: Sequence[torch.Tensor]) -> torch.Tensor:
            return sum(gradients)

    node = create_test_node()
    await node.start()

    # Create aggregation pipeline triggered by message
    msg_input = GraphInput.from_message("aggregate_msg", field="gradients")
    agg_op = _SumAggregator()
    agg_node = GraphNode(name="aggregate", op=agg_op, inputs={"gradients": msg_input})
    graph = ComputationGraph(nodes=[agg_node], outputs=["aggregate"])
    node.application.register_pipeline("aggregate", graph)

    exec_task = asyncio.create_task(node.execute_pipeline("aggregate", {}))
    await asyncio.sleep(0.01)

    # Send gradients via message
    grads = [torch.tensor([1.0]), torch.tensor([2.0])]
    await node.handle_incoming_message("sender", "aggregate_msg", {"gradients": grads})

    result = await exec_task
    assert torch.equal(result["aggregate"], torch.tensor([3.0]))


@pytest.mark.asyncio
async def test_message_driven_pipeline_with_actor_pool():
    """Verify message-driven execution works with actor pool parallelism."""
    from byzpy.engine.graph.operator import Operator
    from byzpy.engine.graph.subtask import SubTask

    class _ParallelOp(Operator):
        supports_subtasks = True

        def create_subtasks(self, inputs, *, context):
            data = inputs["data"]
            return [SubTask(fn=lambda x=x: x * 2, args=(x,)) for x in data]

        def reduce_subtasks(self, partials, inputs, *, context):
            return sum(partials)

    node = create_test_node_with_pool()
    await node.start()

    msg_input = GraphInput.from_message("data_msg", field="values")
    op = _ParallelOp()
    node_op = GraphNode(name="parallel", op=op, inputs={"data": msg_input})
    graph = ComputationGraph(nodes=[node_op], outputs=["parallel"])
    node.application.register_pipeline("parallel", graph)

    exec_task = asyncio.create_task(node.execute_pipeline("parallel", {}))
    await asyncio.sleep(0.01)

    await node.handle_incoming_message("sender", "data_msg", {"values": [1, 2, 3, 4]})

    result = await exec_task
    assert result["parallel"] == 20  # (1+2+3+4) * 2


# Category 8: P2P Aggregation Example


@pytest.mark.asyncio
async def test_simple_p2p_aggregation_triggered_by_message():
    """Verify simple P2P aggregation example from Milestone 2."""
    from typing import Sequence

    from byzpy.aggregators.base import Aggregator

    class _MeanAggregator(Aggregator):
        name = "mean"

        def aggregate(self, gradients: Sequence[torch.Tensor]) -> torch.Tensor:
            stacked = torch.stack(gradients)
            return stacked.mean(dim=0)

    # Node 1: sends gradient
    node1 = create_test_node(node_id="node1")
    await node1.start()

    # Node 2: waits for neighbor gradients and aggregates
    node2 = create_test_node(node_id="node2")
    await node2.start()

    # Node 2's aggregation pipeline waits for neighbor messages
    msg_input = GraphInput.from_message("neighbor_gradient")
    agg_op = _MeanAggregator()
    agg_node = GraphNode(name="aggregate", op=agg_op, inputs={"gradients": msg_input})
    graph = ComputationGraph(nodes=[agg_node], outputs=["aggregate"])
    node2.application.register_pipeline("aggregate", graph)

    # Start aggregation (waits for messages)
    agg_task = asyncio.create_task(node2.execute_pipeline("aggregate", {}))
    await asyncio.sleep(0.01)

    # Node 1 sends its gradient to node 2
    grad1 = torch.tensor([1.0, 2.0])
    # Aggregator expects a sequence, so send as list
    await node1.send_message("node2", "neighbor_gradient", [grad1])

    # Give time for message delivery
    await asyncio.sleep(0.1)

    # Node 2 should receive and aggregate
    result = await agg_task
    assert torch.allclose(result["aggregate"], grad1)
