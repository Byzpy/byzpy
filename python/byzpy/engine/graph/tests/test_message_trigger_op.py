from __future__ import annotations

import asyncio

import pytest

from byzpy.engine.graph.graph import ComputationGraph, GraphNode
from byzpy.engine.graph.operator import MessageTriggerOp
from byzpy.engine.graph.scheduler import MessageAwareNodeScheduler

# Category 2: MessageTriggerOp Operator


def test_messagetriggerop_can_be_created():
    """Verify MessageTriggerOp can be instantiated."""
    op = MessageTriggerOp("test_msg")
    assert op.message_type == "test_msg"
    assert op.timeout is None


def test_messagetriggerop_with_timeout():
    """Verify MessageTriggerOp can be created with timeout."""
    op = MessageTriggerOp("test_msg", timeout=5.0)
    assert op.message_type == "test_msg"
    assert op.timeout == 5.0


@pytest.mark.asyncio
async def test_messagetriggerop_executes_in_graph():
    """Verify MessageTriggerOp waits for message in graph execution."""
    # Create graph with MessageTriggerOp
    trigger_op = MessageTriggerOp("test_msg")
    node = GraphNode(name="trigger", op=trigger_op, inputs={})
    graph = ComputationGraph(nodes=[node], outputs=["trigger"])

    scheduler = MessageAwareNodeScheduler(graph)

    # Start graph execution (will wait for message)
    exec_task = asyncio.create_task(scheduler.run({}))
    await asyncio.sleep(0.01)

    # Deliver message
    scheduler.deliver_message("test_msg", {"payload": "data"})

    # Graph should complete
    result = await exec_task
    assert result["trigger"] == {"payload": "data"}


@pytest.mark.asyncio
async def test_messagetriggerop_requires_scheduler_in_context():
    """Verify MessageTriggerOp raises error if scheduler not in context."""
    from byzpy.engine.graph.operator import OpContext

    op = MessageTriggerOp("test_msg")
    ctx = OpContext(node_name="test", metadata={})  # No scheduler

    with pytest.raises(RuntimeError, match="requires scheduler"):
        await op.run({}, context=ctx, pool=None)


@pytest.mark.asyncio
async def test_messagetriggerop_timeout_raises_error():
    """Verify MessageTriggerOp respects timeout."""
    trigger_op = MessageTriggerOp("test_msg", timeout=0.1)
    node = GraphNode(name="trigger", op=trigger_op, inputs={})
    graph = ComputationGraph(nodes=[node], outputs=["trigger"])

    scheduler = MessageAwareNodeScheduler(graph)

    # Graph execution should timeout
    with pytest.raises(asyncio.TimeoutError):
        await scheduler.run({})


def test_messagetriggerop_empty_message_type():
    """Verify MessageTriggerOp validates message type."""
    with pytest.raises(ValueError):
        MessageTriggerOp("")
