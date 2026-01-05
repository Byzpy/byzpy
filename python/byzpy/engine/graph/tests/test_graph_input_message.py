from __future__ import annotations

import asyncio

import pytest

from byzpy.engine.graph.graph import ComputationGraph, GraphInput, GraphNode, graph_input
from byzpy.engine.graph.operator import Operator
from byzpy.engine.graph.scheduler import MessageAwareNodeScheduler

# Category 3: GraphInput Message Sources


def test_graphinput_from_message_creates_message_source():
    """Verify GraphInput.from_message() creates a message source."""
    msg_input = GraphInput.from_message("test_msg")
    assert hasattr(msg_input, "message_type")
    assert msg_input.message_type == "test_msg"
    assert msg_input.field is None


def test_graphinput_from_message_with_field():
    """Verify GraphInput.from_message() can extract specific field."""
    msg_input = GraphInput.from_message("test_msg", field="data")
    assert msg_input.message_type == "test_msg"
    assert msg_input.field == "data"


@pytest.mark.asyncio
async def test_graphinput_from_message_in_graph_execution():
    """Verify graph with message input waits for message."""

    class _DoubleOp(Operator):
        def compute(self, inputs, *, context):
            return inputs["x"] * 2

    # Create graph with message input
    msg_input = GraphInput.from_message("data_msg")
    op = _DoubleOp()
    node = GraphNode(name="double", op=op, inputs={"x": msg_input})
    graph = ComputationGraph(nodes=[node], outputs=["double"])

    scheduler = MessageAwareNodeScheduler(graph)

    # Start execution
    exec_task = asyncio.create_task(scheduler.run({}))
    await asyncio.sleep(0.01)

    # Deliver message
    scheduler.deliver_message("data_msg", 5)

    # Graph should complete
    result = await exec_task
    assert result["double"] == 10


@pytest.mark.asyncio
async def test_graphinput_from_message_extracts_field():
    """Verify GraphInput.from_message() extracts specific field from message."""

    class _IdentityOp(Operator):
        def compute(self, inputs, *, context):
            return inputs["x"]

    msg_input = GraphInput.from_message("test_msg", field="value")
    op = _IdentityOp()
    node = GraphNode(name="identity", op=op, inputs={"x": msg_input})
    graph = ComputationGraph(nodes=[node], outputs=["identity"])

    scheduler = MessageAwareNodeScheduler(graph)

    exec_task = asyncio.create_task(scheduler.run({}))
    await asyncio.sleep(0.01)

    # Deliver message with nested structure
    scheduler.deliver_message("test_msg", {"value": 42, "other": "ignored"})

    result = await exec_task
    assert result["identity"] == 42


@pytest.mark.asyncio
async def test_graphinput_from_message_missing_field():
    """Verify GraphInput.from_message() handles missing field gracefully."""

    class _IdentityOp(Operator):
        def compute(self, inputs, *, context):
            return inputs["x"]

    msg_input = GraphInput.from_message("test_msg", field="nonexistent")
    op = _IdentityOp()
    node = GraphNode(name="identity", op=op, inputs={"x": msg_input})
    graph = ComputationGraph(nodes=[node], outputs=["identity"])

    scheduler = MessageAwareNodeScheduler(graph)

    exec_task = asyncio.create_task(scheduler.run({}))
    await asyncio.sleep(0.01)

    # Deliver message without the field
    scheduler.deliver_message("test_msg", {"other": "data"})

    # Should raise KeyError or return None
    with pytest.raises((KeyError, TypeError)):
        await exec_task
