from __future__ import annotations

import asyncio

import pytest

from byzpy.engine.graph.graph import ComputationGraph, GraphNode, graph_input
from byzpy.engine.graph.operator import Operator
from byzpy.engine.graph.scheduler import MessageAwareNodeScheduler, NodeScheduler


class _AddOp(Operator):
    def compute(self, inputs, *, context):
        return inputs["lhs"] + inputs["rhs"]


class _NoOp(Operator):
    def compute(self, inputs, *, context):
        return None


def create_simple_graph():
    """Create a simple computation graph for testing."""
    op = _NoOp()
    node = GraphNode(name="dummy", op=op, inputs={})
    return ComputationGraph(nodes=[node], outputs=["dummy"])


def create_message_aware_scheduler(graph=None):
    """Create a MessageAwareNodeScheduler for testing."""
    if graph is None:
        graph = create_simple_graph()
    return MessageAwareNodeScheduler(graph)


# Category 1: MessageAwareNodeScheduler Core Functionality


def test_messageawarenodescheduler_can_be_created():
    """Verify MessageAwareNodeScheduler can be instantiated."""
    graph = create_simple_graph()
    scheduler = MessageAwareNodeScheduler(graph)

    assert isinstance(scheduler, NodeScheduler)
    assert scheduler.graph is graph
    assert hasattr(scheduler, "wait_for_message")
    assert hasattr(scheduler, "deliver_message")


@pytest.mark.asyncio
async def test_messageawarenodescheduler_wait_for_message_blocks_until_delivered():
    """Verify wait_for_message() blocks until message is delivered."""
    scheduler = create_message_aware_scheduler()

    # Start waiting task
    async def wait_task():
        return await scheduler.wait_for_message("test_msg")

    wait_fut = asyncio.create_task(wait_task())

    # Give it a moment to start waiting
    await asyncio.sleep(0.01)

    # Deliver message
    scheduler.deliver_message("test_msg", {"data": 42})

    # Wait should complete
    result = await wait_fut
    assert result == {"data": 42}


@pytest.mark.asyncio
async def test_messageawarenodescheduler_wait_for_message_uses_cache():
    """Verify wait_for_message() returns cached message immediately."""
    scheduler = create_message_aware_scheduler()

    # Deliver message first
    scheduler.deliver_message("test_msg", {"data": 100})

    # Wait should return immediately from cache
    result = await scheduler.wait_for_message("test_msg")
    assert result == {"data": 100}


@pytest.mark.asyncio
async def test_messageawarenodescheduler_wait_for_message_timeout():
    """Verify wait_for_message() raises TimeoutError on timeout."""
    scheduler = create_message_aware_scheduler()

    with pytest.raises(asyncio.TimeoutError):
        await scheduler.wait_for_message("test_msg", timeout=0.1)


@pytest.mark.asyncio
async def test_messageawarenodescheduler_deliver_message_wakes_multiple_waiters():
    """Verify deliver_message() wakes all waiting futures."""
    scheduler = create_message_aware_scheduler()

    # Create multiple waiters
    waiters = [asyncio.create_task(scheduler.wait_for_message("test_msg")) for _ in range(5)]

    await asyncio.sleep(0.01)  # Let them start waiting

    # Deliver message
    scheduler.deliver_message("test_msg", {"data": 999})

    # All waiters should get the message
    results = await asyncio.gather(*waiters)
    assert all(r == {"data": 999} for r in results)


@pytest.mark.asyncio
async def test_messageawarenodescheduler_message_cache_fifo():
    """Verify message cache works in FIFO order."""
    scheduler = create_message_aware_scheduler()

    # Deliver multiple messages
    scheduler.deliver_message("test_msg", {"seq": 1})
    scheduler.deliver_message("test_msg", {"seq": 2})
    scheduler.deliver_message("test_msg", {"seq": 3})

    # Waiters should get them in order
    r1 = await scheduler.wait_for_message("test_msg")
    r2 = await scheduler.wait_for_message("test_msg")
    r3 = await scheduler.wait_for_message("test_msg")

    assert r1["seq"] == 1
    assert r2["seq"] == 2
    assert r3["seq"] == 3


@pytest.mark.asyncio
async def test_messageawarenodescheduler_multiple_message_types_independent():
    """Verify different message types are handled independently."""
    scheduler = create_message_aware_scheduler()

    # Wait for different message types
    task1 = asyncio.create_task(scheduler.wait_for_message("type1"))
    task2 = asyncio.create_task(scheduler.wait_for_message("type2"))

    await asyncio.sleep(0.01)

    # Deliver only type1
    scheduler.deliver_message("type1", {"data": 1})

    # Only task1 should complete
    result1 = await task1
    assert result1 == {"data": 1}

    # task2 should still be waiting
    assert not task2.done()

    # Deliver type2
    scheduler.deliver_message("type2", {"data": 2})
    result2 = await task2
    assert result2 == {"data": 2}


# Category 5: Complex Message-Driven Scenarios


@pytest.mark.asyncio
async def test_graph_with_multiple_message_inputs():
    """Verify graph can wait for multiple message inputs."""
    from byzpy.engine.graph.graph import GraphInput

    class _AddOp(Operator):
        def compute(self, inputs, *, context):
            return inputs["a"] + inputs["b"]

    msg_a = GraphInput.from_message("msg_a")
    msg_b = GraphInput.from_message("msg_b")
    op = _AddOp()
    node = GraphNode(name="sum", op=op, inputs={"a": msg_a, "b": msg_b})
    graph = ComputationGraph(nodes=[node], outputs=["sum"])

    scheduler = MessageAwareNodeScheduler(graph)

    exec_task = asyncio.create_task(scheduler.run({}))
    await asyncio.sleep(0.01)

    # Deliver messages in any order
    scheduler.deliver_message("msg_b", 20)
    await asyncio.sleep(0.01)
    scheduler.deliver_message("msg_a", 10)

    result = await exec_task
    assert result["sum"] == 30


@pytest.mark.asyncio
async def test_graph_with_message_and_regular_inputs():
    """Verify graph can mix message inputs with regular inputs."""
    from byzpy.engine.graph.graph import GraphInput

    class _MultiplyOp(Operator):
        def compute(self, inputs, *, context):
            return inputs["x"] * inputs["y"]

    msg_x = GraphInput.from_message("msg_x")
    regular_y = graph_input("y")
    op = _MultiplyOp()
    node = GraphNode(name="mult", op=op, inputs={"x": msg_x, "y": regular_y})
    graph = ComputationGraph(nodes=[node], outputs=["mult"])

    scheduler = MessageAwareNodeScheduler(graph)

    exec_task = asyncio.create_task(scheduler.run({"y": 5}))
    await asyncio.sleep(0.01)

    scheduler.deliver_message("msg_x", 4)

    result = await exec_task
    assert result["mult"] == 20


@pytest.mark.asyncio
async def test_graph_with_message_dependent_nodes():
    """Verify nodes dependent on message inputs wait correctly."""
    from byzpy.engine.graph.graph import GraphInput

    class _DoubleOp(Operator):
        def compute(self, inputs, *, context):
            return inputs["x"] * 2

    class _AddOp(Operator):
        def compute(self, inputs, *, context):
            return inputs["a"] + inputs["b"]

    # Node 1: waits for message
    msg_input = GraphInput.from_message("data_msg")
    double_op = _DoubleOp()
    node1 = GraphNode(name="double", op=double_op, inputs={"x": msg_input})

    # Node 2: depends on node1
    add_op = _AddOp()
    node2 = GraphNode(name="sum", op=add_op, inputs={"a": "double", "b": graph_input("bias")})

    graph = ComputationGraph(nodes=[node1, node2], outputs=["sum"])
    scheduler = MessageAwareNodeScheduler(graph)

    exec_task = asyncio.create_task(scheduler.run({"bias": 10}))
    await asyncio.sleep(0.01)

    # Deliver message
    scheduler.deliver_message("data_msg", 5)

    result = await exec_task
    assert result["sum"] == 20  # (5 * 2) + 10


@pytest.mark.asyncio
async def test_messageawarenodescheduler_concurrent_deliveries():
    """Verify concurrent message deliveries are handled correctly."""
    scheduler = create_message_aware_scheduler()

    # Create multiple waiters
    waiters = [asyncio.create_task(scheduler.wait_for_message(f"msg_{i}")) for i in range(10)]

    await asyncio.sleep(0.01)

    # Deliver all messages concurrently
    for i in range(10):
        scheduler.deliver_message(f"msg_{i}", {"id": i})

    # All waiters should complete
    results = await asyncio.gather(*waiters)
    assert len(results) == 10
    assert all(r["id"] == i for i, r in enumerate(results))


@pytest.mark.asyncio
async def test_messageawarenodescheduler_wait_cancellation():
    """Verify wait_for_message() can be cancelled."""
    scheduler = create_message_aware_scheduler()

    wait_task = asyncio.create_task(scheduler.wait_for_message("test_msg"))
    await asyncio.sleep(0.01)

    # Cancel the wait
    wait_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await wait_task
