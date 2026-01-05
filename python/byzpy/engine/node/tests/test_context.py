from __future__ import annotations

import asyncio

import pytest

from byzpy.engine.node.context import InProcessContext, NodeContext


class MockDecentralizedNode:
    """Mock node for testing contexts."""

    def __init__(self, node_id: str = "test-node"):
        self.node_id = node_id
        self._inbox = asyncio.Queue()
        self._running = False


# Category 1: NodeContext Interface Tests


@pytest.mark.asyncio
async def test_nodecontext_is_abstract_base_class():
    """Verify NodeContext cannot be instantiated directly."""
    with pytest.raises(TypeError):
        NodeContext()


def test_nodecontext_requires_interface_methods():
    """Verify all required methods are defined in interface."""
    required_methods = ["start", "send_message", "receive_messages", "shutdown"]
    for method in required_methods:
        assert hasattr(NodeContext, method)
        # Verify it's abstract by checking if it raises NotImplementedError
        ctx = InProcessContext()  # Concrete implementation
        # The abstract methods should exist on the base class
        assert callable(getattr(NodeContext, method))


# Category 2: InProcessContext Implementation Tests


def test_inprocesscontext_can_be_instantiated():
    """Verify InProcessContext can be created."""
    ctx = InProcessContext()
    assert ctx is not None
    assert isinstance(ctx, NodeContext)


@pytest.mark.asyncio
async def test_inprocesscontext_start_stores_node_reference():
    """Verify start() stores node reference."""
    ctx = InProcessContext()
    node = MockDecentralizedNode()

    await ctx.start(node)

    assert ctx._node is node


@pytest.mark.asyncio
async def test_inprocesscontext_send_message_enqueues():
    """Verify send_message() enqueues message to target node's inbox."""
    ctx1 = InProcessContext()
    ctx2 = InProcessContext()
    node1 = MockDecentralizedNode("node1")
    node2 = MockDecentralizedNode("node2")
    await ctx1.start(node1)
    await ctx2.start(node2)

    # Send from ctx1 to ctx2
    await ctx1.send_message("node2", "test", "data")

    # Message should be in ctx2's inbox
    assert not ctx2._inbox.empty()
    received = await ctx2._inbox.get()
    assert received == {"from": "node1", "type": "test", "payload": "data"}


@pytest.mark.asyncio
async def test_inprocesscontext_receive_messages_yields_from_inbox():
    """Verify receive_messages() yields messages from inbox."""
    ctx1 = InProcessContext()
    ctx2 = InProcessContext()
    node1 = MockDecentralizedNode("node1")
    node2 = MockDecentralizedNode("node2")
    await ctx1.start(node1)
    await ctx2.start(node2)

    # Send messages from ctx1 to ctx2
    await ctx1.send_message("node2", "msg1", 1)
    await ctx1.send_message("node2", "msg2", 2)

    received = []
    async for msg in ctx2.receive_messages():
        received.append(msg)
        if len(received) == 2:
            break

    assert len(received) == 2
    assert received[0]["type"] == "msg1"
    assert received[1]["type"] == "msg2"


@pytest.mark.asyncio
async def test_inprocesscontext_shutdown_cleans_up():
    """Verify shutdown() cleans up resources."""
    ctx = InProcessContext()
    node = MockDecentralizedNode()
    await ctx.start(node)

    await ctx.shutdown()

    # Verify cleanup
    assert ctx._running is False
    assert ctx._node is None


@pytest.mark.asyncio
async def test_inprocesscontext_handles_concurrent_messages():
    """Verify context handles concurrent message sends."""
    ctx1 = InProcessContext()
    ctx2 = InProcessContext()
    node1 = MockDecentralizedNode("node1")
    node2 = MockDecentralizedNode("node2")
    await ctx1.start(node1)
    await ctx2.start(node2)

    # Send multiple messages concurrently from ctx1 to ctx2
    tasks = [ctx1.send_message("node2", f"msg{i}", i) for i in range(10)]
    await asyncio.gather(*tasks)

    # All messages should be received
    count = 0
    async for msg in ctx2.receive_messages():
        count += 1
        if count == 10:
            break

    assert count == 10


@pytest.mark.asyncio
async def test_inprocesscontext_send_message_raises_when_not_started():
    """Verify send_message() raises error when context not started."""
    ctx = InProcessContext()

    with pytest.raises(RuntimeError, match="not started"):
        await ctx.send_message("target", "test", "data")
