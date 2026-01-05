from __future__ import annotations

import asyncio

import pytest
import torch

from byzpy.engine.graph.pool import ActorPoolConfig
from byzpy.engine.node.application import NodeApplication
from byzpy.engine.node.context import NodeContext, ProcessContext
from byzpy.engine.node.decentralized import DecentralizedNode


def create_test_application():
    """Create a test NodeApplication."""
    return NodeApplication(
        name="test-app",
        actor_pool=[ActorPoolConfig(backend="thread", count=1)],
    )


def create_test_node(node_id: str = "test-node"):
    """Create a test DecentralizedNode."""
    from byzpy.engine.node.context import InProcessContext

    return DecentralizedNode(
        node_id=node_id,
        application=create_test_application(),
        context=InProcessContext(),
    )


# Category 1: ProcessContext Core Functionality


def test_processcontext_can_be_created():
    """Verify ProcessContext can be instantiated."""
    ctx = ProcessContext()
    assert ctx is not None
    assert isinstance(ctx, NodeContext)
    assert not ctx._running


@pytest.mark.asyncio
async def test_processcontext_start_creates_process():
    """Verify start() creates and starts a process."""
    ctx = ProcessContext()
    node = create_test_node()

    await ctx.start(node)

    assert ctx._running
    assert ctx._process is not None
    assert ctx._process.is_alive()

    await ctx.shutdown()


@pytest.mark.asyncio
async def test_processcontext_start_stores_node_reference():
    """Verify start() stores node reference in process."""
    ctx = ProcessContext()
    node = create_test_node()

    await ctx.start(node)

    # Node ID should be stored
    assert ctx._node_id == node.node_id

    await ctx.shutdown()


@pytest.mark.asyncio
async def test_processcontext_shutdown_terminates_process():
    """Verify shutdown() terminates the process."""
    ctx = ProcessContext()
    node = create_test_node()
    await ctx.start(node)

    process = ctx._process
    assert process.is_alive()

    await ctx.shutdown()

    # Process should be terminated
    assert not process.is_alive()
    assert not ctx._running


@pytest.mark.asyncio
async def test_processcontext_restart():
    """Verify ProcessContext can restart after shutdown."""
    ctx = ProcessContext()
    node = create_test_node()

    await ctx.start(node)
    process_id1 = ctx._process.pid

    await ctx.shutdown()

    # Restart
    await ctx.start(node)
    process_id2 = ctx._process.pid

    # Should be a new process
    assert process_id1 != process_id2
    assert ctx._process.is_alive()

    await ctx.shutdown()


@pytest.mark.asyncio
async def test_processcontext_start_idempotent():
    """Verify start() is idempotent."""
    ctx = ProcessContext()
    node = create_test_node()

    await ctx.start(node)
    process_id1 = ctx._process.pid

    # Start again should not create new process
    await ctx.start(node)
    process_id2 = ctx._process.pid

    assert process_id1 == process_id2

    await ctx.shutdown()


@pytest.mark.asyncio
async def test_processcontext_shutdown_idempotent():
    """Verify shutdown() is idempotent."""
    ctx = ProcessContext()
    node = create_test_node()

    await ctx.start(node)
    await ctx.shutdown()

    # Shutdown again should not error
    await ctx.shutdown()

    assert not ctx._running


@pytest.mark.asyncio
async def test_processcontext_send_message_when_not_started():
    """Verify send_message() raises error when not started."""
    ctx = ProcessContext()

    with pytest.raises(RuntimeError, match="not started"):
        await ctx.send_message("target", "test", {})


@pytest.mark.asyncio
async def test_processcontext_receive_messages_when_not_started():
    """Verify receive_messages() raises error when not started."""
    ctx = ProcessContext()

    with pytest.raises((RuntimeError, StopAsyncIteration)):
        async for msg in ctx.receive_messages():
            break
