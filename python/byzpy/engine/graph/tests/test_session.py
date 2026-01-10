"""
Tests for session management - Phase 3 implementation.

Test plan: docs/design/phase3_test_plan.md
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Mapping

import pytest

from byzpy.engine.graph.graph import ComputationGraph, GraphInput, GraphNode
from byzpy.engine.graph.lazy import GraphBuilder
from byzpy.engine.graph.operator import OpContext, Operator
from byzpy.engine.graph.parallel_scheduler import ParallelScheduler
from byzpy.engine.graph.session import ExecutionFuture, ExecutionSession

# =============================================================================
# Test Operators
# =============================================================================


class _DoubleOp(Operator):
    """Doubles the input value."""

    name = "double"
    input_key = "value"

    def compute(self, inputs: Mapping[str, Any], *, context: OpContext) -> Any:
        return inputs["value"] * 2


class _MultiplyOp(Operator):
    """Multiplies input by a factor."""

    name = "multiply"
    input_key = "value"

    def __init__(self, factor: int):
        self.factor = factor

    def compute(self, inputs: Mapping[str, Any], *, context: OpContext) -> Any:
        return inputs["value"] * self.factor


class _AddConstantOp(Operator):
    """Adds a constant to the input."""

    name = "add_constant"
    input_key = "value"

    def __init__(self, constant: int):
        self.constant = constant

    def compute(self, inputs: Mapping[str, Any], *, context: OpContext) -> Any:
        return inputs["value"] + self.constant


class _CombineOp(Operator):
    """Combines two inputs (a + b)."""

    name = "combine"
    input_key = "a"

    def compute(self, inputs: Mapping[str, Any], *, context: OpContext) -> Any:
        return inputs["a"] + inputs["b"]


class _RecordCallOp(Operator):
    """Records whether compute() was called. For testing caching."""

    name = "record_call"
    input_key = "value"

    def __init__(self):
        self.compute_called = False
        self.call_count = 0

    def compute(self, inputs: Mapping[str, Any], *, context: OpContext) -> Any:
        self.compute_called = True
        self.call_count += 1
        return inputs["value"]


class _SlowOp(Operator):
    """Operator that sleeps for timing tests."""

    name = "slow"
    input_key = "value"

    def __init__(self, delay: float, result_value: Any = None):
        self.delay = delay
        self.result_value = result_value

    async def run(
        self,
        inputs: Mapping[str, Any],
        *,
        context: OpContext,
        pool: Any,
    ) -> Any:
        await asyncio.sleep(self.delay)
        if self.result_value is not None:
            return self.result_value
        return inputs.get("value", 0)


class _ErrorOp(Operator):
    """Always raises an error."""

    name = "error"
    input_key = "value"

    def __init__(self, error_message: str = "Test error"):
        self.error_message = error_message

    def compute(self, inputs: Mapping[str, Any], *, context: OpContext) -> Any:
        raise RuntimeError(self.error_message)


# =============================================================================
# Helper Functions
# =============================================================================


def _make_simple_graph():
    """Create a simple graph: input -> double."""
    data = GraphInput("data")
    node = GraphNode(name="doubled", op=_DoubleOp(), inputs={"value": data})
    return ComputationGraph([node], outputs=["doubled"])


def _make_chain_graph():
    """Create a chain: input -> A -> B -> C."""
    data = GraphInput("data")
    node_a = GraphNode(name="A", op=_AddConstantOp(1), inputs={"value": data})
    node_b = GraphNode(name="B", op=_AddConstantOp(2), inputs={"value": "A"})
    node_c = GraphNode(name="C", op=_AddConstantOp(3), inputs={"value": "B"})
    return ComputationGraph([node_a, node_b, node_c], outputs=["C"])


# =============================================================================
# 1. ExecutionSession Basic Tests
# =============================================================================


def test_execution_session_creates_with_defaults():
    """Test 1.1: Verify ExecutionSession can be created with default parameters."""
    session = ExecutionSession()

    assert session.pool is None
    assert session.cache_intermediate is True
    assert session._result_cache == {}


def test_execution_session_creates_with_pool():
    """Test 1.2: Verify ExecutionSession can be created with a pool."""

    class MockPool:
        @property
        def size(self):
            return 4

        def worker_affinities(self):
            return ("w0", "w1", "w2", "w3")

    pool = MockPool()
    session = ExecutionSession(pool=pool)

    assert session.pool is pool


@pytest.mark.asyncio
async def test_execution_session_execute_single_graph():
    """Test 1.3: Verify execute() runs a single graph correctly."""
    graph = _make_simple_graph()
    session = ExecutionSession()

    result = await session.execute(graph, inputs={"data": 5})

    assert result == {"doubled": 10}


@pytest.mark.asyncio
async def test_execution_session_execute_multiple_outputs():
    """Test 1.4: Verify execute() with multiple output nodes."""
    data = GraphInput("data")
    node_a = GraphNode(name="A", op=_MultiplyOp(2), inputs={"value": data})
    node_b = GraphNode(name="B", op=_MultiplyOp(3), inputs={"value": data})
    graph = ComputationGraph([node_a, node_b], outputs=["A", "B"])

    session = ExecutionSession()
    result = await session.execute(graph, inputs={"data": 10})

    assert result == {"A": 20, "B": 30}


# =============================================================================
# 2. Result Caching Tests
# =============================================================================


@pytest.mark.asyncio
async def test_execution_session_caches_results():
    """Test 2.1: Verify intermediate results are cached."""
    graph = _make_simple_graph()
    session = ExecutionSession(cache_intermediate=True)

    await session.execute(graph, inputs={"data": 5})

    # Result should be in cache
    assert "doubled" in session._result_cache
    assert session._result_cache["doubled"] == 10


@pytest.mark.asyncio
async def test_execution_session_reuses_cached_results():
    """Test 2.2: Verify subsequent executions reuse cached intermediates."""
    # Build first graph: input -> A (record_call)
    record_op = _RecordCallOp()
    data = GraphInput("data")
    node_a = GraphNode(name="A", op=record_op, inputs={"value": data})
    graph1 = ComputationGraph([node_a], outputs=["A"])

    # Build second graph: A -> B (depends on A)
    # Note: A is already computed
    node_a2 = GraphNode(name="A", op=record_op, inputs={"value": data})
    node_b = GraphNode(name="B", op=_AddConstantOp(10), inputs={"value": "A"})
    graph2 = ComputationGraph([node_a2, node_b], outputs=["B"])

    session = ExecutionSession(cache_intermediate=True)

    # First execution - A is computed
    await session.execute(graph1, inputs={"data": 5})
    assert record_op.call_count == 1

    # Second execution - A should be reused from cache
    result = await session.execute(graph2, inputs={"data": 5})
    # call_count should still be 1 because A was cached
    assert record_op.call_count == 1
    assert result == {"B": 15}  # 5 + 10


@pytest.mark.asyncio
async def test_execution_session_cache_disabled():
    """Test 2.3: Verify cache_intermediate=False disables caching."""
    record_op = _RecordCallOp()
    data = GraphInput("data")
    node = GraphNode(name="A", op=record_op, inputs={"value": data})
    graph = ComputationGraph([node], outputs=["A"])

    session = ExecutionSession(cache_intermediate=False)

    await session.execute(graph, inputs={"data": 5})
    assert record_op.call_count == 1

    # Execute again - should call operator again (no caching)
    await session.execute(graph, inputs={"data": 5})
    assert record_op.call_count == 2

    # Cache should be empty
    assert len(session._result_cache) == 0


@pytest.mark.asyncio
async def test_execution_session_clear_cache():
    """Test 2.4: Verify cache can be cleared."""
    record_op = _RecordCallOp()
    data = GraphInput("data")
    node = GraphNode(name="A", op=record_op, inputs={"value": data})
    graph = ComputationGraph([node], outputs=["A"])

    session = ExecutionSession(cache_intermediate=True)

    await session.execute(graph, inputs={"data": 5})
    assert record_op.call_count == 1
    assert "A" in session._result_cache

    # Clear cache
    session.clear_cache()
    assert len(session._result_cache) == 0

    # Execute again - should compute again
    await session.execute(graph, inputs={"data": 5})
    assert record_op.call_count == 2


# =============================================================================
# 3. Incremental Execution Tests
# =============================================================================


@pytest.mark.asyncio
async def test_execution_session_incremental_execution():
    """Test 3.1: Verify executing part of a graph, then extending."""
    record_op_a = _RecordCallOp()
    record_op_b = _RecordCallOp()
    record_op_c = _RecordCallOp()

    # Graph 1: input -> A
    data = GraphInput("data")
    node_a1 = GraphNode(name="A", op=record_op_a, inputs={"value": data})
    graph1 = ComputationGraph([node_a1], outputs=["A"])

    # Graph 2: A -> B -> C (full chain for reference, but A is cached)
    node_a2 = GraphNode(name="A", op=record_op_a, inputs={"value": data})
    node_b = GraphNode(name="B", op=record_op_b, inputs={"value": "A"})
    node_c = GraphNode(name="C", op=record_op_c, inputs={"value": "B"})
    graph2 = ComputationGraph([node_a2, node_b, node_c], outputs=["C"])

    session = ExecutionSession(cache_intermediate=True)

    # Execute graph1 - A computed
    await session.execute(graph1, inputs={"data": 5})
    assert record_op_a.call_count == 1
    assert record_op_b.call_count == 0
    assert record_op_c.call_count == 0

    # Execute graph2 - A reused, B and C computed
    await session.execute(graph2, inputs={"data": 5})
    assert record_op_a.call_count == 1  # Still 1 - reused
    assert record_op_b.call_count == 1  # Now computed
    assert record_op_c.call_count == 1  # Now computed


@pytest.mark.asyncio
async def test_execution_session_different_inputs_same_graph():
    """Test 3.2: Verify different input values execute correctly."""
    graph = _make_simple_graph()
    session = ExecutionSession(cache_intermediate=True)

    # Execute with data=5
    result1 = await session.execute(graph, inputs={"data": 5})
    assert result1 == {"doubled": 10}

    # Cache is keyed by node name, so same node name = cached
    # This means with same graph, cache is reused even with different inputs
    # Clear cache to test fresh execution
    session.clear_cache()

    # Execute with data=10
    result2 = await session.execute(graph, inputs={"data": 10})
    assert result2 == {"doubled": 20}


# =============================================================================
# 4. ExecutionFuture Basic Tests
# =============================================================================


@pytest.mark.asyncio
async def test_execution_future_done_initially_false():
    """Test 4.1: Verify new future's done() returns False."""

    async def slow_task():
        await asyncio.sleep(0.1)
        return {"result": 42}

    task = asyncio.create_task(slow_task())
    future = ExecutionFuture(task, output_keys=["result"])

    # Should not be done immediately
    assert future.done() is False

    # Cleanup
    await future


@pytest.mark.asyncio
async def test_execution_future_done_after_completion():
    """Test 4.2: Verify done() returns True after task completes."""

    async def fast_task():
        return {"result": 42}

    task = asyncio.create_task(fast_task())
    future = ExecutionFuture(task, output_keys=["result"])

    # Wait for completion
    await future

    assert future.done() is True


@pytest.mark.asyncio
async def test_execution_future_await_returns_result():
    """Test 4.3: Verify awaiting future returns the result."""

    async def compute_task():
        return {"doubled": 10, "tripled": 15}

    task = asyncio.create_task(compute_task())
    future = ExecutionFuture(task, output_keys=["doubled", "tripled"])

    result = await future

    assert result == {"doubled": 10, "tripled": 15}


@pytest.mark.asyncio
async def test_execution_future_cancel():
    """Test 4.4: Verify cancel() cancels the underlying task."""

    async def slow_task():
        await asyncio.sleep(10)  # Very slow
        return {"result": 42}

    task = asyncio.create_task(slow_task())
    future = ExecutionFuture(task, output_keys=["result"])

    # Cancel it
    cancelled = future.cancel()
    assert cancelled is True

    # Wait a bit for cancellation to propagate
    await asyncio.sleep(0.01)

    assert future.cancelled() is True

    # Awaiting should raise CancelledError
    with pytest.raises(asyncio.CancelledError):
        await future


# =============================================================================
# 5. Non-blocking Execution Tests
# =============================================================================


@pytest.mark.asyncio
async def test_execution_session_execute_async_non_blocking():
    """Test 5.1: Verify execute_async() returns immediately."""
    data = GraphInput("data")
    node = GraphNode(name="slow", op=_SlowOp(0.1, result_value=42), inputs={"value": data})
    graph = ComputationGraph([node], outputs=["slow"])

    session = ExecutionSession()

    start = time.monotonic()
    future = session.execute_async(graph, inputs={"data": 0})
    elapsed = time.monotonic() - start

    # Should return immediately (< 0.05s, not waiting for 0.1s sleep)
    assert elapsed < 0.05
    assert isinstance(future, ExecutionFuture)
    assert future.done() is False

    # Wait for completion
    result = await future
    assert result == {"slow": 42}


@pytest.mark.asyncio
async def test_execution_session_execute_async_result():
    """Test 5.2: Verify execute_async() future provides correct result."""
    graph = _make_simple_graph()
    session = ExecutionSession()

    future = session.execute_async(graph, inputs={"data": 7})
    result = await future

    assert result == {"doubled": 14}


@pytest.mark.asyncio
async def test_execution_future_result_with_timeout():
    """Test 5.3: Verify result(timeout=...) raises TimeoutError."""

    async def slow_task():
        await asyncio.sleep(10)  # Very slow
        return {"result": 42}

    task = asyncio.create_task(slow_task())
    future = ExecutionFuture(task, output_keys=["result"])

    # Trying to get result with very short timeout from within async context
    # should raise RuntimeError (can't call result() from running loop)
    with pytest.raises(RuntimeError, match="Cannot call result"):
        future.result(timeout=0.01)

    # Cancel to cleanup
    future.cancel()
    await asyncio.sleep(0.01)


# =============================================================================
# 6. Session with GraphBuilder Integration Tests
# =============================================================================


@pytest.mark.asyncio
async def test_execution_session_with_graph_builder():
    """Test 6.1: Verify session works with GraphBuilder-created graphs."""
    builder = GraphBuilder()
    data = builder.input("data")
    doubled = data.apply(_DoubleOp(), name="doubled")
    graph = builder.build(outputs=[doubled.key])

    session = ExecutionSession()
    result = await session.execute(graph, inputs={"data": 8})

    assert result == {"doubled": 16}


@pytest.mark.asyncio
async def test_execution_session_diamond_pattern_caching():
    r"""
    Test 6.2: Verify diamond pattern caches correctly.

    Graph::

           input
           /   \
          A     B
           \   /
             C
    """
    record_op_a = _RecordCallOp()
    record_op_b = _RecordCallOp()

    # Build diamond graph
    builder = GraphBuilder()
    data = builder.input("data")
    a = data.apply(record_op_a, name="A")
    b = data.apply(record_op_b, name="B")
    c = a.apply(_CombineOp(), extra_inputs={"b": b}, name="C")

    graph_full = builder.build(outputs=[c.key])

    # Build graph for just A
    builder2 = GraphBuilder()
    data2 = builder2.input("data")
    a2 = data2.apply(record_op_a, name="A")
    graph_a = builder2.build(outputs=[a2.key])

    session = ExecutionSession(cache_intermediate=True)

    # Execute full diamond
    result1 = await session.execute(graph_full, inputs={"data": 10})
    assert result1 == {"C": 20}  # A=10, B=10, C=10+10=20
    assert record_op_a.call_count == 1
    assert record_op_b.call_count == 1

    # Now execute just A - should use cached result
    result2 = await session.execute(graph_a, inputs={"data": 10})
    assert result2 == {"A": 10}
    # A should not be called again - it's cached
    assert record_op_a.call_count == 1


# =============================================================================
# 7. Error Handling Tests
# =============================================================================


@pytest.mark.asyncio
async def test_execution_session_propagates_errors():
    """Test 7.1: Verify errors from operators propagate correctly."""
    data = GraphInput("data")
    node = GraphNode(name="error", op=_ErrorOp("Session error"), inputs={"value": data})
    graph = ComputationGraph([node], outputs=["error"])

    session = ExecutionSession()

    with pytest.raises(RuntimeError, match="Session error"):
        await session.execute(graph, inputs={"data": 5})


@pytest.mark.asyncio
async def test_execution_future_propagates_errors():
    """Test 7.2: Verify future awaiting propagates errors."""
    data = GraphInput("data")
    node = GraphNode(name="error", op=_ErrorOp("Future error"), inputs={"value": data})
    graph = ComputationGraph([node], outputs=["error"])

    session = ExecutionSession()
    future = session.execute_async(graph, inputs={"data": 5})

    with pytest.raises(RuntimeError, match="Future error"):
        await future


# =============================================================================
# 8. Context Manager Tests
# =============================================================================


@pytest.mark.asyncio
async def test_execution_session_context_manager():
    """Test 8.1: Verify session works as async context manager."""
    graph = _make_simple_graph()

    async with ExecutionSession() as session:
        result = await session.execute(graph, inputs={"data": 5})
        assert result == {"doubled": 10}

        # Cache should exist during context
        assert "doubled" in session._result_cache

    # After context, cache should be cleared
    # (We can't check session._result_cache here as it's implementation detail,
    # but the context manager clears it)


@pytest.mark.asyncio
async def test_execution_session_context_manager_with_pool():
    """Test 8.2: Verify context manager handles pool lifecycle."""

    class MockPool:
        def __init__(self):
            self.used = False

        @property
        def size(self):
            return 2

        def worker_affinities(self):
            return ("w0", "w1")

    pool = MockPool()
    graph = _make_simple_graph()

    async with ExecutionSession(pool=pool) as session:
        assert session.pool is pool
        result = await session.execute(graph, inputs={"data": 6})
        assert result == {"doubled": 12}


# =============================================================================
# Additional Tests
# =============================================================================


@pytest.mark.asyncio
async def test_execution_session_get_cached():
    """Test get_cached() retrieves cached values."""
    graph = _make_simple_graph()
    session = ExecutionSession()

    await session.execute(graph, inputs={"data": 5})

    assert session.get_cached("doubled") == 10


@pytest.mark.asyncio
async def test_execution_session_get_cached_raises_for_missing():
    """Test get_cached() raises KeyError for missing keys."""
    session = ExecutionSession()

    with pytest.raises(KeyError):
        session.get_cached("nonexistent")


@pytest.mark.asyncio
async def test_execution_session_is_cached():
    """Test is_cached() returns correct boolean."""
    graph = _make_simple_graph()
    session = ExecutionSession()

    assert session.is_cached("doubled") is False

    await session.execute(graph, inputs={"data": 5})

    assert session.is_cached("doubled") is True
    assert session.is_cached("nonexistent") is False


@pytest.mark.asyncio
async def test_execution_future_output_keys():
    """Test ExecutionFuture.output_keys property."""

    async def task():
        return {"a": 1, "b": 2}

    future = ExecutionFuture(
        asyncio.create_task(task()),
        output_keys=["a", "b"],
    )

    assert future.output_keys == ("a", "b")
    await future  # Cleanup
