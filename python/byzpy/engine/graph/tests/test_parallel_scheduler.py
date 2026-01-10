"""
Tests for ParallelScheduler - Phase 1 implementation.

Test plan: docs/design/phase1_test_plan.md
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Mapping, cast

import pytest

from byzpy.engine.graph.graph import ComputationGraph, GraphNode, graph_input
from byzpy.engine.graph.operator import OpContext, Operator
from byzpy.engine.graph.parallel_scheduler import ParallelScheduler
from byzpy.engine.graph.scheduler import NodeScheduler

# =============================================================================
# Test Operators
# =============================================================================


class _AddOp(Operator):
    """Adds two inputs together."""

    name = "add"

    def compute(self, inputs: Mapping[str, Any], *, context: OpContext) -> Any:
        return inputs["lhs"] + inputs["rhs"]


class _DoubleOp(Operator):
    """Doubles the input value."""

    name = "double"

    def compute(self, inputs: Mapping[str, Any], *, context: OpContext) -> Any:
        return inputs["value"] * 2


class _RecordMetadataOp(Operator):
    """Records metadata from context for verification."""

    name = "record_metadata"

    def __init__(self):
        self.last_metadata = None

    def compute(self, inputs: Mapping[str, Any], *, context: OpContext) -> Any:
        self.last_metadata = context.metadata
        return inputs["value"] * 2


class _SlowOp(Operator):
    """Operator that sleeps for a specified duration."""

    name = "slow"

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


class _TimestampOp(Operator):
    """Records execution timestamp for ordering verification."""

    name = "timestamp"

    def __init__(self):
        self.execution_time = None

    def compute(self, inputs: Mapping[str, Any], *, context: OpContext) -> Any:
        self.execution_time = time.monotonic()
        return inputs.get("value", 0)


class _AddConstantOp(Operator):
    """Adds a constant to the input."""

    name = "add_constant"

    def __init__(self, constant: int):
        self.constant = constant

    def compute(self, inputs: Mapping[str, Any], *, context: OpContext) -> Any:
        return inputs["value"] + self.constant


class _MultiplyOp(Operator):
    """Multiplies input by a factor."""

    name = "multiply"

    def __init__(self, factor: int):
        self.factor = factor

    def compute(self, inputs: Mapping[str, Any], *, context: OpContext) -> Any:
        return inputs["value"] * self.factor


class _CombineOp(Operator):
    """Combines two inputs into a sum."""

    name = "combine"

    def compute(self, inputs: Mapping[str, Any], *, context: OpContext) -> Any:
        return inputs["a"] + inputs["b"]


class _CombineFourOp(Operator):
    """Combines four inputs into a sum."""

    name = "combine_four"

    def compute(self, inputs: Mapping[str, Any], *, context: OpContext) -> Any:
        return inputs["a"] + inputs["b"] + inputs["c"] + inputs["d"]


class _ErrorOp(Operator):
    """Always raises an error."""

    name = "error"

    def __init__(self, error_message: str = "Test error"):
        self.error_message = error_message

    def compute(self, inputs: Mapping[str, Any], *, context: OpContext) -> Any:
        raise RuntimeError(self.error_message)


class _RecordPoolOp(Operator):
    """Records whether pool was passed."""

    name = "record_pool"

    def __init__(self):
        self.received_pool = None
        self.received_metadata = None

    async def run(
        self,
        inputs: Mapping[str, Any],
        *,
        context: OpContext,
        pool: Any,
    ) -> Any:
        self.received_pool = pool
        self.received_metadata = context.metadata
        return inputs.get("value", 0)


# =============================================================================
# Helper Functions
# =============================================================================


def _make_basic_graph():
    """Create a basic graph: input → double → sum (with bias)."""
    data = graph_input("data")
    bias = graph_input("bias")
    doubler = GraphNode(name="double", op=_RecordMetadataOp(), inputs={"value": data})
    summer = GraphNode(
        name="sum",
        op=_AddOp(),
        inputs={"lhs": "double", "rhs": bias},
    )
    graph = ComputationGraph([doubler, summer], outputs=["sum", "double"])
    return graph, doubler, summer


# =============================================================================
# 1. Basic Functionality Tests
# =============================================================================


@pytest.mark.asyncio
async def test_parallel_scheduler_runs_graph_and_returns_outputs():
    """Test 1.1: Verify scheduler executes graph and returns correct outputs."""
    graph, _, _ = _make_basic_graph()
    scheduler = ParallelScheduler(graph, metadata={"phase": "train"})

    outputs = await scheduler.run({"data": 5, "bias": 3})

    assert outputs == {"sum": 13, "double": 10}


@pytest.mark.asyncio
async def test_parallel_scheduler_raises_for_missing_inputs():
    """Test 1.2: Verify proper error handling when required inputs are missing."""
    graph, _, _ = _make_basic_graph()
    scheduler = ParallelScheduler(graph)

    with pytest.raises(ValueError, match="Missing graph inputs"):
        await scheduler.run({"data": 1})


@pytest.mark.asyncio
async def test_parallel_scheduler_passes_metadata_into_context():
    """Test 1.3: Verify metadata is correctly passed through OpContext."""
    graph, double_node, _ = _make_basic_graph()
    doubler = cast(_RecordMetadataOp, double_node.op)
    scheduler = ParallelScheduler(graph, metadata={"stage": "eval"})

    await scheduler.run({"data": 4, "bias": 0})

    assert doubler.last_metadata == {"stage": "eval"}


# =============================================================================
# 2. Parallelism Tests
# =============================================================================


@pytest.mark.asyncio
async def test_parallel_scheduler_executes_independent_nodes_concurrently():
    r"""
    Test 2.1: Verify independent nodes run in parallel.

    Graph::

           input
           /   \
        slow1  slow2  (each sleeps 0.1s)
           \   /
          combine
    """
    data = graph_input("data")
    slow1 = GraphNode(
        name="slow1",
        op=_SlowOp(delay=0.1, result_value=1),
        inputs={"value": data},
    )
    slow2 = GraphNode(
        name="slow2",
        op=_SlowOp(delay=0.1, result_value=2),
        inputs={"value": data},
    )
    combine = GraphNode(
        name="combine",
        op=_CombineOp(),
        inputs={"a": "slow1", "b": "slow2"},
    )
    graph = ComputationGraph([slow1, slow2, combine], outputs=["combine"])
    scheduler = ParallelScheduler(graph)

    start = time.monotonic()
    result = await scheduler.run({"data": 0})
    elapsed = time.monotonic() - start

    # If parallel: ~0.1s; if sequential: ~0.2s
    # Allow some tolerance
    assert elapsed < 0.15, f"Expected parallel execution in ~0.1s, got {elapsed:.3f}s"
    assert result == {"combine": 3}


@pytest.mark.asyncio
async def test_parallel_scheduler_respects_dependencies():
    """
    Test 2.2: Verify dependent nodes wait for their dependencies.

    Graph:
        input
          |
         A (records timestamp)
          |
         B (records timestamp)
    """
    data = graph_input("data")
    op_a = _TimestampOp()
    op_b = _TimestampOp()
    node_a = GraphNode(name="A", op=op_a, inputs={"value": data})
    node_b = GraphNode(name="B", op=op_b, inputs={"value": "A"})
    graph = ComputationGraph([node_a, node_b], outputs=["B"])
    scheduler = ParallelScheduler(graph)

    await scheduler.run({"data": 1})

    # B must execute after A
    assert op_a.execution_time is not None
    assert op_b.execution_time is not None
    assert op_b.execution_time >= op_a.execution_time


@pytest.mark.asyncio
async def test_parallel_scheduler_handles_diamond_dependency():
    r"""
    Test 2.3: Verify diamond-shaped dependency patterns work correctly.

    Graph::

           input
           /   \
          A     B
           \   /
             C
    """
    data = graph_input("data")
    node_a = GraphNode(
        name="A",
        op=_MultiplyOp(factor=2),
        inputs={"value": data},
    )
    node_b = GraphNode(
        name="B",
        op=_MultiplyOp(factor=3),
        inputs={"value": data},
    )
    node_c = GraphNode(
        name="C",
        op=_CombineOp(),
        inputs={"a": "A", "b": "B"},
    )
    graph = ComputationGraph([node_a, node_b, node_c], outputs=["C"])
    scheduler = ParallelScheduler(graph)

    result = await scheduler.run({"data": 10})

    # A = 10 * 2 = 20
    # B = 10 * 3 = 30
    # C = 20 + 30 = 50
    assert result == {"C": 50}


# =============================================================================
# 3. Edge Cases
# =============================================================================


@pytest.mark.asyncio
async def test_parallel_scheduler_single_node_graph():
    """Test 3.1: Verify single-node graphs work correctly."""
    x = graph_input("x")
    node = GraphNode(name="double", op=_DoubleOp(), inputs={"value": x})
    graph = ComputationGraph([node], outputs=["double"])
    scheduler = ParallelScheduler(graph)

    result = await scheduler.run({"x": 5})

    assert result == {"double": 10}


@pytest.mark.asyncio
async def test_parallel_scheduler_linear_chain():
    """
    Test 3.2: Verify purely sequential chains work (no parallelism possible).

    Graph: input → A → B → C → D
    Each adds 1 to the input.
    """
    data = graph_input("data")
    node_a = GraphNode(name="A", op=_AddConstantOp(1), inputs={"value": data})
    node_b = GraphNode(name="B", op=_AddConstantOp(1), inputs={"value": "A"})
    node_c = GraphNode(name="C", op=_AddConstantOp(1), inputs={"value": "B"})
    node_d = GraphNode(name="D", op=_AddConstantOp(1), inputs={"value": "C"})
    graph = ComputationGraph([node_a, node_b, node_c, node_d], outputs=["D"])
    scheduler = ParallelScheduler(graph)

    result = await scheduler.run({"data": 1})

    # 1 + 1 + 1 + 1 + 1 = 5
    assert result == {"D": 5}


@pytest.mark.asyncio
async def test_parallel_scheduler_multiple_independent_chains():
    """
    Test 3.3: Verify multiple independent chains can run in parallel.

    Graph::

        input1 -> A1 -> B1  (chain 1)
        input2 -> A2 -> B2  (chain 2)
    """
    input1 = graph_input("input1")
    input2 = graph_input("input2")
    node_a1 = GraphNode(name="A1", op=_AddConstantOp(1), inputs={"value": input1})
    node_b1 = GraphNode(name="B1", op=_AddConstantOp(1), inputs={"value": "A1"})
    node_a2 = GraphNode(name="A2", op=_AddConstantOp(1), inputs={"value": input2})
    node_b2 = GraphNode(name="B2", op=_AddConstantOp(1), inputs={"value": "A2"})
    graph = ComputationGraph(
        [node_a1, node_b1, node_a2, node_b2],
        outputs=["B1", "B2"],
    )
    scheduler = ParallelScheduler(graph)

    result = await scheduler.run({"input1": 1, "input2": 100})

    # B1 = 1 + 1 + 1 = 3
    # B2 = 100 + 1 + 1 = 102
    assert result == {"B1": 3, "B2": 102}


@pytest.mark.asyncio
async def test_parallel_scheduler_fan_out_fan_in():
    r"""
    Test 3.4: Verify fan-out/fan-in patterns.

    Graph::

             input
           /  |  |  \
          A   B  C   D  (all parallel)
           \  |  |  /
             combine
    """
    data = graph_input("data")
    node_a = GraphNode(name="A", op=_AddConstantOp(1), inputs={"value": data})
    node_b = GraphNode(name="B", op=_AddConstantOp(2), inputs={"value": data})
    node_c = GraphNode(name="C", op=_AddConstantOp(3), inputs={"value": data})
    node_d = GraphNode(name="D", op=_AddConstantOp(4), inputs={"value": data})
    combine = GraphNode(
        name="combine",
        op=_CombineFourOp(),
        inputs={"a": "A", "b": "B", "c": "C", "d": "D"},
    )
    graph = ComputationGraph(
        [node_a, node_b, node_c, node_d, combine],
        outputs=["combine"],
    )
    scheduler = ParallelScheduler(graph)

    result = await scheduler.run({"data": 1})

    # A=2, B=3, C=4, D=5 → combine = 2+3+4+5 = 14
    assert result == {"combine": 14}


# =============================================================================
# 4. Concurrency Control Tests
# =============================================================================


@pytest.mark.asyncio
async def test_parallel_scheduler_max_concurrent_nodes():
    """
    Test 4.1: Verify max_concurrent_nodes parameter limits parallelism.

    4 independent slow nodes (each sleeps 0.1s).
    With limit=2: should take ~0.2s (2 batches of 2)
    Without limit: would take ~0.1s (all 4 parallel)
    """
    data = graph_input("data")
    slow1 = GraphNode(name="slow1", op=_SlowOp(0.1, result_value=1), inputs={"value": data})
    slow2 = GraphNode(name="slow2", op=_SlowOp(0.1, result_value=2), inputs={"value": data})
    slow3 = GraphNode(name="slow3", op=_SlowOp(0.1, result_value=3), inputs={"value": data})
    slow4 = GraphNode(name="slow4", op=_SlowOp(0.1, result_value=4), inputs={"value": data})
    graph = ComputationGraph(
        [slow1, slow2, slow3, slow4],
        outputs=["slow1", "slow2", "slow3", "slow4"],
    )
    scheduler = ParallelScheduler(graph, max_concurrent_nodes=2)

    start = time.monotonic()
    result = await scheduler.run({"data": 0})
    elapsed = time.monotonic() - start

    # With limit=2, 4 nodes should take ~0.2s (two batches)
    # Allow tolerance: should be > 0.15s and < 0.3s
    assert elapsed >= 0.15, f"Expected limited concurrency, but took only {elapsed:.3f}s"
    assert elapsed < 0.3, f"Expected ~0.2s with limit=2, got {elapsed:.3f}s"
    assert result == {"slow1": 1, "slow2": 2, "slow3": 3, "slow4": 4}


# =============================================================================
# 5. Pool Integration Tests
# =============================================================================


@pytest.mark.asyncio
async def test_parallel_scheduler_passes_pool_to_operators():
    """
    Test 5.1: Verify pool is passed to operators and metadata contains pool info.
    """

    # Create a simple mock pool
    class MockPool:
        @property
        def size(self):
            return 4

        def worker_affinities(self):
            return ("worker0", "worker1", "worker2", "worker3")

    data = graph_input("data")
    record_op = _RecordPoolOp()
    node = GraphNode(name="record", op=record_op, inputs={"value": data})
    graph = ComputationGraph([node], outputs=["record"])

    pool = MockPool()
    scheduler = ParallelScheduler(graph, pool=pool)

    await scheduler.run({"data": 42})

    assert record_op.received_pool is pool
    assert record_op.received_metadata["pool_size"] == 4
    assert record_op.received_metadata["worker_affinities"] == (
        "worker0",
        "worker1",
        "worker2",
        "worker3",
    )


# =============================================================================
# 6. Correctness Tests
# =============================================================================


@pytest.mark.asyncio
async def test_parallel_scheduler_produces_same_result_as_sequential():
    """Test 6.1: Verify ParallelScheduler and NodeScheduler produce identical results."""
    graph, _, _ = _make_basic_graph()

    # Run with NodeScheduler (sequential)
    sequential_scheduler = NodeScheduler(graph, metadata={"mode": "test"})
    sequential_result = await sequential_scheduler.run({"data": 7, "bias": 5})

    # Run with ParallelScheduler
    parallel_scheduler = ParallelScheduler(graph, metadata={"mode": "test"})
    parallel_result = await parallel_scheduler.run({"data": 7, "bias": 5})

    assert parallel_result == sequential_result


@pytest.mark.asyncio
async def test_parallel_scheduler_node_execution_order_preserves_data_dependencies():
    """
    Test 6.2: Verify data flows correctly through the graph.

    Graph: input → multiply_by_2 → add_10 → multiply_by_3
    Calculation: ((5 * 2) + 10) * 3 = 60
    """
    data = graph_input("data")
    mult2 = GraphNode(name="mult2", op=_MultiplyOp(2), inputs={"value": data})
    add10 = GraphNode(name="add10", op=_AddConstantOp(10), inputs={"value": "mult2"})
    mult3 = GraphNode(name="mult3", op=_MultiplyOp(3), inputs={"value": "add10"})
    graph = ComputationGraph([mult2, add10, mult3], outputs=["mult3"])
    scheduler = ParallelScheduler(graph)

    result = await scheduler.run({"data": 5})

    # (5 * 2) = 10, + 10 = 20, * 3 = 60
    assert result == {"mult3": 60}


# =============================================================================
# 7. Error Handling Tests
# =============================================================================


@pytest.mark.asyncio
async def test_parallel_scheduler_propagates_operator_error():
    """Test 7.1: Verify operator exceptions propagate correctly."""
    data = graph_input("data")
    error_node = GraphNode(
        name="error",
        op=_ErrorOp("Test error message"),
        inputs={"value": data},
    )
    graph = ComputationGraph([error_node], outputs=["error"])
    scheduler = ParallelScheduler(graph)

    with pytest.raises(RuntimeError, match="Test error message"):
        await scheduler.run({"data": 1})


@pytest.mark.asyncio
async def test_parallel_scheduler_handles_error_in_parallel_branch():
    r"""
    Test 7.2: Verify error handling when one of multiple parallel nodes fails.

    Graph::

           input
           /   \
       success  fail (raises error)
           \   /
          combine
    """
    data = graph_input("data")
    success_node = GraphNode(
        name="success",
        op=_DoubleOp(),
        inputs={"value": data},
    )
    fail_node = GraphNode(
        name="fail",
        op=_ErrorOp("Parallel branch error"),
        inputs={"value": data},
    )
    combine = GraphNode(
        name="combine",
        op=_CombineOp(),
        inputs={"a": "success", "b": "fail"},
    )
    graph = ComputationGraph([success_node, fail_node, combine], outputs=["combine"])
    scheduler = ParallelScheduler(graph)

    with pytest.raises(RuntimeError, match="Parallel branch error"):
        await scheduler.run({"data": 5})


# =============================================================================
# Additional Tests for Resolve Inputs
# =============================================================================


def test_resolve_inputs_detects_missing_dependency():
    """Verify _resolve_inputs raises KeyError for missing dependencies."""
    graph, _, summer = _make_basic_graph()
    scheduler = ParallelScheduler(graph)

    with pytest.raises(KeyError, match="has not been computed"):
        scheduler._resolve_inputs(summer, cache={})


# =============================================================================
# Tests for Per-Scheduler Subtask Limit
# =============================================================================


def test_parallel_scheduler_max_pending_subtasks_default():
    """Verify default max_pending_subtasks is pool.size * 8 when pool provided."""
    graph, _, _ = _make_basic_graph()

    # Without pool, should be None
    scheduler = ParallelScheduler(graph, pool=None)
    assert scheduler.max_pending_subtasks is None

    # With pool, should default to pool.size * 8
    # (We can't easily test this without a real pool, but we can test the param)
    scheduler = ParallelScheduler(graph, pool=None, max_pending_subtasks=16)
    assert scheduler.max_pending_subtasks == 16


def test_parallel_scheduler_max_pending_subtasks_zero_disables():
    """Verify max_pending_subtasks=0 disables the shared limit."""
    graph, _, _ = _make_basic_graph()
    scheduler = ParallelScheduler(graph, pool=None, max_pending_subtasks=0)
    assert scheduler.max_pending_subtasks == 0


@pytest.mark.asyncio
async def test_parallel_scheduler_subtask_semaphore_in_metadata():
    """Verify subtask_semaphore is added to metadata when max_pending_subtasks > 0."""
    # Track what metadata was passed to operators
    received_metadata = []

    class _MetadataTrackingOp(Operator):
        name = "tracking"
        supports_subtasks = False

        def compute(self, inputs, *, context):
            received_metadata.append(dict(context.metadata or {}))
            return inputs["x"] + 1

    data = graph_input("data")
    node = GraphNode(name="track", op=_MetadataTrackingOp(), inputs={"x": data})
    graph = ComputationGraph([node], outputs=["track"])

    # With max_pending_subtasks > 0, semaphore should be in metadata
    scheduler = ParallelScheduler(graph, pool=None, max_pending_subtasks=10)
    await scheduler.run({"data": 5})

    assert len(received_metadata) == 1
    assert "subtask_semaphore" in received_metadata[0]
    assert received_metadata[0]["subtask_semaphore"]._value == 10  # Initial semaphore value

    received_metadata.clear()

    # With max_pending_subtasks=0, no semaphore
    scheduler = ParallelScheduler(graph, pool=None, max_pending_subtasks=0)
    await scheduler.run({"data": 5})

    assert len(received_metadata) == 1
    assert "subtask_semaphore" not in received_metadata[0]
