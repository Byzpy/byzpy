"""
Tests for the lazy evaluation API - Phase 2 implementation.

Test plan: docs/design/phase2_test_plan.md
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Mapping

import pytest

from byzpy.engine.graph.graph import ComputationGraph, GraphInput, GraphNode
from byzpy.engine.graph.lazy import GraphBuilder, LazyNode
from byzpy.engine.graph.operator import OpContext, Operator
from byzpy.engine.graph.parallel_scheduler import ParallelScheduler

# =============================================================================
# Test Operators
# =============================================================================


class _DoubleOp(Operator):
    """Doubles the input value. Uses input_key='value'."""

    name = "double"
    input_key = "value"

    def compute(self, inputs: Mapping[str, Any], *, context: OpContext) -> Any:
        return inputs["value"] * 2


class _AddOp(Operator):
    """Adds two inputs together."""

    name = "add"
    input_key = "lhs"

    def compute(self, inputs: Mapping[str, Any], *, context: OpContext) -> Any:
        return inputs["lhs"] + inputs["rhs"]


class _CombineOp(Operator):
    """Combines two inputs (a + b)."""

    name = "combine"
    input_key = "a"

    def compute(self, inputs: Mapping[str, Any], *, context: OpContext) -> Any:
        return inputs["a"] + inputs["b"]


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


class _RecordCallOp(Operator):
    """Records whether compute() was called. For testing lazy evaluation."""

    name = "record_call"
    input_key = "value"

    def __init__(self):
        self.compute_called = False
        self.call_count = 0

    def compute(self, inputs: Mapping[str, Any], *, context: OpContext) -> Any:
        self.compute_called = True
        self.call_count += 1
        return inputs["value"]


class _NoInputKeyOp(Operator):
    """Operator without input_key attribute to test default behavior."""

    name = "no_input_key"

    def compute(self, inputs: Mapping[str, Any], *, context: OpContext) -> Any:
        return inputs["vectors"]


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


# =============================================================================
# 1. GraphBuilder Basic Tests
# =============================================================================


def test_graph_builder_creates_input_node():
    """Test 1.1: Verify builder.input() creates a LazyNode representing an input."""
    builder = GraphBuilder()

    node = builder.input("data")

    assert isinstance(node, LazyNode)
    assert node.key == "data"
    assert node._is_input is True


def test_graph_builder_reuses_same_input_node():
    """Test 1.2: Verify calling input() twice with same name references same input."""
    builder = GraphBuilder()

    node1 = builder.input("data")
    node2 = builder.input("data")

    # Both should reference the same underlying GraphInput
    assert node1.key == node2.key
    assert builder._inputs["data"] is builder._inputs["data"]


def test_graph_builder_build_creates_computation_graph():
    """Test 1.3: Verify builder.build() creates a valid ComputationGraph."""
    builder = GraphBuilder()
    data = builder.input("data")
    result = data.apply(_DoubleOp())

    graph = builder.build(outputs=[result.key])

    assert isinstance(graph, ComputationGraph)
    assert result.key in graph.outputs
    assert "data" in graph.required_inputs


def test_graph_builder_build_with_empty_nodes_raises():
    """Test 1.4: Verify build() with no operators added raises error."""
    builder = GraphBuilder()
    _ = builder.input("data")  # Only input, no apply()

    with pytest.raises(ValueError, match="requires at least one node"):
        builder.build(outputs=["data"])


# =============================================================================
# 2. LazyNode Tests
# =============================================================================


def test_lazy_node_key_property():
    """Test 2.1: Verify LazyNode.key returns correct key."""
    builder = GraphBuilder()
    node = builder.input("my_data")

    assert node.key == "my_data"


def test_lazy_node_apply_returns_new_lazy_node():
    """Test 2.2: Verify apply() returns a new LazyNode, not the same one."""
    builder = GraphBuilder()
    input_node = builder.input("data")
    op = _DoubleOp()

    result_node = input_node.apply(op)

    assert result_node is not input_node
    assert result_node.key != input_node.key
    assert isinstance(result_node, LazyNode)


def test_lazy_node_apply_does_not_execute_operator():
    """Test 2.3: Verify apply() does NOT execute the operator (lazy evaluation)."""
    builder = GraphBuilder()
    data = builder.input("data")
    op = _RecordCallOp()

    # Apply should NOT call compute()
    _ = data.apply(op)

    assert op.compute_called is False
    assert op.call_count == 0


def test_lazy_node_apply_auto_generates_node_name():
    """Test 2.4: Verify apply() auto-generates unique node names."""
    builder = GraphBuilder()
    data = builder.input("data")

    node1 = data.apply(_DoubleOp())
    node2 = data.apply(_DoubleOp())

    # Each should have unique name
    assert node1.key != node2.key
    assert node1.key.startswith("double_")
    assert node2.key.startswith("double_")


def test_lazy_node_apply_uses_custom_name():
    """Test 2.5: Verify apply(name='custom') uses provided name."""
    builder = GraphBuilder()
    data = builder.input("data")

    node = data.apply(_DoubleOp(), name="my_custom_name")

    assert node.key == "my_custom_name"


def test_lazy_node_apply_detects_input_key_from_operator():
    """Test 2.6: Verify apply() auto-detects input_key from operator.input_key."""
    builder = GraphBuilder()
    data = builder.input("data")
    op = _DoubleOp()  # Has input_key="value"

    node = data.apply(op)
    graph = builder.build(outputs=[node.key])

    # Check the graph node uses "value" as input key
    graph_node = list(graph.nodes_in_order())[0]
    assert "value" in graph_node.inputs


def test_lazy_node_apply_uses_explicit_input_key():
    """Test 2.7: Verify apply(input_key='custom') overrides auto-detection."""
    builder = GraphBuilder()
    data = builder.input("data")

    # Create an operator that expects "custom_key"
    class _CustomKeyOp(Operator):
        name = "custom"

        def compute(self, inputs: Mapping[str, Any], *, context: OpContext) -> Any:
            return inputs["custom_key"]

    op = _CustomKeyOp()
    node = data.apply(op, input_key="custom_key")
    graph = builder.build(outputs=[node.key])

    graph_node = list(graph.nodes_in_order())[0]
    assert "custom_key" in graph_node.inputs


# =============================================================================
# 3. Chaining Tests
# =============================================================================


def test_lazy_node_chain_multiple_applies():
    """Test 3.1: Verify chaining multiple apply() calls builds correct graph."""
    builder = GraphBuilder()
    data = builder.input("data")

    n1 = data.apply(_DoubleOp(), name="op1")
    n2 = n1.apply(_DoubleOp(), name="op2")
    n3 = n2.apply(_DoubleOp(), name="op3")

    graph = builder.build(outputs=[n3.key])

    # Should have 3 nodes
    nodes = list(graph.nodes_in_order())
    assert len(nodes) == 3


def test_lazy_node_chain_preserves_dependencies():
    """Test 3.2: Verify chained nodes have correct input dependencies."""
    builder = GraphBuilder()
    data = builder.input("data")

    n1 = data.apply(_DoubleOp(), name="op1")
    n2 = n1.apply(_DoubleOp(), name="op2")
    n3 = n2.apply(_DoubleOp(), name="op3")

    graph = builder.build(outputs=[n3.key])
    nodes = list(graph.nodes_in_order())

    # op1 depends on data (GraphInput)
    assert isinstance(nodes[0].inputs["value"], GraphInput)
    assert nodes[0].inputs["value"].name == "data"

    # op2 depends on op1 (string reference)
    assert nodes[1].inputs["value"] == "op1"

    # op3 depends on op2 (string reference)
    assert nodes[2].inputs["value"] == "op2"


def test_lazy_node_chain_with_multiple_inputs():
    """Test 3.3: Verify chaining works with multiple graph inputs."""
    builder = GraphBuilder()
    a = builder.input("a")
    b = builder.input("b")

    # a_doubled = a * 2, then combine with b
    a_doubled = a.apply(_DoubleOp(), name="a_doubled")
    result = a_doubled.apply(_CombineOp(), extra_inputs={"b": b}, name="result")

    graph = builder.build(outputs=[result.key])

    # Both inputs should be required
    assert "a" in graph.required_inputs
    assert "b" in graph.required_inputs


# =============================================================================
# 4. Extra Inputs Tests
# =============================================================================


def test_lazy_node_apply_with_extra_input_from_lazy_node():
    """Test 4.1: Verify extra_inputs can reference another LazyNode."""
    builder = GraphBuilder()
    data = builder.input("data")

    branch1 = data.apply(_MultiplyOp(2), name="branch1")
    branch2 = data.apply(_MultiplyOp(3), name="branch2")

    combined = branch1.apply(
        _CombineOp(),
        extra_inputs={"b": branch2},
        name="combined",
    )

    graph = builder.build(outputs=[combined.key])
    combined_node = builder._nodes["combined"]

    # Should depend on both branch1 (via "a") and branch2 (via "b")
    assert combined_node.inputs["a"] == "branch1"
    assert combined_node.inputs["b"] == "branch2"


def test_lazy_node_apply_with_extra_input_from_graph_input():
    """Test 4.2: Verify extra_inputs can reference graph inputs."""
    builder = GraphBuilder()
    data = builder.input("data")
    bias = builder.input("bias")

    result = data.apply(
        _CombineOp(),
        extra_inputs={"b": bias},
        name="result",
    )

    graph = builder.build(outputs=[result.key])
    result_node = builder._nodes["result"]

    # "a" should be GraphInput(data), "b" should be GraphInput(bias)
    assert isinstance(result_node.inputs["a"], GraphInput)
    assert result_node.inputs["a"].name == "data"
    assert isinstance(result_node.inputs["b"], GraphInput)
    assert result_node.inputs["b"].name == "bias"


def test_lazy_node_apply_with_extra_input_string():
    """Test 4.3: Verify extra_inputs can use string node references."""
    builder = GraphBuilder()
    data = builder.input("data")

    n1 = data.apply(_DoubleOp(), name="n1")
    n2 = data.apply(_MultiplyOp(3), name="n2")

    # Reference n2 by string name
    result = n1.apply(
        _CombineOp(),
        extra_inputs={"b": "n2"},
        name="result",
    )

    graph = builder.build(outputs=[result.key])
    result_node = builder._nodes["result"]

    assert result_node.inputs["a"] == "n1"
    assert result_node.inputs["b"] == "n2"


# =============================================================================
# 5. Diamond and Complex Graph Tests
# =============================================================================


def test_lazy_node_diamond_pattern():
    r"""
    Test 5.1: Verify diamond-shaped graphs work correctly.

    Graph::

           input
           /   \
          A     B
           \   /
             C
    """
    builder = GraphBuilder()
    data = builder.input("data")

    a = data.apply(_MultiplyOp(2), name="A")
    b = data.apply(_MultiplyOp(3), name="B")
    c = a.apply(_CombineOp(), extra_inputs={"b": b}, name="C")

    graph = builder.build(outputs=[c.key])

    # Verify structure
    assert len(list(graph.nodes_in_order())) == 3
    assert "data" in graph.required_inputs

    # A and B both depend on data
    assert isinstance(builder._nodes["A"].inputs["value"], GraphInput)
    assert isinstance(builder._nodes["B"].inputs["value"], GraphInput)

    # C depends on A and B
    assert builder._nodes["C"].inputs["a"] == "A"
    assert builder._nodes["C"].inputs["b"] == "B"


def test_lazy_node_fan_out_pattern():
    r"""
    Test 5.2: Verify single input feeding multiple operators.

    Graph::

            input
           / | | \
          A  B C  D
    """
    builder = GraphBuilder()
    data = builder.input("data")

    a = data.apply(_AddConstantOp(1), name="A")
    b = data.apply(_AddConstantOp(2), name="B")
    c = data.apply(_AddConstantOp(3), name="C")
    d = data.apply(_AddConstantOp(4), name="D")

    graph = builder.build(outputs=[a.key, b.key, c.key, d.key])

    # All 4 nodes present
    assert len(list(graph.nodes_in_order())) == 4

    # All outputs present
    assert set(graph.outputs) == {"A", "B", "C", "D"}

    # All depend on same input
    for node_name in ["A", "B", "C", "D"]:
        node = builder._nodes[node_name]
        assert isinstance(node.inputs["value"], GraphInput)
        assert node.inputs["value"].name == "data"


def test_lazy_node_multiple_outputs():
    """Test 5.3: Verify build() can specify multiple output nodes."""
    builder = GraphBuilder()
    a_in = builder.input("a")
    b_in = builder.input("b")

    branch_a = a_in.apply(_DoubleOp(), name="branch_a")
    branch_b = b_in.apply(_MultiplyOp(3), name="branch_b")

    graph = builder.build(outputs=[branch_a.key, branch_b.key])

    assert "branch_a" in graph.outputs
    assert "branch_b" in graph.outputs
    assert len(graph.outputs) == 2


# =============================================================================
# 6. Integration with ParallelScheduler Tests
# =============================================================================


@pytest.mark.asyncio
async def test_graph_builder_with_parallel_scheduler_produces_correct_result():
    """Test 6.1: Verify built graph executes correctly with ParallelScheduler."""
    builder = GraphBuilder()
    data = builder.input("data")

    doubled = data.apply(_DoubleOp(), name="doubled")

    graph = builder.build(outputs=[doubled.key])
    scheduler = ParallelScheduler(graph)

    result = await scheduler.run({"data": 5})

    assert result == {"doubled": 10}


@pytest.mark.asyncio
async def test_graph_builder_complex_graph_executes_correctly():
    r"""
    Test 6.2: Verify complex graph with parallel branches executes correctly.

    Graph::

           data (10)
           /    \
          A(*2)  B(*3)
           \    /
             C(+)
    """
    builder = GraphBuilder()
    data = builder.input("data")

    a = data.apply(_MultiplyOp(2), name="A")
    b = data.apply(_MultiplyOp(3), name="B")
    c = a.apply(_CombineOp(), extra_inputs={"b": b}, name="C")

    graph = builder.build(outputs=[c.key])
    scheduler = ParallelScheduler(graph)

    result = await scheduler.run({"data": 10})

    # A = 10 * 2 = 20
    # B = 10 * 3 = 30
    # C = 20 + 30 = 50
    assert result == {"C": 50}


@pytest.mark.asyncio
async def test_graph_builder_parallel_branches_execute_concurrently():
    """Test 6.3: Verify parallel branches actually run in parallel."""
    builder = GraphBuilder()
    data = builder.input("data")

    # Two slow operators as parallel branches
    slow1 = data.apply(_SlowOp(0.1, result_value=1), name="slow1")
    slow2 = data.apply(_SlowOp(0.1, result_value=2), name="slow2")
    combined = slow1.apply(_CombineOp(), extra_inputs={"b": slow2}, name="combined")

    graph = builder.build(outputs=[combined.key])
    scheduler = ParallelScheduler(graph)

    start = time.monotonic()
    result = await scheduler.run({"data": 0})
    elapsed = time.monotonic() - start

    # If parallel: ~0.1s; if sequential: ~0.2s
    assert elapsed < 0.15, f"Expected parallel execution in ~0.1s, got {elapsed:.3f}s"
    assert result == {"combined": 3}


# =============================================================================
# 7. Error Handling Tests
# =============================================================================


def test_graph_builder_build_with_unknown_output_raises():
    """Test 7.1: Verify build(outputs=['unknown']) raises error."""
    builder = GraphBuilder()
    data = builder.input("data")
    _ = data.apply(_DoubleOp(), name="doubled")

    with pytest.raises(ValueError, match="Unknown output node"):
        builder.build(outputs=["nonexistent"])


def test_lazy_node_apply_with_non_operator_raises():
    """Test 7.2: Verify apply() with non-Operator raises error."""
    builder = GraphBuilder()
    data = builder.input("data")

    with pytest.raises(TypeError, match="must be an Operator instance"):
        data.apply("not an operator")  # type: ignore


# =============================================================================
# 8. Operator Input Key Detection Tests
# =============================================================================


def test_lazy_node_apply_default_input_key_is_vectors():
    """Test 8.1: Verify default input_key is 'vectors' when operator has no input_key."""
    builder = GraphBuilder()
    data = builder.input("data")

    node = data.apply(_NoInputKeyOp(), name="result")
    graph = builder.build(outputs=[node.key])

    graph_node = list(graph.nodes_in_order())[0]
    # Should use default "vectors"
    assert "vectors" in graph_node.inputs


def test_lazy_node_apply_with_value_input_key():
    """Test 8.2: Verify operators with input_key='value' are handled."""
    builder = GraphBuilder()
    data = builder.input("data")

    # _DoubleOp has input_key="value"
    node = data.apply(_DoubleOp(), name="result")
    graph = builder.build(outputs=[node.key])

    graph_node = list(graph.nodes_in_order())[0]
    assert "value" in graph_node.inputs


# =============================================================================
# Additional Integration Tests
# =============================================================================


@pytest.mark.asyncio
async def test_lazy_evaluation_operator_called_only_during_execution():
    """Verify operator compute() is called during execution, not during build."""
    builder = GraphBuilder()
    data = builder.input("data")

    op = _RecordCallOp()
    node = data.apply(op, name="result")

    # Build should not call compute
    graph = builder.build(outputs=[node.key])
    assert op.compute_called is False

    # Creating scheduler should not call compute
    scheduler = ParallelScheduler(graph)
    assert op.compute_called is False

    # Running should call compute
    result = await scheduler.run({"data": 42})
    assert op.compute_called is True
    assert op.call_count == 1
    assert result == {"result": 42}


@pytest.mark.asyncio
async def test_chain_of_three_operators_produces_correct_result():
    """Integration test: chain input -> double -> add10 -> multiply3."""
    builder = GraphBuilder()
    data = builder.input("data")

    doubled = data.apply(_DoubleOp(), name="doubled")
    plus10 = doubled.apply(_AddConstantOp(10), name="plus10")
    final = plus10.apply(_MultiplyOp(3), name="final")

    graph = builder.build(outputs=[final.key])
    scheduler = ParallelScheduler(graph)

    result = await scheduler.run({"data": 5})

    # (5 * 2) = 10, + 10 = 20, * 3 = 60
    assert result == {"final": 60}
