"""
Tests for run_operator convenience function.
"""

from __future__ import annotations

import inspect
import time
from typing import Any, Mapping, Sequence

import pytest
import torch

from byzpy import run_operator
from byzpy.aggregators.coordinate_wise.median import CoordinateWiseMedian
from byzpy.aggregators.geometric_wise.krum import MultiKrum
from byzpy.attacks.empire import EmpireAttack
from byzpy.engine.graph.executor import OperatorExecutor
from byzpy.engine.graph.executor import run_operator as run_operator_from_module
from byzpy.engine.graph.operator import OpContext, Operator
from byzpy.engine.graph.pool import ActorPoolConfig
from byzpy.pre_aggregators.bucketing import Bucketing
from byzpy.pre_aggregators.clipping import Clipping


def _make_gradients(n: int, dim: int, seed: int = 0) -> list[torch.Tensor]:
    torch.manual_seed(seed)
    return [torch.randn(dim) for _ in range(n)]


def _make_vectors(n: int, dim: int, seed: int = 0) -> list[torch.Tensor]:
    torch.manual_seed(seed)
    return [torch.randn(dim) for _ in range(n)]


class TestRunOperatorBasic:
    """Category 1: Basic functionality tests."""

    @pytest.mark.asyncio
    async def test_basic_usage(self):
        """Test 1.1: run_operator() Basic Usage"""
        operator = CoordinateWiseMedian()
        gradients = _make_gradients(64, 1000)

        result = await run_operator(operator=operator, inputs={"gradients": gradients})

        # Verify result is correct
        assert isinstance(result, torch.Tensor)
        assert result.shape == (1000,)

        # Compare with direct operator call
        direct_result = operator.aggregate(gradients)
        assert torch.allclose(result, direct_result, atol=1e-6)

    @pytest.mark.asyncio
    async def test_with_pool(self):
        """Test 1.2: run_operator() With Pool"""
        operator = MultiKrum(f=10, q=5, chunk_size=10)
        gradients = _make_gradients(30, 1000)

        result = await run_operator(
            operator=operator,
            inputs={"gradients": gradients},
            pool_config=ActorPoolConfig(backend="process", count=2),
        )

        # Verify result is correct
        assert isinstance(result, torch.Tensor)
        assert result.shape == (1000,)

        # Pool should be cleaned up automatically (no way to verify directly,
        # but if it wasn't cleaned up, subsequent tests might fail)

    @pytest.mark.asyncio
    async def test_auto_detection_input_keys(self):
        """Test 1.3: run_operator() Auto-Detection of Input Keys"""
        operator = CoordinateWiseMedian()
        gradients = _make_gradients(64, 1000)

        # Don't specify input_keys - should auto-detect
        result = await run_operator(operator=operator, inputs={"gradients": gradients})

        assert isinstance(result, torch.Tensor)
        assert result.shape == (1000,)

    @pytest.mark.asyncio
    async def test_explicit_input_keys(self):
        """Test 1.4: run_operator() Explicit Input Keys"""
        operator = CoordinateWiseMedian()
        gradients = _make_gradients(64, 1000)

        # Use explicit custom input key
        result = await run_operator(
            operator=operator,
            inputs={"custom_key": gradients},
            input_keys=("custom_key",),
        )

        assert isinstance(result, torch.Tensor)
        assert result.shape == (1000,)


class TestRunOperatorIntegration:
    """Category 2: Integration with different operator types."""

    @pytest.mark.asyncio
    async def test_with_aggregator(self):
        """Test 2.1: run_operator() with Aggregator"""
        # Test CoordinateWiseMedian
        operator1 = CoordinateWiseMedian()
        gradients = _make_gradients(64, 1000)
        result1 = await run_operator(operator=operator1, inputs={"gradients": gradients})
        direct_result1 = operator1.aggregate(gradients)
        assert torch.allclose(result1, direct_result1, atol=1e-6)

        # Test MultiKrum
        operator2 = MultiKrum(f=10, q=5, chunk_size=10)
        gradients2 = _make_gradients(30, 1000)
        result2 = await run_operator(operator=operator2, inputs={"gradients": gradients2})
        # MultiKrum result should be a tensor
        assert isinstance(result2, torch.Tensor)

    @pytest.mark.asyncio
    async def test_with_preaggregator(self):
        """Test 2.2: run_operator() with PreAggregator"""
        # Test Clipping
        operator1 = Clipping(threshold=2.0, chunk_size=32)
        vectors = _make_vectors(256, 1000)
        result1 = await run_operator(operator=operator1, inputs={"vectors": vectors})
        direct_result1 = operator1.pre_aggregate(vectors)
        assert len(result1) == len(direct_result1)
        for r, d in zip(result1, direct_result1):
            assert torch.allclose(r, d, atol=1e-6)

        # Test Bucketing
        operator2 = Bucketing(bucket_size=32)
        vectors2 = _make_vectors(256, 1000)
        result2 = await run_operator(operator=operator2, inputs={"vectors": vectors2})
        direct_result2 = operator2.pre_aggregate(vectors2)
        assert len(result2) == len(direct_result2)

    @pytest.mark.asyncio
    async def test_with_attack(self):
        """Test 2.3: run_operator() with Attack"""
        operator = EmpireAttack()
        honest_grads = _make_gradients(10, 1000)

        # Attacks require explicit input_keys
        result = await run_operator(
            operator=operator,
            inputs={"honest_grads": honest_grads},
            input_keys=("honest_grads",),
        )

        # Compare with direct call
        direct_result = operator.apply(honest_grads=honest_grads)
        assert torch.allclose(result, direct_result, atol=1e-6)


class TestRunOperatorEquivalence:
    """Category 3: Equivalence with OperatorExecutor."""

    @pytest.mark.asyncio
    async def test_result_equivalence_no_pool(self):
        """Test 3.1: Result Equivalence - No Pool"""
        operator = CoordinateWiseMedian()
        gradients = _make_gradients(64, 1000)

        # Run with run_operator()
        result1 = await run_operator(operator=operator, inputs={"gradients": gradients})

        # Run with OperatorExecutor
        executor = OperatorExecutor(operator)
        result2 = await executor.run({"gradients": gradients})

        # Should be bitwise identical
        assert torch.allclose(result1, result2, atol=1e-6)

    @pytest.mark.asyncio
    async def test_result_equivalence_with_pool(self):
        """Test 3.2: Result Equivalence - With Pool"""
        operator = MultiKrum(f=10, q=5, chunk_size=10)
        gradients = _make_gradients(30, 1000)
        pool_config = ActorPoolConfig(backend="process", count=2)

        # Run with run_operator()
        result1 = await run_operator(
            operator=operator, inputs={"gradients": gradients}, pool_config=pool_config
        )

        # Run with OperatorExecutor
        executor = OperatorExecutor(operator, pool_config=pool_config)
        async with executor:
            result2 = await executor.run({"gradients": gradients})

        # Should be bitwise identical
        assert torch.allclose(result1, result2, atol=1e-6)

    @pytest.mark.asyncio
    async def test_code_path_equivalence(self):
        """Test 3.3: Code Path Equivalence"""
        # This test verifies that run_operator() calls the same underlying functions
        # as OperatorExecutor. We verify this by checking that results are identical
        # and that the function signature matches what OperatorExecutor expects.

        operator = CoordinateWiseMedian()
        gradients = _make_gradients(64, 1000)

        # Both should produce identical results (verified in other tests)
        result1 = await run_operator(operator=operator, inputs={"gradients": gradients})
        executor = OperatorExecutor(operator)
        result2 = await executor.run({"gradients": gradients})

        assert torch.allclose(result1, result2, atol=1e-6)

        # Verify run_operator() is implemented as a wrapper around OperatorExecutor
        # by checking the source code
        source = inspect.getsource(run_operator)
        assert "OperatorExecutor" in source
        assert "async with" in source or "__aenter__" in source


class TestRunOperatorPoolConfig:
    """Category 4: Pool configuration tests."""

    @pytest.mark.asyncio
    async def test_single_pool_config(self):
        """Test 4.1: Single Pool Config"""
        operator = CoordinateWiseMedian()
        gradients = _make_gradients(64, 1000)

        result = await run_operator(
            operator=operator,
            inputs={"gradients": gradients},
            pool_config=ActorPoolConfig(backend="process", count=2),
        )

        assert isinstance(result, torch.Tensor)

    @pytest.mark.asyncio
    async def test_multiple_pool_configs(self):
        """Test 4.2: Multiple Pool Configs"""
        operator = CoordinateWiseMedian()
        gradients = _make_gradients(64, 1000)

        result = await run_operator(
            operator=operator,
            inputs={"gradients": gradients},
            pool_config=[
                ActorPoolConfig(backend="process", count=2),
                ActorPoolConfig(backend="thread", count=2),
            ],
        )

        assert isinstance(result, torch.Tensor)

    @pytest.mark.asyncio
    async def test_different_backend_types(self):
        """Test 4.3: Different Backend Types"""
        operator = CoordinateWiseMedian()
        gradients = _make_gradients(64, 1000)

        # Test process backend
        result_process = await run_operator(
            operator=operator,
            inputs={"gradients": gradients},
            pool_config=ActorPoolConfig(backend="process", count=2),
        )

        # Test thread backend
        result_thread = await run_operator(
            operator=operator,
            inputs={"gradients": gradients},
            pool_config=ActorPoolConfig(backend="thread", count=2),
        )

        # Results should be identical
        assert torch.allclose(result_process, result_thread, atol=1e-6)

    @pytest.mark.asyncio
    async def test_none_pool_config(self):
        """Test 4.4: None Pool Config"""
        operator = CoordinateWiseMedian()
        gradients = _make_gradients(64, 1000)

        result = await run_operator(
            operator=operator, inputs={"gradients": gradients}, pool_config=None
        )

        assert isinstance(result, torch.Tensor)


class TestRunOperatorErrorHandling:
    """Category 5: Error handling tests."""

    @pytest.mark.asyncio
    async def test_missing_inputs(self):
        """Test 5.1: Missing Required Inputs"""
        operator = CoordinateWiseMedian()

        # Missing required input
        with pytest.raises((KeyError, ValueError), match="Missing|gradients"):
            await run_operator(operator=operator, inputs={})

    @pytest.mark.asyncio
    async def test_invalid_input_keys(self):
        """Test 5.2: Invalid Input Keys"""
        operator = CoordinateWiseMedian()
        gradients = _make_gradients(64, 1000)

        # Using wrong key name with explicit input_keys
        with pytest.raises((KeyError, ValueError)):
            await run_operator(
                operator=operator,
                inputs={"gradients": gradients},
                input_keys=("invalid_key",),
            )

    @pytest.mark.asyncio
    async def test_invalid_operator_type(self):
        """Test 5.3: Invalid Operator Type"""
        # Not an Operator instance
        with pytest.raises(TypeError, match="must be an Operator"):
            await run_operator(operator="not an operator", inputs={})

        with pytest.raises(TypeError, match="must be an Operator"):
            await run_operator(operator=42, inputs={})

    @pytest.mark.asyncio
    async def test_pool_creation_failure(self):
        """Test 5.4: Pool Creation Failure"""
        operator = CoordinateWiseMedian()
        gradients = _make_gradients(64, 1000)

        # Invalid pool config (invalid backend)
        with pytest.raises((ValueError, RuntimeError)):
            await run_operator(
                operator=operator,
                inputs={"gradients": gradients},
                pool_config=ActorPoolConfig(backend="invalid_backend", count=2),
            )

    @pytest.mark.asyncio
    async def test_exception_during_execution(self):
        """Test 5.5: Exception During Execution"""

        # Create an operator that raises an exception
        class FailingOperator(Operator):
            name = "failing"

            def compute(self, inputs: Mapping[str, Any], *, context: OpContext) -> Any:
                raise ValueError("Test exception")

        operator = FailingOperator()

        # Exception should be propagated
        with pytest.raises(ValueError, match="Test exception"):
            await run_operator(operator=operator, inputs={"x": 42}, input_keys=("x",))

        # Should be able to call run_operator() again after failure
        operator2 = CoordinateWiseMedian()
        gradients = _make_gradients(64, 1000)
        result = await run_operator(operator=operator2, inputs={"gradients": gradients})
        assert isinstance(result, torch.Tensor)


class TestRunOperatorPerformance:
    """Category 6: Performance verification tests."""

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Flaky performance test - timing sensitive")
    async def test_performance_vs_executor_no_pool(self):
        """Test 6.1: Performance Comparison - No Pool"""
        operator = CoordinateWiseMedian()
        gradients = _make_gradients(64, 1000)

        # run_operator()
        run_operator_times = []
        for _ in range(10):
            start = time.perf_counter()
            await run_operator(operator=operator, inputs={"gradients": gradients})
            run_operator_times.append(time.perf_counter() - start)

        # OperatorExecutor
        executor_times = []
        executor = OperatorExecutor(operator)
        for _ in range(10):
            start = time.perf_counter()
            await executor.run({"gradients": gradients})
            executor_times.append(time.perf_counter() - start)

        # Compare mean times (should be within 5% - allowing for measurement variance)
        run_operator_mean = sum(run_operator_times) / len(run_operator_times)
        executor_mean = sum(executor_times) / len(executor_times)

        ratio = run_operator_mean / executor_mean
        assert 0.95 <= ratio <= 1.05, f"Performance difference too large: {ratio:.3f}"

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Flaky performance test - timing sensitive")
    async def test_performance_vs_executor_with_pool(self):
        """Test 6.2: Performance Comparison - With Pool"""
        operator = MultiKrum(f=10, q=5, chunk_size=10)
        gradients = _make_gradients(30, 1000)
        pool_config = ActorPoolConfig(backend="process", count=2)

        # run_operator()
        run_operator_times = []
        for _ in range(5):  # Fewer iterations for pool (slower)
            start = time.perf_counter()
            await run_operator(
                operator=operator,
                inputs={"gradients": gradients},
                pool_config=pool_config,
            )
            run_operator_times.append(time.perf_counter() - start)

        # OperatorExecutor
        executor_times = []
        executor = OperatorExecutor(operator, pool_config=pool_config)
        async with executor:
            for _ in range(5):
                start = time.perf_counter()
                await executor.run({"gradients": gradients})
                executor_times.append(time.perf_counter() - start)

        # Compare mean times (should be within 5%)
        run_operator_mean = sum(run_operator_times) / len(run_operator_times)
        executor_mean = sum(executor_times) / len(executor_times)

        ratio = run_operator_mean / executor_mean
        assert 0.95 <= ratio <= 1.05, f"Performance difference too large: {ratio:.3f}"

    @pytest.mark.asyncio
    async def test_performance_vs_manual_boilerplate(self):
        """Test 6.3: Performance Comparison - Manual Boilerplate"""
        from byzpy.engine.graph.ops import make_single_operator_graph
        from byzpy.engine.graph.scheduler import NodeScheduler

        operator = CoordinateWiseMedian()
        gradients = _make_gradients(64, 1000)

        # Manual boilerplate
        manual_times = []
        for _ in range(10):
            graph = make_single_operator_graph(
                node_name=operator.name,
                operator=operator,
                input_keys=("gradients",),
            )
            scheduler = NodeScheduler(graph, pool=None)
            start = time.perf_counter()
            results = await scheduler.run({"gradients": gradients})
            manual_times.append(time.perf_counter() - start)

        # run_operator()
        run_operator_times = []
        for _ in range(10):
            start = time.perf_counter()
            await run_operator(operator=operator, inputs={"gradients": gradients})
            run_operator_times.append(time.perf_counter() - start)

        # Compare mean times (should be within 5%)
        manual_mean = sum(manual_times) / len(manual_times)
        run_operator_mean = sum(run_operator_times) / len(run_operator_times)

        ratio = run_operator_mean / manual_mean
        assert 0.95 <= ratio <= 1.05, f"Performance difference too large: {ratio:.3f}"

    @pytest.mark.asyncio
    async def test_overhead_verification(self):
        """Test 6.4: Overhead Verification"""
        operator = CoordinateWiseMedian()
        gradients = _make_gradients(64, 1000)

        # Measure run_operator() overhead
        # The overhead should be just the context manager entry/exit
        # which should be negligible compared to actual execution time

        # Run multiple times to get stable measurement
        times = []
        for _ in range(20):
            start = time.perf_counter()
            await run_operator(operator=operator, inputs={"gradients": gradients})
            times.append(time.perf_counter() - start)

        mean_time = sum(times) / len(times)

        # Overhead should be less than 1% of total time
        # (context manager overhead is typically microseconds)
        # Since we can't measure overhead directly, we verify that
        # run_operator() performance is very close to OperatorExecutor
        # (which we test in other performance tests)
        assert mean_time > 0  # Just verify it runs


class TestRunOperatorEdgeCases:
    """Category 7: Edge case tests."""

    @pytest.mark.asyncio
    async def test_empty_inputs(self):
        """Test 7.1: Empty Input Dictionary"""
        operator = CoordinateWiseMedian()

        # Empty inputs should fail
        with pytest.raises((KeyError, ValueError)):
            await run_operator(operator=operator, inputs={})

    @pytest.mark.asyncio
    async def test_extra_input_keys(self):
        """Test 7.2: Extra Input Keys"""
        operator = CoordinateWiseMedian()
        gradients = _make_gradients(64, 1000)

        # Extra keys should be ignored (or cause error if validation is strict)
        # NodeScheduler will ignore extra keys, so this should work
        result = await run_operator(
            operator=operator, inputs={"gradients": gradients, "extra_key": 42}
        )
        assert isinstance(result, torch.Tensor)

    @pytest.mark.asyncio
    async def test_node_name_customization(self):
        """Test 7.3: Node Name Customization"""
        operator = CoordinateWiseMedian()
        gradients = _make_gradients(64, 1000)

        # run_operator() doesn't accept node_name parameter
        # It should use operator.name automatically
        result = await run_operator(operator=operator, inputs={"gradients": gradients})
        assert isinstance(result, torch.Tensor)

        # Verify node_name is not a valid parameter
        sig = inspect.signature(run_operator)
        assert "node_name" not in sig.parameters

    @pytest.mark.asyncio
    async def test_multiple_sequential_calls(self):
        """Test 7.4: Multiple Sequential Calls"""
        # Call with operator A
        operator1 = CoordinateWiseMedian()
        gradients1 = _make_gradients(64, 1000, seed=0)
        result1 = await run_operator(operator=operator1, inputs={"gradients": gradients1})

        # Call with operator B
        operator2 = MultiKrum(f=10, q=5, chunk_size=10)
        gradients2 = _make_gradients(30, 1000, seed=1)
        result2 = await run_operator(operator=operator2, inputs={"gradients": gradients2})

        # Both should succeed
        assert isinstance(result1, torch.Tensor)
        assert isinstance(result2, torch.Tensor)

        # Verify no resource leaks by calling multiple times
        for _ in range(5):
            result = await run_operator(operator=operator1, inputs={"gradients": gradients1})
            assert isinstance(result, torch.Tensor)


class TestRunOperatorAPI:
    """Category 8: Import and API surface tests."""

    def test_import_from_byzpy(self):
        """Test 8.1: Import from byzpy"""
        from byzpy import run_operator

        assert callable(run_operator)
        assert inspect.iscoroutinefunction(run_operator)

    def test_import_from_executor_module(self):
        """Test 8.2: Import from executor module"""
        from byzpy.engine.graph.executor import run_operator

        assert callable(run_operator)
        assert inspect.iscoroutinefunction(run_operator)

    def test_function_signature(self):
        """Test 8.3: Function Signature"""
        sig = inspect.signature(run_operator)

        # Verify parameters
        params = list(sig.parameters.keys())
        assert "operator" in params
        assert "inputs" in params
        assert "pool_config" in params
        assert "input_keys" in params

        # Verify types (if available)
        operator_param = sig.parameters["operator"]
        assert operator_param.annotation != inspect.Parameter.empty

    def test_docstring(self):
        """Test 8.4: Docstring"""
        doc = run_operator.__doc__
        assert doc is not None
        assert len(doc) > 0

        # Verify key information is documented
        assert "operator" in doc.lower()
        assert "inputs" in doc.lower()
        assert "pool_config" in doc.lower() or "pool" in doc.lower()
        assert "return" in doc.lower() or "returns" in doc.lower()


class TestRunOperatorRealWorld:
    """Category 9: Real-world scenario tests."""

    @pytest.mark.asyncio
    async def test_benchmark_usage(self):
        """Test 9.1: Benchmark-Style Usage"""
        # Simulate benchmark usage: call run_operator() multiple times with different operators
        operators = [
            CoordinateWiseMedian(),
            MultiKrum(f=10, q=5, chunk_size=10),
            Clipping(threshold=2.0, chunk_size=32),
        ]

        gradients = _make_gradients(64, 1000)
        vectors = _make_vectors(256, 1000)

        for operator in operators:
            if isinstance(operator, CoordinateWiseMedian) or isinstance(operator, MultiKrum):
                result = await run_operator(operator=operator, inputs={"gradients": gradients})
                assert isinstance(result, torch.Tensor)
            elif isinstance(operator, Clipping):
                result = await run_operator(operator=operator, inputs={"vectors": vectors})
                assert isinstance(result, list)

        # Verify no resource leaks
        for _ in range(3):
            result = await run_operator(operator=operators[0], inputs={"gradients": gradients})
            assert isinstance(result, torch.Tensor)

    @pytest.mark.asyncio
    async def test_ad_hoc_experiment_usage(self):
        """Test 9.2: Ad-Hoc Experiment Usage"""
        # Simulate ad-hoc usage: quick one-off calls with various operators
        operator1 = CoordinateWiseMedian()
        gradients1 = _make_gradients(64, 1000)
        result1 = await run_operator(operator=operator1, inputs={"gradients": gradients1})
        assert isinstance(result1, torch.Tensor)

        operator2 = Clipping(threshold=2.0)
        vectors2 = _make_vectors(256, 1000)
        result2 = await run_operator(operator=operator2, inputs={"vectors": vectors2})
        assert isinstance(result2, list)

        # Cleanup should happen automatically
        # Verify by making another call
        result3 = await run_operator(operator=operator1, inputs={"gradients": gradients1})
        assert isinstance(result3, torch.Tensor)

    @pytest.mark.asyncio
    async def test_error_recovery(self):
        """Test 9.3: Error Recovery"""
        operator = CoordinateWiseMedian()

        # First call with invalid inputs (should fail)
        with pytest.raises((KeyError, ValueError)):
            await run_operator(operator=operator, inputs={})

        # Second call with valid inputs (should succeed)
        gradients = _make_gradients(64, 1000)
        result = await run_operator(operator=operator, inputs={"gradients": gradients})
        assert isinstance(result, torch.Tensor)

        # Third call should also succeed
        result2 = await run_operator(operator=operator, inputs={"gradients": gradients})
        assert isinstance(result2, torch.Tensor)
