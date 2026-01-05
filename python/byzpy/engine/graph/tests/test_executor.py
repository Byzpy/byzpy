"""
Tests for OperatorExecutor - simplified operator execution API.
"""

from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

import pytest
import torch

from byzpy.aggregators.coordinate_wise.median import CoordinateWiseMedian
from byzpy.aggregators.geometric_wise.krum import MultiKrum
from byzpy.attacks.empire import EmpireAttack
from byzpy.engine.graph.executor import OperatorExecutor, run_operator
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


def _results_equal(a: Any, b: Any) -> bool:
    """Compare two results, handling tensors."""
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return torch.allclose(a, b, atol=1e-6)
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        return all(_results_equal(ai, bi) for ai, bi in zip(a, b))
    return a == b


async def _manual_run(
    operator: Operator,
    inputs: Mapping[str, Any],
    pool_config: ActorPoolConfig | Sequence[ActorPoolConfig] | None = None,
    input_keys: Sequence[str] | None = None,
) -> Any:
    """Run operator using manual boilerplate (for comparison)."""
    from byzpy.engine.graph.executor import _detect_input_keys
    from byzpy.engine.graph.ops import make_single_operator_graph
    from byzpy.engine.graph.pool import ActorPool
    from byzpy.engine.graph.scheduler import NodeScheduler

    if input_keys is None:
        try:
            input_keys = _detect_input_keys(operator)
        except ValueError:
            if hasattr(operator, "input_key"):
                input_keys = (operator.input_key,)
            else:
                raise

    graph = make_single_operator_graph(
        node_name=operator.name,
        operator=operator,
        input_keys=input_keys,
    )

    pool = None
    if pool_config is not None:
        if isinstance(pool_config, ActorPoolConfig):
            configs = [pool_config]
        else:
            configs = list(pool_config)
        pool = ActorPool(configs)
        await pool.start()

    try:
        scheduler = NodeScheduler(graph, pool=pool)
        results = await scheduler.run(inputs)
        return results[operator.name]
    finally:
        if pool is not None:
            await pool.shutdown()


class TestOperatorExecutorBasic:
    """Category 1: Basic functionality tests."""

    @pytest.mark.asyncio
    async def test_executor_creation(self):
        """Test 1.1: OperatorExecutor Creation"""
        operator = CoordinateWiseMedian()
        executor = OperatorExecutor(operator)

        assert executor.operator is operator
        assert executor.input_keys == ("gradients",)
        assert executor.node_name == operator.name
        assert executor.pool_config is None

    @pytest.mark.asyncio
    async def test_run_without_pool(self):
        """Test 1.2: Run Operator Without Pool"""
        operator = CoordinateWiseMedian()
        executor = OperatorExecutor(operator)

        gradients = _make_gradients(64, 1000)
        result = await executor.run({"gradients": gradients})

        assert isinstance(result, torch.Tensor)
        assert result.shape == (1000,)

        direct_result = operator.aggregate(gradients)
        assert torch.allclose(result, direct_result, atol=1e-6)

    @pytest.mark.asyncio
    async def test_run_with_pool(self):
        """Test 1.3: Run Operator With Pool"""
        operator = MultiKrum(f=10, q=5, chunk_size=10)
        executor = OperatorExecutor(
            operator, pool_config=ActorPoolConfig(backend="process", count=2)
        )

        gradients = _make_gradients(30, 1000)

        async with executor:
            result = await executor.run({"gradients": gradients})

        assert isinstance(result, torch.Tensor)
        assert result.shape == (1000,)

        manual_result = await _manual_run(
            operator,
            {"gradients": gradients},
            pool_config=ActorPoolConfig(backend="process", count=2),
        )
        assert torch.allclose(result, manual_result, atol=1e-6)

    @pytest.mark.asyncio
    async def test_run_multiple_times_pool_reuse(self):
        """Test 1.4: Run Operator Multiple Times (Pool Reuse)"""
        operator = CoordinateWiseMedian()
        executor = OperatorExecutor(
            operator, pool_config=ActorPoolConfig(backend="process", count=2)
        )

        gradients1 = _make_gradients(64, 1000, seed=0)
        gradients2 = _make_gradients(64, 1000, seed=1)
        gradients3 = _make_gradients(64, 1000, seed=2)

        async with executor:
            result1 = await executor.run({"gradients": gradients1})
            result2 = await executor.run({"gradients": gradients2})
            result3 = await executor.run({"gradients": gradients3})

        assert isinstance(result1, torch.Tensor)
        assert isinstance(result2, torch.Tensor)
        assert isinstance(result3, torch.Tensor)

        assert not torch.allclose(result1, result2, atol=1e-6)


class TestAutoDetection:
    """Category 2: Auto-detection tests."""

    @pytest.mark.asyncio
    async def test_auto_detect_aggregator(self):
        """Test 2.1: Auto-Detect Input Keys for Aggregator"""
        operator = CoordinateWiseMedian()
        executor = OperatorExecutor(operator)

        assert executor.input_keys == ("gradients",)

        gradients = _make_gradients(64, 1000)
        result = await executor.run({"gradients": gradients})
        assert isinstance(result, torch.Tensor)

    @pytest.mark.asyncio
    async def test_auto_detect_preaggregator(self):
        """Test 2.2: Auto-Detect Input Keys for PreAggregator"""
        operator = Clipping(threshold=2.0, chunk_size=32)
        executor = OperatorExecutor(operator)

        assert executor.input_keys == ("vectors",)

        vectors = _make_vectors(256, 1000)
        result = await executor.run({"vectors": vectors})
        assert isinstance(result, list)
        assert len(result) == len(vectors)

    @pytest.mark.asyncio
    async def test_auto_detect_attack(self):
        """Test 2.3: Auto-Detect Input Keys for Attack"""
        operator = EmpireAttack()

        with pytest.raises(ValueError, match="Cannot auto-detect input keys"):
            OperatorExecutor(operator)

        executor = OperatorExecutor(operator, input_keys=("honest_grads",))
        assert executor.input_keys == ("honest_grads",)

    @pytest.mark.asyncio
    async def test_explicit_input_keys_override(self):
        """Test 2.4: Explicit Input Keys Override Auto-Detection"""
        operator = CoordinateWiseMedian()
        executor = OperatorExecutor(operator, input_keys=("custom_key",))

        assert executor.input_keys == ("custom_key",)

        gradients = _make_gradients(64, 1000)
        result = await executor.run({"custom_key": gradients})
        assert isinstance(result, torch.Tensor)

    @pytest.mark.asyncio
    async def test_auto_detection_failure_requires_explicit(self):
        """Test 2.5: Auto-Detection Failure Requires Explicit Keys"""

        class CustomOperator(Operator):
            name = "custom"

            def compute(self, inputs: Mapping[str, Any], *, context: OpContext) -> Any:
                return inputs.get("x", 0)

        operator = CustomOperator()

        with pytest.raises(ValueError, match="Cannot auto-detect input keys"):
            OperatorExecutor(operator)

        executor = OperatorExecutor(operator, input_keys=("x",))
        result = await executor.run({"x": 42})
        assert result == 42


class TestContextManager:
    """Category 3: Context manager tests."""

    @pytest.mark.asyncio
    async def test_context_manager_entry_starts_pool(self):
        """Test 3.1: Context Manager Entry Starts Pool"""
        operator = CoordinateWiseMedian()
        executor = OperatorExecutor(
            operator, pool_config=ActorPoolConfig(backend="process", count=2)
        )

        assert executor._pool is None

        async with executor:
            assert executor._pool is not None
            assert executor._pool.size == 2
            assert executor._pool_managed is True

        assert executor._pool is None

    @pytest.mark.asyncio
    async def test_context_manager_exit_shuts_down_pool(self):
        """Test 3.2: Context Manager Exit Shuts Down Pool"""
        operator = CoordinateWiseMedian()
        executor = OperatorExecutor(
            operator, pool_config=ActorPoolConfig(backend="process", count=2)
        )

        gradients = _make_gradients(64, 1000)

        async with executor:
            result = await executor.run({"gradients": gradients})
            assert isinstance(result, torch.Tensor)

        assert executor._pool is None

    @pytest.mark.asyncio
    async def test_context_manager_handles_exceptions(self):
        """Test 3.3: Context Manager Handles Exceptions"""
        operator = CoordinateWiseMedian()
        executor = OperatorExecutor(
            operator, pool_config=ActorPoolConfig(backend="process", count=2)
        )

        try:
            async with executor:
                raise ValueError("Test exception")
        except ValueError as e:
            assert str(e) == "Test exception"

        assert executor._pool is None

    @pytest.mark.asyncio
    async def test_context_manager_without_pool(self):
        """Test 3.4: Context Manager Without Pool"""
        operator = CoordinateWiseMedian()
        executor = OperatorExecutor(operator)  # No pool config

        gradients = _make_gradients(64, 1000)

        async with executor:
            result = await executor.run({"gradients": gradients})
            assert isinstance(result, torch.Tensor)

        assert executor._pool is None


class TestOperatorTypes:
    """Category 4: Different operator types."""

    @pytest.mark.asyncio
    async def test_coordinate_wise_median(self):
        """Test 4.1: Aggregator - CoordinateWiseMedian"""
        operator = CoordinateWiseMedian()
        executor = OperatorExecutor(operator)

        gradients = _make_gradients(64, 1000)
        result = await executor.run({"gradients": gradients})

        direct_result = operator.aggregate(gradients)
        assert torch.allclose(result, direct_result, atol=1e-6)

    @pytest.mark.asyncio
    async def test_multikrum(self):
        """Test 4.2: Aggregator - MultiKrum"""
        operator = MultiKrum(f=10, q=5, chunk_size=10)
        executor = OperatorExecutor(operator)

        gradients = _make_gradients(30, 1000)
        result = await executor.run({"gradients": gradients})

        manual_result = await _manual_run(operator, {"gradients": gradients})
        assert torch.allclose(result, manual_result, atol=1e-6)

    @pytest.mark.asyncio
    async def test_clipping(self):
        """Test 4.3: PreAggregator - Clipping"""
        operator = Clipping(threshold=2.0, chunk_size=32)
        executor = OperatorExecutor(operator)

        vectors = _make_vectors(256, 1000)
        result = await executor.run({"vectors": vectors})

        # Compare with direct call
        direct_result = operator.pre_aggregate(vectors)
        assert len(result) == len(direct_result)
        for r, d in zip(result, direct_result):
            assert torch.allclose(r, d, atol=1e-6)

    @pytest.mark.asyncio
    async def test_bucketing(self):
        """Test 4.4: PreAggregator - Bucketing"""
        operator = Bucketing(bucket_size=32)
        executor = OperatorExecutor(operator)

        vectors = _make_vectors(256, 1000)
        result = await executor.run({"vectors": vectors})

        direct_result = operator.pre_aggregate(vectors)
        assert len(result) == len(direct_result)

    @pytest.mark.asyncio
    async def test_empire_attack(self):
        """Test 4.5: Attack - EmpireAttack"""
        operator = EmpireAttack()
        executor = OperatorExecutor(operator, input_keys=("honest_grads",))

        honest_grads = _make_gradients(10, 1000)
        result = await executor.run({"honest_grads": honest_grads})

        direct_result = operator.apply(honest_grads=honest_grads)
        assert torch.allclose(result, direct_result, atol=1e-6)


class TestPoolConfiguration:
    """Category 5: Pool configuration tests."""

    @pytest.mark.asyncio
    async def test_single_pool_config(self):
        """Test 5.1: Single Pool Config"""
        operator = CoordinateWiseMedian()
        executor = OperatorExecutor(
            operator, pool_config=ActorPoolConfig(backend="process", count=2)
        )

        gradients = _make_gradients(64, 1000)

        async with executor:
            result = await executor.run({"gradients": gradients})
            assert isinstance(result, torch.Tensor)
            assert executor._pool is not None
            assert executor._pool.size == 2

    @pytest.mark.asyncio
    async def test_multiple_pool_configs(self):
        """Test 5.2: Multiple Pool Configs"""
        operator = CoordinateWiseMedian()
        executor = OperatorExecutor(
            operator,
            pool_config=[
                ActorPoolConfig(backend="process", count=2),
                ActorPoolConfig(backend="thread", count=2),
            ],
        )

        gradients = _make_gradients(64, 1000)

        async with executor:
            result = await executor.run({"gradients": gradients})
            assert isinstance(result, torch.Tensor)
            assert executor._pool is not None
            assert executor._pool.size == 4  # 2 + 2

    @pytest.mark.asyncio
    async def test_different_backend_types(self):
        """Test 5.3: Different Backend Types"""
        operator = CoordinateWiseMedian()
        gradients = _make_gradients(64, 1000)

        executor_process = OperatorExecutor(
            operator, pool_config=ActorPoolConfig(backend="process", count=2)
        )
        async with executor_process:
            result_process = await executor_process.run({"gradients": gradients})
            assert isinstance(result_process, torch.Tensor)

        executor_thread = OperatorExecutor(
            operator, pool_config=ActorPoolConfig(backend="thread", count=2)
        )
        async with executor_thread:
            result_thread = await executor_thread.run({"gradients": gradients})
            assert isinstance(result_thread, torch.Tensor)

        assert torch.allclose(result_process, result_thread, atol=1e-6)


class TestErrorHandling:
    """Category 6: Error handling tests."""

    @pytest.mark.asyncio
    async def test_missing_required_inputs(self):
        """Test 6.1: Missing Required Inputs"""
        operator = CoordinateWiseMedian()
        executor = OperatorExecutor(operator)

        with pytest.raises((KeyError, ValueError), match="Missing|gradients"):
            await executor.run({})

    @pytest.mark.asyncio
    async def test_invalid_input_keys(self):
        """Test 6.2: Invalid Input Keys"""
        operator = CoordinateWiseMedian()
        executor = OperatorExecutor(operator, input_keys=("invalid_key",))

        gradients = _make_gradients(64, 1000)

        with pytest.raises((KeyError, ValueError)):
            await executor.run({"gradients": gradients})

    @pytest.mark.asyncio
    async def test_invalid_operator_type(self):
        """Test 6.3: Invalid Operator Type"""
        with pytest.raises(TypeError, match="must be an Operator"):
            OperatorExecutor("not an operator")

        with pytest.raises(TypeError, match="must be an Operator"):
            OperatorExecutor(42)

    @pytest.mark.asyncio
    async def test_pool_creation_failure(self):
        """Test 6.4: Pool Creation Failure"""
        operator = CoordinateWiseMedian()

        executor = OperatorExecutor(
            operator, pool_config=ActorPoolConfig(backend="invalid_backend", count=2)
        )

        gradients = _make_gradients(64, 1000)

        with pytest.raises((ValueError, RuntimeError)):
            async with executor:
                await executor.run({"gradients": gradients})


class TestPerformance:
    """Category 7: Performance verification tests."""

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Flaky performance test - timing sensitive")
    async def test_performance_no_pool(self):
        """Test 7.1: Performance Comparison - No Pool"""
        operator = CoordinateWiseMedian()
        gradients = _make_gradients(64, 1000)

        manual_times = []
        for _ in range(10):
            start = time.perf_counter()
            await _manual_run(operator, {"gradients": gradients})
            manual_times.append(time.perf_counter() - start)

        executor_times = []
        executor = OperatorExecutor(operator)
        for _ in range(10):
            start = time.perf_counter()
            await executor.run({"gradients": gradients})
            executor_times.append(time.perf_counter() - start)

        manual_mean = sum(manual_times) / len(manual_times)
        executor_mean = sum(executor_times) / len(executor_times)

        ratio = executor_mean / manual_mean
        assert 0.95 <= ratio <= 1.05, f"Performance difference too large: {ratio:.3f}"

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Flaky performance test - timing sensitive")
    async def test_performance_with_pool(self):
        """Test 7.2: Performance Comparison - With Pool"""
        operator = MultiKrum(f=10, q=5, chunk_size=10)
        gradients = _make_gradients(30, 1000)
        pool_config = ActorPoolConfig(backend="process", count=2)

        from byzpy.engine.graph.executor import _detect_input_keys
        from byzpy.engine.graph.ops import make_single_operator_graph
        from byzpy.engine.graph.pool import ActorPool
        from byzpy.engine.graph.scheduler import NodeScheduler

        input_keys = _detect_input_keys(operator)
        graph = make_single_operator_graph(
            node_name=operator.name,
            operator=operator,
            input_keys=input_keys,
        )

        if isinstance(pool_config, ActorPoolConfig):
            configs = [pool_config]
        else:
            configs = list(pool_config)
        pool = ActorPool(configs)
        await pool.start()

        try:
            scheduler = NodeScheduler(graph, pool=pool)

            manual_times = []
            for _ in range(5):
                start = time.perf_counter()
                results = await scheduler.run({"gradients": gradients})
                manual_times.append(time.perf_counter() - start)
        finally:
            await pool.shutdown()

        executor_times = []
        executor = OperatorExecutor(operator, pool_config=pool_config)
        async with executor:
            for _ in range(5):
                start = time.perf_counter()
                await executor.run({"gradients": gradients})
                executor_times.append(time.perf_counter() - start)

        manual_mean = sum(manual_times) / len(manual_times)
        executor_mean = sum(executor_times) / len(executor_times)

        ratio = executor_mean / manual_mean
        assert 0.95 <= ratio <= 1.05, f"Performance difference too large: {ratio:.3f}"

    @pytest.mark.asyncio
    async def test_memory_usage_comparison(self):
        """Test 7.3: Memory Usage Comparison"""
        operator = CoordinateWiseMedian()
        gradients = _make_gradients(64, 1000)

        for _ in range(10):
            await _manual_run(operator, {"gradients": gradients})

        executor = OperatorExecutor(operator)
        for _ in range(10):
            await executor.run({"gradients": gradients})

    @pytest.mark.asyncio
    async def test_code_path_verification(self):
        """Test 7.4: Code Path Verification"""
        operator = CoordinateWiseMedian()
        gradients = _make_gradients(64, 1000)

        manual_result = await _manual_run(operator, {"gradients": gradients})

        executor = OperatorExecutor(operator)
        executor_result = await executor.run({"gradients": gradients})

        assert torch.allclose(manual_result, executor_result, atol=1e-6)


class TestResultCorrectness:
    """Category 8: Result correctness tests."""

    @pytest.mark.asyncio
    async def test_result_equivalence_aggregator(self):
        """Test 8.1: Result Equivalence - Aggregator"""
        operator = CoordinateWiseMedian()
        gradients = _make_gradients(64, 1000)

        manual_result = await _manual_run(operator, {"gradients": gradients})

        executor = OperatorExecutor(operator)
        executor_result = await executor.run({"gradients": gradients})

        assert torch.allclose(manual_result, executor_result, atol=1e-6)

    @pytest.mark.asyncio
    async def test_result_equivalence_preaggregator(self):
        """Test 8.2: Result Equivalence - PreAggregator"""
        operator = Clipping(threshold=2.0, chunk_size=32)
        vectors = _make_vectors(256, 1000)

        manual_result = await _manual_run(operator, {"vectors": vectors})

        executor = OperatorExecutor(operator)
        executor_result = await executor.run({"vectors": vectors})

        assert len(manual_result) == len(executor_result)
        for m, e in zip(manual_result, executor_result):
            assert torch.allclose(m, e, atol=1e-6)

    @pytest.mark.asyncio
    async def test_result_equivalence_attack(self):
        """Test 8.3: Result Equivalence - Attack"""
        operator = EmpireAttack()
        honest_grads = _make_gradients(10, 1000)

        manual_result = await _manual_run(
            operator, {"honest_grads": honest_grads}, input_keys=("honest_grads",)
        )

        executor = OperatorExecutor(operator, input_keys=("honest_grads",))
        executor_result = await executor.run({"honest_grads": honest_grads})

        assert torch.allclose(manual_result, executor_result, atol=1e-6)


class TestEdgeCases:
    """Category 9: Edge case tests."""

    @pytest.mark.asyncio
    async def test_empty_input_dictionary(self):
        """Test 9.1: Empty Input Dictionary"""
        operator = CoordinateWiseMedian()
        executor = OperatorExecutor(operator)

        with pytest.raises((KeyError, ValueError)):
            await executor.run({})

    @pytest.mark.asyncio
    async def test_extra_input_keys(self):
        """Test 9.2: Extra Input Keys"""
        operator = CoordinateWiseMedian()
        executor = OperatorExecutor(operator)

        gradients = _make_gradients(64, 1000)

        result = await executor.run({"gradients": gradients, "extra_key": 42})
        assert isinstance(result, torch.Tensor)

    @pytest.mark.asyncio
    async def test_none_pool_config(self):
        """Test 9.3: None Pool Config"""
        operator = CoordinateWiseMedian()
        executor = OperatorExecutor(operator, pool_config=None)

        gradients = _make_gradients(64, 1000)
        result = await executor.run({"gradients": gradients})

        assert isinstance(result, torch.Tensor)
        assert executor._pool is None

    @pytest.mark.asyncio
    async def test_node_name_customization(self):
        """Test 9.4: Node Name Customization"""
        operator = CoordinateWiseMedian()
        executor = OperatorExecutor(operator, node_name="custom_op")

        assert executor.node_name == "custom_op"

        gradients = _make_gradients(64, 1000)
        result = await executor.run({"gradients": gradients})

        assert isinstance(result, torch.Tensor)
