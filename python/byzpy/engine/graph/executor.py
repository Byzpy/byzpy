"""
Simplified operator execution API.

Provides OperatorExecutor and run_operator for easy operator execution without
manual graph creation.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from .operator import Operator
from .ops import make_single_operator_graph
from .pool import ActorPool, ActorPoolConfig
from .scheduler import NodeScheduler

try:
    from byzpy.aggregators.base import Aggregator
except ImportError:
    Aggregator = None

try:
    from byzpy.pre_aggregators.base import PreAggregator
except ImportError:
    PreAggregator = None

try:
    from byzpy.attacks.base import Attack
except ImportError:
    Attack = None


def _detect_input_keys(operator: Operator) -> Sequence[str]:
    """
    Auto-detect input keys based on operator type.

    Args:
        operator: The operator to detect input keys for.

    Returns:
        Sequence of input key names.

    Raises:
        ValueError: If input keys cannot be auto-detected.
    """
    # Check if it's an Aggregator
    if Aggregator is not None and isinstance(operator, Aggregator):
        return (operator.input_key,)

    # Check if it's a PreAggregator
    if PreAggregator is not None and isinstance(operator, PreAggregator):
        return (operator.input_key,)

    # Check if it's an Attack
    if Attack is not None and isinstance(operator, Attack):
        # Attacks have variable inputs based on their flags
        # They typically need: honest_grads, base_grad, model/x/y, etc.
        # Cannot auto-detect reliably - require explicit input_keys
        raise ValueError(
            f"Cannot auto-detect input keys for Attack {type(operator).__name__}. "
            "Attacks have variable input requirements. Please specify input_keys explicitly."
        )

    # Cannot auto-detect
    raise ValueError(
        f"Cannot auto-detect input keys for operator {type(operator).__name__}. "
        "Please specify input_keys explicitly."
    )


class OperatorExecutor:
    """
    Simple executor for running a single operator with optional parallelism.

    Automatically creates computation graph, scheduler, and manages actor pool lifecycle.
    This is a pure convenience wrapper with zero performance overhead - it executes
    the exact same code paths as manual boilerplate.
    """

    def __init__(
        self,
        operator: Operator,
        *,
        input_keys: Sequence[str] | None = None,
        pool_config: ActorPoolConfig | Sequence[ActorPoolConfig] | None = None,
        node_name: str | None = None,
    ):
        """
        Args:
            operator: The operator to execute (Aggregator, PreAggregator, Attack, etc.)
            input_keys: Names of input parameters. If None, auto-detect from operator type.
            pool_config: Optional actor pool configuration for parallelism.
            node_name: Optional name for the graph node. Defaults to operator name.
        """
        if not isinstance(operator, Operator):
            raise TypeError(f"operator must be an Operator instance, got {type(operator)}")

        self.operator = operator
        self.pool_config = pool_config
        self.node_name = node_name or operator.name

        # Auto-detect input keys if not provided
        if input_keys is None:
            self.input_keys = _detect_input_keys(operator)
        else:
            self.input_keys = tuple(input_keys)

        # Pre-compute whether input mapping is needed (to avoid overhead in run())
        self._needs_input_mapping = False
        self._operator_input_key = None
        if hasattr(operator, "input_key") and len(self.input_keys) == 1:
            self._operator_input_key = operator.input_key
            if self.input_keys[0] != operator.input_key:
                self._needs_input_mapping = True

        # Pool is created on-demand (lazy initialization)
        self._pool: ActorPool | None = None
        self._pool_managed = False  # Track if we created the pool (for cleanup)

        # Cache graph and scheduler for reuse (matching manual boilerplate pattern)
        self._graph = None
        self._scheduler = None

    async def __aenter__(self):
        """Context manager entry - starts actor pool if needed."""
        if self.pool_config is not None and self._pool is None:
            await self._ensure_pool()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - shuts down actor pool if needed."""
        await self._cleanup_pool()

    async def _ensure_pool(self) -> None:
        """Ensure actor pool is created and started."""
        if self._pool is not None:
            return

        # Create pool from config
        if isinstance(self.pool_config, ActorPoolConfig):
            configs = [self.pool_config]
        else:
            configs = list(self.pool_config)

        self._pool = ActorPool(configs)
        await self._pool.start()
        self._pool_managed = True

    async def _cleanup_pool(self) -> None:
        """Clean up actor pool if we created it."""
        if self._pool is not None and self._pool_managed:
            await self._pool.shutdown()
            self._pool = None
            self._pool_managed = False
            # Invalidate cached scheduler when pool is cleaned up
            self._scheduler = None

    async def run(self, inputs: Mapping[str, Any]) -> Any:
        """
        Execute the operator with given inputs.

        Args:
            inputs: Dictionary mapping input keys to values.

        Returns:
            The result of operator execution.
        """
        # Ensure pool is started if needed
        if self.pool_config is not None:
            await self._ensure_pool()

        # Reuse cached graph and scheduler if pool is configured (matching manual boilerplate pattern)
        # When pool is None, create fresh graph/scheduler on every call (like manual boilerplate)
        # When pool is configured, reuse scheduler across calls (like manual boilerplate in pool test)
        should_cache = self.pool_config is not None

        if (
            should_cache
            and self._graph is not None
            and self._scheduler is not None
            and self._scheduler.pool == self._pool
        ):
            # Use cached scheduler
            scheduler = self._scheduler
        else:
            # Create graph and scheduler (matching manual boilerplate)
            # Check if we need to map input keys to operator's expected keys
            # This happens when user provides custom input_keys that differ from
            # the operator's default input_key (e.g., Aggregator expects "gradients")
            if self._needs_input_mapping:
                # Create a wrapper operator that maps custom keys to operator's expected keys
                # We need to use a proper Operator subclass, not CallableOp, to preserve context
                from .operator import OpContext, Operator

                class _InputMappingOperator(Operator):
                    """Wrapper operator that maps input keys."""

                    def __init__(self, wrapped_op: Operator, custom_key: str, op_key: str):
                        self.wrapped_op = wrapped_op
                        self.custom_key = custom_key
                        self.op_key = op_key
                        self.name = wrapped_op.name
                        self.supports_subtasks = wrapped_op.supports_subtasks
                        self.max_subtasks_inflight = wrapped_op.max_subtasks_inflight
                        self.supports_barriered_subtasks = wrapped_op.supports_barriered_subtasks

                    def compute(self, inputs: Mapping[str, Any], *, context: OpContext) -> Any:
                        # Map custom key to operator's expected key
                        if self.custom_key not in inputs:
                            raise KeyError(f"Missing input key {self.custom_key!r}")
                        mapped_inputs = {self.op_key: inputs[self.custom_key]}
                        return self.wrapped_op.compute(mapped_inputs, context=context)

                    def create_subtasks(
                        self, inputs: Mapping[str, Any], *, context: OpContext
                    ) -> Iterable[Any]:
                        mapped_inputs = {self.op_key: inputs[self.custom_key]}
                        return self.wrapped_op.create_subtasks(mapped_inputs, context=context)

                    def reduce_subtasks(
                        self,
                        partials: Sequence[Any],
                        inputs: Mapping[str, Any],
                        *,
                        context: OpContext,
                    ) -> Any:
                        mapped_inputs = {self.op_key: inputs[self.custom_key]}
                        return self.wrapped_op.reduce_subtasks(
                            partials, mapped_inputs, context=context
                        )

                # Create wrapper operator
                wrapper_op = _InputMappingOperator(
                    self.operator, self.input_keys[0], self._operator_input_key
                )

                # Create graph with wrapper
                graph = make_single_operator_graph(
                    node_name=self.node_name,
                    operator=wrapper_op,
                    input_keys=self.input_keys,
                )
            else:
                # Create graph (identical to manual boilerplate)
                graph = make_single_operator_graph(
                    node_name=self.node_name,
                    operator=self.operator,
                    input_keys=self.input_keys,
                )

            # Create scheduler (identical to manual boilerplate)
            scheduler = NodeScheduler(graph, pool=self._pool)

            # Cache if we should
            if should_cache:
                self._graph = graph
                self._scheduler = scheduler

        # Run scheduler (identical to manual boilerplate)
        results = await scheduler.run(inputs)

        # Return the result (graph outputs a single value for single-operator graphs)
        return results[self.node_name]


async def run_operator(
    operator: Operator,
    inputs: Mapping[str, Any],
    *,
    pool_config: ActorPoolConfig | Sequence[ActorPoolConfig] | None = None,
    input_keys: Sequence[str] | None = None,
) -> Any:
    """
    Run an operator with given inputs, optionally using an actor pool.

    This is a convenience function that creates an OperatorExecutor,
    runs the operator, and cleans up automatically.

    Args:
        operator: The operator to execute.
        inputs: Dictionary mapping input keys to values.
        pool_config: Optional actor pool configuration.
        input_keys: Optional input key names (auto-detected if None).

    Returns:
        The result of operator execution.
    """
    async with OperatorExecutor(
        operator, pool_config=pool_config, input_keys=input_keys
    ) as executor:
        return await executor.run(inputs)


__all__ = ["OperatorExecutor", "run_operator"]
