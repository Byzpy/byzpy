"""
Execution session management for lazy evaluation.

This module provides ExecutionSession for managing execution context,
caching intermediate results, and supporting incremental execution.
It also provides ExecutionFuture for non-blocking execution.

Classes
-------
ExecutionSession
    Manages execution context and caches intermediate results.
ExecutionFuture
    Future-like object for non-blocking execution.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .graph import ComputationGraph, GraphInput, GraphNode
from .lazy import GraphBuilder, LazyNode
from .parallel_scheduler import ParallelScheduler
from .pool import ActorPool


class ExecutionFuture:
    """
    Future-like object for non-blocking execution.

    This class wraps an asyncio.Task to provide a convenient interface
    for non-blocking graph execution. Results can be retrieved by
    awaiting the future or calling result() with an optional timeout.

    Parameters
    ----------
    task : asyncio.Task
        The underlying asyncio task performing the execution.
    output_keys : Sequence[str]
        Keys of the output nodes being computed.

    Examples
    --------
    >>> future = session.execute_async(graph, inputs={"data": values})
    >>> # Do other work...
    >>> if future.done():
    ...     result = await future
    >>> # Or block with timeout:
    >>> result = future.result(timeout=5.0)
    """

    def __init__(
        self,
        task: asyncio.Task,
        output_keys: Sequence[str],
    ) -> None:
        self._task = task
        self._output_keys = tuple(output_keys)

    @property
    def output_keys(self) -> Tuple[str, ...]:
        """The keys of output nodes being computed."""
        return self._output_keys

    def done(self) -> bool:
        """
        Return True if the execution has completed.

        Returns
        -------
        bool
            True if the task is done (completed, cancelled, or failed).
        """
        return self._task.done()

    def cancel(self) -> bool:
        """
        Attempt to cancel the execution.

        Returns
        -------
        bool
            True if the task was successfully cancelled.
        """
        return self._task.cancel()

    def cancelled(self) -> bool:
        """
        Return True if the execution was cancelled.

        Returns
        -------
        bool
            True if the task was cancelled.
        """
        return self._task.cancelled()

    async def wait(self) -> Dict[str, Any]:
        """
        Await the execution result.

        Returns
        -------
        Dict[str, Any]
            The execution results mapping output keys to values.

        Raises
        ------
        asyncio.CancelledError
            If the execution was cancelled.
        Exception
            Any exception raised during execution.
        """
        return await self._task

    def __await__(self):
        """Allow awaiting the future directly."""
        return self._task.__await__()

    def result(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Block until result is available.

        This method runs the event loop until the task completes or
        the timeout is exceeded.

        Parameters
        ----------
        timeout : float | None, optional
            Maximum time to wait in seconds. If None, waits indefinitely.

        Returns
        -------
        Dict[str, Any]
            The execution results.

        Raises
        ------
        asyncio.TimeoutError
            If the timeout is exceeded.
        asyncio.CancelledError
            If the execution was cancelled.
        Exception
            Any exception raised during execution.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            # We're inside an event loop - can't use run_until_complete
            # Create a new future and wait on it
            raise RuntimeError(
                "Cannot call result() from within a running event loop. "
                "Use 'await future' instead."
            )

        # No running loop - create one
        return asyncio.run(asyncio.wait_for(self._task, timeout=timeout))


class ExecutionSession:
    """
    Manages execution context and caches intermediate results.

    ExecutionSession provides a stateful execution environment that can
    cache intermediate computation results. This enables incremental
    execution where subsequent computations can reuse previously
    computed values.

    Parameters
    ----------
    pool : ActorPool | None, optional
        Optional actor pool for parallel operator execution.
    cache_intermediate : bool, optional
        Whether to cache intermediate results (default True).
    metadata : Mapping[str, Any] | None, optional
        Metadata to include in operator contexts.

    Examples
    --------
    >>> async with ExecutionSession() as session:
    ...     # First execution - computes everything
    ...     result1 = await session.execute(graph1, inputs={"data": x})
    ...
    ...     # Second execution - may reuse cached intermediates
    ...     result2 = await session.execute(graph2, inputs={"data": y})

    Notes
    -----
    - Results are cached by node name (key)
    - Cache is keyed by node name, not by input values
    - Use clear_cache() to reset cached results
    """

    def __init__(
        self,
        pool: ActorPool | None = None,
        cache_intermediate: bool = True,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.pool = pool
        self.cache_intermediate = cache_intermediate
        self.metadata = dict(metadata or {})
        self._result_cache: Dict[str, Any] = {}

    async def __aenter__(self) -> "ExecutionSession":
        """Enter async context manager."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context manager."""
        # Clear cache on exit
        self._result_cache.clear()

    async def execute(
        self,
        graph: ComputationGraph,
        inputs: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute a computation graph.

        If caching is enabled, results from previous executions may be
        reused. New results are added to the cache.

        Parameters
        ----------
        graph : ComputationGraph
            The computation graph to execute.
        inputs : Mapping[str, Any]
            Input data mapping input names to values.

        Returns
        -------
        Dict[str, Any]
            Dictionary mapping output node names to their computed values.
        """
        # Build pruned graph if caching is enabled
        if self.cache_intermediate:
            pruned_graph, cached_inputs = self._prune_cached_nodes(graph)
            # Merge cached values into inputs
            all_inputs = dict(inputs)
            all_inputs.update(cached_inputs)

            if pruned_graph is None:
                # All nodes are cached - return from cache
                return {name: self._result_cache[name] for name in graph.outputs}

            # Execute pruned graph with ALL nodes as outputs (for caching)
            # We need all intermediate results to be returned for caching
            all_node_names = [node.name for node in pruned_graph.nodes_in_order()]
            pruned_graph_all_outputs = ComputationGraph(
                list(pruned_graph.nodes_in_order()),
                outputs=all_node_names,
            )

            scheduler = ParallelScheduler(
                pruned_graph_all_outputs,
                pool=self.pool,
                metadata=self.metadata,
            )
            results = await scheduler.run(all_inputs)

            # Update cache with ALL computed results (including intermediates)
            self._result_cache.update(results)

            # Return only the requested outputs (may include cached values)
            return {name: self._result_cache.get(name, results.get(name)) for name in graph.outputs}
        else:
            # No caching - execute full graph
            scheduler = ParallelScheduler(
                graph,
                pool=self.pool,
                metadata=self.metadata,
            )
            return await scheduler.run(inputs)

    def execute_async(
        self,
        graph: ComputationGraph,
        inputs: Mapping[str, Any],
    ) -> ExecutionFuture:
        """
        Execute a computation graph asynchronously (non-blocking).

        Returns immediately with an ExecutionFuture that can be awaited
        or polled for completion.

        Parameters
        ----------
        graph : ComputationGraph
            The computation graph to execute.
        inputs : Mapping[str, Any]
            Input data mapping input names to values.

        Returns
        -------
        ExecutionFuture
            A future representing the pending execution.
        """
        task = asyncio.create_task(self.execute(graph, inputs))
        return ExecutionFuture(task, output_keys=graph.outputs)

    def _prune_cached_nodes(
        self,
        graph: ComputationGraph,
    ) -> Tuple[Optional[ComputationGraph], Dict[str, Any]]:
        """
        Create a pruned graph that excludes already-cached nodes.

        Returns a new graph with cached nodes removed (replaced by inputs),
        and a dict of cached values to provide as inputs.

        Parameters
        ----------
        graph : ComputationGraph
            The original computation graph.

        Returns
        -------
        Tuple[Optional[ComputationGraph], Dict[str, Any]]
            (pruned_graph, cached_inputs) where pruned_graph is None if
            all nodes are cached, or a new graph with cached nodes removed.
            cached_inputs contains the cached values to use as inputs.
        """
        # Collect nodes that need to be computed
        nodes_to_compute = []
        cached_inputs: Dict[str, Any] = {}

        for node in graph.nodes_in_order():
            if node.name in self._result_cache:
                # Node is cached - will be provided as input
                cached_inputs[node.name] = self._result_cache[node.name]
            else:
                # Node needs computation - remap inputs
                new_inputs = {}
                for arg, dep in node.inputs.items():
                    if isinstance(dep, GraphInput):
                        new_inputs[arg] = dep
                    elif isinstance(dep, str):
                        if dep in self._result_cache:
                            # Dependency is cached - convert to GraphInput
                            new_inputs[arg] = GraphInput(dep)
                        else:
                            # Dependency will be computed
                            new_inputs[arg] = dep
                    else:
                        new_inputs[arg] = dep

                new_node = GraphNode(
                    name=node.name,
                    op=node.op,
                    inputs=new_inputs,
                )
                nodes_to_compute.append(new_node)

        if not nodes_to_compute:
            # All nodes are cached
            return None, cached_inputs

        # Determine which outputs are in the new graph
        new_outputs = [name for name in graph.outputs if name not in self._result_cache]

        # If all outputs are cached, return None
        if not new_outputs:
            return None, cached_inputs

        return ComputationGraph(nodes_to_compute, outputs=new_outputs), cached_inputs

    def clear_cache(self) -> None:
        """
        Clear all cached results.

        After calling this method, subsequent executions will recompute
        all nodes.
        """
        self._result_cache.clear()

    def get_cached(self, key: str) -> Any:
        """
        Get a cached result by key.

        Parameters
        ----------
        key : str
            The node name to retrieve.

        Returns
        -------
        Any
            The cached value.

        Raises
        ------
        KeyError
            If the key is not in the cache.
        """
        return self._result_cache[key]

    def is_cached(self, key: str) -> bool:
        """
        Check if a key is in the cache.

        Parameters
        ----------
        key : str
            The node name to check.

        Returns
        -------
        bool
            True if the key is cached.
        """
        return key in self._result_cache


__all__ = ["ExecutionSession", "ExecutionFuture"]
