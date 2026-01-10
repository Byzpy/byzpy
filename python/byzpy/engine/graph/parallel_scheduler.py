"""
Parallel scheduler for executing computation graphs with inter-node parallelism.

This module provides ParallelScheduler, which executes independent graph nodes
concurrently using asyncio, while respecting data dependencies between nodes.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

from .graph import ComputationGraph, GraphInput, GraphNode
from .operator import OpContext
from .pool import ActorPool


class ParallelScheduler:
    """
    Scheduler that executes independent graph nodes concurrently.

    This scheduler provides inter-node parallelism by running independent nodes
    in parallel using asyncio. It tracks node dependencies using in-degree counts
    and schedules nodes for execution as soon as all their dependencies complete.

    Parameters
    ----------
    graph : ComputationGraph
        The computation graph to execute.
    pool : ActorPool | None, optional
        Optional actor pool for intra-operator parallel execution (subtasks).
        If None, operators run on the main thread.
    metadata : Optional[Mapping[str, Any]], optional
        Metadata to include in operator contexts (e.g., node name, pool size).
    max_concurrent_nodes : int | None, optional
        Maximum number of nodes to execute concurrently. If None, no limit is
        applied and all ready nodes run in parallel.
    max_pending_subtasks : int | None, optional
        Maximum number of subtasks pending across all concurrent operators.
        This provides a per-scheduler limit to prevent overwhelming the pool
        when multiple operators run in parallel. If None, defaults to
        pool.size * 8 when a pool is provided. Set to 0 to disable the
        shared limit (each operator uses its own local limit).

    Examples
    --------
    >>> from byzpy.engine.graph.graph import ComputationGraph, GraphNode, graph_input
    >>> from byzpy.engine.graph.parallel_scheduler import ParallelScheduler
    >>> graph = ComputationGraph([...])
    >>> scheduler = ParallelScheduler(graph)
    >>> results = await scheduler.run({"input": data})

    Notes
    -----
    - Independent nodes (nodes with no dependencies on each other) execute concurrently
    - A node is scheduled when all its dependencies have completed
    - The scheduler uses asyncio.gather() for efficient concurrent execution
    - Errors in any node are propagated immediately
    - The max_pending_subtasks limit is shared across all concurrent operators,
      preventing queue overflow when many operators run in parallel.
    """

    def __init__(
        self,
        graph: ComputationGraph,
        *,
        pool: ActorPool | None = None,
        metadata: Optional[Mapping[str, Any]] = None,
        max_concurrent_nodes: int | None = None,
        max_pending_subtasks: int | None = None,
    ) -> None:
        self.graph = graph
        self.pool = pool
        self.metadata = dict(metadata or {})
        self.max_concurrent_nodes = max_concurrent_nodes

        # Per-scheduler subtask limit
        # Default: pool.size * 8 when pool is provided
        if max_pending_subtasks is None and pool is not None:
            max_pending_subtasks = pool.size * 8
        self.max_pending_subtasks = max_pending_subtasks

        # Create shared semaphore for subtask coordination
        # (created lazily in run() to ensure we're in an event loop)
        self._subtask_semaphore: Optional[asyncio.Semaphore] = None

        # Build dependency tracking structures
        self._node_map: Dict[str, GraphNode] = {}
        self._in_degree: Dict[str, int] = {}
        self._dependents: Dict[str, List[str]] = defaultdict(list)

        self._build_dependency_graph()

    def _build_dependency_graph(self) -> None:
        """Build in-degree counts and dependent lists for parallel scheduling."""
        # Initialize all nodes with in-degree 0
        for node in self.graph.nodes_in_order():
            self._node_map[node.name] = node
            self._in_degree[node.name] = 0

        # Calculate actual in-degrees by counting node dependencies
        for node in self._node_map.values():
            for dep in node.inputs.values():
                if isinstance(dep, str) and dep in self._node_map:
                    # dep is a node name (dependency on another node)
                    self._in_degree[node.name] += 1
                    self._dependents[dep].append(node.name)

    async def run(self, inputs: Mapping[str, Any]) -> Dict[str, Any]:
        """
        Execute the computation graph with maximum parallelism.

        Independent nodes are executed concurrently. When a node completes,
        its dependents are checked and scheduled if all their dependencies
        are satisfied.

        Parameters
        ----------
        inputs : Mapping[str, Any]
            Input data mapping input names to values.

        Returns
        -------
        Dict[str, Any]
            Dictionary mapping output node names to their computed values.

        Raises
        ------
        ValueError
            If required inputs are missing.
        Exception
            Any exception raised by an operator is propagated.
        """
        # Validate inputs
        missing = [name for name in self.graph.required_inputs if name not in inputs]
        if missing:
            raise ValueError(f"Missing graph inputs: {missing}")

        # Initialize cache with inputs
        cache: Dict[str, Any] = dict(inputs)

        # Track remaining in-degrees (mutable copy)
        remaining: Dict[str, int] = dict(self._in_degree)

        # Find initially ready nodes (in-degree 0)
        ready: List[str] = [name for name, degree in remaining.items() if degree == 0]

        # Create shared subtask semaphore (if limit is set and positive)
        subtask_semaphore: Optional[asyncio.Semaphore] = None
        if self.max_pending_subtasks is not None and self.max_pending_subtasks > 0:
            subtask_semaphore = asyncio.Semaphore(self.max_pending_subtasks)

        # Build operator context once (reused for all nodes)
        base_metadata = dict(self.metadata)
        if self.pool is not None:
            base_metadata.setdefault("pool_size", self.pool.size)
            base_metadata.setdefault("worker_affinities", tuple(self.pool.worker_affinities()))

        # Add shared semaphore to metadata for operators to use
        if subtask_semaphore is not None:
            base_metadata["subtask_semaphore"] = subtask_semaphore

        while ready:
            # Execute all ready nodes
            if len(ready) == 1:
                # Single node: run directly without asyncio overhead
                node_name = ready.pop()
                name, result = await self._execute_node(node_name, cache, base_metadata)
                cache[name] = result
                # Update dependents
                for dependent in self._dependents[name]:
                    remaining[dependent] -= 1
                    if remaining[dependent] == 0:
                        ready.append(dependent)
            else:
                # Multiple ready nodes: run concurrently with gather
                # Apply concurrency limit if specified
                batch = ready
                if self.max_concurrent_nodes is not None and self.max_concurrent_nodes > 0:
                    batch = ready[: self.max_concurrent_nodes]
                    ready = ready[self.max_concurrent_nodes :]
                else:
                    ready = []

                # Execute batch concurrently
                tasks = [self._execute_node(node_name, cache, base_metadata) for node_name in batch]
                results = await asyncio.gather(*tasks)

                # Process results and update dependents
                for name, result in results:
                    cache[name] = result
                    for dependent in self._dependents[name]:
                        remaining[dependent] -= 1
                        if remaining[dependent] == 0:
                            ready.append(dependent)

        return {name: cache[name] for name in self.graph.outputs}

    async def _execute_node(
        self,
        node_name: str,
        cache: Dict[str, Any],
        base_metadata: Dict[str, Any],
    ) -> Tuple[str, Any]:
        """
        Execute a single graph node.

        Parameters
        ----------
        node_name : str
            Name of the node to execute.
        cache : Dict[str, Any]
            Cache containing computed values from previous nodes and inputs.
        base_metadata : Dict[str, Any]
            Base metadata for operator context.

        Returns
        -------
        Tuple[str, Any]
            Tuple of (node_name, result).
        """
        node = self._node_map[node_name]

        # Resolve inputs from cache
        node_inputs = self._resolve_inputs(node, cache)

        # Build context
        ctx = OpContext(node_name=node.name, metadata=base_metadata)

        # Execute operator (with intra-node parallelism via pool if available)
        result = await node.op.run(node_inputs, context=ctx, pool=self.pool)

        return node_name, result

    def _resolve_inputs(
        self,
        node: GraphNode,
        cache: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Resolve node inputs from the cache.

        Parameters
        ----------
        node : GraphNode
            The node whose inputs need resolution.
        cache : Dict[str, Any]
            Cache containing computed values.

        Returns
        -------
        Dict[str, Any]
            Resolved input values.

        Raises
        ------
        KeyError
            If a required dependency is not in the cache.
        """
        resolved: Dict[str, Any] = {}
        for arg, dep in node.inputs.items():
            if isinstance(dep, GraphInput):
                resolved[arg] = cache[dep.name]
            elif isinstance(dep, str):
                if dep not in cache:
                    raise KeyError(
                        f"Graph node {node.name} depends on {dep!r}, which has not been computed."
                    )
                resolved[arg] = cache[dep]
            else:
                # MessageSource or other - handle in subclass if needed
                # For now, treat as direct value (shouldn't happen in typical usage)
                resolved[arg] = dep
        return resolved


__all__ = ["ParallelScheduler"]
