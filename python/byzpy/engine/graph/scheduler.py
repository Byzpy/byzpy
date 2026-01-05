from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any, Dict, List, Mapping, MutableMapping, Optional

from .graph import ComputationGraph, GraphInput, GraphNode
from .operator import OpContext
from .pool import ActorPool


class NodeScheduler:
    """
    Scheduler for executing computation graphs.

    This class evaluates a computation graph by executing nodes in topological
    order. It dispatches operators to an actor pool for parallel execution
    when subtasks are supported.

    Parameters
    ----------
    graph : ComputationGraph
        The computation graph to execute.
    pool : ActorPool | None, optional
        Optional actor pool for parallel operator execution. If None, operators
        run synchronously on the main thread.
    metadata : Optional[Mapping[str, Any]], optional
        Metadata to include in operator contexts (e.g., node name, pool size).

    Examples
    --------
    >>> from byzpy.engine.graph.graph import ComputationGraph
    >>> from byzpy.engine.graph.pool import ActorPool, ActorPoolConfig
    >>> graph = make_single_operator_graph(...)
    >>> pool = ActorPool([ActorPoolConfig(backend="thread", count=4)])
    >>> scheduler = NodeScheduler(graph, pool=pool)
    >>> results = await scheduler.run({"gradients": gradients})
    """

    def __init__(
        self,
        graph: ComputationGraph,
        *,
        pool: ActorPool | None = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.graph = graph
        self.pool = pool
        self.metadata = dict(metadata or {})

    async def run(self, inputs: Mapping[str, Any]) -> Dict[str, Any]:
        missing = [name for name in self.graph.required_inputs if name not in inputs]
        if missing:
            raise ValueError(f"Missing graph inputs: {missing}")

        cache: Dict[str, Any] = dict(inputs)

        for node in self.graph.nodes_in_order():
            node_inputs = self._resolve_inputs(node, cache)
            metadata = dict(self.metadata)
            if self.pool is not None:
                metadata.setdefault("pool_size", self.pool.size)
                metadata.setdefault("worker_affinities", tuple(self.pool.worker_affinities()))
            ctx = OpContext(node_name=node.name, metadata=metadata)
            result = await node.op.run(node_inputs, context=ctx, pool=self.pool)
            cache[node.name] = result

        return {name: cache[name] for name in self.graph.outputs}

    def _resolve_inputs(self, node: GraphNode, cache: MutableMapping[str, Any]) -> Dict[str, Any]:
        resolved: Dict[str, Any] = {}
        for arg, dep in node.inputs.items():
            if isinstance(dep, GraphInput):
                resolved[arg] = cache[dep.name]
            else:
                if dep not in cache:
                    raise KeyError(
                        f"Graph node {node.name} depends on {dep!r}, which has not been computed."
                    )
                resolved[arg] = cache[dep]
        return resolved


class MessageSource:
    """Represents a graph input that comes from a message."""

    def __init__(
        self,
        message_type: str,
        field: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        self.message_type = message_type
        self.field = field
        self.timeout = timeout


class MessageAwareNodeScheduler(NodeScheduler):
    """
    Extended NodeScheduler with message-driven execution support.

    This scheduler extends NodeScheduler to support message-driven computation
    graphs where nodes can wait for messages from other nodes before proceeding.
    This is essential for decentralized peer-to-peer training where nodes
    communicate asynchronously.

    The scheduler maintains a message cache and waiters, allowing operators
    to use MessageTriggerOp to wait for specific message types.

    Examples
    --------
    >>> scheduler = MessageAwareNodeScheduler(graph, pool=pool)
    >>> # In another task, deliver a message
    >>> scheduler.deliver_message("gradient", payload=gradient_vector)
    >>> # In graph execution, wait for message
    >>> message = await scheduler.wait_for_message("gradient")
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._message_waiters: Dict[str, List[asyncio.Future]] = {}
        self._message_cache: Dict[str, List[Any]] = defaultdict(list)

    async def wait_for_message(
        self,
        message_type: str,
        *,
        timeout: Optional[float] = None,
    ) -> Any:
        """
        Wait for a specific message type to arrive.

        This method blocks until a message of the specified type is delivered
        via :meth:`deliver_message`. If a message is already cached, it returns
        immediately.

        Parameters
        ----------
        message_type : str
            Type identifier for the message to wait for.
        timeout : Optional[float], optional
            Maximum time to wait in seconds. If None, waits indefinitely.
            Default is None.

        Returns
        -------
        Any
            The message payload.

        Raises
        ------
        asyncio.TimeoutError
            If timeout is exceeded before message arrives.
        """
        # Check cache first
        if self._message_cache.get(message_type):
            return self._message_cache[message_type].pop(0)

        # Wait for message
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        self._message_waiters.setdefault(message_type, []).append(fut)

        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            # Remove from waiters if timeout
            waiters = self._message_waiters.get(message_type, [])
            if fut in waiters:
                waiters.remove(fut)
            raise

    def deliver_message(self, message_type: str, payload: Any) -> None:
        """
        Deliver a message, waking up any waiters.

        This method delivers a message of the specified type. If there are
        any tasks waiting for this message type, they are immediately woken
        up. Otherwise, the message is cached for future waiters.

        Parameters
        ----------
        message_type : str
            Type identifier for the message.
        payload : Any
            The message payload to deliver.
        """
        waiters = self._message_waiters.pop(message_type, [])
        for fut in waiters:
            if not fut.done():
                fut.set_result(payload)

        # Cache for future waiters
        self._message_cache[message_type].append(payload)

    async def run(self, inputs: Mapping[str, Any]) -> Dict[str, Any]:
        """Override to handle message-driven inputs."""
        # Resolve message sources in inputs dict
        resolved_inputs = {}
        for key, value in inputs.items():
            if isinstance(value, MessageSource):
                # Wait for message
                msg = await self.wait_for_message(value.message_type, timeout=value.timeout)
                if value.field:
                    if isinstance(msg, dict):
                        if value.field not in msg:
                            raise KeyError(
                                f"Message field '{value.field}' not found in message payload"
                            )
                        resolved_inputs[key] = msg[value.field]
                    else:
                        raise TypeError(
                            f"Cannot extract field '{value.field}' from non-dict message"
                        )
                else:
                    resolved_inputs[key] = msg
            else:
                resolved_inputs[key] = value

        # Now resolve message sources in graph node inputs
        missing = [name for name in self.graph.required_inputs if name not in resolved_inputs]
        if missing:
            raise ValueError(f"Missing graph inputs: {missing}")

        cache: Dict[str, Any] = dict(resolved_inputs)

        for node in self.graph.nodes_in_order():
            node_inputs = await self._resolve_inputs(node, cache)
            metadata = dict(self.metadata)
            if self.pool is not None:
                metadata.setdefault("pool_size", self.pool.size)
                metadata.setdefault("worker_affinities", tuple(self.pool.worker_affinities()))
            # Add scheduler to metadata for MessageTriggerOp
            metadata["scheduler"] = self
            ctx = OpContext(node_name=node.name, metadata=metadata)
            result = await node.op.run(node_inputs, context=ctx, pool=self.pool)
            cache[node.name] = result

        return {name: cache[name] for name in self.graph.outputs}

    async def _resolve_inputs(
        self, node: GraphNode, cache: MutableMapping[str, Any]
    ) -> Dict[str, Any]:
        """Override to handle MessageSource in node inputs."""
        resolved: Dict[str, Any] = {}
        for arg, dep in node.inputs.items():
            if isinstance(dep, MessageSource):
                # Direct MessageSource in node inputs
                msg = await self.wait_for_message(dep.message_type, timeout=dep.timeout)
                if dep.field:
                    if isinstance(msg, dict):
                        if dep.field not in msg:
                            raise KeyError(
                                f"Message field '{dep.field}' not found in message payload"
                            )
                        resolved[arg] = msg[dep.field]
                    else:
                        raise TypeError(f"Cannot extract field '{dep.field}' from non-dict message")
                else:
                    resolved[arg] = msg
            elif isinstance(dep, GraphInput):
                resolved[arg] = cache[dep.name]
            else:
                if dep not in cache:
                    raise KeyError(
                        f"Graph node {node.name} depends on {dep!r}, which has not been computed."
                    )
                resolved[arg] = cache[dep]
        return resolved


__all__ = ["NodeScheduler", "MessageAwareNodeScheduler", "MessageSource"]
