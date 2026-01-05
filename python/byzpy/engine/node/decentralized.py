from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, List, Mapping, Optional, Union

from ..graph.scheduler import MessageAwareNodeScheduler
from .application import NodeApplication
from .context import NodeContext
from .router import MessageRouter


class DecentralizedNode:
    """
    Unified node runtime that integrates NodeScheduler, NodeApplication,
    and message routing for fully decentralized execution.

    This is the core component for Milestone 1, providing:
    - Integration of NodeScheduler and NodeApplication
    - Basic message inbox/outbox
    - State management
    """

    def __init__(
        self,
        *,
        node_id: Union[int, str],
        application: NodeApplication,
        context: NodeContext,
        topology: Optional[Any] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        node_id_map: Optional[Dict[int, str]] = None,
    ):
        if node_id is None or (isinstance(node_id, str) and node_id == ""):
            raise ValueError("node_id cannot be empty")

        self.node_id = node_id
        self.application = application
        self.context = context
        self.topology = topology
        self._node_id_map = node_id_map

        # Create scheduler (using MessageAwareNodeScheduler for Milestone 2)
        # Note: NodeScheduler requires a graph, so we'll create a dummy one
        # and replace it per pipeline execution
        merged_metadata = dict(metadata or {})
        merged_metadata["node_id"] = node_id

        # Create a minimal dummy graph for initialization (will be replaced per pipeline)
        from ..graph.graph import ComputationGraph, GraphNode
        from ..graph.operator import Operator

        class _NoOp(Operator):
            def compute(self, inputs, *, context):
                return None

        dummy_node = GraphNode(
            name="dummy",
            op=_NoOp(),
            inputs={},
        )
        dummy_graph = ComputationGraph(nodes=[dummy_node], outputs=["dummy"])

        self.scheduler = MessageAwareNodeScheduler(
            graph=dummy_graph,
            pool=application.pool,
            metadata=merged_metadata,
        )

        # Topology-aware message router (Milestone 4)
        self.message_router = MessageRouter(
            topology=topology, node_id=node_id, node_id_map=node_id_map
        )

        # State and handlers
        self._state: Dict[str, Any] = {}
        self._message_handlers: Dict[str, Callable[[str, Any], Awaitable[None]]] = {}
        self._running = False
        self._message_task: Optional[asyncio.Task] = None
        self._autonomous_tasks: Dict[str, asyncio.Task] = {}  # Track autonomous background tasks

        # Register default message handlers
        self._register_default_handlers()

    async def start(self) -> None:
        """Start the node's event loop and message processing."""
        if self._running:
            return

        self._running = True
        await self.context.start(self)

        # Start message processing task
        self._message_task = asyncio.create_task(self._message_processing_loop())

    async def _message_processing_loop(self) -> None:
        """Continuously process incoming messages."""
        try:
            async for msg in self.context.receive_messages():
                if not self._running:
                    break
                await self.handle_incoming_message(
                    from_node_id=msg.get("from", "unknown"),
                    message_type=msg.get("type", "unknown"),
                    payload=msg.get("payload"),
                )
        except asyncio.CancelledError:
            pass

    async def handle_incoming_message(
        self,
        from_node_id: str,
        message_type: str,
        payload: Any,
    ) -> None:
        """Process incoming message and trigger graph execution if needed."""
        # Deliver to scheduler (for message-driven graph nodes)
        if isinstance(self.scheduler, MessageAwareNodeScheduler):
            self.scheduler.deliver_message(message_type, payload)

        # Call registered handler
        handler = self._message_handlers.get(message_type)
        if handler:
            await handler(from_node_id, payload)

    async def send_message(
        self,
        to_node_id: Union[int, str],
        message_type: str,
        payload: Any,
    ) -> None:
        """Send message to another node via topology."""
        if not self._running:
            raise RuntimeError("Node not started")

        if not self.message_router.can_send_to(to_node_id):
            raise ValueError(f"Cannot send to {to_node_id} (not a neighbor)")

        await self.message_router.route_direct(
            to_node_id,
            message_type,
            payload,
            self.context,
        )

    async def broadcast_message(
        self,
        message_type: str,
        payload: Any,
    ) -> None:
        """Broadcast message to all neighbors."""
        if not self._running:
            raise RuntimeError("Node not started")

        await self.message_router.route_broadcast(
            message_type,
            payload,
            self.context,
        )

    async def multicast_message(
        self,
        to_node_ids: List[Union[int, str]],
        message_type: str,
        payload: Any,
    ) -> None:
        """Multicast message to specified neighbors."""
        if not self._running:
            raise RuntimeError("Node not started")

        await self.message_router.route_multicast(
            to_node_ids,
            message_type,
            payload,
            self.context,
        )

    def get_neighbors(self) -> List[Union[int, str]]:
        """Get list of outgoing neighbors."""
        return self.message_router.get_out_neighbors()

    def get_in_neighbors(self) -> List[Union[int, str]]:
        """Get list of incoming neighbors."""
        return self.message_router.get_in_neighbors()

    async def execute_pipeline(
        self,
        pipeline_name: str,
        inputs: Mapping[str, Any],
        *,
        triggered_by: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute a pipeline via the node's scheduler."""
        if not self._running:
            raise RuntimeError("Node not started")

        pipeline = self.application._pipelines.get(pipeline_name)
        if not pipeline:
            raise KeyError(f"Unknown pipeline: {pipeline_name}")

        # Update scheduler's graph
        self.scheduler.graph = pipeline.graph

        # Update scheduler metadata to include scheduler reference for MessageTriggerOp
        if isinstance(self.scheduler, MessageAwareNodeScheduler):
            self.scheduler.metadata["scheduler"] = self.scheduler

        # Execute
        return await self.scheduler.run(inputs)

    def register_message_handler(
        self,
        message_type: str,
        handler: Callable[[str, Any], Awaitable[None]],
    ) -> None:
        """Register a handler for a specific message type."""
        self._message_handlers[message_type] = handler

    def _register_default_handlers(self) -> None:
        """Register default message handlers."""
        # Can be extended for common patterns
        pass

    async def start_autonomous_task(
        self,
        task_coro: Awaitable[Any],
        name: str = "autonomous_task",
    ) -> asyncio.Task:
        """
        Start an autonomous background task in this node's event loop.

        Args:
            task_coro: Coroutine to run as an autonomous task
            name: Name for the task (used for tracking)

        Returns:
            The asyncio.Task that was created

        Example:
            >>> async def training_loop(node):
            ...     while node._running:
            ...         # Training logic
            ...         await asyncio.sleep(0.1)
            >>> task = await node.start_autonomous_task(training_loop(node), "training")
        """
        if not self._running:
            raise RuntimeError("Node must be started before starting autonomous tasks")

        if name in self._autonomous_tasks:
            raise ValueError(f"Autonomous task with name '{name}' already exists")

        task = asyncio.create_task(task_coro)
        self._autonomous_tasks[name] = task
        return task

    async def shutdown(self) -> None:
        """Gracefully shutdown the node."""
        if not self._running:
            return

        self._running = False

        # Cancel all autonomous tasks
        for name, task in self._autonomous_tasks.items():
            if not task.done():
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=1.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
        self._autonomous_tasks.clear()

        # Cancel message processing task
        if self._message_task and not self._message_task.done():
            self._message_task.cancel()
            try:
                await asyncio.wait_for(self._message_task, timeout=1.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        await self.context.shutdown()
        await self.application.shutdown()


__all__ = ["DecentralizedNode"]
