from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, Optional

if TYPE_CHECKING:
    from .decentralized import DecentralizedNode


class NodeContext(ABC):
    """
    Abstract base class for node execution contexts.

    Defines the interface for different execution contexts (process, remote, in-process).
    """

    @abstractmethod
    async def start(self, node: "DecentralizedNode") -> None:
        """
        Start the context and associate it with a node.

        Args:
            node: The DecentralizedNode instance to associate with this context.
        """
        raise NotImplementedError

    @abstractmethod
    async def send_message(self, to_node_id: str, message_type: str, payload: Any) -> None:
        """
        Send a message to another node.

        Args:
            to_node_id: The ID of the target node.
            message_type: The type of message.
            payload: The message payload.
        """
        raise NotImplementedError

    @abstractmethod
    async def receive_messages(self) -> AsyncIterator[Any]:
        """
        Receive messages from the context.

        Yields:
            Messages in the format: {"from": str, "type": str, "payload": Any}
        """
        raise NotImplementedError

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the context and clean up resources."""
        raise NotImplementedError


class InProcessContext(NodeContext):
    """
    Node context for in-process execution (for testing and single-process scenarios).

    Uses asyncio queues for message passing within the same process.
    """

    _registry: Dict[str, "InProcessContext"] = {}

    def __init__(self) -> None:
        self._node: Optional["DecentralizedNode"] = None
        self._inbox: asyncio.Queue = asyncio.Queue()
        self._running = False

    async def start(self, node: "DecentralizedNode") -> None:
        """Start the context and store node reference."""
        if self._running:
            return
        self._node = node
        self._running = True
        InProcessContext._registry[node.node_id] = self

    async def send_message(self, to_node_id: str, message_type: str, payload: Any) -> None:
        """Send a message to another node."""
        if not self._running:
            raise RuntimeError("InProcessContext is not started.")
        # Get target context from registry
        target_context = InProcessContext._registry.get(to_node_id)
        if not target_context or not target_context._running:
            raise ValueError(f"Target node {to_node_id} not found or not running.")
        await target_context._inbox.put(
            {
                "from": self._node.node_id if self._node else "unknown",
                "type": message_type,
                "payload": payload,
            }
        )

    async def receive_messages(self) -> AsyncIterator[Any]:
        """Yield messages from the inbox."""
        while self._running:
            try:
                msg = await asyncio.wait_for(self._inbox.get(), timeout=0.1)
                yield msg
            except asyncio.TimeoutError:
                # Check if still running
                if not self._running:
                    break
                # Continue to check again
                continue
            except asyncio.CancelledError:
                # Task was cancelled, exit gracefully
                break

    async def shutdown(self) -> None:
        """Shutdown the context."""
        if not self._running:
            return
        self._running = False
        if self._node and self._node.node_id in InProcessContext._registry:
            del InProcessContext._registry[self._node.node_id]
        self._node = None
        # Drain any remaining messages
        while not self._inbox.empty():
            try:
                self._inbox.get_nowait()
            except asyncio.QueueEmpty:
                break


class ProcessContext(NodeContext):
    """
    Node context for process-based execution.

    Runs DecentralizedNode in a separate OS process with message passing via queues.
    """

    _registry: Dict[str, "ProcessContext"] = {}

    def __init__(self) -> None:
        import multiprocessing as mp

        self._node_id: Optional[str] = None
        self._process: Optional[mp.Process] = None
        self._inbox_q: Optional[mp.Queue] = None
        self._outbox_q: Optional[mp.Queue] = None
        self._cmd_q: Optional[mp.Queue] = None
        self._running = False

    async def start(self, node: "DecentralizedNode") -> None:
        """Start the context and create process for the node."""
        import multiprocessing as mp

        import cloudpickle

        if self._running:
            return

        self._node_id = node.node_id
        self._running = True

        # Create queues for communication
        self._inbox_q = mp.Queue()
        self._outbox_q = mp.Queue()
        self._cmd_q = mp.Queue()

        # Register in registry for routing
        ProcessContext._registry[node.node_id] = self

        # Serialize node configuration (not the node itself with queues)
        node_config = {
            "node_id": node.node_id,
            "application": node.application,
            "topology": node.topology,
            "metadata": node.scheduler.metadata if hasattr(node, "scheduler") else {},
            "node_id_map": getattr(
                node, "_node_id_map", None
            ),  # Include node_id_map for topology routing
        }

        # Include node objects in config (for P2P training)
        # Node objects (the actual node instances, not actors) can be pickled
        if hasattr(node, "_p2p_node_objects"):
            try:
                node_config["_node_objects"] = node._p2p_node_objects
            except Exception:
                pass  # Skip if can't be included

        # Include initialization callback if provided (for autonomous training loops)
        # This callback is executed in the subprocess after node creation
        if hasattr(node, "_init_callback") and node._init_callback is not None:
            try:
                node_config["init_callback"] = node._init_callback
            except Exception:
                pass  # Skip if callback can't be pickled

        config_blob = cloudpickle.dumps(node_config)

        # Start process
        self._process = mp.Process(
            target=_process_node_main,
            args=(config_blob, self._inbox_q, self._outbox_q, self._cmd_q),
        )
        self._process.start()

        # Give process time to start - increased for slower environments like pytest
        await asyncio.sleep(0.3)

    async def send_message(self, to_node_id: str, message_type: str, payload: Any) -> None:
        """Send a message to another node (across processes)."""
        import cloudpickle

        if not self._running:
            raise RuntimeError("ProcessContext is not started.")

        # Get target context
        target_context = ProcessContext._registry.get(to_node_id)
        if not target_context or not target_context._running:
            raise ValueError(f"Target node {to_node_id} not found or not running.")

        # Serialize and send message to target's inbox
        msg = {"from": self._node_id, "type": message_type, "payload": payload}
        serialized_msg = cloudpickle.dumps(msg)
        target_context._inbox_q.put(serialized_msg)

    async def receive_messages(self) -> AsyncIterator[Any]:
        """Receive messages from the process (non-blocking async)."""
        import queue

        import cloudpickle

        if not self._running:
            raise RuntimeError("ProcessContext is not started.")

        loop = asyncio.get_running_loop()

        while self._running:
            try:
                # Use run_in_executor to avoid blocking the event loop
                def _get_message():
                    try:
                        return self._outbox_q.get(timeout=0.01)
                    except queue.Empty:
                        return None

                serialized_msg = await loop.run_in_executor(None, _get_message)

                if serialized_msg is None:
                    # Allow other tasks to run with minimal delay
                    await asyncio.sleep(0.001)
                    continue

                msg = cloudpickle.loads(serialized_msg)

                # Check if this is a routing request from subprocess
                if msg.get("_route_request"):
                    # Route to target node
                    to_node_id = msg.get("to")
                    target_context = ProcessContext._registry.get(to_node_id)
                    if target_context and target_context._running:
                        # Forward to target's inbox (remove routing flag)
                        forward_msg = {
                            "from": msg["from"],
                            "type": msg["type"],
                            "payload": msg["payload"],
                        }
                        target_context._inbox_q.put(cloudpickle.dumps(forward_msg))
                    continue

                # Check if this is a message notification from subprocess
                if msg.get("type") == "_message_received":
                    # Yield in format expected by DecentralizedNode's handler
                    yield {
                        "from": msg["from"],
                        "type": msg["message_type"],
                        "payload": msg["payload"],
                    }
                    continue

                # Regular message - yield to caller
                yield msg
            except asyncio.CancelledError:
                break
            except Exception:
                # Handle deserialization errors - minimal delay before retry
                await asyncio.sleep(0.001)
                continue

    async def shutdown(self) -> None:
        """Shutdown the process and clean up."""
        if not self._running:
            return

        self._running = False

        # Send stop command to process
        if self._cmd_q:
            try:
                self._cmd_q.put(("stop", None), timeout=1.0)
            except Exception:
                pass

        # Wait for process to terminate
        process_was_alive = False
        if self._process:
            process_was_alive = self._process.is_alive()
            if process_was_alive:
                self._process.join(timeout=2.0)
                if self._process.is_alive():
                    self._process.terminate()
                    self._process.join(timeout=1.0)

        # Remove from registry
        if self._node_id and self._node_id in ProcessContext._registry:
            del ProcessContext._registry[self._node_id]

        # Clean up queues
        self._inbox_q = None
        self._outbox_q = None
        self._cmd_q = None
        self._process = None


def _process_node_main(config_blob: bytes, inbox_q, outbox_q, cmd_q) -> None:
    """
    Main function for the node process.

    Runs a full DecentralizedNode with its own asyncio event loop.
    Messages are received via inbox_q, processed by the node, and results
    sent back via outbox_q.
    """
    import asyncio
    import queue

    import cloudpickle

    # Deserialize node configuration
    config = cloudpickle.loads(config_blob)
    node_id = config["node_id"]
    application = config["application"]
    topology = config.get("topology")
    metadata = config.get("metadata", {})
    node_id_map = config.get("node_id_map")

    # Populate node object registry if node objects are provided (for P2P training)
    # Node objects (not actors) can be pickled and are stored in the registry
    node_objects = config.get("_node_objects", {})
    if node_objects:
        try:
            from byzpy.engine.peer_to_peer.runner import _NODE_OBJECT_REGISTRY

            _NODE_OBJECT_REGISTRY.update(node_objects)
        except Exception:
            pass  # Registry might not be available, continue without it

    # Create new event loop for this process
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def run_node():
        """Run the node with message processing."""
        from .decentralized import DecentralizedNode

        # Create a subprocess-specific context that bridges queues
        subprocess_context = _SubprocessBridgeContext(
            node_id=node_id,
            inbox_q=inbox_q,
            outbox_q=outbox_q,
        )

        # Create the actual DecentralizedNode in this process
        node = DecentralizedNode(
            node_id=node_id,
            application=application,
            context=subprocess_context,
            topology=topology,
            metadata=metadata,
            node_id_map=node_id_map,  # Pass node_id_map for topology routing
        )

        # Start the node
        await node.start()

        # Execute initialization callback if provided (for autonomous training loops)
        # This allows handlers and training loops to be set up in the subprocess
        init_callback = config.get("init_callback")
        if init_callback is not None:
            try:
                await init_callback(node)
            except Exception as e:
                # Send error to parent process
                outbox_q.put(
                    cloudpickle.dumps(
                        {
                            "type": "init_error",
                            "error": str(e),
                        }
                    )
                )
                # Continue even if initialization fails (node still processes messages)

        # Message processing loop
        running = True
        while running:
            processed_any = False

            # Check for stop command (non-blocking)
            try:
                cmd, payload = cmd_q.get_nowait()
                if cmd == "stop":
                    running = False
                    continue
                elif cmd == "execute_pipeline":
                    # Execute pipeline and send result back
                    try:
                        result = await node.execute_pipeline(
                            payload["pipeline_name"],
                            payload["inputs"],
                        )
                        outbox_q.put(
                            cloudpickle.dumps(
                                {
                                    "type": "pipeline_result",
                                    "request_id": payload.get("request_id"),
                                    "result": result,
                                }
                            )
                        )
                    except Exception as e:
                        outbox_q.put(
                            cloudpickle.dumps(
                                {
                                    "type": "pipeline_error",
                                    "request_id": payload.get("request_id"),
                                    "error": str(e),
                                }
                            )
                        )
                processed_any = True
            except queue.Empty:
                pass

            # Process ALL available messages (drain the queue)
            while True:
                try:
                    serialized_msg = inbox_q.get_nowait()
                    msg = cloudpickle.loads(serialized_msg)

                    # Handle the message in the subprocess node
                    await node.handle_incoming_message(
                        from_node_id=msg.get("from", "unknown"),
                        message_type=msg.get("type", "unknown"),
                        payload=msg.get("payload"),
                    )

                    # Also forward to parent for parent-side handlers
                    # (handlers registered on parent DecentralizedNode)
                    outbox_q.put(
                        cloudpickle.dumps(
                            {
                                "type": "_message_received",
                                "from": msg.get("from", "unknown"),
                                "message_type": msg.get("type", "unknown"),
                                "payload": msg.get("payload"),
                            }
                        )
                    )
                    processed_any = True
                except queue.Empty:
                    break
                except Exception as e:
                    # Log error but continue
                    break

            # Only sleep if no work was done - minimal delay for responsiveness
            if not processed_any:
                await asyncio.sleep(0.001)

        # Shutdown node
        await node.shutdown()

    try:
        loop.run_until_complete(run_node())
    except Exception as e:
        # Send error to parent
        outbox_q.put(
            cloudpickle.dumps(
                {
                    "type": "process_error",
                    "error": str(e),
                }
            )
        )
    finally:
        loop.close()


class _SubprocessBridgeContext(NodeContext):
    """
    A NodeContext that bridges subprocess queues to the DecentralizedNode.

    Used inside the subprocess to connect the node to the parent process
    via multiprocessing queues.
    """

    _subprocess_registry: Dict[str, "_SubprocessBridgeContext"] = {}

    def __init__(self, node_id: str, inbox_q, outbox_q) -> None:
        self._node_id = node_id
        self._inbox_q = inbox_q
        self._outbox_q = outbox_q
        self._running = False
        self._node: Optional["DecentralizedNode"] = None

    async def start(self, node: "DecentralizedNode") -> None:
        """Start the bridge context."""
        if self._running:
            return
        self._node = node
        self._running = True
        _SubprocessBridgeContext._subprocess_registry[self._node_id] = self

    async def send_message(self, to_node_id: str, message_type: str, payload: Any) -> None:
        """Send message via parent process routing."""
        import cloudpickle

        if not self._running:
            raise RuntimeError("Context not started.")

        # Package message for parent to route
        msg = {
            "from": self._node_id,
            "to": to_node_id,
            "type": message_type,
            "payload": payload,
            "_route_request": True,  # Signal to parent to route this
        }
        self._outbox_q.put(cloudpickle.dumps(msg))

    async def receive_messages(self) -> AsyncIterator[Any]:
        """
        Receive messages - in subprocess, messages come via inbox_q.
        This is called by DecentralizedNode's message processing loop.
        """
        import queue

        while self._running:
            try:
                # Non-blocking check
                import cloudpickle

                serialized_msg = self._inbox_q.get_nowait()
                msg = cloudpickle.loads(serialized_msg)
                yield msg
            except queue.Empty:
                await asyncio.sleep(0.001)
            except asyncio.CancelledError:
                break

    async def shutdown(self) -> None:
        """Shutdown the bridge context."""
        if not self._running:
            return
        self._running = False
        if self._node_id in _SubprocessBridgeContext._subprocess_registry:
            del _SubprocessBridgeContext._subprocess_registry[self._node_id]
        self._node = None


class RemoteContext(NodeContext):
    """
    Node context for remote execution.

    Connects to a RemoteNodeServer to communicate with nodes on that server.
    """

    def __init__(self, host: str, port: int):
        """
        Initialize remote context.

        Args:
            host: Remote server hostname or IP
            port: Remote server port
        """
        self.host = host
        self.port = port
        self._client = None
        self._node: Optional["DecentralizedNode"] = None
        self._running = False
        self._receive_task: Optional[asyncio.Task] = None

    async def start(self, node: "DecentralizedNode") -> None:
        """
        Start the context and connect to remote server.

        Args:
            node: The DecentralizedNode instance to associate with this context.
        """
        if self._running:
            return

        from .remote_client import RemoteNodeClient

        self._node = node
        self._client = RemoteNodeClient(host=self.host, port=self.port)

        try:
            await self._client.connect(timeout=5.0)
        except (ConnectionError, OSError, asyncio.TimeoutError) as e:
            raise ConnectionError(f"Failed to connect to {self.host}:{self.port}") from e

        self._running = True

        # Register this node with the server
        if self._node:
            await self._client.register_node(self._node.node_id)

        self._receive_task = asyncio.create_task(self._message_processing_loop())

    async def send_message(self, to_node_id: str, message_type: str, payload: Any) -> None:
        """
        Send a message to a node on the remote server.

        Args:
            to_node_id: Target node ID
            message_type: Message type
            payload: Message payload
        """
        if not self._running:
            raise RuntimeError("RemoteContext is not started")

        if not self._client or not self._client.is_connected():
            raise RuntimeError("Not connected to remote server")

        # Include sender information - need to modify client to accept from_node_id
        # For now, we'll send it in the payload and let server extract it
        # Actually, let's update the client call to include from
        if self._node:
            await self._client.send_message(
                to_node_id, message_type, payload, from_node_id=self._node.node_id
            )
        else:
            await self._client.send_message(to_node_id, message_type, payload)

    async def receive_messages(self) -> AsyncIterator[Any]:
        """
        Receive messages from the remote server.

        Yields:
            Messages in format: {"from": str, "type": str, "payload": Any}
        """
        if not self._running:
            raise RuntimeError("RemoteContext is not started")

        while self._running:
            try:
                msg = await asyncio.wait_for(self._client.receive_message(timeout=0.1), timeout=0.1)
                if msg:
                    # Convert server message format to expected format
                    yield {
                        "from": msg.get("from", "unknown"),
                        "type": msg.get("type", "unknown"),
                        "payload": msg.get("payload"),
                    }
            except asyncio.TimeoutError:
                # Continue to check if still running
                continue
            except asyncio.CancelledError:
                break
            except Exception:
                # Handle connection errors
                if not self._running:
                    break
                await asyncio.sleep(0.01)
                continue

    async def _message_processing_loop(self) -> None:
        """Background task to process incoming messages."""
        try:
            async for msg in self.receive_messages():
                if not self._running:
                    break
                if self._node:
                    await self._node.handle_incoming_message(
                        from_node_id=msg.get("from", "unknown"),
                        message_type=msg.get("type", "unknown"),
                        payload=msg.get("payload"),
                    )
        except asyncio.CancelledError:
            pass

    async def shutdown(self) -> None:
        """Shutdown the context and disconnect."""
        if not self._running:
            return

        self._running = False

        if self._receive_task and not self._receive_task.done():
            self._receive_task.cancel()
            try:
                await asyncio.wait_for(self._receive_task, timeout=1.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        if self._client:
            await self._client.disconnect()

        self._node = None
        self._client = None


__all__ = ["NodeContext", "InProcessContext", "ProcessContext", "RemoteContext"]
