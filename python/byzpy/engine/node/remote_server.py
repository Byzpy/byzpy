"""Remote node server for hosting DecentralizedNode instances."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Dict, Optional

from .context import InProcessContext, NodeContext
from .remote_client import deserialize_message, serialize_message

if TYPE_CHECKING:
    from .decentralized import DecentralizedNode


class ServerNodeContext(NodeContext):
    """
    Context for nodes running on a RemoteNodeServer.

    Routes messages to local nodes via InProcessContext, and to remote clients via the server.
    """

    def __init__(self, server: "RemoteNodeServer", in_process_context: InProcessContext):
        """
        Initialize server node context.

        Args:
            server: The RemoteNodeServer instance
            in_process_context: InProcessContext for local communication
        """
        self._server = server
        self._in_process = in_process_context
        self._node: Optional["DecentralizedNode"] = None

    async def start(self, node: "DecentralizedNode") -> None:
        """Start the context."""
        self._node = node
        await self._in_process.start(node)

    async def send_message(self, to_node_id: str, message_type: str, payload: Any) -> None:
        """
        Send message to target node.

        Routes to local node if available, otherwise routes through server to remote client.
        """
        # Check if target is a local node
        if to_node_id in self._server._nodes:
            # Use in-process context
            await self._in_process.send_message(to_node_id, message_type, payload)
        elif to_node_id in self._server._client_connections:
            # Target is a remote client - route through server
            client_writer = self._server._client_connections[to_node_id]
            from_node_id = self._node.node_id if self._node else "unknown"
            await self._server.send_message_to_client(
                client_writer, from_node_id, message_type, payload
            )
        else:
            raise ValueError(f"Node {to_node_id} not found (not local and not registered remote)")

    async def receive_messages(self):
        """Receive messages - delegate to in-process context."""
        async for msg in self._in_process.receive_messages():
            yield msg

    async def shutdown(self) -> None:
        """Shutdown the context."""
        await self._in_process.shutdown()
        self._node = None


class RemoteNodeServer:
    """
    Server that hosts DecentralizedNode instances remotely.

    Accepts connections from other nodes and routes messages to local nodes.
    """

    def __init__(self, host: str = "localhost", port: int = 8888):
        """
        Initialize remote node server.

        Args:
            host: Hostname or IP to bind to
            port: Port number to listen on
        """
        self.host = host
        self.port = port
        self._nodes: Dict[str, "DecentralizedNode"] = {}
        self._server: Optional[asyncio.Server] = None
        self._running = False
        self._client_connections: Dict[str, asyncio.StreamWriter] = {}  # node_id -> writer
        self._client_writers: Dict[asyncio.StreamWriter, str] = {}  # writer -> node_id

    async def register_node(self, node: "DecentralizedNode") -> None:
        """
        Register a node on this server.

        Args:
            node: DecentralizedNode instance to register

        Raises:
            ValueError: If node ID already registered
        """
        if node.node_id in self._nodes:
            raise ValueError(f"Node {node.node_id} is already registered on this server")

        # Replace node's context with ServerNodeContext for routing
        in_process_context = InProcessContext()
        server_context = ServerNodeContext(self, in_process_context)
        node.context = server_context

        self._nodes[node.node_id] = node
        await node.start()

    async def serve(self) -> None:
        """Start the server and accept connections."""
        if self._running:
            return

        self._server = await asyncio.start_server(self._handle_client, self.host, self.port)
        self._running = True

        async with self._server:
            await self._server.serve_forever()

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """
        Handle incoming client connection.

        Args:
            reader: Stream reader for receiving data
            writer: Stream writer for sending data
        """
        client_addr = writer.get_extra_info("peername")
        client_id = f"{client_addr[0]}:{client_addr[1]}"
        remote_node_id: Optional[str] = None

        try:
            while self._running:
                try:
                    # Read length prefix
                    length_bytes = await asyncio.wait_for(reader.readexactly(4), timeout=1.0)
                    length = int.from_bytes(length_bytes, byteorder="big")

                    # Read message
                    data = await asyncio.wait_for(reader.readexactly(length), timeout=5.0)

                    msg = deserialize_message(data)

                    # Handle registration message
                    if msg.get("type") == "_register_node":
                        remote_node_id = msg.get("node_id")
                        if remote_node_id:
                            self._client_connections[remote_node_id] = writer
                            self._client_writers[writer] = remote_node_id
                        continue

                    # Route message to target node
                    to_node_id = msg.get("to")
                    if to_node_id and to_node_id in self._nodes:
                        target_node = self._nodes[to_node_id]
                        await target_node.handle_incoming_message(
                            from_node_id=msg.get("from", "unknown"),
                            message_type=msg.get("type", "unknown"),
                            payload=msg.get("payload"),
                        )
                    elif to_node_id and to_node_id in self._client_connections:
                        # Target is a remote client - forward message
                        client_writer = self._client_connections[to_node_id]
                        await self.send_message_to_client(
                            client_writer,
                            msg.get("from", "unknown"),
                            msg.get("type", "unknown"),
                            msg.get("payload"),
                        )
                    else:
                        # Node not found - send error response
                        error_msg = {
                            "from": "server",
                            "type": "error",
                            "payload": {"error": f"Node {to_node_id} not found"},
                        }
                        serialized = serialize_message(error_msg)
                        length = len(serialized).to_bytes(4, byteorder="big")
                        writer.write(length + serialized)
                        await writer.drain()

                except asyncio.TimeoutError:
                    # Continue to check if still running
                    continue
                except asyncio.IncompleteReadError:
                    # Client disconnected
                    break
                except Exception as e:
                    # Handle other errors
                    if self._running:
                        # Log error but continue
                        continue
                    break

        except asyncio.CancelledError:
            pass
        finally:
            # Clean up writer, but don't await if loop is closed
            try:
                writer.close()
                try:
                    await writer.wait_closed()
                except (RuntimeError, Exception):
                    # RuntimeError: Event loop is closed - ignore
                    # Other exceptions also ignored for cleanup
                    pass
            except RuntimeError:
                # Event loop is closed, can't close writer properly
                # Just skip writer cleanup
                pass
            except Exception:
                pass

            if remote_node_id and remote_node_id in self._client_connections:
                del self._client_connections[remote_node_id]
            if writer in self._client_writers:
                del self._client_writers[writer]

    async def send_message_to_client(
        self,
        client_writer: asyncio.StreamWriter,
        from_node_id: str,
        message_type: str,
        payload: Any,
    ) -> None:
        """
        Send a message to a connected client.

        Args:
            client_writer: Client's stream writer
            from_node_id: Source node ID
            message_type: Message type
            payload: Message payload
        """
        msg = {
            "from": from_node_id,
            "type": message_type,
            "payload": payload,
        }

        try:
            serialized = serialize_message(msg)
            length = len(serialized).to_bytes(4, byteorder="big")
            client_writer.write(length + serialized)
            await client_writer.drain()
        except Exception:
            # Connection error - client may have disconnected
            pass

    async def shutdown(self) -> None:
        """Shutdown the server and clean up."""
        self._running = False

        if self._server:
            self._server.close()
            await self._server.wait_closed()

        # Shutdown all registered nodes
        for node in self._nodes.values():
            if node._running:
                await node.shutdown()

        self._nodes.clear()
        self._client_connections.clear()


__all__ = ["RemoteNodeServer"]
