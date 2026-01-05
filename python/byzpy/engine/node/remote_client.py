"""Remote node client for network communication."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

import cloudpickle


def serialize_message(msg: Dict[str, Any]) -> bytes:
    """
    Serialize a message for network transport.

    Args:
        msg: Message dictionary with 'from', 'type', 'payload' keys

    Returns:
        Serialized bytes
    """
    return cloudpickle.dumps(msg)


def deserialize_message(data: bytes) -> Dict[str, Any]:
    """
    Deserialize a message from network transport.

    Args:
        data: Serialized message bytes

    Returns:
        Message dictionary
    """
    return cloudpickle.loads(data)


class RemoteNodeClient:
    """
    Client for connecting to a remote node server.

    Handles TCP connection, message sending/receiving, and connection management.
    """

    def __init__(self, host: str, port: int):
        """
        Initialize remote node client.

        Args:
            host: Server hostname or IP address
            port: Server port number
        """
        self.host = host
        self.port = port
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._connected = False
        self._receive_task: Optional[asyncio.Task] = None
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._running = False

    async def connect(self, timeout: float = 5.0) -> None:
        """
        Connect to the remote server.

        Args:
            timeout: Connection timeout in seconds

        Raises:
            ConnectionError: If connection fails
            asyncio.TimeoutError: If connection times out
        """
        if self._connected:
            return

        try:
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_connection(self.host, self.port), timeout=timeout
            )
            self._connected = True
            self._running = True
            self._receive_task = asyncio.create_task(self._receive_loop())
        except (OSError, ConnectionRefusedError) as e:
            raise ConnectionError(f"Failed to connect to {self.host}:{self.port}") from e
        except asyncio.TimeoutError:
            raise asyncio.TimeoutError(f"Connection to {self.host}:{self.port} timed out")

    async def disconnect(self) -> None:
        """Disconnect from the server."""
        self._running = False
        self._connected = False

        if self._receive_task and not self._receive_task.done():
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass

        if self._writer:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                pass

        self._reader = None
        self._writer = None

    def is_connected(self) -> bool:
        """Check if client is connected to server."""
        if not self._connected:
            return False
        if self._writer is None:
            self._connected = False
            return False
        try:
            # Check if writer is closing or closed
            if self._writer.is_closing():
                self._connected = False
                return False
            # Check transport state
            transport = self._writer.transport
            if transport is None or transport.is_closing():
                self._connected = False
                return False
            return True
        except Exception:
            # Writer may be in invalid state
            self._connected = False
            return False

    async def send_message(
        self,
        to_node_id: str,
        message_type: str,
        payload: Any,
        from_node_id: Optional[str] = None,
    ) -> None:
        """
        Send a message to a node on the remote server.

        Args:
            to_node_id: Target node ID
            message_type: Message type
            payload: Message payload
            from_node_id: Optional sender node ID

        Raises:
            RuntimeError: If not connected
        """
        if not self.is_connected():
            raise RuntimeError("Client is not connected")

        msg = {
            "to": to_node_id,
            "type": message_type,
            "payload": payload,
        }
        if from_node_id:
            msg["from"] = from_node_id

        try:
            # Check connection before sending
            if not self.is_connected():
                raise RuntimeError("Client is not connected")

            serialized = serialize_message(msg)
            # Send length prefix
            length = len(serialized).to_bytes(4, byteorder="big")
            self._writer.write(length + serialized)
            await self._writer.drain()
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            self._connected = False
            raise RuntimeError(f"Connection lost: {e}") from e
        except Exception as e:
            # Check if it's a connection-related error
            error_str = str(e).lower()
            if any(keyword in error_str for keyword in ["closed", "broken", "reset", "connection"]):
                self._connected = False
            raise RuntimeError(f"Failed to send message: {e}") from e

    async def receive_message(self, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        Receive a message from the server.

        Args:
            timeout: Optional timeout in seconds

        Returns:
            Message dictionary or None if timeout
        """
        try:
            if timeout is not None:
                return await asyncio.wait_for(self._message_queue.get(), timeout=timeout)
            else:
                return await self._message_queue.get()
        except asyncio.TimeoutError:
            return None

    async def register_node(self, node_id: str) -> None:
        """
        Register this client's node ID with the server.

        Args:
            node_id: The node ID to register
        """
        if not self.is_connected():
            raise RuntimeError("Client is not connected")

        registration_msg = {
            "type": "_register_node",
            "node_id": node_id,
        }

        try:
            serialized = serialize_message(registration_msg)
            length = len(serialized).to_bytes(4, byteorder="big")
            self._writer.write(length + serialized)
            await self._writer.drain()
        except Exception as e:
            raise RuntimeError(f"Failed to register node: {e}") from e

    async def _receive_loop(self) -> None:
        """Background task to receive messages from server."""
        try:
            while self._running:
                # Check connection status - verify writer is still valid
                if self._writer is None or self._writer.is_closing():
                    self._connected = False
                    break
                if not self._connected:
                    break

                try:
                    # Read length prefix with shorter timeout for faster disconnection detection
                    length_bytes = await asyncio.wait_for(self._reader.readexactly(4), timeout=0.05)
                    length = int.from_bytes(length_bytes, byteorder="big")

                    # Read message
                    data = await asyncio.wait_for(self._reader.readexactly(length), timeout=5.0)

                    msg = deserialize_message(data)
                    await self._message_queue.put(msg)

                except asyncio.TimeoutError:
                    # Continue loop to check if still running and connected
                    continue
                except (
                    asyncio.IncompleteReadError,
                    ConnectionResetError,
                    BrokenPipeError,
                    OSError,
                ):
                    # Connection closed
                    self._connected = False
                    break
                except Exception as e:
                    # Handle other errors
                    if self._running:
                        # Check if it's a connection error
                        error_str = str(e).lower()
                        if any(
                            keyword in error_str
                            for keyword in ["closed", "broken", "reset", "connection"]
                        ):
                            self._connected = False
                            break
                        # Log error but continue
                        continue
                    break

        except asyncio.CancelledError:
            pass
        finally:
            self._connected = False


__all__ = ["RemoteNodeClient", "serialize_message", "deserialize_message"]
