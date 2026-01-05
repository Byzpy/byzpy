"""Topology-aware message router for decentralized nodes."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from .context import NodeContext


class MessageRouter:
    """
    Topology-aware message router that enforces communication constraints.

    Provides routing patterns:
    - Direct: Send to specific neighbor
    - Broadcast: Send to all neighbors
    - Multicast: Send to subset of neighbors
    - Reply: Send back to message sender
    """

    def __init__(
        self,
        *,
        topology: Optional[Any] = None,
        node_id: Union[int, str],
        node_id_map: Optional[Dict[int, str]] = None,
    ):
        """
        Initialize MessageRouter.

        Args:
            topology: Topology object defining communication graph (None = allow all)
            node_id: ID of the node this router belongs to
            node_id_map: Optional mapping from topology integer IDs to string node IDs
        """
        self.topology = topology
        self.node_id = node_id
        self._node_id_map = node_id_map or {}

        # Build reverse map (string -> int)
        self._reverse_id_map: Dict[str, int] = {v: k for k, v in self._node_id_map.items()}

        # Validate node_id exists in topology (only if using integer IDs directly)
        if topology is not None and isinstance(node_id, int):
            if node_id not in range(topology.n):
                raise ValueError(f"node_id {node_id} is not in topology (n={topology.n})")
        # For string node_ids, validation is deferred until node_id_map is set

    def _to_internal_id(self, node_id: Union[int, str]) -> int:
        """Convert external node ID to internal topology ID."""
        if isinstance(node_id, int):
            return node_id
        return self._reverse_id_map.get(node_id, -1) if self._reverse_id_map else -1

    def _to_external_id(self, internal_id: int) -> Union[int, str]:
        """Convert internal topology ID to external node ID."""
        return self._node_id_map.get(internal_id, internal_id)

    def get_out_neighbors(self) -> List[Union[int, str]]:
        """
        Get list of outgoing neighbors (nodes this node can send to).

        Returns:
            List of neighbor node IDs
        """
        if self.topology is None:
            return []

        internal_id = self._to_internal_id(self.node_id)
        if internal_id < 0:
            return []

        out_neighbors = self.topology.out.get(internal_id, [])
        return [self._to_external_id(n) for n in out_neighbors]

    def get_in_neighbors(self) -> List[Union[int, str]]:
        """
        Get list of incoming neighbors (nodes that can send to this node).

        Returns:
            List of neighbor node IDs
        """
        if self.topology is None:
            return []

        internal_id = self._to_internal_id(self.node_id)
        if internal_id < 0:
            return []

        in_neighbors = self.topology.in_.get(internal_id, [])
        return [self._to_external_id(n) for n in in_neighbors]

    def can_send_to(self, target_node_id: Union[int, str]) -> bool:
        """
        Check if this node can send to target node.

        Args:
            target_node_id: Target node ID

        Returns:
            True if send is allowed
        """
        if self.topology is None:
            # No topology = allow all sends
            return True

        target_internal = self._to_internal_id(target_node_id)
        if target_internal < 0:
            # Unknown target - check if it's an integer in range
            if isinstance(target_node_id, int) and 0 <= target_node_id < self.topology.n:
                target_internal = target_node_id
            else:
                return False

        return target_internal in self.get_out_neighbors_internal()

    def get_out_neighbors_internal(self) -> List[int]:
        """Get out neighbors as internal IDs."""
        if self.topology is None:
            return []
        internal_id = self._to_internal_id(self.node_id)
        if internal_id < 0 and isinstance(self.node_id, int):
            internal_id = self.node_id
        return self.topology.out.get(internal_id, [])

    async def route_direct(
        self,
        target_node_id: Union[int, str],
        message_type: str,
        payload: Any,
        context: NodeContext,
    ) -> None:
        """
        Route message directly to a specific neighbor.

        Args:
            target_node_id: Target node ID (must be a neighbor)
            message_type: Type of the message
            payload: Message payload
            context: NodeContext for sending

        Raises:
            ValueError: If target is not a neighbor or is self
        """
        # Check for self-send
        if target_node_id == self.node_id:
            raise ValueError("cannot send to self")

        # Validate neighbor
        if self.topology is not None and not self.can_send_to(target_node_id):
            raise ValueError(f"Target {target_node_id} is not a neighbor of {self.node_id}")

        await context.send_message(target_node_id, message_type, payload)

    async def route_broadcast(
        self,
        message_type: str,
        payload: Any,
        context: Optional[NodeContext],
    ) -> None:
        """
        Broadcast message to all neighbors.

        Args:
            message_type: Type of the message
            payload: Message payload
            context: NodeContext for sending (can be None if no neighbors)

        Note:
            This method is resilient to individual send failures - it will continue
            sending to remaining neighbors even if some fail.
        """
        neighbors = self.get_out_neighbors()

        if not neighbors or context is None:
            return

        # Deduplicate neighbors (topology might have duplicate edges)
        unique_neighbors = list(dict.fromkeys(neighbors))  # Preserves order

        for neighbor_id in unique_neighbors:
            try:
                await context.send_message(neighbor_id, message_type, payload)
            except Exception:
                # Continue broadcasting to other neighbors even if one fails
                pass

    async def route_multicast(
        self,
        target_node_ids: List[Union[int, str]],
        message_type: str,
        payload: Any,
        context: Optional[NodeContext],
    ) -> None:
        """
        Multicast message to a subset of neighbors.

        Args:
            target_node_ids: List of target node IDs (must all be neighbors)
            message_type: Type of the message
            payload: Message payload
            context: NodeContext for sending

        Raises:
            ValueError: If any target is not a neighbor
        """
        if not target_node_ids:
            return

        # Validate all targets are neighbors
        if self.topology is not None:
            for target_id in target_node_ids:
                if not self.can_send_to(target_id):
                    raise ValueError(f"Target {target_id} is not a neighbor of {self.node_id}")

        if context is None:
            return

        for target_id in target_node_ids:
            await context.send_message(target_id, message_type, payload)

    async def route_reply(
        self,
        original_message: Dict[str, Any],
        message_type: str,
        payload: Any,
        context: NodeContext,
    ) -> None:
        """
        Reply to the sender of an original message.

        Args:
            original_message: The original message to reply to
            message_type: Type of the reply message
            payload: Reply payload
            context: NodeContext for sending

        Raises:
            ValueError: If sender is not a valid neighbor
        """
        sender_id = original_message.get("from")
        if sender_id is None:
            raise ValueError("Original message has no 'from' field")

        await self.route_direct(sender_id, message_type, payload, context)

    async def route_message(
        self,
        target_node_id: Union[int, str],
        message_type: str,
        payload: Any,
        context: NodeContext,
    ) -> None:
        """
        Route message to target (alias for route_direct for backward compatibility).
        """
        await self.route_direct(target_node_id, message_type, payload, context)


__all__ = ["MessageRouter"]
