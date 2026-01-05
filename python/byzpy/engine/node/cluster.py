"""Cluster management for decentralized nodes."""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

from .application import NodeApplication
from .context import NodeContext, ProcessContext
from .decentralized import DecentralizedNode


class DecentralizedCluster:
    """
    Manages a cluster of DecentralizedNode instances running in separate processes.

    Provides convenience methods for creating, starting, and managing multiple nodes.
    """

    def __init__(self) -> None:
        self.nodes: Dict[Union[int, str], DecentralizedNode] = {}
        self._node_id_map: Dict[int, Union[int, str]] = {}

    async def add_node(
        self,
        node_id: Union[int, str],
        application: NodeApplication,
        *,
        topology: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        context: Optional[NodeContext] = None,
    ) -> DecentralizedNode:
        """
        Add a node to the cluster.

        Args:
            node_id: Unique identifier for the node (can be int or str)
            application: NodeApplication instance for the node
            topology: Optional topology for the node
            metadata: Optional metadata for the node
            context: Optional NodeContext (defaults to ProcessContext)

        Returns:
            The created DecentralizedNode

        Raises:
            ValueError: If node_id already exists
        """
        if node_id in self.nodes:
            raise ValueError(f"Node {node_id} already exists in cluster.")

        # Build node_id_map for topology routing
        # Map from topology index (int) to actual node_id
        node_index = len(self.nodes)
        self._node_id_map[node_index] = node_id

        # Create node with provided context or default to ProcessContext
        if context is None:
            context = ProcessContext()

        node = DecentralizedNode(
            node_id=node_id,
            application=application,
            context=context,
            topology=topology,
            metadata=metadata,
            node_id_map=None,  # Will be set by update_node_id_maps after all nodes added
        )

        self.nodes[node_id] = node
        return node

    def _update_node_id_maps(self) -> None:
        """Update all nodes with the complete node_id_map."""
        for node in self.nodes.values():
            node._node_id_map = self._node_id_map.copy()
            node.message_router._node_id_map = self._node_id_map.copy()
            # Build reverse map
            node.message_router._reverse_id_map = {v: k for k, v in self._node_id_map.items()}

    async def start_all(self) -> None:
        """Start all nodes in the cluster."""
        # Update all nodes with complete node_id_map before starting
        self._update_node_id_maps()

        for node in self.nodes.values():
            if not node._running:
                await node.start()

    async def shutdown_all(self) -> None:
        """Shutdown all nodes in the cluster."""
        for node in self.nodes.values():
            if node._running:
                await node.shutdown()

    def get_node(self, node_id: Union[int, str]) -> Optional[DecentralizedNode]:
        """Get a node by ID."""
        return self.nodes.get(node_id)

    async def remove_node(self, node_id: Union[int, str]) -> None:
        """
        Remove a node from the cluster and shutdown.

        Args:
            node_id: ID of node to remove
        """
        node = self.nodes.pop(node_id, None)
        if node and node._running:
            await node.shutdown()


__all__ = ["DecentralizedCluster"]
