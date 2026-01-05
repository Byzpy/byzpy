"""
Lightweight cluster helper to manage multiple NodeRunner instances.

This is a minimal scaffold to exercise decentralized scheduling: each runner
executes in its own process and can receive messages via the cluster broker.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict

from .node_runner import NodeRunner


class NodeCluster:
    """Manage a set of NodeRunner instances keyed by node_id."""

    def __init__(self, transport=None) -> None:
        self._nodes: Dict[str, NodeRunner] = {}
        self._transport = transport

    def add_node(
        self,
        node_id: str,
        step_fn: Callable[[dict], dict],
        msg_handler: Callable[[dict, Any], dict],
        *,
        init_state: dict | None = None,
    ) -> None:
        if node_id in self._nodes:
            raise ValueError(f"Node {node_id} already exists")
        runner = NodeRunner(step_fn, msg_handler, init_state=init_state)
        self._nodes[node_id] = runner
        if self._transport is not None:
            self._transport.register(node_id, runner.send_message)

    def start_all(self) -> None:
        for node in self._nodes.values():
            node.start()

    def stop_all(self) -> None:
        for node in self._nodes.values():
            node.stop()

    def start_auto(self, node_id: str, interval_sec: float) -> None:
        self._nodes[node_id].start_auto(interval_sec)

    def send(self, to_id: str, msg: Any) -> None:
        if self._transport is not None:
            self._transport.send(to_id, msg)
        else:
            self._nodes[to_id].send_message(msg)

    def state(self, node_id: str) -> dict:
        return self._nodes[node_id].state()

    def barrier(self, duration: float) -> None:
        """Sleep helper to let nodes make progress."""
        time.sleep(duration)
