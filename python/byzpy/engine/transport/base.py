from __future__ import annotations

"""
Transport abstraction for decentralized runners.
"""
from typing import Any, Protocol


class Transport(Protocol):
    def register(self, node_id: str, handler: callable) -> None:
        """Register a local message handler for node_id."""
        ...

    def send(self, to_id: str, payload: Any) -> None:
        """Send a payload to a logical node id."""
        ...
