from __future__ import annotations

"""
Local in-process transport that dispatches to registered handlers.
"""
from typing import Any, Callable, Dict

from .base import Transport


class LocalTransport(Transport):
    def __init__(self) -> None:
        self._handlers: Dict[str, Callable[[Any], None]] = {}

    def register(self, node_id: str, handler: Callable[[Any], None]) -> None:
        self._handlers[node_id] = handler

    def send(self, to_id: str, payload: Any) -> None:
        fn = self._handlers.get(to_id)
        if fn is None:
            raise KeyError(f"Unknown node_id {to_id}")
        fn(payload)
