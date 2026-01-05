"""
Shared registry and routing helpers for actor backends.
The existing actor backends relied on importing each other's module level
registries (e.g. ``_THREAD_REG``) to deliver messages across schemes. That
created tight coupling and import cycles.  This module centralizes the
registration of live backends and provides lightweight lookup helpers that
any backend can use without importing its peers.  It intentionally avoids
referencing concrete backend classes to remain dependency-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class BackendRecord:
    scheme: str
    actor_id: str
    backend: Any


class ChannelRouter:
    """Process-local registry for actor backends keyed by scheme/actor_id."""

    def __init__(self) -> None:
        self._backends: Dict[str, Dict[str, BackendRecord]] = {}

    def register(self, scheme: str, actor_id: str, backend: Any) -> None:
        """Record that ``backend`` owns ``(scheme, actor_id)`` locally."""
        reg = self._backends.setdefault(scheme, {})
        reg[actor_id] = BackendRecord(scheme=scheme, actor_id=actor_id, backend=backend)

    def unregister(self, scheme: str, actor_id: str) -> None:
        reg = self._backends.get(scheme)
        if not reg:
            return
        reg.pop(actor_id, None)
        if not reg:
            self._backends.pop(scheme, None)

    def resolve(self, scheme: str, actor_id: str) -> Optional[Any]:
        """Return the backend bound to ``(scheme, actor_id)`` if present."""
        reg = self._backends.get(scheme)
        if not reg:
            return None
        rec = reg.get(actor_id)
        if rec is None:
            return None
        return rec.backend


# Global singleton used by all backends
channel_router = ChannelRouter()
