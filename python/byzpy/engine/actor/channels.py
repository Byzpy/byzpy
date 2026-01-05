# byzpy/engine/actor/channels.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .ipc import unwrap_payload


# -------------------------------------------------------------------
# Public endpoint type (imported by backends)
# -------------------------------------------------------------------
@dataclass(frozen=True)
class Endpoint:
    """A globally addressable location for an actor."""

    scheme: str  # "thread" | "process" | "tcp"
    address: str  # "" for local thread/process; "host:port" for tcp
    actor_id: str  # unique within scheme/address


# -------------------------------------------------------------------
# Universal IPC unwrap for tensor payloads (safe on coordinator side)
# -------------------------------------------------------------------
def _unwrap_ipc(obj: Any) -> Any:
    return unwrap_payload(obj)


# -------------------------------------------------------------------
# Channel handle bound to a local endpoint
# -------------------------------------------------------------------
class ChannelRef:
    """
    Typed, awaitable channel handle bound to a *local* actor endpoint.
    - `send(to, payload)` delivers to a remote/local Endpoint.
    - `recv(timeout=...)` gets the next payload, with universal IPC unwrapping.
    """

    __slots__ = ("_backend", "_local", "_name")

    def __init__(self, backend, local_ep: Endpoint, name: str):
        self._backend = backend
        self._local = local_ep
        self._name = name

    async def send(self, to: Endpoint, payload: Any) -> None:
        await self._backend.chan_put(
            from_ep=self._local, to_ep=to, name=self._name, payload=payload
        )

    async def recv(self, *, timeout: Optional[float] = None) -> Any:
        raw = await self._backend.chan_get(ep=self._local, name=self._name, timeout=timeout)
        return _unwrap_ipc(raw)


# -------------------------------------------------------------------
# Convenience helper to open/bind a channel on a backend
# -------------------------------------------------------------------
async def open_channel(backend, name: str) -> ChannelRef:
    """
    Creates (or ensures) a mailbox named `name` on `backend`, returns a ChannelRef
    bound to the backend's local endpoint for that mailbox.
    """
    local_ep: Endpoint = await backend.chan_open(name)
    return ChannelRef(backend, local_ep, name)
