from __future__ import annotations

import asyncio
from typing import Any, Optional

from .._wire import recv_obj, send_obj

__all__ = ["chan_put", "chan_get"]


def _split_hostport(address: str) -> tuple[str, int]:
    host, port_str = address.rsplit(":", 1)
    return host, int(port_str)


async def _request(address: str, payload: dict) -> dict:
    host, port = _split_hostport(address)
    reader, writer = await asyncio.open_connection(host, port)
    try:
        await send_obj(writer, payload)
        return await recv_obj(reader)
    finally:
        writer.close()
        await writer.wait_closed()


async def chan_put(
    address: str,
    *,
    from_ep: Any,
    to_ep: Any,
    name: str,
    payload: Any,
) -> None:
    rep = await _request(
        address,
        {
            "op": "chan_put",
            "from": getattr(from_ep, "__dict__", from_ep),
            "to": getattr(to_ep, "__dict__", to_ep),
            "name": name,
            "payload": payload,
        },
    )
    if not rep.get("ok", False):
        raise RuntimeError(rep)


async def chan_get(
    address: str,
    *,
    name: str,
    timeout: Optional[float],
    actor_id: str,
) -> Any:
    rep = await _request(
        address,
        {
            "op": "chan_get",
            "name": name,
            "timeout": timeout,
            "actor_id": actor_id,
        },
    )
    if rep.get("ok", False):
        return rep["payload"]
    raise RuntimeError(rep)
