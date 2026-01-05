from __future__ import annotations

import asyncio
import concurrent.futures
import uuid
from typing import Any, Dict, Optional

from ..base import ActorBackend
from ..channels import Endpoint
from ..router import channel_router
from ..transports import tcp, ucx


class ThreadActorBackend(ActorBackend):
    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._obj = None
        self._actor_id = str(uuid.uuid4())
        self._queues: Dict[str, asyncio.Queue] = {}
        channel_router.register("thread", self._actor_id, self)

    async def start(self):
        if self._loop is None:
            self._loop = asyncio.get_running_loop()

    async def construct(self, cls_or_factory: Any, *, args: tuple, kwargs: dict) -> None:
        loop = self._loop or asyncio.get_running_loop()

        def _build():
            target = cls_or_factory
            return (
                target(*args, **kwargs) if not isinstance(target, type) else target(*args, **kwargs)
            )

        self._obj = await loop.run_in_executor(self._pool, _build)

    async def call(self, method: str, *args, **kwargs):
        loop = self._loop or asyncio.get_running_loop()

        def _invoke():
            fn = getattr(self._obj, method)
            return fn(*args, **kwargs)

        return await loop.run_in_executor(self._pool, _invoke)

    async def close(self) -> None:
        channel_router.unregister("thread", self._actor_id)
        self._pool.shutdown(wait=True)

    # ---- decentralized channels ----
    async def get_endpoint(self) -> Endpoint:
        return Endpoint(scheme="thread", address="", actor_id=self._actor_id)

    async def chan_open(self, name: str) -> Endpoint:
        if name not in self._queues:
            self._queues[name] = asyncio.Queue()
        return await self.get_endpoint()

    async def _deliver_local(self, name: str, from_ep: Endpoint, payload: Any) -> None:
        q = self._queues.setdefault(name, asyncio.Queue())
        await q.put((from_ep, payload))

    async def chan_put(
        self, *, from_ep: Endpoint, to_ep: Endpoint, name: str, payload: Any
    ) -> None:
        if to_ep.scheme == "thread":
            if to_ep.actor_id == self._actor_id:
                await self._deliver_local(name, from_ep, payload)
                return
            peer = channel_router.resolve("thread", to_ep.actor_id)
            if peer is None:
                raise RuntimeError(f"no local thread actor {to_ep.actor_id}")
            await peer._deliver_local(name, from_ep, payload)
            return

        if to_ep.scheme == "process":
            peer = channel_router.resolve("process", to_ep.actor_id)
            if peer is None:
                raise RuntimeError(f"no local process actor {to_ep.actor_id}")
            await peer.chan_put(from_ep=from_ep, to_ep=to_ep, name=name, payload=payload)
            return

        if to_ep.scheme == "gpu":
            peer = channel_router.resolve("gpu", to_ep.actor_id)
            if peer is None:
                raise RuntimeError(f"no local gpu actor {to_ep.actor_id}")
            await peer.chan_put(from_ep=from_ep, to_ep=to_ep, name=name, payload=payload)
            return

        if to_ep.scheme == "ucx":
            if not ucx.have_ucx():
                raise RuntimeError("UCX requested but UCX is not available (need ucxx or ucp)")
            host, port_str = to_ep.address.rsplit(":", 1)

            async def _op(e):
                await ucx.send_control(
                    e,
                    {
                        "op": "chan_put",
                        "from": from_ep.__dict__,
                        "to": to_ep.__dict__,
                        "name": name,
                    },
                )
                tag, desc = ucx.pack_payload(payload)
                await ucx.send_payload(e, tag, desc, payload)
                rep = await ucx.recv_control(e)
                if not rep.get("ok", False):
                    raise RuntimeError(rep)

            await ucx.call(host, int(port_str), _op)
            return

        if to_ep.scheme == "tcp":
            await tcp.chan_put(
                to_ep.address,
                from_ep=from_ep,
                to_ep=to_ep,
                name=name,
                payload=payload,
            )
            return

        raise RuntimeError(f"ThreadActorBackend cannot route to {to_ep.scheme!r}")

    async def chan_get(self, *, ep: Endpoint, name: str, timeout: Optional[float]) -> Any:
        if ep.scheme == "thread" and ep.actor_id == self._actor_id:
            q = self._queues.setdefault(name, asyncio.Queue())
            if timeout is None:
                _, payload = await q.get()
                return payload
            try:
                _, payload = await asyncio.wait_for(q.get(), timeout=timeout)
                return payload
            except asyncio.TimeoutError:
                return None

        if ep.scheme == "ucx":
            if not ucx.have_ucx():
                raise RuntimeError("UCX requested but UCX is not available (need ucxx or ucp)")
            host, port_str = ep.address.rsplit(":", 1)

            async def _op(e):
                await ucx.send_control(
                    e,
                    {
                        "op": "chan_get",
                        "name": name,
                        "timeout": timeout,
                        "actor_id": ep.actor_id,
                    },
                )
                rep = await ucx.recv_control(e)
                if not rep.get("ok", False):
                    raise RuntimeError(rep)
                if rep.get("payload_is_none", False):
                    return None
                return await ucx.recv_payload(e)

            return await ucx.call(host, int(port_str), _op)

        if ep.scheme == "tcp":
            return await tcp.chan_get(
                ep.address,
                name=name,
                timeout=timeout,
                actor_id=ep.actor_id,
            )

        raise RuntimeError("Endpoint mismatch")
