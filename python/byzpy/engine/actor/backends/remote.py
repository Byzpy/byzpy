# byzpy/engine/actor/backends/remote.py
from __future__ import annotations

import asyncio
import traceback
import uuid
from typing import Any, Dict, Optional, Tuple

import cloudpickle

from .._wire import recv_obj, send_obj
from ..base import ActorBackend
from ..channels import Endpoint
from ..ipc import unwrap_payload, wrap_payload
from ..router import channel_router
from ..transports import tcp, ucx


class RemoteActorBackend(ActorBackend):
    def __init__(self, host: str, port: int):
        self._host = host
        self._port = int(port)
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._actor_id: Optional[str] = None
        self._io_lock = asyncio.Lock()

    async def start(self) -> None:
        if self._reader is None or self._writer is None:
            self._reader, self._writer = await asyncio.open_connection(self._host, self._port)

    def _ensure_open(self) -> Tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        r, w = self._reader, self._writer
        if r is None or w is None:
            raise RuntimeError("RemoteActorBackend not started. Call await start() first.")
        return r, w

    async def close(self) -> None:
        async with self._io_lock:
            if self._writer is None or self._reader is None:
                self._actor_id = None
                return
            try:
                if self._actor_id is not None:
                    await send_obj(self._writer, {"op": "close", "actor_id": self._actor_id})
                    await recv_obj(self._reader)
            except Exception:
                pass
            finally:
                try:
                    self._writer.close()
                    await self._writer.wait_closed()
                except Exception:
                    pass
                self._reader = None
                self._writer = None
                self._actor_id = None

    async def construct(self, cls_or_factory: Any, *, args: tuple, kwargs: dict) -> None:
        async with self._io_lock:
            await self.start()
            r, w = self._ensure_open()
            blob = cloudpickle.dumps(cls_or_factory)
            await send_obj(
                w,
                {
                    "op": "construct",
                    "blob": blob,
                    "args": wrap_payload(args),
                    "kwargs": wrap_payload(kwargs),
                },
            )
            rep = await recv_obj(r)
            if not rep.get("ok", False):
                et, em, tb = rep["payload"]
                raise RuntimeError(f"[RemoteActor construct {et}] {em}\n{tb}")
            self._actor_id = rep["actor_id"]

    async def call(self, method: str, *args, **kwargs):
        async with self._io_lock:
            if not self._actor_id:
                raise RuntimeError("Remote actor not constructed.")
            await self.start()
            r, w = self._ensure_open()
            await send_obj(
                w,
                {
                    "op": "call",
                    "actor_id": self._actor_id,
                    "method": method,
                    "args": wrap_payload(args),
                    "kwargs": wrap_payload(kwargs),
                },
            )
            rep = await recv_obj(r)
            if rep.get("ok", False):
                return unwrap_payload(rep["payload"])
            et, em, tb = rep["payload"]
            raise RuntimeError(f"[RemoteActor call {et}] {em}\n{tb}")

    async def get_endpoint(self) -> Endpoint:
        async with self._io_lock:
            await self.start()
            r, w = self._ensure_open()
            await send_obj(w, {"op": "get_ep"})
            rep = await recv_obj(r)
            if not rep.get("ok", False):
                raise RuntimeError(rep)
            return Endpoint(**rep["payload"])

    async def chan_open(self, name: str) -> Endpoint:
        async with self._io_lock:
            await self.start()
            r, w = self._ensure_open()
            await send_obj(w, {"op": "chan_open", "name": name})
            rep = await recv_obj(r)
            if not rep.get("ok", False):
                raise RuntimeError(rep)
            return Endpoint(**rep["payload"])

    async def chan_put(
        self, *, from_ep: Endpoint, to_ep: Endpoint, name: str, payload: Any
    ) -> None:
        local_payload = unwrap_payload(payload)
        if to_ep.scheme == "thread":
            peer = channel_router.resolve("thread", to_ep.actor_id)
            if peer is None:
                raise RuntimeError(f"no local thread actor {to_ep.actor_id}")
            await peer.chan_put(from_ep=from_ep, to_ep=to_ep, name=name, payload=local_payload)
            return

        if to_ep.scheme == "process":
            peer = channel_router.resolve("process", to_ep.actor_id)
            if peer is None:
                raise RuntimeError(f"no local process actor {to_ep.actor_id}")
            await peer.chan_put(from_ep=from_ep, to_ep=to_ep, name=name, payload=local_payload)
            return

        if to_ep.scheme == "gpu":
            peer = channel_router.resolve("gpu", to_ep.actor_id)
            if peer is None:
                raise RuntimeError(f"no local gpu actor {to_ep.actor_id}")
            await peer.chan_put(from_ep=from_ep, to_ep=to_ep, name=name, payload=local_payload)
            return

        dest_hostport = to_ep.address
        my_hostport = f"{self._host}:{self._port}"
        wrapped = wrap_payload(payload)
        if to_ep.scheme == "tcp" and dest_hostport == my_hostport:
            async with self._io_lock:
                await self.start()
                r, w = self._ensure_open()
                await send_obj(
                    w,
                    {
                        "op": "chan_put",
                        "from": from_ep.__dict__,
                        "to": to_ep.__dict__,
                        "name": name,
                        "payload": wrapped,
                    },
                )
                rep = await recv_obj(r)
                if not rep.get("ok", False):
                    raise RuntimeError(rep)
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
                tag, desc = ucx.pack_payload(wrapped)
                await ucx.send_payload(e, tag, desc, wrapped)
                rep = await ucx.recv_control(e)
                if not rep.get("ok", False):
                    raise RuntimeError(rep)

            await ucx.call(host, int(port_str), _op)
            return

        if to_ep.scheme == "tcp":
            await tcp.chan_put(
                dest_hostport,
                from_ep=from_ep,
                to_ep=to_ep,
                name=name,
                payload=wrapped,
            )
            return

        raise RuntimeError(f"RemoteActorBackend cannot route to {to_ep.scheme!r}")

    async def chan_get(self, *, ep: Endpoint, name: str, timeout: Optional[float]):
        if ep.scheme == "thread":
            peer = channel_router.resolve("thread", ep.actor_id)
            if peer is None:
                raise RuntimeError(f"no local thread actor {ep.actor_id}")
            return await peer.chan_get(ep=ep, name=name, timeout=timeout)

        if ep.scheme == "process":
            peer = channel_router.resolve("process", ep.actor_id)
            if peer is None:
                raise RuntimeError(f"no local process actor {ep.actor_id}")
            return await peer.chan_get(ep=ep, name=name, timeout=timeout)

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

        if ep.scheme != "tcp":
            raise RuntimeError("RemoteActorBackend only resolves tcp endpoints")
        my_hostport = f"{self._host}:{self._port}"

        if ep.address == my_hostport:
            async with self._io_lock:
                await self.start()
                r, w = self._ensure_open()
            await send_obj(w, {"op": "chan_get", "name": name, "timeout": timeout})
            rep = await recv_obj(r)
            if rep.get("ok", False):
                return unwrap_payload(rep["payload"])
            raise RuntimeError(rep)

        payload = await tcp.chan_get(
            ep.address,
            name=name,
            timeout=timeout,
            actor_id=ep.actor_id,
        )
        return unwrap_payload(payload)


class RemoteActorServer:
    def __init__(self, host: str, port: int):
        self._host, self._port = host, port
        self._actors: Dict[str, Any] = {}
        self._mb: Dict[str, Dict[str, asyncio.Queue]] = {}
        self._srv: Optional[asyncio.AbstractServer] = None
        self._addr_cache: Optional[str] = None

    def _hostport(self) -> str:
        if self._addr_cache:
            return self._addr_cache
        srv = self._srv
        if srv is not None:
            sockets = getattr(srv, "sockets", None)
            if sockets:
                host, port = sockets[0].getsockname()[0], sockets[0].getsockname()[1]
                self._addr_cache = f"{host}:{port}"
                return self._addr_cache
        return f"{self._host}:{self._port}"

    async def serve(self):
        self._srv = await asyncio.start_server(self._handle, self._host, self._port)
        sockets = getattr(self._srv, "sockets", None)
        if sockets:
            host, port = sockets[0].getsockname()[0], sockets[0].getsockname()[1]
            self._addr_cache = f"{host}:{port}"
        print(f"[RemoteActorServer] Serving on {self._hostport()}")
        async with self._srv:
            await self._srv.serve_forever()

    async def _ensure_mb(self, aid: str, name: str) -> asyncio.Queue:
        chs = self._mb.setdefault(aid, {})
        q = chs.get(name)
        if q is None:
            q = asyncio.Queue()
            chs[name] = q
        return q

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        bound_aid: Optional[str] = None

        try:
            while True:
                req = await recv_obj(reader)
                op = req.get("op")

                if op == "construct":
                    target = cloudpickle.loads(req["blob"])
                    args = unwrap_payload(req.get("args", ()))
                    kwargs = unwrap_payload(req.get("kwargs", {}) or {})
                    obj = target(*args, **kwargs)
                    aid = str(uuid.uuid4())
                    self._actors[aid] = obj
                    bound_aid = aid
                    await send_obj(writer, {"ok": True, "actor_id": aid})

                elif op == "get_ep":
                    if not bound_aid:
                        raise RuntimeError("no actor on this connection")
                    await send_obj(
                        writer,
                        {
                            "ok": True,
                            "payload": {
                                "scheme": "tcp",
                                "address": self._hostport(),
                                "actor_id": bound_aid,
                            },
                        },
                    )

                elif op == "chan_open":
                    if not bound_aid:
                        raise RuntimeError("no actor on this connection")
                    name = req["name"]
                    await self._ensure_mb(bound_aid, name)
                    await send_obj(
                        writer,
                        {
                            "ok": True,
                            "payload": {
                                "scheme": "tcp",
                                "address": self._hostport(),
                                "actor_id": bound_aid,
                            },
                        },
                    )

                elif op == "chan_put":
                    to = req["to"]
                    payload = req["payload"]
                    name = req["name"]
                    to_aid = to["actor_id"]
                    q = await self._ensure_mb(to_aid, name)
                    await q.put((req["from"], payload))
                    await send_obj(writer, {"ok": True})

                elif op == "chan_get":
                    aid = bound_aid or req.get("actor_id")
                    if not aid:
                        raise RuntimeError("actor_id missing for chan_get")
                    name = req["name"]
                    timeout = req.get("timeout")
                    q = await self._ensure_mb(aid, name)
                    if timeout is None:
                        fr, payload = await q.get()
                    else:
                        try:
                            fr, payload = await asyncio.wait_for(q.get(), timeout=timeout)
                        except asyncio.TimeoutError:
                            payload = None
                    await send_obj(writer, {"ok": True, "payload": payload})

                elif op == "call":
                    if not bound_aid:
                        raise RuntimeError("no actor on this connection")
                    obj = self._actors[bound_aid]
                    import inspect

                    call_args = unwrap_payload(req.get("args", ()))
                    call_kwargs = unwrap_payload(req.get("kwargs", {}) or {})
                    if inspect.iscoroutinefunction(getattr(obj, req["method"])):
                        res = await getattr(obj, req["method"])(*call_args, **call_kwargs)
                    else:
                        loop = asyncio.get_running_loop()
                        res = await loop.run_in_executor(
                            None,
                            lambda: getattr(obj, req["method"])(*call_args, **call_kwargs),
                        )
                    await send_obj(writer, {"ok": True, "payload": wrap_payload(res)})

                elif op == "close":
                    if bound_aid:
                        self._actors.pop(bound_aid, None)
                        self._mb.pop(bound_aid, None)
                    await send_obj(writer, {"ok": True})

                else:
                    await send_obj(
                        writer,
                        {
                            "ok": False,
                            "payload": ("RuntimeError", f"unknown op {op}", ""),
                        },
                    )

        except asyncio.IncompleteReadError:
            pass
        except Exception as e:
            try:
                await send_obj(
                    writer,
                    {
                        "ok": False,
                        "payload": (type(e).__name__, str(e), traceback.format_exc()),
                    },
                )
            except Exception:
                pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass


async def start_actor_server(host: str, port: int):
    server = RemoteActorServer(host=host, port=port)
    await server.serve()
