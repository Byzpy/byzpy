from __future__ import annotations

import asyncio
import traceback
import uuid
from typing import Any, Dict, Optional

import cloudpickle

from ..base import ActorBackend
from ..channels import Endpoint
from ..router import channel_router
from ..transports import tcp, ucx


def _resolve_backend(scheme: str, actor_id: str):
    return channel_router.resolve(scheme, actor_id)


# ----------------------------- GPU local backend --------------------------------


class GPUActorBackend(ActorBackend):
    """
    Local, single-process actor that prefers CUDA tensors, exposes endpoint scheme 'gpu'.
    Mailboxes are in-process asyncio. Cross-backend routing:
      - to 'gpu' in same process: enqueue (zero-copy PyTorch tensors)
      - to 'thread'/'process': delegate to their registries
      - to 'tcp': use your TCP server protocol
      - to 'ucx': UCX fast path (GPU→GPU) using UCX-Py
    """

    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._obj = None
        self._actor_id = str(uuid.uuid4())
        self._queues: Dict[str, asyncio.Queue] = {}
        channel_router.register("gpu", self._actor_id, self)

    async def start(self):
        if self._loop is None:
            self._loop = asyncio.get_running_loop()

    async def construct(self, cls_or_factory: Any, *, args: tuple, kwargs: dict) -> None:
        target = cls_or_factory
        self._obj = (
            target(*args, **kwargs) if not isinstance(target, type) else target(*args, **kwargs)
        )

    async def call(self, method: str, *args, **kwargs):
        fn = getattr(self._obj, method)
        if asyncio.iscoroutinefunction(fn):
            return await fn(*args, **kwargs)
        loop = self._loop or asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))

    async def close(self) -> None:
        channel_router.unregister("gpu", self._actor_id)

    # ---- decentralized channels ----
    async def get_endpoint(self) -> Endpoint:
        return Endpoint(scheme="gpu", address="", actor_id=self._actor_id)

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
        # 1) target == local gpu actor
        if to_ep.scheme == "gpu":
            if to_ep.actor_id == self._actor_id:
                await self._deliver_local(name, from_ep, payload)
                return
            peer = _resolve_backend("gpu", to_ep.actor_id)
            if peer is None:
                raise RuntimeError(f"no local gpu actor {to_ep.actor_id}")
            await peer._deliver_local(name, from_ep, payload)
            return

        # 2) route to local thread/process
        if to_ep.scheme == "thread":
            peer = _resolve_backend("thread", to_ep.actor_id)
            if peer is None:
                raise RuntimeError(f"no local thread actor {to_ep.actor_id}")
            await peer.chan_put(from_ep=from_ep, to_ep=to_ep, name=name, payload=payload)
            return

        if to_ep.scheme == "process":
            peer = _resolve_backend("process", to_ep.actor_id)
            if peer is None:
                raise RuntimeError(f"no local process actor {to_ep.actor_id}")
            await peer.chan_put(from_ep=from_ep, to_ep=to_ep, name=name, payload=payload)
            return

        # 3) UCX fast path (GPU→GPU) to remote UCX server
        if to_ep.scheme == "ucx":
            if not ucx.have_ucx():
                raise RuntimeError("UCX requested but UCX is not available (need ucxx or ucp)")
            host, port = to_ep.address.rsplit(":", 1)

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

            await ucx.call(host, int(port), _op)
            return

        # 4) TCP fallback
        if to_ep.scheme == "tcp":
            await tcp.chan_put(
                to_ep.address,
                from_ep=from_ep,
                to_ep=to_ep,
                name=name,
                payload=payload,
            )
            return

        raise RuntimeError(f"GPUActorBackend cannot route to {to_ep.scheme!r}")

    async def chan_get(self, *, ep: Endpoint, name: str, timeout: Optional[float]):
        # 1) local GPU mailbox
        if ep.scheme == "gpu" and ep.actor_id == self._actor_id:
            q = self._queues.setdefault(name, asyncio.Queue())
            if timeout is None:
                _, payload = await q.get()
                return payload
            try:
                _, payload = await asyncio.wait_for(q.get(), timeout=timeout)
                return payload
            except asyncio.TimeoutError:
                return None

        # 2) read from local thread/process backends (same Python runtime)
        if ep.scheme == "thread":
            peer = _resolve_backend("thread", ep.actor_id)
            if peer is None:
                raise RuntimeError(f"no local thread actor {ep.actor_id}")
            return await peer.chan_get(ep=ep, name=name, timeout=timeout)

        if ep.scheme == "process":
            peer = _resolve_backend("process", ep.actor_id)
            if peer is None:
                raise RuntimeError(f"no local process actor {ep.actor_id}")
            return await peer.chan_get(ep=ep, name=name, timeout=timeout)

        # 3) read from a UCX server mailbox
        if ep.scheme == "ucx":
            if not ucx.have_ucx():
                raise RuntimeError("UCX requested but UCX is not available (need ucxx or ucp)")
            host, port = ep.address.rsplit(":", 1)

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

            return await ucx.call(host, int(port), _op)

        # 4) TCP read
        if ep.scheme == "tcp":
            return await tcp.chan_get(
                ep.address,
                name=name,
                timeout=timeout,
                actor_id=ep.actor_id,
            )

        raise RuntimeError("Endpoint mismatch")


# ----------------------------- UCX remote (optional) ----------------------------


class UCXRemoteActorBackend(ActorBackend):
    """
    Client for a UCXRemoteActorServer. Lets you do:
      backend = UCXRemoteActorBackend(host, port)
      ref = ActorRef(backend)  # normal construct/call
    Also provides UCX-aware chan_put/chan_get for GPU GPU transfers.
    """

    def __init__(self, host: str, port: int):
        self._host, self._port = host, int(port)
        self._ep = None
        self._actor_id: Optional[str] = None
        self._io_lock = asyncio.Lock()
        self._advertised_address: Optional[str] = None

    async def start(self) -> None:
        if self._ep is None:
            if not ucx.have_ucx():
                raise RuntimeError("UCX is not available (install ucxx or ucp)")
            # Persistent endpoint dedicated to this backend (no sharing).
            self._ep = await ucx.create_endpoint(self._host, self._port)

    async def close(self) -> None:
        async with self._io_lock:
            if self._ep is None:
                self._actor_id = None
                return
            try:
                if self._actor_id is not None:
                    await ucx.send_control(self._ep, {"op": "close", "actor_id": self._actor_id})
                    await ucx.recv_control(self._ep)
            except Exception:
                pass
            finally:
                try:
                    await self._ep.close()
                except Exception:
                    pass
                self._ep = None
                self._actor_id = None
                self._advertised_address = None

    async def construct(self, cls_or_factory: Any, *, args: tuple, kwargs: dict) -> None:
        async with self._io_lock:
            await self.start()
            blob = cloudpickle.dumps(cls_or_factory)
            await ucx.send_control(
                self._ep,
                {"op": "construct", "blob": blob, "args": args, "kwargs": kwargs},
            )
            rep = await ucx.recv_control(self._ep)
            if not rep.get("ok", False):
                et, em, tb = rep["payload"]
                raise RuntimeError(f"[UCXRemote construct {et}] {em}\n{tb}")
            self._actor_id = rep["actor_id"]

    async def call(self, method: str, *args, **kwargs):
        async with self._io_lock:
            if not self._actor_id:
                raise RuntimeError("UCX remote actor not constructed.")
            await self.start()
            await ucx.send_control(
                self._ep,
                {
                    "op": "call",
                    "actor_id": self._actor_id,
                    "method": method,
                    "args": args,
                    "kwargs": kwargs,
                },
            )
            rep = await ucx.recv_control(self._ep)
            if rep.get("ok", False):
                return rep["payload"]
            et, em, tb = rep["payload"]
            raise RuntimeError(f"[UCXRemote call {et}] {em}\n{tb}")

    async def get_endpoint(self) -> Endpoint:
        async with self._io_lock:
            await self.start()
            await ucx.send_control(self._ep, {"op": "get_ep"})
            rep = await ucx.recv_control(self._ep)
            if not rep.get("ok", False):
                raise RuntimeError(rep)
            payload = rep["payload"]
            addr = payload.get("address")
            if addr:
                self._advertised_address = addr
            return Endpoint(**payload)

    async def chan_open(self, name: str) -> Endpoint:
        async with self._io_lock:
            await self.start()
            await ucx.send_control(self._ep, {"op": "chan_open", "name": name})
            rep = await ucx.recv_control(self._ep)
            if not rep.get("ok", False):
                raise RuntimeError(rep)
            payload = rep["payload"]
            addr = payload.get("address")
            if addr:
                self._advertised_address = addr
            return Endpoint(**payload)

    def _is_same_server(self, address: str) -> bool:
        if self._advertised_address and address == self._advertised_address:
            return True
        try:
            host, port_str = address.rsplit(":", 1)
            port = int(port_str)
        except ValueError:
            return False
        if port != self._port:
            return False
        if host == self._host:
            return True
        if self._host in ("0.0.0.0", "") and host in ("127.0.0.1", "localhost"):
            return True
        if host in ("0.0.0.0", "") and self._host in ("127.0.0.1", "localhost"):
            return True
        return False

    async def chan_put(
        self, *, from_ep: Endpoint, to_ep: Endpoint, name: str, payload: Any
    ) -> None:
        # route to local thread/process/gpu
        if to_ep.scheme == "thread":
            peer = _resolve_backend("thread", to_ep.actor_id)
            if peer is None:
                raise RuntimeError(f"no local thread actor {to_ep.actor_id}")
            await peer.chan_put(from_ep=from_ep, to_ep=to_ep, name=name, payload=payload)
            return
        if to_ep.scheme == "process":
            peer = _resolve_backend("process", to_ep.actor_id)
            if peer is None:
                raise RuntimeError(f"no local process actor {to_ep.actor_id}")
            await peer.chan_put(from_ep=from_ep, to_ep=to_ep, name=name, payload=payload)
            return
        if to_ep.scheme == "gpu":
            peer = _resolve_backend("gpu", to_ep.actor_id)
            if peer is None:
                raise RuntimeError(f"no local gpu actor {to_ep.actor_id}")
            await peer.chan_put(from_ep=from_ep, to_ep=to_ep, name=name, payload=payload)
            return

        # same UCX server via persistent ep (already serialized by _io_lock)
        if to_ep.scheme == "ucx" and self._is_same_server(to_ep.address):
            async with self._io_lock:
                await self.start()
                await ucx.send_control(
                    self._ep,
                    {
                        "op": "chan_put",
                        "from": from_ep.__dict__,
                        "to": to_ep.__dict__,
                        "name": name,
                    },
                )
                tag, desc = ucx.pack_payload(payload)
                await ucx.send_payload(self._ep, tag, desc, payload)
                rep = await ucx.recv_control(self._ep)
                if not rep.get("ok", False):
                    raise RuntimeError(rep)
            return

        # different UCX server via pooled ep
        if to_ep.scheme == "ucx":
            host, port = to_ep.address.rsplit(":", 1)

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

            await ucx.call(host, int(port), _op)
            return

        # tcp fallback
        if to_ep.scheme == "tcp":
            await tcp.chan_put(
                to_ep.address,
                from_ep=from_ep,
                to_ep=to_ep,
                name=name,
                payload=payload,
            )
            return

        raise RuntimeError(f"UCXRemoteActorBackend cannot route to {to_ep.scheme!r}")

    async def chan_get(self, *, ep: Endpoint, name: str, timeout: Optional[float]):
        # local thread/process/gpu reads
        if ep.scheme == "thread":
            peer = _resolve_backend("thread", ep.actor_id)
            if peer is None:
                raise RuntimeError(f"no local thread actor {ep.actor_id}")
            return await peer.chan_get(ep=ep, name=name, timeout=timeout)
        if ep.scheme == "process":
            peer = _resolve_backend("process", ep.actor_id)
            if peer is None:
                raise RuntimeError(f"no local process actor {ep.actor_id}")
            return await peer.chan_get(ep=ep, name=name, timeout=timeout)
        if ep.scheme == "gpu":
            peer = _resolve_backend("gpu", ep.actor_id)
            if peer is None:
                raise RuntimeError(f"no local gpu actor {ep.actor_id}")
            return await peer.chan_get(ep=ep, name=name, timeout=timeout)

        # same UCX server via persistent ep
        if ep.scheme == "ucx" and self._is_same_server(ep.address):
            async with self._io_lock:
                await self.start()
                await ucx.send_control(
                    self._ep, {"op": "chan_get", "name": name, "timeout": timeout}
                )
                rep = await ucx.recv_control(self._ep)
                if not rep.get("ok", False):
                    raise RuntimeError(rep)
                if rep.get("payload_is_none", False):
                    return None
                return await ucx.recv_payload(self._ep)

        # different UCX server
        if ep.scheme == "ucx":
            host, port = ep.address.rsplit(":", 1)

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

            return await ucx.call(host, int(port), _op)

        # tcp
        if ep.scheme == "tcp":
            return await tcp.chan_get(
                ep.address,
                name=name,
                timeout=timeout,
                actor_id=ep.actor_id,
            )

        raise RuntimeError("Endpoint mismatch")


# ----------------------------- UCX remote server --------------------------------


class UCXRemoteActorServer:
    """
    UCX multi-actor server with CUDA-aware channels.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 0):
        if not ucx.have_ucx():
            raise RuntimeError("UCX (ucp) is not available in this environment.")
        self._host, self._port = host, int(port)
        self._actors: Dict[str, Any] = {}
        self._mb: Dict[str, Dict[str, asyncio.Queue]] = {}
        self._listener = None
        self._addr_cache: Optional[str] = None

    def address(self) -> str:
        return self._addr_cache or f"{self._host}:{self._port}"

    async def _send_error(self, ep, exc: Exception) -> None:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        payload = (type(exc).__name__, str(exc), tb)
        try:
            await ucx.send_control(ep, {"ok": False, "payload": payload})
        except Exception:
            pass

    def _log_error(self, op: Any, exc: Exception) -> None:
        try:
            opname = op if isinstance(op, str) else repr(op)
            print(f"[UCXRemoteActorServer] error during {opname}: {type(exc).__name__}: {exc}")
        except Exception:
            pass

    async def serve(self):
        async def _handle(ep):
            bound_aid: Optional[str] = None
            try:
                while True:
                    req = await ucx.recv_control(ep)
                    op = req.get("op")

                    try:
                        if op == "construct":
                            target = cloudpickle.loads(req["blob"])
                            args = req.get("args", ())
                            kwargs = req.get("kwargs", {}) or {}
                            obj = target(*args, **kwargs)
                            aid = str(uuid.uuid4())
                            self._actors[aid] = obj
                            bound_aid = aid
                            await ucx.send_control(ep, {"ok": True, "actor_id": aid})

                        elif op == "get_ep":
                            if not bound_aid:
                                raise RuntimeError("no actor on this connection")
                            await ucx.send_control(
                                ep,
                                {
                                    "ok": True,
                                    "payload": {
                                        "scheme": "ucx",
                                        "address": self.address(),
                                        "actor_id": bound_aid,
                                    },
                                },
                            )

                        elif op == "chan_open":
                            if not bound_aid:
                                raise RuntimeError("no actor on this connection")
                            name = req["name"]
                            await self._ensure_mb(bound_aid, name)
                            await ucx.send_control(
                                ep,
                                {
                                    "ok": True,
                                    "payload": {
                                        "scheme": "ucx",
                                        "address": self.address(),
                                        "actor_id": bound_aid,
                                    },
                                },
                            )

                        elif op == "chan_put":
                            to = req["to"]
                            name = req["name"]
                            to_aid = to["actor_id"]
                            q = await self._ensure_mb(to_aid, name)
                            payload = await ucx.recv_payload(ep)
                            await q.put((req["from"], payload))
                            await ucx.send_control(ep, {"ok": True})

                        elif op == "chan_get":
                            aid = bound_aid or req.get("actor_id")
                            if not aid:
                                raise RuntimeError("actor_id missing for chan_get")
                            name = req["name"]
                            timeout = req.get("timeout")
                            q = await self._ensure_mb(aid, name)
                            try:
                                if timeout is None:
                                    fr, payload = await q.get()
                                else:
                                    fr, payload = await asyncio.wait_for(q.get(), timeout=timeout)
                                await ucx.send_control(ep, {"ok": True, "payload_is_none": False})
                                tag, desc = ucx.pack_payload(payload)
                                await ucx.send_payload(ep, tag, desc, payload)
                            except asyncio.TimeoutError:
                                await ucx.send_control(ep, {"ok": True, "payload_is_none": True})

                        elif op == "call":
                            if not bound_aid:
                                raise RuntimeError("no actor on this connection")
                            obj = self._actors[bound_aid]
                            import inspect

                            if inspect.iscoroutinefunction(getattr(obj, req["method"])):
                                res = await getattr(obj, req["method"])(
                                    *req.get("args", ()),
                                    **(req.get("kwargs", {}) or {}),
                                )
                            else:
                                loop = asyncio.get_running_loop()
                                res = await loop.run_in_executor(
                                    None,
                                    lambda: getattr(obj, req["method"])(
                                        *req.get("args", ()),
                                        **(req.get("kwargs", {}) or {}),
                                    ),
                                )
                            await ucx.send_control(ep, {"ok": True, "payload": res})

                        elif op == "close":
                            if bound_aid:
                                self._actors.pop(bound_aid, None)
                                self._mb.pop(bound_aid, None)
                            await ucx.send_control(ep, {"ok": True})

                        else:
                            raise RuntimeError(f"unknown op {op}")
                    except Exception as exc:
                        self._log_error(op, exc)
                        await self._send_error(ep, exc)
                        if op == "construct":
                            # construction failures leave the connection unbound
                            bound_aid = None

            except Exception:
                # best-effort error reply already sent above if possible
                pass
            finally:
                try:
                    await ep.close()
                except Exception:
                    pass

        def _host_ip() -> str:
            import socket

            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                ip = s.getsockname()[0]
                s.close()
                return ip
            except Exception:
                return "127.0.0.1"

        self._listener = ucx.create_listener(_handle, host=self._host, port=self._port)
        host = self._host if self._host not in ("0.0.0.0", "") else _host_ip()
        self._addr_cache = f"{host}:{int(self._listener.port)}"
        print(f"[UCXRemoteActorServer] Serving on ucx://{self._addr_cache}")
        while True:
            await asyncio.sleep(3600)

    async def _ensure_mb(self, aid: str, name: str) -> asyncio.Queue:
        chs = self._mb.setdefault(aid, {})
        q = chs.get(name)
        if q is None:
            q = asyncio.Queue()
            chs[name] = q
        return q


async def start_ucx_actor_server(host: str = "0.0.0.0", port: int = 0):
    srv = UCXRemoteActorServer(host=host, port=port)
    await srv.serve()
