from __future__ import annotations

import asyncio
import threading
import traceback
import uuid
from multiprocessing import get_context
from typing import Any, Dict, List, Optional, Tuple

import cloudpickle

from ..base import ActorBackend
from ..channels import Endpoint
from ..ipc import unwrap_payload, wrap_payload
from ..router import channel_router
from ..transports import tcp, ucx


def _worker(conn):
    obj = None
    import queue as qmod

    mailboxes: Dict[str, qmod.Queue] = {}

    def _ensure_q(name: str) -> qmod.Queue:
        q = mailboxes.get(name)
        if q is None:
            q = qmod.Queue()
            mailboxes[name] = q
        return q

    try:
        while True:
            msg = conn.recv()
            if msg is None:
                break
            try:
                op = msg["op"]

                if op == "construct":
                    target = cloudpickle.loads(msg["blob"])
                    args = unwrap_payload(msg["args"])
                    kwargs = unwrap_payload(msg["kwargs"])
                    obj = (
                        target(*args, **kwargs)
                        if not isinstance(target, type)
                        else target(*args, **kwargs)
                    )
                    conn.send(
                        {
                            "ok": True,
                            "id": msg["id"],
                            "payload": None,
                            "actor_id": msg["actor_id"],
                        }
                    )

                elif op == "call":
                    if obj is None:
                        raise RuntimeError("ProcessActor not initialized; call construct() first.")
                    fn = getattr(obj, msg["method"])
                    args = unwrap_payload(msg["args"])
                    kwargs = unwrap_payload(msg["kwargs"])
                    res = fn(*args, **kwargs)
                    conn.send({"ok": True, "id": msg["id"], "payload": wrap_payload(res)})

                elif op == "chan_open":
                    _ensure_q(msg["name"])
                    conn.send({"ok": True, "id": msg["id"]})

                elif op == "chan_deliver":
                    name = msg["name"]
                    payload = msg["payload"]
                    _ensure_q(name).put(payload)
                    conn.send({"ok": True, "id": msg["id"]})

                elif op == "chan_get":
                    name = msg["name"]
                    timeout = msg.get("timeout", None)
                    q = _ensure_q(name)
                    try:
                        item = q.get(timeout=timeout) if timeout is not None else q.get()
                    except qmod.Empty:
                        payload = None
                    else:
                        if isinstance(item, tuple) and len(item) == 2:
                            _, raw_payload = item
                            payload = raw_payload
                        else:
                            payload = item
                    conn.send({"ok": True, "id": msg["id"], "payload": payload})

                else:
                    raise RuntimeError(f"unknown op {op!r}")

            except Exception as e:
                conn.send(
                    {
                        "ok": False,
                        "id": msg.get("id"),
                        "payload": (type(e).__name__, str(e), traceback.format_exc()),
                    }
                )
    finally:
        try:
            conn.close()
        except Exception:
            pass


class ProcessActorBackend(ActorBackend):
    def __init__(self) -> None:
        ctx = get_context("spawn")
        parent, child = ctx.Pipe(duplex=True)
        self._parent = parent
        self._proc = ctx.Process(target=_worker, args=(child,), daemon=True)
        self._proc.start()
        child.close()

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._actor_id = str(uuid.uuid4())
        channel_router.register("process", self._actor_id, self)
        self._io_lock = threading.Lock()

    async def start(self):
        if self._loop is None:
            self._loop = asyncio.get_running_loop()

    async def construct(self, cls_or_factory: Any, *, args: tuple, kwargs: dict) -> None:
        blob = cloudpickle.dumps(cls_or_factory)
        req = {
            "op": "construct",
            "id": str(uuid.uuid4()),
            "blob": blob,
            "args": wrap_payload(args),
            "kwargs": wrap_payload(kwargs),
            "actor_id": self._actor_id,
        }
        await self._send(req)

    async def call(self, method: str, *args, **kwargs):
        req = {
            "op": "call",
            "id": str(uuid.uuid4()),
            "method": method,
            "args": wrap_payload(args),
            "kwargs": wrap_payload(kwargs),
        }
        return await self._send(req)

    async def close(self):
        try:

            def _rt_close():
                with self._io_lock:
                    try:
                        self._parent.send(None)
                    except Exception:
                        pass
                    try:
                        self._parent.close()
                    except Exception:
                        pass

            loop = self._loop or asyncio.get_running_loop()
            await loop.run_in_executor(None, _rt_close)
        except Exception:
            pass
        self._proc.join(timeout=5)
        channel_router.unregister("process", self._actor_id)

    async def get_endpoint(self) -> Endpoint:
        return Endpoint(scheme="process", address="", actor_id=self._actor_id)

    async def chan_open(self, name: str) -> Endpoint:
        await self._send({"op": "chan_open", "id": str(uuid.uuid4()), "name": name})
        return await self.get_endpoint()

    async def chan_put(
        self, *, from_ep: Endpoint, to_ep: Endpoint, name: str, payload: Any
    ) -> None:
        if to_ep.scheme == "process":
            safe_payload = wrap_payload(payload)
            if to_ep.actor_id == self._actor_id:
                await self._send(
                    {
                        "op": "chan_deliver",
                        "id": str(uuid.uuid4()),
                        "name": name,
                        "payload": (from_ep, safe_payload),
                    }
                )
                return
            peer = channel_router.resolve("process", to_ep.actor_id)
            if peer is None:
                raise RuntimeError(f"no local process actor {to_ep.actor_id}")
            await peer._send(
                {
                    "op": "chan_deliver",
                    "id": str(uuid.uuid4()),
                    "name": name,
                    "payload": (from_ep, safe_payload),
                }
            )
            return

        if to_ep.scheme == "thread":
            peer = channel_router.resolve("thread", to_ep.actor_id)
            if peer is None:
                raise RuntimeError(f"no local thread actor {to_ep.actor_id}")
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

        if to_ep.scheme == "gpu":
            peer = channel_router.resolve("gpu", to_ep.actor_id)
            if peer is None:
                raise RuntimeError(f"no local gpu actor {to_ep.actor_id}")
            await peer.chan_put(from_ep=from_ep, to_ep=to_ep, name=name, payload=payload)
            return

        raise RuntimeError(f"ProcessActorBackend cannot route to {to_ep.scheme!r}")

    async def chan_get(self, *, ep: Endpoint, name: str, timeout: Optional[float]):
        if ep.scheme == "process" and ep.actor_id == self._actor_id:
            raw = await self._send(
                {
                    "op": "chan_get",
                    "id": str(uuid.uuid4()),
                    "name": name,
                    "timeout": timeout,
                }
            )
            return unwrap_payload(raw)

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

    async def _send(self, msg: Dict):
        loop = self._loop or asyncio.get_running_loop()

        def _rt():
            with self._io_lock:
                self._parent.send(msg)
                rep = self._parent.recv()
            if rep["ok"]:
                return unwrap_payload(rep.get("payload"))
            et, emsg, tb = rep["payload"]
            raise RuntimeError(f"[ProcessActor Remote {et}] {emsg}\n{tb}")

        try:
            return await loop.run_in_executor(None, _rt)
        except (EOFError, BrokenPipeError) as e:
            raise RuntimeError(
                "ProcessActor worker terminated while handling request "
                "(likely due to concurrent Pipe writes without locking; now fixed)."
            ) from e
