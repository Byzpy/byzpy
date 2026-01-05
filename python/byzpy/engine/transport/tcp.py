from __future__ import annotations

"""
Transport implementation backed by simple TCP mailboxes.
Suitable for local/loopback simulations; not optimized for production.
"""
import threading
import time
from typing import Any, Callable, Dict, Tuple

from .base import Transport
from .tcp_simple import TcpMailbox, send_message


class TcpTransport(Transport):
    def __init__(self) -> None:
        self._mailboxes: Dict[str, TcpMailbox] = {}
        self._handlers: Dict[str, Callable[[Any], None]] = {}
        self._threads: Dict[str, threading.Thread] = {}
        self._stop = threading.Event()

    def register(self, node_id: str, handler: Callable[[Any], None]) -> None:
        if node_id in self._mailboxes:
            raise ValueError(f"Node {node_id} already registered")
        mbox = TcpMailbox()
        self._mailboxes[node_id] = mbox
        self._handlers[node_id] = handler
        t = threading.Thread(target=self._poll_loop, args=(node_id,), daemon=True)
        self._threads[node_id] = t
        t.start()

    def _poll_loop(self, node_id: str) -> None:
        mbox = self._mailboxes[node_id]
        handler = self._handlers[node_id]
        while not self._stop.is_set():
            try:
                msg = mbox.recv(timeout=0.1)
            except Exception:
                continue
            try:
                handler(msg)
            except Exception:
                continue

    def send(self, to_id: str, payload: Any) -> None:
        mbox = self._mailboxes.get(to_id)
        if mbox is None:
            raise KeyError(f"Unknown node_id {to_id}")
        addr: Tuple[str, int] = ("127.0.0.1", mbox.port)
        send_message(addr, payload)

    def close(self) -> None:
        self._stop.set()
        for m in self._mailboxes.values():
            m.close()
        for t in self._threads.values():
            t.join(timeout=1.0)
