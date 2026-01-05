from __future__ import annotations

"""Very simple TCP transport for message passing (loopback-friendly).

This is a minimal, length-prefixed, pickle-based transport intended for tests
and local simulations. It is not optimized for production or large payloads.
"""

import pickle
import queue
import socket
import threading
from typing import Any, Optional, Tuple


def _recv_all(conn: socket.socket, n: int) -> bytes:
    data = bytearray()
    while len(data) < n:
        chunk = conn.recv(n - len(data))
        if not chunk:
            raise ConnectionError("connection closed before receiving payload")
        data.extend(chunk)
    return bytes(data)


def send_message(addr: Tuple[str, int], payload: Any) -> None:
    buf = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    length = len(buf).to_bytes(8, byteorder="big")
    with socket.create_connection(addr, timeout=5.0) as sock:
        sock.sendall(length)
        sock.sendall(buf)


class TcpMailbox:
    def __init__(self, host: str = "127.0.0.1", port: int = 0) -> None:
        self.host = host
        self.port = port
        self._q: queue.Queue[Any] = queue.Queue()
        self._stop = threading.Event()
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind((host, port))
        self._server.listen()
        self.port = self._server.getsockname()[1]
        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()

    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._server.settimeout(0.2)
                conn, _ = self._server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(target=self._handle_conn, args=(conn,), daemon=True).start()

    def _handle_conn(self, conn: socket.socket) -> None:
        with conn:
            try:
                header = _recv_all(conn, 8)
                length = int.from_bytes(header, byteorder="big")
                buf = _recv_all(conn, length)
                obj = pickle.loads(buf)
                self._q.put(obj)
            except Exception:
                # Drop malformed messages
                return

    def recv(self, timeout: Optional[float] = None) -> Any:
        return self._q.get(timeout=timeout)

    def close(self) -> None:
        self._stop.set()
        try:
            self._server.close()
        except Exception:
            pass
        self._thread.join(timeout=1.0)
