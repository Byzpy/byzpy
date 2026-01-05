"""Per-node runner for decentralized scheduling experiments.

This module provides a small, process-backed runner that executes a user
supplied step function and message handler in its own OS process. It is
intended as a minimal building block toward fully decentralized nodes where
each node advances based on local state and incoming messages.

The runner is intentionally simple (blocking loop with periodic message/command
polling) to avoid entangling with the existing event-loop orchestration. It can
be extended later to host a NodeScheduler or richer control logic.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import cloudpickle

# --------------------------- Runner process loop ---------------------------


@dataclass
class _Command:
    op: str
    payload: Any = None


def _runner_main(cmd_q: mp.Queue, inbox_q: mp.Queue, result_q: mp.Queue, blob: bytes) -> None:
    step_fn, msg_handler, init_state = cloudpickle.loads(blob)
    state = init_state if init_state is not None else {}
    auto = False
    auto_interval = 0.0
    last_auto = time.time()

    def _step() -> None:
        nonlocal state
        state = step_fn(state)

    def _handle_msg(msg: Any) -> None:
        nonlocal state
        state = msg_handler(state, msg)

    running = True
    while running:
        now = time.time()
        # Handle auto step if enabled
        if auto and (now - last_auto) >= auto_interval:
            _step()
            last_auto = now

        # Drain inbox quickly
        try:
            while True:
                msg = inbox_q.get_nowait()
                _handle_msg(msg)
        except queue.Empty:
            pass

        # Handle control commands
        try:
            cmd: _Command = cmd_q.get(timeout=0.01)
        except queue.Empty:
            continue

        if cmd.op == "stop":
            running = False
            result_q.put(("stopped", None))
        elif cmd.op == "step":
            _step()
            result_q.put(("step", None))
        elif cmd.op == "start_auto":
            auto = True
            auto_interval = float(cmd.payload)
            last_auto = time.time()
            result_q.put(("start_auto", auto_interval))
        elif cmd.op == "stop_auto":
            auto = False
            result_q.put(("stop_auto", None))
        elif cmd.op == "state":
            result_q.put(("state", state))
        else:
            result_q.put(("error", f"unknown op {cmd.op}"))


# --------------------------- Public API ---------------------------


class NodeRunner:
    """Run a node loop in its own process with message passing.

    step_fn: Callable[[state], state]
    msg_handler: Callable[[state, msg], state]
    """

    def __init__(
        self,
        step_fn: Callable[[dict], dict],
        msg_handler: Callable[[dict, Any], dict],
        *,
        init_state: Optional[dict] = None,
    ) -> None:
        self._cmd_q: mp.Queue = mp.Queue()
        self._inbox_q: mp.Queue = mp.Queue()
        self._result_q: mp.Queue = mp.Queue()
        payload = cloudpickle.dumps((step_fn, msg_handler, init_state))
        self._proc = mp.Process(
            target=_runner_main,
            args=(self._cmd_q, self._inbox_q, self._result_q, payload),
        )
        self._pump_thread: threading.Thread | None = None
        self._pump_stop = threading.Event()

    def start(self) -> None:
        self._proc.start()

    def stop(self) -> None:
        self._cmd_q.put(_Command("stop"))
        self._result_q.get(timeout=5)
        self._proc.join(timeout=5)
        self._stop_pump_thread()

    def step(self) -> None:
        self._cmd_q.put(_Command("step"))
        self._result_q.get(timeout=5)

    def start_auto(self, interval_sec: float) -> None:
        self._cmd_q.put(_Command("start_auto", payload=interval_sec))
        self._result_q.get(timeout=5)

    def stop_auto(self) -> None:
        self._cmd_q.put(_Command("stop_auto"))
        self._result_q.get(timeout=5)

    def send_message(self, msg: Any) -> None:
        self._inbox_q.put(msg)

    def state(self) -> dict:
        self._cmd_q.put(_Command("state"))
        _, st = self._result_q.get(timeout=5)
        return st

    def pump_once(self) -> None:
        """Process inbox and any pending commands once (non-blocking)."""
        # trigger a step; runner loop will drain inbox first
        self._cmd_q.put(_Command("step"))
        try:
            self._result_q.get(timeout=1.0)
        except queue.Empty:
            pass

    def start_async(self, interval_sec: float = 0.01) -> None:
        """Continuously pump step commands on a background thread."""
        if self._pump_thread and self._pump_thread.is_alive():
            return
        self._pump_stop.clear()

        def _loop():
            while not self._pump_stop.is_set():
                self.pump_once()
                time.sleep(interval_sec)

        self._pump_thread = threading.Thread(target=_loop, daemon=True)
        self._pump_thread.start()

    def _stop_pump_thread(self) -> None:
        self._pump_stop.set()
        if self._pump_thread and self._pump_thread.is_alive():
            self._pump_thread.join(timeout=1.0)
        self._pump_thread = None
