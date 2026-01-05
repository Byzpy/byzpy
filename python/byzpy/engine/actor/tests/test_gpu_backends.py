from __future__ import annotations

import asyncio
import contextlib
import math
import socket
import threading
from typing import Tuple

import pytest
import pytest_asyncio
import torch

from byzpy.engine.actor.backends.gpu import (
    GPUActorBackend,
    UCXRemoteActorBackend,
    start_ucx_actor_server,
)
from byzpy.engine.actor.backends.process import ProcessActorBackend
from byzpy.engine.actor.backends.remote import RemoteActorBackend, start_actor_server
from byzpy.engine.actor.backends.thread import ThreadActorBackend
from byzpy.engine.actor.base import ActorRef
from byzpy.engine.actor.channels import open_channel
from byzpy.engine.actor.transports.ucx import _ucx_mod


def _have_cuda() -> bool:
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


def _have_cupy() -> bool:
    try:
        import cupy as _

        return True
    except Exception:
        return False


def _have_ucx() -> bool:
    return _ucx_mod() is not None


CUDA_OK = _have_cuda()
CUPY_OK = _have_cupy()
UCX_OK = _have_ucx()
UCX_GPU_OK = UCX_OK and CUDA_OK and CUPY_OK


class Worker:
    def __init__(self, bias: float = 0.0):
        self.bias = float(bias)

    def make(self, n: int) -> torch.Tensor:
        return torch.arange(n, dtype=torch.float32).contiguous()

    def score(self, x: torch.Tensor) -> float:
        return float((x * x).sum().item() + self.bias)


async def _wait_until_listening_tcp(host: str, port: int, timeout: float = 3.0) -> None:
    """Poll-connect until the TCP server accepts, or timeout."""
    deadline = asyncio.get_event_loop().time() + timeout
    last_err = None
    while True:
        try:
            r, w = await asyncio.open_connection(host, port)
            w.close()
            with contextlib.suppress(Exception):
                await w.wait_closed()
            return
        except Exception as e:
            last_err = e
            if asyncio.get_event_loop().time() >= deadline:
                raise TimeoutError(f"TCP server {host}:{port} did not start: {last_err!r}")
            await asyncio.sleep(0.05)


async def _wait_until_listening_ucx(host: str, port: int, timeout: float = 5.0) -> None:
    """Poll-create UCX endpoint until the UCX server accepts, or timeout."""
    m = _ucx_mod()
    if m is None:
        raise RuntimeError("UCX module not available")
    deadline = asyncio.get_event_loop().time() + timeout
    last_err = None
    while True:
        try:
            ep = await m.create_endpoint(host, int(port))
            with contextlib.suppress(Exception):
                await ep.close()
            return
        except Exception as e:
            last_err = e
            if asyncio.get_event_loop().time() >= deadline:
                raise TimeoutError(f"UCX server {host}:{port} did not start: {last_err!r}")
            await asyncio.sleep(0.05)


@pytest_asyncio.fixture(scope="module")
async def tcp_server_addr():
    """
    Run RemoteActorServer on a dedicated event loop in a background thread.
    """
    host = "127.0.0.1"
    with socket.socket() as s:
        s.bind((host, 0))
        port = s.getsockname()[1]

    started_evt = threading.Event()
    stop_evt = threading.Event()

    def _server_thread():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def _runner():
            started_evt.set()
            await start_actor_server(host, port)

        task = loop.create_task(_runner())

        def _stop():
            if not task.done():
                task.cancel()

        def _watch_stop():
            stop_evt.wait()
            loop.call_soon_threadsafe(_stop)
            loop.call_soon_threadsafe(loop.stop)

        watcher = threading.Thread(target=_watch_stop, daemon=True)
        watcher.start()

        try:
            loop.run_forever()
        finally:
            with contextlib.suppress(asyncio.CancelledError):
                loop.run_until_complete(task)
            loop.close()

    th = threading.Thread(target=_server_thread, daemon=True)
    th.start()

    started_evt.wait(timeout=2.0)
    await _wait_until_listening_tcp(host, port)

    try:
        yield host, port
    finally:
        stop_evt.set()
        th.join(timeout=3.0)


@pytest_asyncio.fixture(scope="module")
async def ucx_server_addr():
    """
    Run UCXRemoteActorServer on a dedicated event loop in a background thread.
    Skipped entirely if UCX/CUDA/CuPy are not all available.
    """
    if not UCX_GPU_OK:
        pytest.skip("UCX/CUDA/CuPy not available")

    host = "127.0.0.1"
    with socket.socket() as s:
        s.bind((host, 0))
        port = s.getsockname()[1]

    started_evt = threading.Event()
    stop_evt = threading.Event()

    def _server_thread():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def _runner():
            started_evt.set()
            await start_ucx_actor_server(host=host, port=port)

        task = loop.create_task(_runner())

        def _stop():
            if not task.done():
                task.cancel()

        def _watch_stop():
            stop_evt.wait()
            loop.call_soon_threadsafe(_stop)
            loop.call_soon_threadsafe(loop.stop)

        watcher = threading.Thread(target=_watch_stop, daemon=True)
        watcher.start()

        try:
            loop.run_forever()
        finally:
            with contextlib.suppress(asyncio.CancelledError):
                loop.run_until_complete(task)
            loop.close()

    th = threading.Thread(target=_server_thread, daemon=True)
    th.start()

    started_evt.wait(timeout=2.0)
    # actively probe UCX server
    await _wait_until_listening_ucx(host, port)

    try:
        yield host, port
    finally:
        stop_evt.set()
        th.join(timeout=3.0)


async def _make_backend(kind: str, tcp_addr: Tuple[str, int], ucx_addr: Tuple[str, int] | None):
    if kind == "thread":
        return ThreadActorBackend()
    if kind == "process":
        return ProcessActorBackend()
    if kind == "remote":
        host, port = tcp_addr
        return RemoteActorBackend(host, port)
    if kind == "gpu":
        return GPUActorBackend()
    if kind == "ucx":
        if not UCX_GPU_OK:
            pytest.skip("UCX/CUDA/CuPy not available")
        assert ucx_addr is not None
        host, port = ucx_addr
        return UCXRemoteActorBackend(host, port)
    raise RuntimeError(kind)


@pytest.mark.asyncio
async def test_gpu_basic_make_and_score_cpu_tensor():
    A = GPUActorBackend()
    await A.start()
    await A.construct(Worker, args=(), kwargs={"bias": 2.0})
    try:
        t = await A.call("make", 4)
        s = await A.call("score", t)
        assert isinstance(t, torch.Tensor) and t.shape == (4,)
        assert math.isfinite(s)
    finally:
        await A.close()


@pytest.mark.asyncio
async def test_gpu_local_channel_roundtrip_cpu_tensor():
    A = GPUActorBackend()
    await A.start()
    await A.construct(Worker, args=(), kwargs={})
    try:
        epA = await A.get_endpoint()
        chA = await open_channel(A, "gpu_local")
        payload = await A.call("make", 5)
        await chA.send(epA, payload)
        got = await chA.recv(timeout=1.0)
        assert torch.equal(got, torch.arange(5, dtype=torch.float32))
    finally:
        await A.close()


_GPU_MATRIX = [
    ("gpu", "thread"),
    ("thread", "gpu"),
    ("gpu", "process"),
    ("process", "gpu"),
    ("gpu", "remote"),
    ("remote", "gpu"),
    ("gpu", "gpu"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("kindA,kindB", _GPU_MATRIX)
async def test_gpu_cross_backend_matrix_cpu(kindA, kindB, tcp_server_addr, ucx_server_addr=None):
    A = await _make_backend(kindA, tcp_server_addr, ucx_server_addr)
    B = await _make_backend(kindB, tcp_server_addr, ucx_server_addr)

    await A.start()
    await B.start()
    await A.construct(Worker, args=(), kwargs={})
    await B.construct(Worker, args=(), kwargs={})
    try:
        ch_name = f"mb_gpu_cpu_{kindA}_{kindB}_{id(A)}_{id(B)}"
        chA = await open_channel(A, ch_name)
        chB = await open_channel(B, ch_name)

        epB = await B.get_endpoint()
        payload = await A.call("make", 7)
        await chA.send(epB, payload)
        got = await chB.recv(timeout=1.0)

        assert isinstance(got, torch.Tensor)
        assert torch.equal(got, torch.arange(7, dtype=torch.float32))
    finally:
        await A.close()
        await B.close()


_UCX_MATRIX = [
    ("gpu", "ucx"),
    ("ucx", "gpu"),
    ("ucx", "ucx"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("kindA,kindB", _UCX_MATRIX)
@pytest.mark.skipif(not UCX_GPU_OK, reason="UCX/CUDA/CuPy not available")
async def test_gpu_ucx_cross_backend_cuda(kindA, kindB, tcp_server_addr, ucx_server_addr):
    """
    UCX path with CUDA tensors: exercises GPU→UCX and UCX→GPU, plus UCX→UCX.
    """
    A = await _make_backend(kindA, tcp_server_addr, ucx_server_addr)
    B = await _make_backend(kindB, tcp_server_addr, ucx_server_addr)

    await A.start()
    await B.start()
    await A.construct(Worker, args=(), kwargs={})
    await B.construct(Worker, args=(), kwargs={})

    try:
        ch_name = f"mb_ucx_cuda_{kindA}_{kindB}_{id(A)}_{id(B)}"
        chA = await open_channel(A, ch_name)
        chB = await open_channel(B, ch_name)

        epB = await B.get_endpoint()

        payload = torch.arange(9, dtype=torch.float32, device="cuda")
        await chA.send(epB, payload)

        got = await chB.recv(timeout=2.0)
        assert isinstance(got, torch.Tensor)
        assert torch.equal(got.to("cpu"), torch.arange(9, dtype=torch.float32))
    finally:
        await A.close()
        await B.close()


@pytest.mark.asyncio
async def test_gpu_actorref_local_loopback():
    A = GPUActorBackend()
    ref = ActorRef(A)

    async with ref:
        await ref._backend.start()
        await ref._backend.construct(Worker, args=(), kwargs={"bias": 3.0})
        t = await ref.make(4)
        s = await ref.score(t)
        assert isinstance(s, float) and s > 0.0

        ch = await ref.open_channel("gpu_ref")
        ep = await ref.endpoint()
        await ch.send(ep, torch.tensor([1.0, 2.0, 3.0]))
        got = await ch.recv(timeout=1.0)
        assert torch.equal(got, torch.tensor([1.0, 2.0, 3.0]))


@pytest.mark.asyncio
@pytest.mark.skipif(not UCX_GPU_OK, reason="UCX/CUDA/CuPy not available")
async def test_gpu_actorref_ucx_roundtrip(ucx_server_addr, tcp_server_addr):
    """
    ActorRef sugar across GPU <-> UCX using a CUDA tensor over UCX.
    """
    ga = GPUActorBackend()
    ub = UCXRemoteActorBackend(*ucx_server_addr)
    refA = ActorRef(ga)
    refB = ActorRef(ub)

    async with refA, refB:
        await refA._backend.start()
        await refB._backend.start()
        await refA._backend.construct(Worker, args=(), kwargs={})
        await refB._backend.construct(Worker, args=(), kwargs={})

        ch_name = f"gpu_ucx_ref_{id(ga)}_{id(ub)}"
        chA = await refA.open_channel(ch_name)
        chB = await refB.open_channel(ch_name)

        epB = await refB.endpoint()
        payload = torch.arange(11, dtype=torch.float32, device="cuda")
        await chA.send(epB, payload)

        got = await chB.recv(timeout=2.0)
        assert isinstance(got, torch.Tensor)
        assert torch.equal(got.to("cpu"), torch.arange(11, dtype=torch.float32))
