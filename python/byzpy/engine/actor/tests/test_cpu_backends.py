import asyncio
import contextlib
import math
import socket
import threading

import pytest
import pytest_asyncio
import torch

from byzpy.engine.actor.backends.process import ProcessActorBackend
from byzpy.engine.actor.backends.remote import RemoteActorBackend, start_actor_server
from byzpy.engine.actor.backends.thread import ThreadActorBackend
from byzpy.engine.actor.base import ActorRef
from byzpy.engine.actor.channels import open_channel


class Worker:
    def __init__(self, bias: float = 0.0):
        self.bias = float(bias)

    def make(self, n: int) -> torch.Tensor:
        return torch.arange(n, dtype=torch.float32).contiguous()

    def add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x.contiguous() + y.contiguous()

    def score(self, x: torch.Tensor) -> float:
        return float((x * x).sum().item() + self.bias)


async def _wait_until_listening(host: str, port: int, timeout: float = 3.0) -> None:
    """Poll-connect until the server socket accepts, or timeout."""
    deadline = asyncio.get_event_loop().time() + timeout
    while True:
        try:
            r, w = await asyncio.open_connection(host, port)
            w.close()
            with contextlib.suppress(Exception):
                await w.wait_closed()
            return
        except Exception as e:
            if asyncio.get_event_loop().time() >= deadline:
                raise TimeoutError(f"Server {host}:{port} did not start: {e!r}")
            await asyncio.sleep(0.05)


@pytest_asyncio.fixture(scope="module")
async def tcp_server_addr():
    """
    Run the RemoteActorServer on a dedicated event loop in a background thread.
    This avoids any scheduling/race issues with the test loop.
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
            try:
                await start_actor_server(host, port)
            finally:
                pass

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
    await _wait_until_listening(host, port)

    try:
        yield host, port
    finally:
        stop_evt.set()
        th.join(timeout=3.0)


@pytest_asyncio.fixture(params=["thread", "process", "remote"])
async def backend_pair(request, tcp_server_addr):
    kind = request.param
    if kind == "thread":
        A = ThreadActorBackend()
        B = ThreadActorBackend()
    elif kind == "process":
        A = ProcessActorBackend()
        B = ProcessActorBackend()
    elif kind == "remote":
        host, port = tcp_server_addr
        A = RemoteActorBackend(host, port)
        B = RemoteActorBackend(host, port)
    else:
        raise RuntimeError(kind)

    await A.start()
    await B.start()
    await A.construct(Worker, args=(), kwargs={"bias": 3.0})
    await B.construct(Worker, args=(), kwargs={"bias": 5.0})

    try:
        yield A, B
    finally:
        await A.close()
        await B.close()


@pytest.mark.asyncio
async def test_basic_rpc_make_and_score(backend_pair):
    A, B = backend_pair
    tA = await A.call("make", 8)
    tB = await B.call("make", 8)
    assert isinstance(tA, torch.Tensor) and tA.shape == (8,)
    assert isinstance(tB, torch.Tensor) and tB.shape == (8,)
    sA = await A.call("score", tA)
    sB = await B.call("score", tB)
    ss = sum(i * i for i in range(8))
    assert math.isclose(sA, ss + 3.0, rel_tol=1e-6)
    assert math.isclose(sB, ss + 5.0, rel_tol=1e-6)


@pytest.mark.asyncio
async def test_channel_roundtrip_same_backend(backend_pair):
    A, B = backend_pair
    epB = await B.get_endpoint()
    chA = await open_channel(A, "data")
    chB = await open_channel(B, "data")

    payload = await A.call("make", 4)
    await chA.send(epB, payload)

    got = await chB.recv(timeout=1.0)
    assert isinstance(got, torch.Tensor)
    assert torch.equal(got, torch.tensor([0.0, 1.0, 2.0, 3.0]))


@pytest.mark.asyncio
async def test_channel_timeout_returns_none(backend_pair):
    _, B = backend_pair
    chB = await open_channel(B, "empty")
    out = await chB.recv(timeout=0.05)  # nobody sends
    assert out is None


@pytest.mark.asyncio
async def test_cross_backend_thread_to_process():
    A = ThreadActorBackend()
    B = ProcessActorBackend()
    await A.start()
    await B.start()
    await A.construct(Worker, args=(), kwargs={})
    await B.construct(Worker, args=(), kwargs={})
    try:
        epB = await B.get_endpoint()
        chA = await open_channel(A, "x")
        chB = await open_channel(B, "x")
        payload = await A.call("make", 5)
        await chA.send(epB, payload)
        got = await chB.recv(timeout=1.0)
        assert torch.equal(got, torch.arange(5, dtype=torch.float32))
    finally:
        await A.close()
        await B.close()


@pytest.mark.asyncio
async def test_cross_backend_process_to_thread():
    A = ProcessActorBackend()
    B = ThreadActorBackend()
    await A.start()
    await B.start()
    await A.construct(Worker, args=(), kwargs={})
    await B.construct(Worker, args=(), kwargs={})
    try:
        epB = await B.get_endpoint()
        chA = await open_channel(A, "y")
        chB = await open_channel(B, "y")
        t = await A.call("make", 6)
        await chA.send(epB, t)
        got = await chB.recv(timeout=1.0)
        assert torch.equal(got, torch.arange(6, dtype=torch.float32))
    finally:
        await A.close()
        await B.close()


@pytest.mark.asyncio
async def test_remote_backend_via_tcp_server(tcp_server_addr):
    host, port = tcp_server_addr
    A = RemoteActorBackend(host, port)
    B = RemoteActorBackend(host, port)
    await A.start()
    await B.start()
    await A.construct(Worker, args=(), kwargs={"bias": 1.5})
    await B.construct(Worker, args=(), kwargs={"bias": 2.5})

    try:
        t = await A.call("make", 3)
        s = await B.call("score", t)
        assert isinstance(s, float)

        ch_name = f"tcp_{id(A)}_{id(B)}"
        chA = await open_channel(A, ch_name)
        chB = await open_channel(B, ch_name)
        epB = await B.get_endpoint()

        await chA.send(epB, torch.tensor([10.0, 20.0]))
        got = await chB.recv(timeout=1.0)
        assert torch.equal(got, torch.tensor([10.0, 20.0]))
    finally:
        await A.close()
        await B.close()


@pytest.mark.asyncio
async def test_actor_ref_context_and_proxy(tcp_server_addr):
    host, port = tcp_server_addr
    ref = ActorRef(RemoteActorBackend(host, port))

    async with ref:
        await ref._backend.start()
        await ref._backend.construct(Worker, args=(), kwargs={"bias": 7.0})
        t = await ref.make(4)
        val = await ref.score(t)
        assert isinstance(val, float) and val > 0.0

        ch = await ref.open_channel("z_ref")
        ep = await ref.endpoint()
        await ch.send(ep, torch.tensor([1.0]))
        got = await ch.recv(timeout=1.0)
        assert torch.equal(got, torch.tensor([1.0]))


async def _make_backend(kind: str, tcp_server_addr):
    if kind == "thread":
        return ThreadActorBackend()
    if kind == "process":
        return ProcessActorBackend()
    if kind == "remote":
        host, port = tcp_server_addr
        return RemoteActorBackend(host, port)
    raise RuntimeError(kind)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kindA,kindB",
    [
        ("thread", "thread"),
        ("process", "process"),
        ("remote", "remote"),
        ("thread", "process"),
        ("process", "thread"),
        ("thread", "remote"),
        ("remote", "thread"),
        ("process", "remote"),
        ("remote", "process"),
    ],
)
async def test_cross_backend_matrix(kindA, kindB, tcp_server_addr):
    A = await _make_backend(kindA, tcp_server_addr)
    B = await _make_backend(kindB, tcp_server_addr)

    await A.start()
    await B.start()

    await A.construct(Worker, args=(), kwargs={})
    await B.construct(Worker, args=(), kwargs={})

    try:
        ch_name = f"mb_{kindA}_{kindB}_{id(A)}_{id(B)}"
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kindA,kindB",
    [
        ("thread", "thread"),
        ("process", "process"),
        ("remote", "remote"),
        ("thread", "process"),
        ("process", "thread"),
        ("thread", "remote"),
        ("remote", "thread"),
        ("process", "remote"),
        ("remote", "process"),
    ],
)
async def test_cross_backend_matrix_actorref(kindA, kindB, tcp_server_addr):
    A = await _make_backend(kindA, tcp_server_addr)
    B = await _make_backend(kindB, tcp_server_addr)

    refA = ActorRef(A)
    refB = ActorRef(B)

    async with refA, refB:
        await refA._backend.start()
        await refB._backend.start()
        await refA._backend.construct(Worker, args=(), kwargs={})
        await refB._backend.construct(Worker, args=(), kwargs={})

        ch_name = f"mb_ref_{kindA}_{kindB}_{id(A)}_{id(B)}"
        chA = await refA.open_channel(ch_name)
        chB = await refB.open_channel(ch_name)

        epB = await refB.endpoint()

        payload = await refA.make(7)
        await chA.send(epB, payload)

        got = await chB.recv(timeout=1.0)
        assert isinstance(got, torch.Tensor)
        assert torch.equal(got, torch.arange(7, dtype=torch.float32))
