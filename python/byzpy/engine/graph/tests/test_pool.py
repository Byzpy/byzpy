from __future__ import annotations

import asyncio
import contextlib
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
from byzpy.engine.actor.backends.remote import start_actor_server
from byzpy.engine.actor.transports import ucx as ucx_transport
from byzpy.engine.actor.transports.ucx import _ucx_mod
from byzpy.engine.graph import pool as pool_mod
from byzpy.engine.graph.pool import ActorPool, ActorPoolConfig, _infer_capabilities
from byzpy.engine.graph.subtask import SubTask


def _scale(value: int, *, factor: int) -> int:
    return value * factor


class _FakeBackend:
    def __init__(self):
        self.worker = None
        self.started = False
        self.closed = False

    async def start(self) -> None:
        self.started = True

    async def construct(self, cls_or_factory, *, args: tuple, kwargs: dict) -> None:
        self.worker = cls_or_factory(*args, **kwargs)

    async def call(self, method: str, *args, **kwargs):
        if not self.worker:
            raise RuntimeError("SubTask worker not constructed.")
        return getattr(self.worker, method)(*args, **kwargs)

    async def close(self) -> None:
        self.closed = True

    async def get_endpoint(self):
        raise NotImplementedError

    async def chan_open(self, name: str):
        raise NotImplementedError

    async def chan_put(self, **kwargs):
        raise NotImplementedError

    async def chan_get(self, **kwargs):
        raise NotImplementedError


@pytest.fixture(autouse=True)
def _patch_backend(monkeypatch, request):
    if request.node.get_closest_marker("real_actor_backends") is not None:
        # Tests can opt out of patching to exercise the real backends.
        return []

    created: list[_FakeBackend] = []

    def _factory(spec) -> _FakeBackend:
        backend = _FakeBackend()
        created.append(backend)
        return backend

    monkeypatch.setattr(pool_mod, "resolve_backend", _factory)
    return created


def test_infer_capabilities_from_string_and_backend():
    class _GpuBackend:
        pass

    assert _infer_capabilities("gpu") == ("gpu",)
    assert _infer_capabilities("ucx://host:1337") == ("gpu",)
    assert _infer_capabilities(GPUActorBackend()) == ("gpu",)
    assert _infer_capabilities(UCXRemoteActorBackend("0.0.0.0", 12345)) == ("gpu",)
    assert _infer_capabilities("thread") == ("cpu",)
    assert _infer_capabilities(_GpuBackend()) == ("cpu",)


def _have_cuda() -> bool:
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


def _have_cupy() -> bool:
    try:
        import cupy as _  # type: ignore

        return True
    except Exception:
        return False


def _have_ucx() -> bool:
    return _ucx_mod() is not None


CUDA_OK = _have_cuda()
CUPY_OK = _have_cupy()
UCX_OK = _have_ucx()
UCX_GPU_OK = CUDA_OK and CUPY_OK and UCX_OK


async def _wait_until_listening_tcp(host: str, port: int, timeout: float = 3.0) -> None:
    deadline = asyncio.get_event_loop().time() + timeout
    last_err = None
    while True:
        try:
            reader, writer = await asyncio.open_connection(host, port)
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()
            return
        except Exception as err:
            last_err = err
            if asyncio.get_event_loop().time() >= deadline:
                raise TimeoutError(f"TCP server {host}:{port} did not start: {last_err!r}")
            await asyncio.sleep(0.05)


async def _wait_until_listening_ucx(host: str, port: int, timeout: float = 5.0) -> None:
    if not UCX_OK:
        raise RuntimeError("UCX module not available")
    deadline = asyncio.get_event_loop().time() + timeout
    last_err = None
    while True:
        try:

            async def _probe(ep):
                await ucx_transport.send_control(
                    ep,
                    {
                        "op": "chan_get",
                        "name": "__probe__",
                        "timeout": 0.0,
                        "actor_id": "__probe__",
                    },
                )
                with contextlib.suppress(Exception):
                    await ucx_transport.recv_control(ep)

            await ucx_transport.call(host, int(port), _probe)
            return
        except Exception as err:
            last_err = err
            if asyncio.get_event_loop().time() >= deadline:
                raise TimeoutError(f"UCX server {host}:{port} did not start: {last_err!r}")
            await asyncio.sleep(0.05)


@pytest_asyncio.fixture(scope="module")
async def tcp_server_addr():
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
    await _wait_until_listening_ucx(host, port)

    try:
        yield host, port
    finally:
        stop_evt.set()
        th.join(timeout=3.0)


def test_actor_pool_config_with_explicit_capabilities():
    cfg = ActorPoolConfig(backend="thread", count=2, capabilities=("gpu",), name="trainer")
    assert cfg.resolved_capabilities() == ("gpu",)


def _pool_backend_spec(
    kind: str,
    tcp_addr: Tuple[str, int] | None = None,
    ucx_addr: Tuple[str, int] | None = None,
) -> str:
    if kind == "thread":
        return "thread"
    if kind == "process":
        return "process"
    if kind == "remote":
        if tcp_addr is None:
            raise RuntimeError("TCP server address required for remote backend.")
        host, port = tcp_addr
        return f"tcp://{host}:{port}"
    if kind == "gpu":
        if not CUDA_OK:
            pytest.skip("CUDA not available")
        return "gpu"
    if kind == "ucx":
        if not UCX_GPU_OK:
            pytest.skip("UCX/CUDA/CuPy not available")
        if ucx_addr is None:
            raise RuntimeError("UCX server address required for UCX backend.")
        host, port = ucx_addr
        return f"ucx://{host}:{port}"
    raise RuntimeError(kind)


_POOL_CPU_MATRIX = [
    ("thread", "thread"),
    ("thread", "process"),
    ("process", "thread"),
    ("process", "process"),
    ("thread", "remote"),
    ("remote", "thread"),
    ("process", "remote"),
    ("remote", "process"),
    ("remote", "remote"),
]

_POOL_GPU_MATRIX = [
    ("gpu", "thread"),
    ("thread", "gpu"),
    ("gpu", "process"),
    ("process", "gpu"),
    ("gpu", "remote"),
    ("remote", "gpu"),
    ("gpu", "gpu"),
]

_POOL_UCX_MATRIX = [
    ("gpu", "ucx"),
    ("ucx", "gpu"),
    ("ucx", "ucx"),
]


@pytest.mark.asyncio
async def test_actor_pool_run_many_executes_subtasks(_patch_backend):
    pool = ActorPool([ActorPoolConfig(backend="thread", count=2)])

    subtasks = [
        SubTask(fn=_scale, args=(2,), kwargs={"factor": 3}),
        SubTask(fn=_scale, args=(5,), kwargs={"factor": 2}),
    ]

    results = await pool.run_many(subtasks)

    assert results == [6, 10]
    assert pool.size == 2

    await pool.shutdown()
    assert all(backend.closed for backend in _patch_backend)


@pytest.mark.asyncio
async def test_actor_pool_run_many_handles_empty_subtask_list():
    pool = ActorPool([ActorPoolConfig(backend="thread", count=1)])

    results = await pool.run_many([])

    assert results == []
    assert pool.size == 1  # Start still materializes workers.

    await pool.shutdown()


@pytest.mark.asyncio
async def test_actor_pool_acquire_requires_matching_affinity():
    pool = ActorPool([ActorPoolConfig(backend="thread", count=1, capabilities=("cpu",))])
    await pool.start()

    with pytest.raises(RuntimeError, match="No actor in the pool"):
        await pool._acquire("gpu")  # type: ignore[attr-defined]

    await pool.shutdown()


@pytest.mark.asyncio
async def test_actor_pool_acquire_returns_worker_with_matching_affinity():
    pool = ActorPool(
        [
            ActorPoolConfig(backend="thread", count=1, capabilities=("cpu",)),
            ActorPoolConfig(backend="gpu", count=1, capabilities=("gpu",)),
        ]
    )
    await pool.start()

    gpu_worker = await pool._acquire("gpu")  # type: ignore[attr-defined]
    assert "gpu" in gpu_worker.capabilities

    pool._available.put_nowait(gpu_worker)

    cpu_worker = await pool._acquire("cpu")  # type: ignore[attr-defined]
    assert "cpu" in cpu_worker.capabilities

    await pool.shutdown()


@pytest.mark.asyncio
@pytest.mark.real_actor_backends
async def test_actor_pool_channel_cross_worker_communication():
    pool = ActorPool(
        [
            ActorPoolConfig(backend="thread", count=2, name="worker"),
        ]
    )

    channel = await pool.open_channel("pool_comm")
    workers = list(channel.workers)

    assert len(workers) == 2
    sender, recipient = workers[0], workers[1]

    # Ensure endpoint lookup succeeds for each worker.
    ep_sender = channel.endpoint(sender)
    ep_recipient = channel.endpoint(recipient)
    assert ep_sender.actor_id != ep_recipient.actor_id

    recv_task = asyncio.create_task(channel.recv(recipient, timeout=1.0))
    await asyncio.sleep(0)

    payload = {"hello": "world"}
    await channel.send(sender, recipient, payload)

    received = await recv_task
    assert received == payload

    await pool.shutdown()


@pytest.mark.asyncio
@pytest.mark.real_actor_backends
async def test_actor_pools_can_exchange_messages_via_channels():
    pool_a = ActorPool(
        [
            ActorPoolConfig(backend="thread", count=1, name="left"),
        ]
    )
    pool_b = ActorPool(
        [
            ActorPoolConfig(backend="thread", count=1, name="right"),
        ]
    )

    try:
        channel_a = await pool_a.open_channel("cross_pool")
        channel_b = await pool_b.open_channel("cross_pool")

        worker_a = channel_a.workers[0]
        worker_b = channel_b.workers[0]

        endpoint_a = channel_a.endpoint(worker_a)
        endpoint_b = channel_b.endpoint(worker_b)

        recv_from_b = asyncio.create_task(channel_a.recv(worker_a, timeout=1.0))
        recv_from_a = asyncio.create_task(channel_b.recv(worker_b, timeout=1.0))
        await asyncio.sleep(0)

        payload_from_b = {"origin": "pool_b"}
        payload_from_a = {"origin": "pool_a"}

        await channel_b.channel(worker_b).send(endpoint_a, payload_from_b)
        await channel_a.channel(worker_a).send(endpoint_b, payload_from_a)

        got_from_b = await recv_from_b
        got_from_a = await recv_from_a

        assert got_from_b == payload_from_b
        assert got_from_a == payload_from_a
    finally:
        await pool_a.shutdown()
        await pool_b.shutdown()


@pytest.mark.asyncio
@pytest.mark.real_actor_backends
@pytest.mark.parametrize("kind_a,kind_b", _POOL_CPU_MATRIX)
async def test_actor_pools_cross_backend_matrix_cpu(kind_a, kind_b, tcp_server_addr):
    channel_name = f"pool_cpu_{kind_a}_{kind_b}_{asyncio.get_running_loop().time()}"
    pool_a = ActorPool(
        [
            ActorPoolConfig(
                backend=_pool_backend_spec(kind_a, tcp_addr=tcp_server_addr),
                count=1,
                name=f"A-{kind_a}",
            )
        ]
    )
    pool_b = ActorPool(
        [
            ActorPoolConfig(
                backend=_pool_backend_spec(kind_b, tcp_addr=tcp_server_addr),
                count=1,
                name=f"B-{kind_b}",
            )
        ]
    )

    try:
        chan_a = await pool_a.open_channel(channel_name)
        chan_b = await pool_b.open_channel(channel_name)

        worker_a = chan_a.workers[0]
        worker_b = chan_b.workers[0]

        ch_a = chan_a.channel(worker_a)
        ch_b = chan_b.channel(worker_b)

        ep_a = chan_a.endpoint(worker_a)
        ep_b = chan_b.endpoint(worker_b)

        payload_ab = torch.arange(8, dtype=torch.float32)
        await ch_a.send(ep_b, payload_ab)
        got_b = await chan_b.recv(worker_b, timeout=5.0)
        assert isinstance(got_b, torch.Tensor)
        assert torch.equal(got_b, payload_ab)

        payload_ba = torch.arange(4, dtype=torch.float32) + 10.0
        await ch_b.send(ep_a, payload_ba)
        got_a = await chan_a.recv(worker_a, timeout=5.0)
        assert isinstance(got_a, torch.Tensor)
        assert torch.equal(got_a, payload_ba)
    finally:
        await pool_a.shutdown()
        await pool_b.shutdown()


@pytest.mark.asyncio
@pytest.mark.real_actor_backends
@pytest.mark.parametrize("kind_a,kind_b", _POOL_GPU_MATRIX)
async def test_actor_pools_cross_backend_matrix_gpu(kind_a, kind_b, tcp_server_addr):
    if not CUDA_OK:
        pytest.skip("CUDA not available")

    channel_name = f"pool_gpu_{kind_a}_{kind_b}_{asyncio.get_running_loop().time()}"
    pool_a = ActorPool(
        [
            ActorPoolConfig(
                backend=_pool_backend_spec(kind_a, tcp_addr=tcp_server_addr),
                count=1,
                name=f"A-{kind_a}",
            )
        ]
    )
    pool_b = ActorPool(
        [
            ActorPoolConfig(
                backend=_pool_backend_spec(kind_b, tcp_addr=tcp_server_addr),
                count=1,
                name=f"B-{kind_b}",
            )
        ]
    )

    try:
        chan_a = await pool_a.open_channel(channel_name)
        chan_b = await pool_b.open_channel(channel_name)

        worker_a = chan_a.workers[0]
        worker_b = chan_b.workers[0]

        ch_a = chan_a.channel(worker_a)
        ch_b = chan_b.channel(worker_b)

        ep_a = chan_a.endpoint(worker_a)
        ep_b = chan_b.endpoint(worker_b)

        payload_ab = torch.arange(6, dtype=torch.float32)
        await ch_a.send(ep_b, payload_ab)
        got_b = await chan_b.recv(worker_b, timeout=5.0)
        assert isinstance(got_b, torch.Tensor)
        assert torch.equal(got_b, payload_ab)

        payload_ba = torch.arange(3, dtype=torch.float32) + 20.0
        await ch_b.send(ep_a, payload_ba)
        got_a = await chan_a.recv(worker_a, timeout=5.0)
        assert isinstance(got_a, torch.Tensor)
        assert torch.equal(got_a, payload_ba)
    finally:
        await pool_a.shutdown()
        await pool_b.shutdown()


@pytest.mark.asyncio
@pytest.mark.real_actor_backends
@pytest.mark.parametrize("kind_a,kind_b", _POOL_UCX_MATRIX)
@pytest.mark.skipif(not UCX_GPU_OK, reason="UCX/CUDA/CuPy not available")
async def test_actor_pools_cross_backend_matrix_ucx(
    kind_a, kind_b, tcp_server_addr, ucx_server_addr
):
    channel_name = f"pool_ucx_{kind_a}_{kind_b}_{asyncio.get_running_loop().time()}"
    pool_a = ActorPool(
        [
            ActorPoolConfig(
                backend=_pool_backend_spec(
                    kind_a, tcp_addr=tcp_server_addr, ucx_addr=ucx_server_addr
                ),
                count=1,
                name=f"A-{kind_a}",
            )
        ]
    )
    pool_b = ActorPool(
        [
            ActorPoolConfig(
                backend=_pool_backend_spec(
                    kind_b, tcp_addr=tcp_server_addr, ucx_addr=ucx_server_addr
                ),
                count=1,
                name=f"B-{kind_b}",
            )
        ]
    )

    try:
        chan_a = await pool_a.open_channel(channel_name)
        chan_b = await pool_b.open_channel(channel_name)

        worker_a = chan_a.workers[0]
        worker_b = chan_b.workers[0]

        ch_a = chan_a.channel(worker_a)
        ch_b = chan_b.channel(worker_b)

        ep_a = chan_a.endpoint(worker_a)
        ep_b = chan_b.endpoint(worker_b)

        payload_ab = torch.arange(5, dtype=torch.float32, device="cuda")
        await ch_a.send(ep_b, payload_ab)
        got_b = await chan_b.recv(worker_b, timeout=5.0)
        assert isinstance(got_b, torch.Tensor)
        assert torch.equal(got_b.to("cuda"), payload_ab)

        payload_ba = torch.arange(7, dtype=torch.float32, device="cuda") + 30.0
        await ch_b.send(ep_a, payload_ba)
        got_a = await chan_a.recv(worker_a, timeout=5.0)
        assert isinstance(got_a, torch.Tensor)
        assert torch.equal(got_a.to("cuda"), payload_ba)
    finally:
        await pool_a.shutdown()
        await pool_b.shutdown()
