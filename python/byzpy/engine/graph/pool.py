from __future__ import annotations

import asyncio
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    List,
    Mapping,
    MutableSequence,
    Optional,
    Sequence,
    Union,
)

import cloudpickle

from ..actor.backends.gpu import GPUActorBackend, UCXRemoteActorBackend
from ..actor.base import ActorBackend, ActorRef
from ..actor.channels import ChannelRef, Endpoint
from ..actor.factory import resolve_backend
from .subtask import SubTask


def _infer_capabilities(spec: Union[str, ActorBackend]) -> Sequence[str]:
    if isinstance(spec, str) and (spec == "gpu" or spec.startswith("ucx://")):
        return ("gpu",)
    if isinstance(spec, GPUActorBackend) or isinstance(spec, UCXRemoteActorBackend):
        return ("gpu",)
    return ("cpu",)


@dataclass(frozen=True)
class ActorPoolConfig:
    """
    Configuration for a pool of actors of the same type.

    This dataclass specifies how to create a group of actor workers with
    the same backend and capabilities.

    Parameters
    ----------
    backend : Union[str, ActorBackend]
        Backend specification. Can be a string ("thread", "process", "gpu",
        "tcp://host:port", "ucx://host:port") or an ActorBackend instance.
    count : int, optional
        Number of workers to create. Default is 1.
    capabilities : Sequence[str] | None, optional
        Explicit capability tags for these workers. If None, capabilities are
        inferred from the backend. Default is None.
    name : Optional[str], optional
        Optional name prefix for worker identification. Default is None.

    Examples
    --------
    >>> config = ActorPoolConfig(backend="thread", count=4, name="cpu-workers")
    >>> pool = ActorPool([config])
    >>> await pool.start()
    """

    backend: Union[str, ActorBackend]
    count: int = 1
    capabilities: Sequence[str] | None = None
    name: Optional[str] = None

    def resolved_capabilities(self) -> Sequence[str]:
        return (
            self.capabilities
            if self.capabilities is not None
            else _infer_capabilities(self.backend)
        )


class ActorPool:
    """
    Pool of actor workers for parallel task execution.

    This class manages a collection of actor workers from one or more backend
    configurations. It schedules subtasks across workers based on task
    affinity and worker capabilities, enabling efficient parallel execution
    of computation graph operators.

    The pool supports heterogeneous workers (e.g., mix of CPU and GPU workers)
    and automatically routes tasks to appropriate workers based on their
    declared capabilities.

    Parameters
    ----------
    configs : Sequence[ActorPoolConfig]
        Sequence of pool configurations. Each config creates a group of
        workers with the same backend and capabilities.

    Examples
    --------
    >>> from byzpy.engine.graph.pool import ActorPool, ActorPoolConfig
    >>> configs = [
    ...     ActorPoolConfig(backend="thread", count=4),
    ...     ActorPoolConfig(backend="gpu", count=2)
    ... ]
    >>> pool = ActorPool(configs)
    >>> await pool.start()
    >>> # Use pool with NodeScheduler...
    >>> await pool.shutdown()

    Notes
    -----
    - Workers are lazily started when :meth:`start` is called.
    - Tasks are scheduled based on SubTask affinity hints.
    - The pool supports channel-based communication between workers.
    """

    def __init__(self, configs: Sequence[ActorPoolConfig]) -> None:
        self.configs = list(configs)
        self._workers: List[_PoolWorker] = []
        self._available: asyncio.Queue[_PoolWorker] = asyncio.Queue()
        self._waiting: Dict[str | None, Deque[asyncio.Future[_PoolWorker]]] = defaultdict(deque)
        self._started = False
        self._channel_cache: Dict[str, ActorPoolChannel] = {}
        self._worker_affinity_caps: List[str] = []

    @property
    def size(self) -> int:
        return len(self._workers) if self._started else sum(cfg.count for cfg in self.configs)

    async def start(self) -> None:
        if self._started:
            return
        for cfg in self.configs:
            for idx in range(cfg.count):
                backend = resolve_backend(cfg.backend)
                affinity_cap = f"worker::{cfg.name or 'actor'}-{idx}"
                caps = set(cfg.resolved_capabilities())
                caps.add(affinity_cap)
                worker = _PoolWorker(
                    backend=backend,
                    capabilities=caps,
                    name=f"{cfg.name or 'actor'}-{idx}",
                )
                await worker.start()
                self._workers.append(worker)
                await self._release(worker)
                self._worker_affinity_caps.append(affinity_cap)
        self._started = True

    async def shutdown(self) -> None:
        for worker in self._workers:
            await worker.close()
        self._workers.clear()
        self._started = False
        self._channel_cache.clear()
        self._worker_affinity_caps.clear()
        while not self._available.empty():
            self._available.get_nowait()
        for waiters in self._waiting.values():
            while waiters:
                fut = waiters.popleft()
                if not fut.done():
                    fut.set_exception(RuntimeError("ActorPool shutdown"))
        self._waiting.clear()

    async def open_channel(self, name: str) -> "ActorPoolChannel":
        """
        Bind a channel with the given ``name`` on every worker in the pool and return
        a helper that exposes per-worker send/receive helpers. Reuses existing bindings
        when called multiple times for the same name.
        """
        await self.start()

        cached = self._channel_cache.get(name)
        if cached is not None:
            return cached

        channel_map: Dict[str, ChannelRef] = {}
        endpoint_map: Dict[str, Endpoint] = {}

        for worker in self._workers:
            channel_map[worker.name] = await worker.open_channel(name)
            endpoint_map[worker.name] = await worker.endpoint()

        pool_channel = ActorPoolChannel(
            name=name,
            channels=channel_map,
            endpoints=endpoint_map,
        )
        self._channel_cache[name] = pool_channel
        return pool_channel

    async def run_many(self, subtasks: Sequence[SubTask]) -> List[Any]:
        await self.start()
        if not subtasks:
            return []
        coros = [self._run_subtask(st) for st in subtasks]
        return await asyncio.gather(*coros)

    async def run_subtask(self, subtask: SubTask) -> Any:
        await self.start()
        return await self._run_subtask(subtask)

    async def _run_subtask(self, subtask: SubTask) -> Any:
        attempts = 0
        last_exc: Exception | None = None
        max_attempts = max(0, subtask.max_retries) + 1
        while attempts < max_attempts:
            worker = await self._acquire(subtask.affinity)
            try:
                return await worker.run(subtask)
            except Exception as exc:
                last_exc = exc
                attempts += 1
                if attempts >= max_attempts:
                    raise
            finally:
                await self._release(worker)
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Subtask failed without exception")  # pragma: no cover

    def worker_affinities(self) -> Sequence[str]:
        return tuple(self._worker_affinity_caps)

    async def _acquire(self, affinity: Optional[str]) -> "_PoolWorker":
        if not self._workers:
            raise RuntimeError("ActorPool has no workers configured.")
        if affinity is None:
            return await self._available.get()

        # fast path: try to grab a matching worker from the available queue without starving others
        rotations = 0
        size = self._available.qsize()
        while rotations < size:
            worker = await self._available.get()
            if affinity in worker.capabilities:
                return worker
            await self._available.put(worker)
            rotations += 1

        # Check if any worker in the pool has the requested affinity
        # before waiting
        has_affinity = any(affinity in worker.capabilities for worker in self._workers)
        if not has_affinity:
            raise RuntimeError("No actor in the pool")

        # no idle worker with that affinity; wait until one is released
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[_PoolWorker] = loop.create_future()
        self._waiting[affinity].append(fut)
        return await fut

    async def _release(self, worker: "_PoolWorker") -> None:
        served_waiter = False
        for cap in worker.capabilities:
            waiters = self._waiting.get(cap)
            while waiters:
                fut = waiters.popleft()
                if not fut.done():
                    fut.set_result(worker)
                    served_waiter = True
                    break
            if served_waiter:
                break
        if not served_waiter:
            waiters = self._waiting.get(None)
            while waiters:
                fut = waiters.popleft()
                if not fut.done():
                    fut.set_result(worker)
                    served_waiter = True
                    break
        if not served_waiter:
            await self._available.put(worker)


class _PoolWorker:
    """
    A single actor worker in a pool
    """

    def __init__(self, *, backend: ActorBackend, capabilities: set[str], name: str) -> None:
        self.backend = backend
        self.capabilities = frozenset(capabilities)
        self.name = name
        self._ref = ActorRef(self.backend)
        self._endpoint: Endpoint | None = None
        self._channels: Dict[str, ChannelRef] = {}
        self._fn_cache: "OrderedDict[Callable[..., Any], bytes]" = OrderedDict()
        self._fn_cache_limit = 64

    async def start(self) -> None:
        await self._ref._backend.start()
        await self._ref._backend.construct(_SubTaskWorker, args=(), kwargs={})

    async def endpoint(self) -> Endpoint:
        if self._endpoint is None:
            self._endpoint = await self._ref._backend.get_endpoint()
        return self._endpoint

    async def open_channel(self, name: str) -> ChannelRef:
        if name not in self._channels:
            ep = await self._ref._backend.chan_open(name)
            self._channels[name] = ChannelRef(self._ref._backend, ep, name)
        return self._channels[name]

    async def run(self, subtask: SubTask) -> Any:
        payload = self._serialized_fn(subtask.fn)
        args = tuple(subtask.args)
        kwargs = dict(subtask.kwargs)
        return await self._ref.execute(payload, args, kwargs)

    async def close(self) -> None:
        await self._ref._backend.close()

    def _serialized_fn(self, fn: Callable[..., Any]) -> bytes:
        try:
            payload = self._fn_cache.pop(fn)
            self._fn_cache[fn] = payload
            return payload
        except KeyError:
            payload = cloudpickle.dumps(fn)
            self._fn_cache[fn] = payload
            if len(self._fn_cache) > self._fn_cache_limit:
                self._fn_cache.popitem(last=False)
            return payload


class _SubTaskWorker:
    def execute(self, payload: bytes, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> Any:
        fn = cloudpickle.loads(payload)
        return fn(*args, **dict(kwargs))


class ActorPoolChannel:
    """
    Wrapper around a channel bound on each worker in an ActorPool. Provides
    convenient `send`/`recv` helpers keyed by worker name.
    """

    def __init__(
        self,
        *,
        name: str,
        channels: Mapping[str, ChannelRef],
        endpoints: Mapping[str, Endpoint],
    ) -> None:
        self.name = name
        self._channels = dict(channels)
        self._endpoints = dict(endpoints)

    @property
    def workers(self) -> Sequence[str]:
        return tuple(self._channels.keys())

    def channel(self, worker: str) -> ChannelRef:
        try:
            return self._channels[worker]
        except KeyError as exc:
            raise KeyError(f"No channel bound for worker {worker!r}") from exc

    def endpoint(self, worker: str) -> Endpoint:
        try:
            return self._endpoints[worker]
        except KeyError as exc:
            raise KeyError(f"No endpoint known for worker {worker!r}") from exc

    async def send(self, sender: str, recipient: str, payload: Any) -> None:
        ch = self.channel(sender)
        target = self.endpoint(recipient)
        await ch.send(target, payload)

    async def recv(self, worker: str, *, timeout: Optional[float] = None) -> Any:
        ch = self.channel(worker)
        return await ch.recv(timeout=timeout)
