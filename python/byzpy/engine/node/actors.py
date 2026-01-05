from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Type, Union

from ..actor.backends.gpu import GPUActorBackend, UCXRemoteActorBackend
from ..actor.backends.process import ProcessActorBackend
from ..actor.backends.remote import RemoteActorBackend
from ..actor.backends.thread import ThreadActorBackend
from ..actor.base import ActorBackend, ActorRef
from .base import ByzantineNode, HonestNode


def _choose_backend(backend: Union[str, ActorBackend]) -> ActorBackend:
    """
    Accepts:
      - "thread"  -> ThreadActorBackend()
      - "process" -> ProcessActorBackend()
      - "tcp://HOST:PORT" -> RemoteActorBackend(HOST, PORT)
      - ActorBackend instance -> returned as-is
    """
    if isinstance(backend, str):
        if backend == "thread":
            return ThreadActorBackend()
        if backend == "process":
            return ProcessActorBackend()
        if backend == "gpu":
            return GPUActorBackend()
        if backend.startswith("tcp://"):
            hostport = backend[len("tcp://") :]
            host, port_str = hostport.rsplit(":", 1)
            return RemoteActorBackend(host, int(port_str))
        if backend.startswith("ucx://"):
            hostport = backend[len("ucx://") :]
            host, port_str = hostport.rsplit(":", 1)
            return UCXRemoteActorBackend(host, int(port_str))
        raise ValueError(f"Unknown backend spec: {backend!r}")
    # assume it's already an ActorBackend
    return backend


class NodeActor:
    def __init__(self, ref: ActorRef) -> None:
        self._ref = ref

    def __getattr__(self, name: str):
        """Delegate attribute access to the underlying reference."""
        return getattr(self._ref, name)


class HonestNodeActor(NodeActor):
    @classmethod
    async def spawn(
        cls,
        node_cls: Type[HonestNode] | Any,
        *,
        backend: ActorBackend,
        args: Tuple[Any, ...] = (),
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> "HonestNodeActor":
        """
        node_cls is shipped **by value** (cloudpickle) to the chosen backend.
        """
        if kwargs is None:
            kwargs = {}
        be = backend
        ref = ActorRef(be)
        await ref._backend.start()
        await ref._backend.construct(node_cls, args=args, kwargs=kwargs)
        return cls(ref)


class ByzantineNodeActor(NodeActor):
    @classmethod
    async def spawn(
        cls,
        node_cls: Type[ByzantineNode] | Any,
        *,
        backend: Union[str, ActorBackend] = "process",
        args: Tuple[Any, ...] = (),
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> "ByzantineNodeActor":
        """
        node_cls is shipped **by value** (cloudpickle) to the chosen backend.
        """
        if kwargs is None:
            kwargs = {}
        be = _choose_backend(backend)
        ref = ActorRef(be)
        await ref._backend.start()
        await ref._backend.construct(node_cls, args=args, kwargs=kwargs)
        return cls(ref)
