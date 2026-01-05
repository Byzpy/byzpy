from __future__ import annotations

from ..engine.actor.backends.gpu import GPUActorBackend, UCXRemoteActorBackend
from ..engine.actor.backends.process import ProcessActorBackend
from ..engine.actor.backends.remote import RemoteActorBackend
from ..engine.actor.backends.thread import ThreadActorBackend
from ..engine.actor.base import ActorBackend


def set_actor(spec: str) -> ActorBackend:
    """
    "thread"  -> ThreadActorBackend()
    "process" -> ProcessActorBackend()
    "tcp://host:port" -> RemoteActorBackend(host, port)
    """
    if spec == "thread":
        return ThreadActorBackend()
    if spec == "process":
        return ProcessActorBackend()
    if spec == "gpu":
        return GPUActorBackend()
    if spec.startswith("tcp://"):
        hostport = spec[len("tcp://") :]
        host, port = hostport.rsplit(":", 1)
        return RemoteActorBackend(host, int(port))
    if spec.startswith("ucx://"):
        hostport = spec[len("ucx://") :]
        host, port_str = hostport.rsplit(":", 1)
        return UCXRemoteActorBackend(host, int(port_str))
    raise ValueError(f"Unknown backend spec: {spec!r}")
