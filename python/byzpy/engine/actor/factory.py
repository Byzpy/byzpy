"""Shared helpers for turning user specs into actor backend instances."""

from __future__ import annotations

from typing import Union

from .backends.gpu import GPUActorBackend, UCXRemoteActorBackend
from .backends.process import ProcessActorBackend
from .backends.remote import RemoteActorBackend
from .backends.thread import ThreadActorBackend
from .base import ActorBackend


def resolve_backend(spec: Union[str, ActorBackend]) -> ActorBackend:
    """
    Resolve an actor backend from a specification string or instance.

    This function converts backend specification strings into ActorBackend
    instances. It supports various backend types including threads, processes,
    GPUs, and remote actors.

    Parameters
    ----------
    spec : Union[str, ActorBackend]
        Backend specification. Can be:
        - "thread": Thread-based actors in the current process
        - "process": Process-based actors in separate processes
        - "gpu": GPU-based actors for CUDA execution
        - "tcp://host:port": Remote actors via TCP
        - "ucx://host:port": Remote actors via UCX (for GPU clusters)
        - An existing ActorBackend instance (returned as-is)

    Returns
    -------
    ActorBackend
        The resolved actor backend instance.

    Raises
    ------
    ValueError
        If the specification string is not recognized.

    Examples
    --------
    >>> backend = resolve_backend("thread")
    >>> isinstance(backend, ThreadActorBackend)
    True
    >>> backend = resolve_backend("tcp://localhost:29000")
    >>> isinstance(backend, RemoteActorBackend)
    True
    """

    if isinstance(spec, str):
        if spec == "thread":
            return ThreadActorBackend()
        if spec == "process":
            return ProcessActorBackend()
        if spec == "gpu":
            return GPUActorBackend()
        if spec.startswith("tcp://"):
            host, port_str = spec[len("tcp://") :].rsplit(":", 1)
            return RemoteActorBackend(host, int(port_str))
        if spec.startswith("ucx://"):
            host, port_str = spec[len("ucx://") :].rsplit(":", 1)
            return UCXRemoteActorBackend(host, int(port_str))
        raise ValueError(f"Unknown actor backend spec: {spec!r}")
    return spec


__all__ = ["resolve_backend"]
