from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

from ..engine.backend.ndarray import _Backend, _TorchBackend
from ..engine.backend.ndarray.base import Tensor

_BACKEND: Optional[_Backend] = None


def set_backend(backend: _Backend | str) -> None:
    """
    Set the global array backend used by all aggregators.
    """

    global _BACKEND

    if isinstance(backend, str):
        key = backend.lower()
        if key in ("torch", "pytorch"):
            _BACKEND = _TorchBackend()
        else:
            raise ValueError(f"Unknown backend string: {backend!r}")
    elif isinstance(backend, _Backend) or (
        hasattr(backend, "asarray") and hasattr(backend, "stack") and hasattr(backend, "median")
    ):
        _BACKEND = backend  # custom implementation
    else:
        raise TypeError("unrecognized backend")


def get_backend() -> _Backend:
    return _TorchBackend()


@contextmanager
def use_backend(backend: _Backend | str):
    """
    Temporary backend override.
    """

    global _BACKEND
    prev = _BACKEND
    set_backend(backend)
    try:
        yield
    finally:
        _BACKEND = prev
