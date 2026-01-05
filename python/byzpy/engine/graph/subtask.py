from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence


@dataclass(frozen=True)
class SubTask:
    """
    Unit of work that can be scheduled on a worker actor.
    """

    fn: Callable[..., Any]
    args: Sequence[Any] = field(default_factory=tuple)
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    name: Optional[str] = None
    affinity: Optional[str] = None  # e.g. "gpu"/"cpu"
    max_retries: int = 0
