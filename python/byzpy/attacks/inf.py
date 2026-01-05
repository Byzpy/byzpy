from __future__ import annotations

from typing import Any, Iterable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from byzpy.aggregators._chunking import select_adaptive_chunk_size
from byzpy.aggregators.coordinate_wise._tiling import flatten_gradients
from byzpy.attacks.base import Attack
from byzpy.configs.backend import get_backend
from byzpy.engine.graph.subtask import SubTask
from byzpy.engine.storage.shared_store import (
    SharedTensorHandle,
    cleanup_tensor,
    open_tensor,
    register_tensor,
)


def _to_like(arr: np.ndarray, like: Any) -> Any:
    if isinstance(like, torch.Tensor):
        return torch.from_numpy(arr).to(dtype=like.dtype, device=like.device)
    be = get_backend()
    return be.asarray(arr, like=like)


def _fill_inf_chunk(handle: SharedTensorHandle, start: int, end: int):
    with open_tensor(handle) as flat:
        flat[start:end] = np.inf
    return start, None


class InfAttack(Attack):
    """
    Return a vector filled with +inf matching the gradient shape.
    """

    uses_honest_grads = True
    supports_subtasks = True
    max_subtasks_inflight = 0

    def __init__(self, *, chunk_size: int = 8192) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        self.chunk_size = int(chunk_size)
        self._handle: SharedTensorHandle | None = None
        self._flat_shape: tuple[int, ...] | None = None
        self._like_template: Any | None = None

    def apply(
        self,
        *,
        model: Optional[nn.Module] = None,
        x: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        honest_grads: Optional[List[Any]] = None,
        base_grad: Optional[Any] = None,
    ) -> Any:
        if not honest_grads:
            raise ValueError("InfAttack requires honest_grads.")
        flat_shape, _ = flatten_gradients(honest_grads)
        arr = np.full(flat_shape, np.inf, dtype=np.float64)
        return _to_like(arr, honest_grads[0])

    def create_subtasks(self, inputs, *, context):  # type: ignore[override]
        grads = inputs.get("honest_grads")
        if not isinstance(grads, Sequence) or not grads:
            return []

        flat_shape, flat = flatten_gradients(grads)
        feature_dim = flat.shape[1]
        buffer = np.zeros(feature_dim, dtype=np.float64)
        handle = register_tensor(buffer)
        self._handle = handle
        self._flat_shape = flat_shape
        self._like_template = grads[0]

        metadata = getattr(context, "metadata", None) or {}
        pool_size = int(metadata.get("pool_size") or 0)
        chunk = select_adaptive_chunk_size(
            feature_dim, self.chunk_size, pool_size=pool_size, allow_small_chunks=True
        )

        def _iter() -> Iterable[SubTask]:
            chunk_id = 0
            for start in range(0, feature_dim, chunk):
                end = min(feature_dim, start + chunk)
                yield SubTask(
                    fn=_fill_inf_chunk,
                    args=(handle, start, end),
                    kwargs={},
                    name=f"inf_chunk_{chunk_id}",
                )
                chunk_id += 1

        return _iter()

    def reduce_subtasks(self, partials, inputs, *, context):  # type: ignore[override]
        if not partials:
            return super().compute(inputs, context=context)
        if self._handle is None or self._flat_shape is None or self._like_template is None:
            raise RuntimeError("InfAttack missing state for reduction.")

        try:
            with open_tensor(self._handle) as flat:
                data = np.array(flat, copy=True)
            reshaped = data.reshape(self._flat_shape)
            reshaped.fill(np.inf)
            return _to_like(reshaped, self._like_template)
        finally:
            cleanup_tensor(self._handle)
            self._handle = None
            self._flat_shape = None
            self._like_template = None


__all__ = ["InfAttack"]
