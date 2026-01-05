from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

from ...configs.backend import get_backend
from ...engine.graph.subtask import SubTask
from ...engine.storage.shared_store import (
    SharedTensorHandle,
    cleanup_tensor,
    open_tensor,
    register_tensor,
)
from .._chunking import select_adaptive_chunk_size
from ..base import Aggregator
from ..coordinate_wise._tiling import flatten_gradients

try:  # optional torch dependency
    import torch

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    _HAS_TORCH = False


class ComparativeGradientElimination(Aggregator):
    """
    Comparative Gradient Elimination (CGE).

    Sort inputs by their L2 norm and average the (n - f) vectors
    with the smallest norms.

    Args (constructor):
        f: number of vectors to drop by norm (must be >= 0)
    """

    name = "comparative-gradient-elimination"
    supports_subtasks = True
    max_subtasks_inflight = 0

    def __init__(self, f: int, *, chunk_size: int = 8192) -> None:
        if f < 0:
            raise ValueError("f must be >= 0")
        self.f = int(f)
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        self.chunk_size = int(chunk_size)
        self._active_handle: SharedTensorHandle | None = None
        self._flat_shape: tuple[int, ...] | None = None
        self._n: int | None = None

    def aggregate(self, gradients: Sequence[Any]) -> Any:
        """
        Remove the ``f`` gradients with the largest L2 norms and average the rest.

        Args:
            gradients: Sequence of tensors to compare by L2 norm.
        """
        if not gradients:
            raise ValueError("gradients must be a non-empty sequence")

        n = len(gradients)
        if self.f >= n:
            raise ValueError(f"f must satisfy 0 <= f < n (got n={n}, f={self.f})")

        be = get_backend()
        like = gradients[0]
        X = be.stack([be.asarray(g, like=like) for g in gradients], axis=0)  # (n, ...)

        axes_feat = tuple(range(1, X.ndim))
        norms = be.sqrt(be.sum(X * X, axis=axes_feat))  # (n,)

        order = be.argsort(norms, axis=0)  # (n,)
        keep_idx = order[: n - self.f]  # (n-f,)

        kept = be.index_select(X, axis=0, indices=keep_idx)  # (n-f, ...)
        return be.mean(kept, axis=0)

    def create_subtasks(self, inputs, *, context):  # type: ignore[override]
        gradients = inputs.get(self.input_key)
        if not isinstance(gradients, Sequence) or not gradients:
            return []

        n = len(gradients)
        if self.f >= n:
            raise ValueError(f"f must satisfy 0 <= f < n (got n={n}, f={self.f})")

        flat_shape, flat = flatten_gradients(gradients)
        self._flat_shape = flat_shape
        self._n = n
        handle = register_tensor(flat.astype(np.float64, copy=False))
        self._active_handle = handle
        features = flat.shape[1]
        metadata = getattr(context, "metadata", None) or {}
        pool_size = int(metadata.get("pool_size") or 0)
        chunk = select_adaptive_chunk_size(features, self.chunk_size, pool_size=pool_size)

        def _iter() -> Iterable[SubTask]:
            chunk_id = 0
            for start in range(0, features, chunk):
                end = min(features, start + chunk)
                yield SubTask(
                    fn=_cge_norm_chunk,
                    args=(handle, start, end),
                    kwargs={},
                    name=f"cge_chunk_{chunk_id}",
                )
                chunk_id += 1

        return _iter()

    def reduce_subtasks(self, partials, inputs, *, context):  # type: ignore[override]
        if not partials:
            return super().compute(inputs, context=context)

        if self._active_handle is None or self._flat_shape is None or self._n is None:
            raise RuntimeError("ComparativeGradientElimination missing state for reduction.")

        n = self._n
        totals = np.zeros(n, dtype=np.float64)
        for item in partials:
            totals += np.asarray(item, dtype=np.float64)

        norms = np.sqrt(totals)
        keep_idx = np.argsort(norms)[: n - self.f]

        try:
            with open_tensor(self._active_handle) as flat:
                kept = flat[keep_idx]
                mean = kept.mean(axis=0)
            reshaped = mean.reshape(self._flat_shape)
            like = inputs[self.input_key][0]
            return _to_like(reshaped, like)
        finally:
            cleanup_tensor(self._active_handle)
            self._active_handle = None
            self._flat_shape = None
            self._n = None


def _cge_norm_chunk(handle: SharedTensorHandle, start: int, end: int) -> np.ndarray:
    with open_tensor(handle) as flat:
        view = flat[:, start:end]
        partial = np.sum(view * view, axis=1)
    return partial


def _to_like(arr: np.ndarray, like: Any) -> Any:
    if _HAS_TORCH and isinstance(like, torch.Tensor):  # type: ignore[arg-type]
        return torch.from_numpy(arr).to(dtype=like.dtype)
    be = get_backend()
    return be.asarray(arr, like=like)
