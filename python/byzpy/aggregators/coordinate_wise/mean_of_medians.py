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
from ._tiling import flatten_gradients

try:  # optional torch dependency
    import torch

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    _HAS_TORCH = False


class MeanOfMedians(Aggregator):
    """
    Coordinate-wise Mean of Medians (MeaMed).

    For each coordinate ``k`` this aggregator:

    * Computes the median ``m_k`` of ``{x_{1k}, ..., x_{nk}}``.
    * Keeps the ``(n - f)`` values whose ``|x_{ik} - m_k|`` are smallest.
    * Returns the mean of the retained values.
    """

    name = "mean-of-medians"
    supports_subtasks = True
    max_subtasks_inflight = 0

    def __init__(self, f: int, *, chunk_size: int = 8192) -> None:
        if f < 0:
            raise ValueError("f must be >= 0")
        self.f = f
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        self.chunk_size = int(chunk_size)
        self._active_handle: SharedTensorHandle | None = None
        self._flat_shape: tuple[int, ...] | None = None

    def aggregate(self, gradients: Sequence[Any]) -> Any:
        """
        Drop the ``f`` farthest values per coordinate and average the rest.

        Args:
            gradients: Sequence of gradient tensors of identical shape/dtype.

        Returns:
            Tensor whose entries are the mean of the ``n - f`` closest values
            to the coordinate-wise medians.
        """
        if not gradients:
            raise ValueError("gradients must be a non-empty sequence")

        n = len(gradients)
        if self.f >= n:
            raise ValueError(f"f must satisfy 0 <= f < n (got n={n}, f={self.f})")

        be = get_backend()
        like = gradients[0]
        X = be.stack([be.asarray(g, like=like) for g in gradients], axis=0)  # (n, ...)

        med = be.median(X, axis=0)  # (...)
        deviations = be.abs(X - med)  # (n, ...)
        order = be.argsort(deviations, axis=0)  # (n, ...)
        keep_idx = order[: n - self.f]  # (n - f, ...)

        nearest_vals = be.take_along_axis(X, keep_idx, axis=0)  # (n - f, ...)
        return be.mean(nearest_vals, axis=0)

    def create_subtasks(self, inputs, *, context):  # type: ignore[override]
        gradients = inputs.get(self.input_key)
        if not isinstance(gradients, Sequence) or not gradients:
            return []

        n = len(gradients)
        if self.f >= n:
            raise ValueError(f"f must satisfy 0 <= f < n (got n={n}, f={self.f})")

        flat_shape, flat = flatten_gradients(gradients)
        self._flat_shape = flat_shape
        handle = register_tensor(flat)
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
                    fn=_meamed_chunk,
                    args=(handle, n, self.f, start, end, gradients[0]),
                    kwargs={},
                    name=f"meamed_chunk_{chunk_id}",
                )
                chunk_id += 1

        return _iter()

    def reduce_subtasks(self, partials, inputs, *, context):  # type: ignore[override]
        if not partials:
            return super().compute(inputs, context=context)

        like = inputs[self.input_key][0]
        if self._flat_shape is None:
            raise RuntimeError("MeanOfMedians reduce_subtasks missing shape state.")
        feature_dim = int(np.prod(self._flat_shape))

        try:
            assembled = np.zeros(feature_dim, dtype=np.float64)
            for start, chunk in sorted(partials, key=lambda x: x[0]):
                end = start + chunk.shape[0]
                assembled[start:end] = chunk
            reshaped = assembled.reshape(self._flat_shape)
            return _to_like(reshaped, like)
        finally:
            handle = self._active_handle
            if handle is not None:
                cleanup_tensor(handle)
            self._active_handle = None


def _meamed_chunk(
    handle: SharedTensorHandle, n: int, f: int, start: int, end: int, like_template
) -> tuple[int, np.ndarray]:
    with open_tensor(handle) as flat:
        view = flat[:, start:end]
        chunk = np.array(view, copy=False)
        chunk_copy = np.array(view, copy=True)
        mid = (n - 1) // 2
        med = np.partition(chunk_copy, mid, axis=0)[mid, :]
        deviations = np.abs(chunk - med)
        keep = n - f
        if keep >= n:
            indices = np.broadcast_to(np.arange(n)[:, None], (n, deviations.shape[1]))
        else:
            indices = np.argpartition(deviations, keep - 1, axis=0)[:keep, :]
        selected = np.take_along_axis(chunk, indices, axis=0)
        mean = np.mean(selected, axis=0)
    return start, mean


def _to_like(arr: np.ndarray, like: Any) -> Any:
    if _HAS_TORCH and isinstance(like, torch.Tensor):  # type: ignore[arg-type]
        return torch.from_numpy(arr).to(dtype=like.dtype)
    be = get_backend()
    return be.asarray(arr, like=like)
