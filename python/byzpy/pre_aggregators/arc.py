from __future__ import annotations

import math
from typing import Any, Iterable, List, Sequence

import numpy as np

from ..aggregators._chunking import select_adaptive_chunk_size
from ..aggregators.coordinate_wise._tiling import flatten_gradients
from ..configs.backend import get_backend
from ..engine.graph.subtask import SubTask
from ..engine.storage.shared_store import (
    SharedTensorHandle,
    cleanup_tensor,
    open_tensor,
    register_tensor,
)
from .base import PreAggregator

try:
    import torch

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    _HAS_TORCH = False


def _to_like(arr: np.ndarray, like: Any) -> Any:
    if _HAS_TORCH and isinstance(like, torch.Tensor):  # type: ignore[arg-type]
        return torch.from_numpy(arr).to(dtype=like.dtype, device=like.device)
    be = get_backend()
    return be.asarray(arr, like=like)


def _compute_threshold(norms: np.ndarray, f: int) -> float:
    n = norms.shape[0]
    if n == 0:
        raise ValueError("norms must be non-empty")
    nb_clipped = int(math.floor((2.0 * f / n) * (n - f)))
    nb_clipped = max(0, min(nb_clipped, n - 1))
    cut_off = n - nb_clipped
    sorted_norms = np.sort(norms)
    return float(sorted_norms[max(0, cut_off - 1)])


def _arc_clip_rows(flat: np.ndarray, threshold: float) -> np.ndarray:
    norms = np.linalg.norm(flat, axis=1, keepdims=True)
    denom = np.maximum(norms, 1e-12)
    factors = np.minimum(1.0, threshold / denom)
    return flat * factors


def _arc_clip_chunk(handle: SharedTensorHandle, start: int, end: int, threshold: float):
    with open_tensor(handle) as flat:
        chunk = np.array(flat[start:end], copy=False)
        norms = np.linalg.norm(chunk, axis=1, keepdims=True)
        denom = np.maximum(norms, 1e-12)
        factors = np.minimum(1.0, threshold / denom)
        chunk *= factors
    return start, None


class ARC(PreAggregator):
    """
    Adaptive Robust Clipping (ARC) pre-aggregator.

    Clips the ``⌊2 f / n * (n - f)⌋`` largest-norm vectors to the norm of the
    next-largest remaining vector.
    """

    name = "pre-agg/arc"
    supports_subtasks = True
    max_subtasks_inflight = 0

    def __init__(self, f: int = 0, *, chunk_size: int = 32) -> None:
        if f < 0:
            raise ValueError("f must be >= 0")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        self.f = int(f)
        self.chunk_size = int(chunk_size)
        self._handle: SharedTensorHandle | None = None
        self._flat_shape: tuple[int, ...] | None = None
        self._like_template: Any | None = None
        self._threshold: float | None = None

    def pre_aggregate(self, xs: Sequence[Any]) -> List[Any]:
        if not xs:
            raise ValueError("xs must be non-empty")
        n = len(xs)
        if self.f > n:
            raise ValueError(f"f must be <= number of vectors (got f={self.f}, n={n})")
        flat_shape, flat = flatten_gradients(xs)
        norms = np.linalg.norm(flat, axis=1)
        threshold = _compute_threshold(norms, self.f)
        clipped = _arc_clip_rows(flat, threshold)
        outputs: List[Any] = []
        for row in clipped:
            reshaped = row.reshape(flat_shape)
            outputs.append(_to_like(reshaped, xs[0]))
        return outputs

    def create_subtasks(self, inputs, *, context):  # type: ignore[override]
        xs = inputs.get(self.input_key)
        if not isinstance(xs, Sequence) or not xs:
            return []
        n = len(xs)
        if self.f > n:
            raise ValueError(f"f must be <= number of vectors (got f={self.f}, n={n})")
        flat_shape, flat = flatten_gradients(xs)
        norms = np.linalg.norm(flat, axis=1)
        threshold = _compute_threshold(norms, self.f)
        self._flat_shape = flat_shape
        self._like_template = xs[0]
        self._threshold = threshold

        handle = register_tensor(flat)
        self._handle = handle

        metadata = getattr(context, "metadata", None) or {}
        pool_size = int(metadata.get("pool_size") or 0)
        chunk = select_adaptive_chunk_size(n, self.chunk_size, pool_size=pool_size)

        def _iter() -> Iterable[SubTask]:
            chunk_id = 0
            for start in range(0, n, chunk):
                end = min(n, start + chunk)
                yield SubTask(
                    fn=_arc_clip_chunk,
                    args=(handle, start, end, threshold),
                    kwargs={},
                    name=f"arc_chunk_{chunk_id}",
                )
                chunk_id += 1

        return _iter()

    def reduce_subtasks(self, partials, inputs, *, context):  # type: ignore[override]
        if not partials:
            return super().compute(inputs, context=context)
        if self._handle is None or self._flat_shape is None or self._like_template is None:
            raise RuntimeError("ARC missing state for reduction.")

        try:
            with open_tensor(self._handle) as flat:
                data = np.array(flat, copy=False)
                outputs: List[Any] = []
                for row in data:
                    reshaped = np.array(row, copy=True).reshape(self._flat_shape)
                    outputs.append(_to_like(reshaped, self._like_template))
                return outputs
        finally:
            cleanup_tensor(self._handle)
            self._handle = None
            self._flat_shape = None
            self._like_template = None
            self._threshold = None


__all__ = ["ARC"]
