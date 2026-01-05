from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

from ...configs.backend import get_backend
from ...engine.graph.operator import OpContext
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

try:  # optional torch for conversions
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


class MoNNA(Aggregator):
    """
    MoNNA aggregator: average of the n-f nearest neighbors of a designated
    reference (trusted) vector.
    """

    name = "monna"
    supports_subtasks = True
    max_subtasks_inflight = 0

    def __init__(self, f: int, *, reference_index: int = 0, chunk_size: int = 32) -> None:
        if f < 0:
            raise ValueError("f must be >= 0")
        if reference_index < 0:
            raise ValueError("reference_index must be >= 0")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        self.f = int(f)
        self.reference_index = int(reference_index)
        self.chunk_size = int(chunk_size)
        self._handle: SharedTensorHandle | None = None
        self._flat_shape: tuple[int, ...] | None = None

    def aggregate(self, gradients: Sequence[Any]) -> Any:
        if not gradients:
            raise ValueError("gradients must be non-empty")
        n = len(gradients)
        if self.f * 2 >= n:
            raise ValueError(f"Cannot tolerate 2f ≥ n (got n={n}, f={self.f}).")
        if not 0 <= self.reference_index < n:
            raise ValueError(
                f"reference_index must be between 0 and {n-1} (got {self.reference_index})."
            )

        be = get_backend()
        like = gradients[0]
        stacked = be.stack([be.asarray(g, like=like) for g in gradients], axis=0)
        ref = be.asarray(gradients[self.reference_index], like=like)
        diff = stacked - ref
        sq = diff * diff
        axes = tuple(range(1, sq.ndim))
        dists = be.sum(sq, axis=axes)
        k = n - self.f
        order = be.argsort(dists)
        indices = order[:k]
        chosen = be.index_select(stacked, axis=0, indices=indices)
        return be.mean(chosen, axis=0)

    def create_subtasks(self, inputs, *, context: OpContext):  # type: ignore[override]
        gradients = inputs.get(self.input_key)
        if not isinstance(gradients, Sequence) or not gradients:
            return []

        n = len(gradients)
        if self.f * 2 >= n:
            raise ValueError(f"Cannot tolerate 2f ≥ n (got n={n}, f={self.f}).")
        if not 0 <= self.reference_index < n:
            raise ValueError(
                f"reference_index must be between 0 and {n-1} (got {self.reference_index})."
            )

        flat_shape, flat = flatten_gradients(gradients)
        self._flat_shape = flat_shape
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
                    fn=_monna_chunk,
                    args=(
                        handle,
                        n,
                        self.reference_index,
                        start,
                        end,
                    ),
                    kwargs={},
                    name=f"monna_chunk_{chunk_id}",
                )
                chunk_id += 1

        return _iter()

    def reduce_subtasks(self, partials, inputs, *, context: OpContext):  # type: ignore[override]
        if not partials:
            return super().compute(inputs, context=context)

        gradients = inputs[self.input_key]
        like = gradients[0]
        ref_idx = self.reference_index
        if self._flat_shape is None:
            raise RuntimeError("MoNNA missing flat shape state for reduction.")

        all_distances: list[tuple[float, int]] = []
        for part in partials:
            start, dists = part
            for offset, val in enumerate(dists):
                all_distances.append((float(val), start + offset))

        n = len(gradients)
        k = n - self.f
        best = sorted(all_distances, key=lambda x: (x[1] != ref_idx, x[0]))[:k]
        indices = [idx for _, idx in best]

        try:
            handle = self._handle
            if handle is None:
                raise RuntimeError("MoNNA missing tensor handle.")
            with open_tensor(handle) as flat:
                flat_view = np.array(flat, copy=False)
                chosen = flat_view[indices, :]
                mean = np.mean(chosen, axis=0, dtype=flat_view.dtype)
            reshaped = mean.reshape(self._flat_shape)
            return _to_like(reshaped, like)
        finally:
            if self._handle is not None:
                cleanup_tensor(self._handle)
            self._handle = None
            self._flat_shape = None


def _monna_chunk(
    handle: SharedTensorHandle,
    n: int,
    ref_idx: int,
    start: int,
    end: int,
) -> tuple[int, np.ndarray]:
    with open_tensor(handle) as flat:
        flat_view = np.array(flat, copy=False)
        ref = flat_view[ref_idx]
        chunk = flat_view[start:end]
        diff = chunk - ref
        sq = diff * diff
        dists = np.sum(sq, axis=1, dtype=chunk.dtype)
        return start, dists
