from __future__ import annotations

import math
from itertools import combinations
from typing import Any, Iterable, Sequence, Tuple

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

try:  # optional torch import for dtype/device mirroring
    import torch

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    _HAS_TORCH = False


def _to_numpy(grad: Any) -> np.ndarray:
    if _HAS_TORCH and isinstance(grad, torch.Tensor):  # type: ignore[arg-type]
        return grad.detach().cpu().numpy()
    if isinstance(grad, np.ndarray):
        return grad
    if isinstance(grad, SharedTensorHandle):
        with open_tensor(grad) as arr:
            return np.array(arr, copy=True)
    if isinstance(grad, dict) and {"name", "shape", "dtype"} <= grad.keys():
        handle = SharedTensorHandle(**grad)
        with open_tensor(handle) as arr:
            return np.array(arr, copy=True)
    return np.asarray(grad)


def _to_like(arr: np.ndarray, like: Any) -> Any:
    if _HAS_TORCH and isinstance(like, torch.Tensor):  # type: ignore[arg-type]
        return torch.from_numpy(arr).to(dtype=like.dtype, device=like.device)
    if isinstance(like, SharedTensorHandle):
        return np.array(arr, dtype=np.dtype(like.dtype), copy=True)
    if isinstance(like, dict) and {"name", "shape", "dtype"} <= like.keys():
        return np.array(arr, dtype=np.dtype(like["dtype"]), copy=True)
    be = get_backend()
    return be.asarray(arr, like=like)


def _flatten_gradients(gradients: Sequence[Any]) -> tuple[tuple[int, ...], np.ndarray]:
    arrays = [_to_numpy(g) for g in gradients]
    stacked = np.stack(arrays, axis=0)
    grad_shape = stacked.shape[1:]
    flat = stacked.reshape(stacked.shape[0], -1)
    return grad_shape, flat


def _centered_max_eigval(gram_subset: np.ndarray) -> float:
    """
    Return the largest eigenvalue of the covariance matrix for the subset
    represented by its Gram matrix.
    """
    m = gram_subset.shape[0]
    if m <= 1:
        return 0.0
    H = np.eye(m, dtype=gram_subset.dtype) - np.full((m, m), 1.0 / m, dtype=gram_subset.dtype)
    centered = H @ gram_subset @ H
    vals = np.linalg.eigvalsh(centered)
    return float(max(vals[-1].real, 0.0) / m)


def _best_subset(flat: np.ndarray, gram: np.ndarray, n: int, m: int) -> tuple[float, np.ndarray]:
    best_val: float | None = None
    best_mean: np.ndarray | None = None
    for combo in combinations(range(n), m):
        sub_gram = gram[np.ix_(combo, combo)]
        eig = _centered_max_eigval(sub_gram)
        if (best_val is None) or (eig < best_val):
            best_val = eig
            best_mean = flat[list(combo)].mean(axis=0)
    if best_val is None or best_mean is None:  # pragma: no cover - guarded by caller
        raise RuntimeError("SMEA failed to evaluate any subset.")
    return best_val, best_mean


def _smea_chunk(
    flat_handle: SharedTensorHandle,
    gram_handle: SharedTensorHandle,
    combos: Tuple[tuple[int, ...], ...],
) -> tuple[float, np.ndarray]:
    with open_tensor(flat_handle) as flat, open_tensor(gram_handle) as gram:
        best_val: float | None = None
        best_mean: np.ndarray | None = None
        for combo in combos:
            sub_gram = gram[np.ix_(combo, combo)]
            eig = _centered_max_eigval(sub_gram)
            if (best_val is None) or (eig < best_val):
                best_val = eig
                best_mean = flat[list(combo)].mean(axis=0)
        if best_val is None or best_mean is None:  # pragma: no cover - defensive
            raise RuntimeError("SMEA chunk did not process any combinations.")
        return best_val, best_mean


class SMEA(Aggregator):
    """
    Smallest Maximum Eigenvalue Averaging (SMEA).

    Enumerates all ``n - f`` sized subsets, selects the one whose empirical
    covariance has the smallest maximum eigenvalue, and returns that subset's
    mean. Uses an m×m eigenproblem (Gram matrix) for efficiency and supports
    chunked evaluation via the task scheduler.
    """

    name = "smea"
    supports_subtasks = True
    max_subtasks_inflight = 0

    def __init__(self, f: int, *, chunk_size: int = 256) -> None:
        if f < 0:
            raise ValueError("f must be >= 0")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        self.f = int(f)
        self.chunk_size = int(chunk_size)
        self._flat_shape: tuple[int, ...] | None = None
        self._handles: tuple[SharedTensorHandle, SharedTensorHandle] | None = None

    def aggregate(self, gradients: Sequence[Any]) -> Any:
        if not gradients:
            raise ValueError("gradients must be a non-empty sequence")

        n = len(gradients)
        f = self.f
        if 2 * f >= n:
            raise ValueError(f"2f must be < n (got n={n}, f={f})")

        like = gradients[0]
        grad_shape, flat = _flatten_gradients(gradients)
        gram = flat @ flat.T
        m = n - f
        _, best_mean = _best_subset(flat, gram, n, m)
        result = best_mean.reshape(grad_shape)
        return _to_like(result, like)

    def create_subtasks(self, inputs, *, context):  # type: ignore[override]
        gradients = inputs.get(self.input_key)
        if not isinstance(gradients, Sequence) or not gradients:
            return []

        n = len(gradients)
        f = self.f
        if 2 * f >= n:
            raise ValueError(f"2f must be < n (got n={n}, f={f})")

        grad_shape, flat = _flatten_gradients(gradients)
        gram = flat @ flat.T
        self._flat_shape = grad_shape

        flat_handle = register_tensor(flat)
        gram_handle = register_tensor(gram)
        self._handles = (flat_handle, gram_handle)

        m = n - f
        total_combos = math.comb(n, m)
        metadata = getattr(context, "metadata", None) or {}
        pool_size = int(metadata.get("pool_size") or 0)
        chunk = select_adaptive_chunk_size(total_combos, self.chunk_size, pool_size=pool_size)

        def _iter_subtasks() -> Iterable[SubTask]:
            chunk_id = 0
            batch: list[tuple[int, ...]] = []
            for combo in combinations(range(n), m):
                batch.append(combo)
                if len(batch) < chunk:
                    continue
                yield SubTask(
                    fn=_smea_chunk,
                    args=(flat_handle, gram_handle, tuple(batch)),
                    kwargs={},
                    name=f"smea_chunk_{chunk_id}",
                )
                chunk_id += 1
                batch = []
            if batch:
                yield SubTask(
                    fn=_smea_chunk,
                    args=(flat_handle, gram_handle, tuple(batch)),
                    kwargs={},
                    name=f"smea_chunk_{chunk_id}",
                )

        return _iter_subtasks()

    def reduce_subtasks(self, partials, inputs, *, context):  # type: ignore[override]
        if not partials:
            return super().compute(inputs, context=context)

        if self._flat_shape is None:
            raise RuntimeError("SMEA reduce_subtasks missing flattened shape.")
        best_val: float | None = None
        best_mean: np.ndarray | None = None
        for eig, mean in partials:
            if (best_val is None) or (eig < best_val):
                best_val = eig
                best_mean = mean

        try:
            if best_mean is None:
                raise RuntimeError("SMEA reduce_subtasks missing partial results.")
            like = inputs[self.input_key][0]
            reshaped = np.array(best_mean, copy=False).reshape(self._flat_shape)
            return _to_like(reshaped, like)
        finally:
            if self._handles is not None:
                flat_h, gram_h = self._handles
                cleanup_tensor(flat_h)
                cleanup_tensor(gram_h)
            self._handles = None
            self._flat_shape = None


__all__ = ["SMEA"]
