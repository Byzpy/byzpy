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


class CAF(Aggregator):
    """Covariance-bound Agnostic Filter (CAF)."""

    name = "caf"
    supports_subtasks = True
    max_subtasks_inflight = 0

    def __init__(self, f: int, *, chunk_size: int = 256, power_iters: int = 3) -> None:
        if f < 0:
            raise ValueError("f must be >= 0")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if power_iters <= 0:
            raise ValueError("power_iters must be > 0")
        self.f = int(f)
        self.chunk_size = int(chunk_size)
        self.power_iters = int(power_iters)
        self._flat_shape: tuple[int, ...] | None = None
        self._handle: SharedTensorHandle | None = None

    def aggregate(self, gradients: Sequence[Any]) -> Any:
        if not gradients:
            raise ValueError("gradients must be non-empty")
        n = len(gradients)
        if self.f * 2 >= n:
            raise ValueError(f"Cannot tolerate 2f ≥ n (got n={n}, f={self.f}).")

        flat_shape, flat = flatten_gradients(gradients)
        data = _as_float_array(flat, like=gradients[0])
        result = _caf_compute(data, self.f, power_iters=self.power_iters)
        reshaped = result.reshape(flat_shape)
        return _to_like(reshaped, gradients[0])

    def create_subtasks(self, inputs, *, context: OpContext):  # type: ignore[override]
        gradients = inputs.get(self.input_key)
        if not isinstance(gradients, Sequence) or not gradients:
            return []
        n = len(gradients)
        if self.f * 2 >= n:
            raise ValueError(f"Cannot tolerate 2f ≥ n (got n={n}, f={self.f}).")

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
                    fn=_caf_chunk_fetch,
                    args=(handle, start, end),
                    kwargs={},
                    name=f"caf_chunk_{chunk_id}",
                )
                chunk_id += 1

        return _iter()

    def reduce_subtasks(self, partials, inputs, *, context: OpContext):  # type: ignore[override]
        if not partials:
            return super().compute(inputs, context=context)
        if self._flat_shape is None:
            raise RuntimeError("CAF missing flat shape state for reduction.")

        dtype = _dtype_from_like(inputs[self.input_key][0])
        parts = sorted(partials, key=lambda item: item[0])
        chunks = [np.array(chunk, dtype=dtype, copy=True) for _, chunk in parts]
        data = np.concatenate(chunks, axis=0)

        try:
            result = _caf_compute(data, self.f, power_iters=self.power_iters)
            reshaped = result.reshape(self._flat_shape)
            return _to_like(reshaped, inputs[self.input_key][0])
        finally:
            if self._handle is not None:
                cleanup_tensor(self._handle)
            self._handle = None
            self._flat_shape = None


def _dtype_from_like(like: Any) -> np.dtype:
    arr = np.asarray(like)
    if not np.issubdtype(arr.dtype, np.floating):
        return np.dtype(np.float32)
    return arr.dtype


def _as_float_array(arr: np.ndarray, *, like: Any) -> np.ndarray:
    dtype = _dtype_from_like(like)
    return np.array(arr, dtype=dtype, copy=True)


def _caf_compute(data: np.ndarray, f: int, *, power_iters: int) -> np.ndarray:
    n = data.shape[0]
    weights = np.ones(n, dtype=data.dtype)
    total = weights.sum()
    best_mu = np.mean(data, axis=0, dtype=data.dtype)
    best_lambda = np.inf if data.dtype != np.float16 else np.finfo(np.float32).max

    while total > n - 2 * f:
        mu = np.sum(weights[:, None] * data, axis=0, dtype=data.dtype) / total
        diffs = data - mu
        lam, vec = _dominant_eigenpair(diffs, weights, power_iters)
        if lam < best_lambda:
            best_lambda = lam
            best_mu = mu.copy()

        proj = diffs @ vec
        tau = proj**2
        tau_max = float(tau.max())
        if tau_max <= 1e-12:
            break
        weights *= 1.0 - tau / tau_max
        weights = np.clip(weights, 0.0, None)
        total = float(weights.sum())
        if total <= 0:
            break

    return best_mu


def _dominant_eigenpair(
    diffs: np.ndarray, weights: np.ndarray, iters: int
) -> tuple[float, np.ndarray]:
    rng = np.random.default_rng(0)
    vec = rng.normal(size=diffs.shape[1]).astype(diffs.dtype, copy=False)
    norm = np.linalg.norm(vec)
    if norm == 0:
        vec = np.ones(diffs.shape[1], dtype=diffs.dtype)
        norm = np.linalg.norm(vec)
    vec /= norm

    for _ in range(iters):
        proj = diffs @ vec
        weighted = (weights * proj)[:, None] * diffs
        next_vec = weighted.sum(axis=0)
        next_norm = np.linalg.norm(next_vec)
        if next_norm <= 1e-12:
            break
        vec = next_vec / next_norm

    proj = diffs @ vec
    eig = float(np.sum(weights * (proj**2)) / max(1e-12, weights.sum()))
    return eig, vec


def _caf_chunk_fetch(handle: SharedTensorHandle, start: int, end: int) -> tuple[int, np.ndarray]:
    with open_tensor(handle) as flat:
        chunk = np.array(flat[start:end], copy=True)
    return start, chunk


__all__ = ["CAF"]
