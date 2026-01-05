from __future__ import annotations

import math
from typing import Any, Iterable, List, Optional

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


def _majority_needed(N: int, f: int) -> int:
    if N <= 0:
        raise ValueError("N must be positive")
    return max(1, (N // 2) + 1 - f)


def _ndtri(p: float) -> float:
    eps = 1e-12
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be in [0, 1]")
    p = min(max(p, eps), 1.0 - eps)
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]
    plow, phigh = 0.02425, 1.0 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        num = ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        den = (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        return num / den
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        num = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        den = (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        return num / den
    q = p - 0.5
    r = q * q
    num = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
    den = ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
    return num / den


class LittleAttack(Attack):
    """
    'Little is Enough': g_mal = μ + z_max * σ with
      s = floor(N/2)+1 - f
      z_max = Φ^{-1}((N - s)/N)
    """

    uses_honest_grads = True
    supports_subtasks = True
    max_subtasks_inflight = 0

    def __init__(self, f: int, N: Optional[int] = None, *, chunk_size: int = 8192) -> None:
        if f < 0:
            raise ValueError("f must be >= 0")
        self.f, self.N = f, N
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        self.chunk_size = int(chunk_size)
        self._active_handle: SharedTensorHandle | None = None
        self._flat_shape: tuple[int, ...] | None = None
        self._n: int | None = None
        self._like: Any | None = None

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
            raise ValueError("LittleAttack requires honest_grads.")
        total = len(honest_grads) + self.f if self.N is None else self.N
        if total < self.f:
            raise ValueError(f"N must be >= f (got N={total}, f={self.f})")
        s = _majority_needed(total, self.f)
        p = (total - s) / float(total)
        z = _ndtri(p)

        be = get_backend()
        like = honest_grads[0]
        X = be.stack([be.asarray(g, like=like) for g in honest_grads], axis=0)
        mu = be.mean(X, axis=0)
        diff = X - mu
        var = be.mean(diff * diff, axis=0)
        sigma = be.sqrt(var)

        out = mu + z * sigma
        return be.copy(out)

    def create_subtasks(self, inputs, *, context):  # type: ignore[override]
        grads = inputs.get("honest_grads")
        if not isinstance(grads, List) or not grads:
            return []

        n = len(grads)
        if self.f >= n:
            raise ValueError("LittleAttack requires len(honest_grads) > f.")

        flat_shape, flat = flatten_gradients(grads)
        self._flat_shape = flat_shape
        self._n = n
        self._like = grads[0]
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
                    fn=_little_partial_stats,
                    args=(handle, start, end),
                    kwargs={},
                    name=f"little_chunk_{chunk_id}",
                )
                chunk_id += 1

        return _iter()

    def reduce_subtasks(self, partials, inputs, *, context):  # type: ignore[override]
        if not partials:
            return super().compute(inputs, context=context)

        if (
            self._active_handle is None
            or self._flat_shape is None
            or self._n is None
            or self._like is None
        ):
            raise RuntimeError("LittleAttack missing chunk state.")

        feature_dim = int(np.prod(self._flat_shape))
        sum_x = np.zeros(feature_dim, dtype=np.float64)
        sum_sq = np.zeros(feature_dim, dtype=np.float64)

        for idx, item in enumerate(partials):
            try:
                start, chunk_sum, chunk_sq = item
            except Exception as exc:  # pragma: no cover
                raise ValueError(
                    f"LittleAttack received malformed partial at index {idx}: {item!r}"
                ) from exc
            end = start + chunk_sum.shape[0]
            sum_x[start:end] = chunk_sum
            sum_sq[start:end] = chunk_sq

        n = float(self._n)
        mu = sum_x / n
        var = np.maximum(sum_sq / n - mu * mu, 0.0)
        sigma = np.sqrt(var)

        total = len(inputs["honest_grads"]) + self.f if self.N is None else self.N
        s = _majority_needed(total, self.f)
        p = (total - s) / float(total)
        z = _ndtri(p)

        out = mu + z * sigma
        reshaped = out.reshape(self._flat_shape)

        like = self._like
        result = _to_like(reshaped, like)

        cleanup_tensor(self._active_handle)
        self._active_handle = None
        self._flat_shape = None
        self._n = None
        self._like = None

        return result


def _little_partial_stats(handle: SharedTensorHandle, start: int, end: int):
    with open_tensor(handle) as flat:
        view = flat[:, start:end]
        sum_x = np.sum(view, axis=0)
        sum_sq = np.sum(view * view, axis=0)
    return start, sum_x, sum_sq


def _to_like(arr: np.ndarray, like: Any):
    if isinstance(like, torch.Tensor):
        return torch.from_numpy(arr).to(device=like.device, dtype=like.dtype)
    be = get_backend()
    return be.asarray(arr, like=like)
