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

try:  # optional torch dependency
    import torch

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    _HAS_TORCH = False


def _to_float(x: Any) -> float:
    return x.item() if hasattr(x, "item") else float(x)


class GeometricMedian(Aggregator):
    """
    Geometric Median via Weiszfeld's algorithm.

    Minimizes sum_i ||x - g_i||_2. Uses backend ops via `get_backend()`.
    """

    name = "geometric-median"
    supports_barriered_subtasks = True
    max_subtasks_inflight = 0

    def __init__(
        self,
        *,
        tol: float = 1e-6,
        max_iter: int = 256,
        eps: float = 1e-12,
        init: str = "median",  # "median" or "mean"
        chunk_size: int = 32,
    ) -> None:
        if tol <= 0:
            raise ValueError("tol must be > 0")
        if max_iter <= 0:
            raise ValueError("max_iter must be > 0")
        if eps <= 0:
            raise ValueError("eps must be > 0")
        if init not in {"median", "mean"}:
            raise ValueError("init must be 'median' or 'mean'")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        self.tol = tol
        self.max_iter = max_iter
        self.eps = eps
        self.init = init
        self.chunk_size = int(chunk_size)

    def aggregate(self, gradients: Sequence[Any]) -> Any:
        """
        Run Weiszfeld iterations until convergence or ``max_iter`` steps.

        Args:
            gradients: Sequence of gradient tensors to summarize.
        """
        if not gradients:
            raise ValueError("gradients must be a non-empty sequence")

        be = get_backend()
        like = gradients[0]
        X = be.stack([be.asarray(g, like=like) for g in gradients], axis=0)  # (n, ...)

        z = be.median(X, axis=0) if self.init == "median" else be.mean(X, axis=0)

        axes_feat = tuple(range(1, X.ndim))

        for _ in range(self.max_iter):
            z_prev = z
            diff = X - z
            dist = be.sqrt(be.sum(diff * diff, axis=axes_feat))  # (n,)
            dist = be.maximum(dist, dist * 0 + self.eps)

            w = 1.0 / dist  # (n,)
            w = be.reshape(w, (w.shape[0],) + (1,) * (X.ndim - 1))

            num = be.sum(w * X, axis=0)
            den = be.sum(w, axis=0)
            z = num / den

            delta = be.sqrt(be.sum((z - z_prev) * (z - z_prev), axis=tuple(range(z.ndim))))
            if _to_float(delta) <= self.tol:
                break

        return z

    async def run_barriered_subtasks(self, inputs, *, context: OpContext, pool):  # type: ignore[override]
        gradients = inputs.get(self.input_key)
        if not isinstance(gradients, Sequence) or not gradients:
            raise ValueError("GeometricMedian requires a non-empty gradient list.")

        backend = get_backend()
        backend_name = getattr(backend, "name", "")
        flat_shape, flat = flatten_gradients(gradients)
        data = np.asarray(flat, dtype=np.float64)
        like = gradients[0]
        grad_handle = register_tensor(data)
        z = _init_center(data, init=self.init, backend_name=backend_name)
        center_handle = register_tensor(z.copy())
        metadata = getattr(context, "metadata", None) or {}
        pool_size = getattr(pool, "size", 1)
        chunk = max(1, min(self.chunk_size, data.shape[0]))

        try:
            for _ in range(self.max_iter):
                _write_handle(center_handle, z)
                subtasks: list[SubTask] = []
                chunk_id = 0
                for start in range(0, data.shape[0], chunk):
                    end = min(data.shape[0], start + chunk)
                    subtasks.append(
                        SubTask(
                            fn=_geom_median_chunk,
                            args=(grad_handle, center_handle, start, end, self.eps),
                            kwargs={},
                            name=f"geom_median_chunk_{chunk_id}",
                        )
                    )
                    chunk_id += 1
                partials = await self._run_subtasks(pool, subtasks, self.max_subtasks_inflight, context)  # type: ignore[arg-type]
                if not partials:
                    break
                weighted_sum = np.zeros_like(z)
                total_weight = 0.0
                for part_sum, part_weight in partials:
                    weighted_sum += np.asarray(part_sum, dtype=np.float64)
                    total_weight += float(part_weight)
                if total_weight <= 0:
                    break
                new_z = weighted_sum / total_weight
                delta = np.linalg.norm(new_z - z)
                z = new_z
                if delta <= self.tol:
                    break
            reshaped = z.reshape(flat_shape)
            return _to_like(reshaped, like)
        finally:
            cleanup_tensor(grad_handle)
            cleanup_tensor(center_handle)


def _geom_median_chunk(
    grad_handle: SharedTensorHandle,
    center_handle: SharedTensorHandle,
    start: int,
    end: int,
    eps: float,
) -> tuple[np.ndarray, float]:
    with open_tensor(grad_handle) as data_arr, open_tensor(center_handle) as center_arr:
        data = np.array(data_arr, copy=False)
        center = np.array(center_arr, copy=False)
        rows = data[start:end]
        if rows.size == 0:
            return np.zeros_like(center), 0.0
        if _HAS_TORCH:
            torch_rows = torch.from_numpy(rows)
            torch_center = torch.from_numpy(center)
            diff = torch_rows - torch_center
            dist = torch.linalg.norm(diff, dim=1)
            dist = torch.clamp(dist, min=eps)
            weights = 1.0 / dist
            weighted = (weights[:, None] * torch_rows).sum(dim=0)
            return weighted.detach().cpu().numpy(), float(weights.sum().item())
        diff = rows - center
        dist = np.linalg.norm(diff, axis=1)
        dist = np.maximum(dist, eps)
        weights = 1.0 / dist
        weighted = (weights[:, None] * rows).sum(axis=0)
        return weighted, float(weights.sum())


def _init_center(data: np.ndarray, *, init: str, backend_name: str) -> np.ndarray:
    if init == "mean":
        return data.mean(axis=0)
    if backend_name == "torch":
        chunk = np.array(data, copy=True)
        n = chunk.shape[0]
        mid = n // 2
        if n % 2 == 1:
            return np.partition(chunk, mid, axis=0)[mid, :]
        return np.partition(chunk, mid - 1, axis=0)[mid - 1, :]
    return np.median(data, axis=0)


def _write_handle(handle: SharedTensorHandle, values: np.ndarray) -> None:
    with open_tensor(handle) as arr:
        arr[:] = values


def _to_like(arr: np.ndarray, like: Any) -> Any:
    if _HAS_TORCH and isinstance(like, torch.Tensor):  # type: ignore[arg-type]
        return torch.from_numpy(arr).to(dtype=like.dtype, device=like.device)
    be = get_backend()
    return be.asarray(arr, like=like)
