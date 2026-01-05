from __future__ import annotations

import heapq
import math
import os
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

try:  # optional torch for _to_like conversion
    import torch

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    _HAS_TORCH = False


def _pairwise_sq_dists(stacked: Any) -> Any:
    """
    stacked: shape (n, ...)
    returns: shape (n, n) of squared Euclidean distances
    """
    be = get_backend()
    be_name = getattr(be, "name", "")
    if be_name == "torch":
        import torch

        flat = stacked.reshape(stacked.shape[0], -1)
        norms = torch.sum(flat * flat, dim=1, keepdim=True)
        D2 = norms + norms.T - 2.0 * (flat @ flat.T)
        return torch.clamp(D2, min=0.0)

    if be_name == "numpy":
        arr = np.asarray(stacked)
        flat = arr.reshape(arr.shape[0], -1)
        norms = np.sum(flat * flat, axis=1, keepdims=True)
        gram = flat @ flat.T
        D2 = norms + norms.T - 2.0 * gram
        np.maximum(D2, 0.0, out=D2)
        return D2

    diff = stacked[:, None, ...] - stacked[None, :, ...]  # (n, n, ...)
    sq = diff * diff
    axes = tuple(range(2, sq.ndim))  # sum over feature dims
    return be.sum(sq, axis=axes)  # (n, n)


def _materialize_gradients(gradients: Sequence[Any]) -> tuple[Any, Sequence[Any]]:
    """Ensure gradients are concrete arrays/tensors (handles -> numpy)."""
    if _HAS_TORCH and isinstance(gradients[0], torch.Tensor):  # type: ignore[arg-type]
        return gradients[0], gradients
    if isinstance(gradients[0], np.ndarray):
        return gradients[0], gradients
    flat_shape, flat = flatten_gradients(gradients)
    arrays = [flat[i].reshape(flat_shape) for i in range(flat.shape[0])]
    return arrays[0], arrays


def _to_like(arr: np.ndarray, like: Any) -> Any:
    if _HAS_TORCH and isinstance(like, torch.Tensor):  # type: ignore[arg-type]
        return torch.from_numpy(arr).to(dtype=like.dtype, device=like.device)
    be = get_backend()
    try:
        return be.asarray(arr, like=like)
    except TypeError:
        return be.asarray(arr)


class MultiKrum(Aggregator):
    """
    Multi-Krum aggregator for Byzantine-robust gradient aggregation.

    Multi-Krum is a geometric aggregation method that selects the q gradients
    with the smallest sum of squared distances to their nearest neighbors,
    then returns their mean. This makes it robust to up to f Byzantine nodes.

    Algorithm:
    1. For each gradient i, compute the sum of squared Euclidean distances to
       its (n - f - 1) nearest neighbors (excluding itself).
    2. Select the q gradients with the smallest sums (Krum scores).
    3. Return the mean of the selected gradients.

    Parameters
    ----------
    f : int
        Maximum number of Byzantine nodes to tolerate. Must satisfy
        0 <= f < n-1 where n is the number of gradients.
    q : int
        Number of gradients to select. Must satisfy 1 <= q <= n - f.
    chunk_size : int, optional
        Size of chunks for parallel distance computation. Default is 32.

    Examples
    --------
    >>> aggregator = MultiKrum(f=2, q=5, chunk_size=16)
    >>> gradients = [torch.randn(100) for _ in range(10)]
    >>> result = aggregator.aggregate(gradients)
    >>> assert result.shape == (100,)

    Notes
    -----
    - Robust to up to f Byzantine nodes when honest gradients are well-separated.
    - Time complexity: O(n^2 * d) for distance computation, where n is number
      of gradients and d is dimension. With subtasks: O(n^2 * d / workers).
    - Memory complexity: O(n^2) for distance matrix, O(n * d) for gradients.
    - The original Krum algorithm corresponds to q=1.

    References
    ----------
    .. [1] Blanchard, P., El Mhamdi, E. M., Guerraoui, R., & Stainer, J.
       (2017). Machine learning with adversaries: Byzantine tolerant gradient
       descent. NeurIPS.
    """

    name = "multi-krum"
    supports_subtasks = True
    max_subtasks_inflight = 0

    def __init__(self, f: int, q: int, *, chunk_size: int = 32) -> None:
        if f < 0:
            raise ValueError("f must be >= 0")
        if q < 1:
            raise ValueError("q must be >= 1")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        self.f = f
        self.q = q
        self.chunk_size = int(chunk_size)
        self._active_handle: SharedTensorHandle | None = None
        self._norms_handle: SharedTensorHandle | None = None
        self._flat_shape: tuple[int, ...] | None = None
        self._active_workers: int = 1

    def aggregate(self, gradients: Sequence[Any]) -> Any:
        """
        Select the q most consensus gradients using the Krum score.

        Parameters
        ----------
        gradients : Sequence[Any]
            Sequence of gradient tensors. All must have the same shape and
            backend. The sequence length n must satisfy f < n-1 and q <= n-f.

        Returns
        -------
        Any
            Mean of the q selected gradients. Same shape and backend as inputs.

        Raises
        ------
        ValueError
            If gradients is empty, or if f >= n-1, or if q > n-f.
        """
        if not gradients:
            raise ValueError("gradients must be a non-empty sequence")

        n = len(gradients)
        f, q = self.f, self.q
        if f >= n - 1:
            raise ValueError(f"f must satisfy 0 <= f < n-1 (got n={n}, f={f})")
        if q > n - f:
            raise ValueError(f"q must satisfy 1 <= q <= n - f (got n={n}, f={f}, q={q})")

        be = get_backend()
        like, mats = _materialize_gradients(gradients)
        use_like = _HAS_TORCH and isinstance(like, torch.Tensor)
        to_stack = [be.asarray(g, like=like) if use_like else be.asarray(g) for g in mats]
        X = be.stack(to_stack, axis=0)  # (n, ...)

        D = _pairwise_sq_dists(X)  # (n, n)

        order = be.argsort(D, axis=1)  # (n, n)
        neigh_idx = order[:, 1 : (n - f)]  # (n, n-f-1)
        neigh_d = be.take_along_axis(D, neigh_idx, axis=1)  # (n, n-f-1)

        scores = be.sum(neigh_d, axis=1)  # (n,)

        winner_order = be.argsort(scores, axis=0)[:q]  # (q,)
        chosen = be.index_select(X, axis=0, indices=winner_order)  # (q, ...)

        return be.mean(chosen, axis=0)

    def create_subtasks(self, inputs, *, context):  # type: ignore[override]
        gradients = inputs.get(self.input_key)
        if not isinstance(gradients, Sequence) or not gradients:
            return []

        n = len(gradients)
        f, q = self.f, self.q
        if f >= n - 1:
            raise ValueError(f"f must satisfy 0 <= f < n-1 (got n={n}, f={f})")
        if q > n - f:
            raise ValueError(f"q must satisfy 1 <= q <= n - f (got n={n}, f={f}, q={q})")

        flat_shape, flat = flatten_gradients(gradients)
        flat = np.ascontiguousarray(flat, dtype=np.float32)
        self._flat_shape = flat_shape
        handle = register_tensor(flat)
        norms = np.sum(flat * flat, axis=1, dtype=np.float64)
        norms_handle = register_tensor(norms)
        self._active_handle = handle
        self._norms_handle = norms_handle
        metadata = getattr(context, "metadata", None) or {}
        pool_size = int(metadata.get("pool_size") or 0)
        chunk = select_adaptive_chunk_size(n, self.chunk_size, pool_size=pool_size)
        chunks = max(1, math.ceil(n / max(1, chunk)))
        self._active_workers = max(1, min(pool_size or 1, chunks))

        def _iter() -> Iterable[SubTask]:
            chunk_id = 0
            for start in range(0, n, chunk):
                end = min(n, start + chunk)
                yield SubTask(
                    fn=_multikrum_chunk,
                    args=(
                        handle,
                        norms_handle,
                        n,
                        self.f,
                        start,
                        end,
                        self._active_workers,
                    ),
                    kwargs={},
                    name=f"multikrum_chunk_{chunk_id}",
                )
                chunk_id += 1

        return _iter()

    def reduce_subtasks(self, partials, inputs, *, context):  # type: ignore[override]
        if not partials:
            try:
                return super().compute(inputs, context=context)
            finally:
                self._cleanup_handles()

        gradients = inputs[self.input_key]
        like, _ = _materialize_gradients(gradients)
        q = self.q
        if self._flat_shape is None:
            raise RuntimeError("MultiKrum reduce_subtasks missing flat shape state.")

        candidates: list[tuple[float, int]] = []
        for idx, part in enumerate(partials):
            try:
                start, scores = part
            except Exception as exc:  # pragma: no cover
                raise ValueError(
                    f"MultiKrum received malformed partial at index {idx}: {part!r}"
                ) from exc
            for offset, score in enumerate(scores):
                candidates.append((float(score), start + offset))

        if len(candidates) < q:
            raise RuntimeError(
                f"MultiKrum expected >= {q} candidates but received {len(candidates)}."
            )

        best = heapq.nsmallest(q, candidates, key=lambda x: x[0])
        best_indices = [idx for _, idx in best]

        try:
            handle = self._active_handle
            if handle is None:
                raise RuntimeError("MultiKrum missing shared tensor handle.")
            with open_tensor(handle) as flat:
                flat_view = np.array(flat, copy=False)
                chosen = flat_view[best_indices, :]
                mean = np.mean(chosen, axis=0)
            reshaped = mean.reshape(self._flat_shape)
            return _to_like(reshaped, like)
        finally:
            self._cleanup_handles()

    def _cleanup_handles(self) -> None:
        handle = self._active_handle
        if handle is not None:
            cleanup_tensor(handle)
        norms_handle = self._norms_handle
        if norms_handle is not None:
            cleanup_tensor(norms_handle)
        self._active_handle = None
        self._norms_handle = None
        self._flat_shape = None
        self._active_workers = 1


class Krum(Aggregator):
    """
    Krum aggregator (Multi-Krum with q=1).

    This is the original Krum algorithm, which selects the single gradient
    with the smallest Krum score (sum of distances to nearest neighbors) and
    returns it directly (without averaging).

    Parameters
    ----------
    f : int
        Maximum number of Byzantine nodes to tolerate. Must satisfy
        0 <= f < n-1 where n is the number of gradients.
    chunk_size : int, optional
        Size of chunks for parallel distance computation. Default is 32.

    Examples
    --------
    >>> aggregator = Krum(f=2, chunk_size=16)
    >>> gradients = [torch.randn(100) for _ in range(10)]
    >>> result = aggregator.aggregate(gradients)
    >>> assert result.shape == (100,)

    Notes
    -----
    - Equivalent to MultiKrum(f=f, q=1).
    - Returns a single selected gradient rather than the mean of q gradients.
    - Robust to up to f Byzantine nodes.

    See Also
    --------
    MultiKrum : The general version that selects q gradients.
    """

    name = "krum"
    supports_subtasks = True
    max_subtasks_inflight = 0

    def __init__(self, f: int, *, chunk_size: int = 32) -> None:
        if f < 0:
            raise ValueError("f must be >= 0")
        self.f = f
        self.krum = MultiKrum(f=self.f, q=1, chunk_size=chunk_size)

    def aggregate(self, gradients: Sequence[Any]) -> Any:
        """
        Return the single gradient picked by the underlying Multi-Krum.

        Parameters
        ----------
        gradients : Sequence[Any]
            Sequence of gradient tensors. All must have the same shape and
            backend.

        Returns
        -------
        Any
            The single selected gradient (not averaged). Same shape and backend
            as inputs.
        """
        return self.krum.aggregate(gradients)

    def create_subtasks(self, inputs, *, context):  # type: ignore[override]
        return self.krum.create_subtasks(inputs, context=context)

    def reduce_subtasks(self, partials, inputs, *, context):  # type: ignore[override]
        return self.krum.reduce_subtasks(partials, inputs, context=context)


def _multikrum_chunk(
    data_handle: SharedTensorHandle,
    norms_handle: SharedTensorHandle,
    n: int,
    f: int,
    start: int,
    end: int,
    worker_target: int,
) -> tuple[int, np.ndarray]:
    count = end - start
    k = max(0, n - f - 1)
    if count <= 0:
        return start, np.zeros(0, dtype=np.float64)
    with open_tensor(data_handle) as flat, open_tensor(norms_handle) as norms_arr:
        data = np.array(flat, copy=False)
        norms = np.array(norms_arr, copy=False)
        if _HAS_TORCH:
            scores = _multikrum_chunk_torch(
                data,
                norms,
                n=n,
                f=f,
                start=start,
                end=end,
                worker_target=worker_target,
            )
        else:
            scores = _multikrum_chunk_numpy(data, norms, n=n, f=f, start=start, end=end)
    return start, scores


def _multikrum_chunk_numpy(
    data: np.ndarray,
    norms: np.ndarray,
    *,
    n: int,
    f: int,
    start: int,
    end: int,
) -> np.ndarray:
    chunk = data[start:end, :]
    chunk_norms = norms[start:end]
    gram = chunk @ data.T  # (count, n)
    dists = chunk_norms[:, None] + norms[None, :] - 2.0 * gram
    np.maximum(dists, 0.0, out=dists)
    row_idx = np.arange(end - start)
    dists[row_idx, np.arange(start, end)] = np.inf
    k = max(0, n - f - 1)
    if k <= 0:
        return np.zeros(end - start, dtype=np.float64)
    nearest = np.partition(dists, kth=k - 1, axis=1)[:, :k]
    return np.sum(nearest, axis=1, dtype=np.float64)


def _multikrum_chunk_torch(
    data: np.ndarray,
    norms: np.ndarray,
    *,
    n: int,
    f: int,
    start: int,
    end: int,
    worker_target: int,
) -> np.ndarray:
    count = end - start
    if count <= 0:
        return np.zeros(0, dtype=np.float64)
    torch_data = torch.from_numpy(data)
    norms32 = norms.astype(np.float32)
    torch_norms = torch.from_numpy(norms32)
    chunk = torch_data[start:end, :]
    chunk_norms = torch_norms[start:end]
    _maybe_limit_torch_threads(worker_target)
    gram = torch.matmul(chunk, torch_data.T)
    dists = chunk_norms[:, None] + torch_norms[None, :] - 2.0 * gram
    dists = dists.float()
    idx = torch.arange(count, device=dists.device)
    idx_full = torch.arange(start, end, device=dists.device)
    dists[idx, idx_full] = float("inf")
    k = max(0, n - f - 1)
    if k <= 0:
        return np.zeros(count, dtype=np.float64)
    nearest = torch.topk(dists, k, dim=1, largest=False).values
    scores = nearest.sum(dim=1)
    return scores.to(dtype=torch.float64).cpu().numpy()


_TORCH_THREAD_TARGET: int | None = None


def _maybe_limit_torch_threads(worker_target: int) -> None:
    global _TORCH_THREAD_TARGET
    if not _HAS_TORCH:
        return
    if worker_target <= 0:
        return
    cpu = os.cpu_count() or 1
    limit = max(1, cpu // worker_target)
    if _TORCH_THREAD_TARGET == limit:
        return
    try:
        torch.set_num_threads(limit)
    except Exception:
        return
    _TORCH_THREAD_TARGET = limit
