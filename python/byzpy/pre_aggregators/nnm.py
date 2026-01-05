from __future__ import annotations

from typing import Any, Iterable, List, Sequence

import numpy as np
import torch

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


class NearestNeighborMixing(PreAggregator):
    """
    NNM (Nearest-Neighbor Mixing):
      For each i, replace x_i with the average of its k = n - f nearest neighbors
      (including itself) by Euclidean distance.

    Constructor args:
        f: int with 0 <= f < n (validated at call time since n is unknown at init)

    Call:
        pre_aggregate(xs) -> List[y_1, ..., y_n] (same length as xs)
    """

    name = "pre-agg/nnm"
    supports_subtasks = True
    max_subtasks_inflight = 0

    def __init__(self, f: int, *, feature_chunk_size: int = 8192) -> None:
        if f < 0:
            raise ValueError("f must be >= 0")
        self.f = int(f)
        if feature_chunk_size <= 0:
            raise ValueError("feature_chunk_size must be > 0")
        self.feature_chunk_size = int(feature_chunk_size)
        self._active_handle: SharedTensorHandle | None = None
        self._flat_shape: tuple[int, ...] | None = None
        self._like_template: Any | None = None
        self._n: int | None = None

    def pre_aggregate(self, xs: Sequence[Any]) -> List[Any]:
        if not xs:
            raise ValueError("xs must be a non-empty sequence")
        n = len(xs)
        if not (0 <= self.f < n):
            raise ValueError("f must satisfy 0 <= f < n")

        k = n - self.f
        first = xs[0]
        if isinstance(first, torch.Tensor):
            return self._pre_aggregate_torch(xs, k=k)
        return self._pre_aggregate_numpy(xs, k=k)

    def _pre_aggregate_numpy(self, xs: Sequence[Any], *, k: int) -> List[Any]:
        be = get_backend()
        like = xs[0]
        X = be.stack([be.asarray(x, like=like) for x in xs], axis=0)  # (n, ...)
        flat = X.reshape(len(xs), -1)
        dtype = np.result_type(flat.dtype, np.float32)
        flat = flat.astype(dtype, copy=False)
        norms = np.sum(flat * flat, axis=1, keepdims=True)
        D2 = norms + norms.T - 2.0 * (flat @ flat.T)

        idx = np.argpartition(D2, kth=k - 1, axis=1)[:, :k]
        mask = np.zeros(D2.shape, dtype=dtype)
        rows = np.arange(D2.shape[0])[:, None]
        mask[rows, idx] = 1.0
        sums = mask @ flat
        means = sums / float(k)
        reshaped = means.reshape((len(xs),) + X.shape[1:])
        return [_to_like(reshaped[i], like) for i in range(len(xs))]

    def _pre_aggregate_torch(self, xs: Sequence[torch.Tensor], *, k: int) -> List[Any]:
        X = torch.stack(xs, dim=0)
        flat = X.reshape(len(xs), -1)
        work_dtype = torch.float32 if flat.dtype in (torch.float16, torch.bfloat16) else flat.dtype
        flat = flat.to(dtype=work_dtype)
        norms = torch.sum(flat * flat, dim=1, keepdim=True)
        D2 = norms + norms.T - 2.0 * (flat @ flat.T)
        idx = torch.topk(D2, k=k, largest=False).indices  # (n, k)
        mask = torch.zeros_like(D2)
        mask.scatter_(1, idx, 1.0)
        sums = mask @ flat
        means = sums / float(k)
        if means.dtype != X.dtype:
            means = means.to(dtype=X.dtype)
        reshaped = means.view((len(xs),) + X.shape[1:])
        return [reshaped[i] for i in range(len(xs))]

    def create_subtasks(self, inputs, *, context):  # type: ignore[override]
        xs = inputs.get(self.input_key)
        if not isinstance(xs, Sequence) or not xs:
            return []

        n = len(xs)
        if self.f >= n:
            raise ValueError("f must satisfy 0 <= f < n")

        flat_shape, flat = flatten_gradients(xs)
        # Keep computation in the input dtype to avoid expensive float64 upcasts.
        data = flat.astype(np.result_type(flat.dtype, np.float32), copy=False)
        handle = register_tensor(data)
        self._active_handle = handle
        self._flat_shape = flat_shape
        self._like_template = xs[0]
        self._n = n

        dim = data.shape[1]
        metadata = getattr(context, "metadata", None) or {}
        pool_size = int(metadata.get("pool_size") or 0)
        chunk = select_adaptive_chunk_size(dim, self.feature_chunk_size, pool_size=pool_size)

        def _iter() -> Iterable[SubTask]:
            chunk_id = 0
            for start in range(0, dim, chunk):
                end = min(dim, start + chunk)
                yield SubTask(
                    fn=_nnm_partial_distances,
                    args=(handle, start, end),
                    kwargs={},
                    name=f"nnm_chunk_{chunk_id}",
                )
                chunk_id += 1

        return _iter()

    def reduce_subtasks(self, partials, inputs, *, context):  # type: ignore[override]
        if not partials:
            return super().compute(inputs, context=context)

        if (
            self._active_handle is None
            or self._flat_shape is None
            or self._like_template is None
            or self._n is None
        ):
            raise RuntimeError("NearestNeighborMixing missing chunk state.")

        n = self._n
        dtype = np.result_type(np.asarray(inputs[self.input_key][0]).dtype, np.float32)
        D2 = np.zeros((n, n), dtype=dtype)
        for idx, item in enumerate(partials):
            try:
                contrib = item
            except Exception as exc:  # pragma: no cover
                raise ValueError(
                    f"NNM received malformed partial at index {idx}: {item!r}"
                ) from exc
            D2 += np.asarray(contrib, dtype=dtype)

        try:
            with open_tensor(self._active_handle) as flat:
                data = np.array(flat, copy=False)
                k = n - self.f
                if k <= 0:
                    raise ValueError("k must be > 0")

                results: List[Any] = []
                for i in range(n):
                    idx = np.argpartition(D2[i], kth=k - 1)[:k]
                    mean = data[idx].mean(axis=0, dtype=data.dtype)
                    reshaped = mean.reshape(self._flat_shape)
                    results.append(_to_like(reshaped, self._like_template))
                return results
        finally:
            cleanup_tensor(self._active_handle)
            self._active_handle = None
            self._flat_shape = None
            self._like_template = None
            self._n = None


def _nnm_partial_distances(handle: SharedTensorHandle, start: int, end: int) -> np.ndarray:
    with open_tensor(handle) as flat:
        chunk = flat[:, start:end]
        norms = np.sum(chunk * chunk, axis=1, keepdims=True)
        contrib = norms + norms.T - 2.0 * (chunk @ chunk.T)
    return contrib


def _to_like(arr: np.ndarray, like: Any) -> Any:
    if isinstance(like, torch.Tensor):
        return torch.from_numpy(arr).to(device=like.device, dtype=like.dtype)
    be = get_backend()
    return be.asarray(arr, like=like)
