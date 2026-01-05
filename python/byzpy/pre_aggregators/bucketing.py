from __future__ import annotations

import random
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..aggregators._chunking import select_adaptive_chunk_size
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


class Bucketing(PreAggregator):
    """
    Bucketing pre-aggregator: group vectors into buckets and average.

    This pre-aggregator randomly permutes input vectors, splits them into
    consecutive buckets of a specified size, and returns the mean of each
    bucket. This reduces the number of vectors while preserving some
    statistical properties.

    Algorithm:
    1. Randomly permute the input vectors (or use provided permutation).
    2. Split into consecutive buckets of size ``bucket_size``.
    3. Return the mean of each bucket.

    Parameters
    ----------
    bucket_size : int
        Number of vectors per bucket. Must be >= 1. The last bucket may be
        smaller if the number of vectors is not divisible by bucket_size.
    feature_chunk_size : int, optional
        Size of feature chunks for parallel processing. Default is 8192.
    perm : Optional[Iterable[int]], optional
        Explicit permutation of indices. If None, a random permutation is
        generated. Must be a permutation of range(n) where n is the number
        of input vectors.
    rng : Optional[random.Random], optional
        Random number generator for shuffling. If None, a new generator is
        created.

    Examples
    --------
    >>> preagg = Bucketing(bucket_size=4)
    >>> vectors = [torch.randn(100) for _ in range(10)]
    >>> result = preagg.pre_aggregate(vectors)
    >>> len(result)  # 10 vectors -> ceil(10/4) = 3 buckets
    3
    >>> assert all(v.shape == (100,) for v in result)

    Notes
    -----
    - Output length is ceil(n / bucket_size) where n is input length.
    - Supports parallel execution via subtasks for large feature dimensions.
    - Time complexity: O(n * d) where n is number of vectors and d is
      dimension. With subtasks: O(n * d / workers).
    - Memory complexity: O(n * d) for stacking vectors.
    """

    name = "pre-agg/bucketing"
    supports_subtasks = True
    max_subtasks_inflight = 0

    def __init__(
        self,
        bucket_size: int,
        *,
        feature_chunk_size: int = 8192,
        perm: Optional[Iterable[int]] = None,
        rng: Optional[random.Random] = None,
    ) -> None:
        if bucket_size < 1:
            raise ValueError("bucket_size must be >= 1")
        self.bucket_size = int(bucket_size)
        if feature_chunk_size <= 0:
            raise ValueError("feature_chunk_size must be > 0")
        self.feature_chunk_size = int(feature_chunk_size)
        self.perm = None if perm is None else [int(i) for i in perm]
        self.rng = rng or random.Random()
        self._active_handle: SharedTensorHandle | None = None
        self._flat_shape: tuple[int, ...] | None = None
        self._bucket_slices: tuple[tuple[int, int], ...] | None = None
        self._bucket_count: int | None = None
        self._like_template: Any | None = None

    def pre_aggregate(self, xs: Sequence[Any]) -> List[Any]:
        if not xs:
            raise ValueError("xs must be a non-empty sequence")

        n = len(xs)
        be = get_backend()
        like = xs[0]

        order = self._resolve_order(n)

        arrs = [be.asarray(xs[i], like=like) for i in order]
        X = be.stack(arrs, axis=0)

        out: List[Any] = []
        for start in range(0, n, self.bucket_size):
            stop = min(start + self.bucket_size, n)
            chunk = X[start:stop]
            mean = be.mean(chunk, axis=0)
            out.append(mean)
        return out

    def create_subtasks(self, inputs, *, context):  # type: ignore[override]
        xs = inputs.get(self.input_key)
        if not isinstance(xs, Sequence) or not xs:
            return []

        n = len(xs)
        order = self._resolve_order(n)

        arrays = [_to_numpy(xs[i]) for i in order]
        stacked = np.stack(arrays, axis=0)
        self._flat_shape = stacked.shape[1:]
        flat = stacked.reshape(stacked.shape[0], -1)
        handle = register_tensor(flat)
        self._active_handle = handle
        self._like_template = xs[0]

        bucket_slices = []
        for start in range(0, n, self.bucket_size):
            stop = min(start + self.bucket_size, n)
            bucket_slices.append((start, stop))
        self._bucket_slices = tuple(bucket_slices)
        bucket_count = len(bucket_slices)
        self._bucket_count = bucket_count

        metadata = getattr(context, "metadata", None) or {}
        pool_size = int(metadata.get("pool_size") or 0)
        bucket_chunk = select_adaptive_chunk_size(
            bucket_count,
            max(1, self.feature_chunk_size),
            pool_size=pool_size,
            allow_small_chunks=True,
        )

        def _iter() -> Iterable[SubTask]:
            chunk_id = 0
            for start in range(0, bucket_count, bucket_chunk):
                end = min(bucket_count, start + bucket_chunk)
                yield SubTask(
                    fn=_bucketing_bucket_chunk,
                    args=(handle, self._bucket_slices[start:end], start),
                    kwargs={},
                    name=f"bucketing_chunk_{chunk_id}",
                )
                chunk_id += 1

        return _iter()

    def reduce_subtasks(self, partials, inputs, *, context):  # type: ignore[override]
        if not partials:
            return super().compute(inputs, context=context)

        if (
            self._active_handle is None
            or self._flat_shape is None
            or self._bucket_slices is None
            or self._bucket_count is None
            or self._like_template is None
        ):
            raise RuntimeError("Bucketing missing reduction state.")

        feature_dim = int(np.prod(self._flat_shape))
        buckets = self._bucket_count
        assembled = np.zeros((buckets, feature_dim), dtype=np.float64)

        for idx, part in enumerate(partials):
            try:
                offset, chunk = part
            except Exception as exc:  # pragma: no cover
                raise ValueError(
                    f"Bucketing received malformed partial at index {idx}: {part!r}"
                ) from exc
            rows = chunk.shape[0]
            assembled[offset : offset + rows, :] = np.asarray(chunk, dtype=np.float64)

        results: List[Any] = []
        for bucket_idx in range(buckets):
            flat_bucket = assembled[bucket_idx]
            reshaped = flat_bucket.reshape(self._flat_shape)
            results.append(_to_like(reshaped, self._like_template))

        cleanup_tensor(self._active_handle)
        self._active_handle = None
        self._flat_shape = None
        self._bucket_slices = None
        self._bucket_count = None
        self._like_template = None

        return results

    def _resolve_order(self, n: int) -> List[int]:
        if self.perm is None:
            order = list(range(n))
            self.rng.shuffle(order)
            return order
        if len(self.perm) != n or sorted(self.perm) != list(range(n)):
            raise ValueError("perm must be a permutation of range(n)")
        return list(self.perm)


def _bucketing_bucket_chunk(
    handle: SharedTensorHandle,
    bucket_slices: Tuple[tuple[int, int], ...],
    offset: int,
) -> tuple[int, np.ndarray]:
    with open_tensor(handle) as flat:
        view = flat[:, :]
        feature_dim = view.shape[1]
        out = np.zeros((len(bucket_slices), feature_dim), dtype=view.dtype)
        for idx, (b_start, b_end) in enumerate(bucket_slices):
            chunk = view[b_start:b_end]
            out[idx, :] = np.mean(chunk, axis=0)
    return offset, out


def _to_numpy(x: Any) -> np.ndarray:
    if _HAS_TORCH and isinstance(x, torch.Tensor):  # type: ignore[arg-type]
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def _to_like(arr: np.ndarray, like: Any) -> Any:
    if _HAS_TORCH and isinstance(like, torch.Tensor):  # type: ignore[arg-type]
        return torch.from_numpy(arr).to(dtype=like.dtype)
    be = get_backend()
    return be.asarray(arr, like=like)
