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


class CoordinateWiseMedian(Aggregator):
    """
    Coordinate-wise median aggregator.

    This aggregator computes the median independently for each coordinate
    (dimension) across all input gradients. It is robust to up to 50% Byzantine
    nodes when the honest gradients are well-separated.

    The median is computed coordinate-wise, meaning for each dimension d, the
    output is the median of all gradients' d-th coordinate. This makes it
    computationally efficient and suitable for high-dimensional gradients.

    The aggregator supports parallel execution via subtasks, chunking the
    computation across multiple workers for improved performance on large
    gradients.

    Parameters
    ----------
    chunk_size : int, optional
        Size of chunks for parallel processing. Larger values reduce overhead
        but may limit parallelism. Default is 8192.

    Examples
    --------
    >>> aggregator = CoordinateWiseMedian(chunk_size=4096)
    >>> gradients = [torch.randn(1000) for _ in range(10)]
    >>> result = aggregator.aggregate(gradients)
    >>> assert result.shape == (1000,)

    Notes
    -----
    - Robust to up to floor((n-1)/2) Byzantine nodes where n is the total
      number of gradients.
    - Time complexity: O(n * d) where n is number of gradients and d is
      dimension. With subtasks: O(n * d / workers).
    - Memory complexity: O(n * d) for stacking gradients.
    """

    name = "coordinate-wise-median"
    supports_subtasks = True
    max_subtasks_inflight = 0

    def __init__(self, *, chunk_size: int = 8192) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        self.chunk_size = int(chunk_size)
        self._active_handle: SharedTensorHandle | None = None
        self._flat_shape: tuple[int, ...] | None = None

    def aggregate(self, gradients: Sequence[Any]) -> Any:
        """
        Compute coordinate-wise median of gradients.

        Parameters
        ----------
        gradients : Sequence[Any]
            Non-empty sequence of gradient tensors/arrays. All must have the
            same shape and be from the same backend (NumPy, PyTorch, etc.).

        Returns
        -------
        Any
            Aggregated gradient tensor with the same shape and backend as
            inputs. Each coordinate is the median of that coordinate across
            all input gradients.

        Raises
        ------
        ValueError
            If gradients sequence is empty.
        """
        if not gradients:
            raise ValueError("gradients must be a non-empty sequence")

        be = get_backend()
        like = gradients[0]
        arrs = [be.asarray(g, like=like) for g in gradients]
        stacked = be.stack(arrs, axis=0)  # (n, ...)
        return be.median(stacked, axis=0)

    def create_subtasks(self, inputs, *, context):  # type: ignore[override]
        gradients = inputs.get(self.input_key)
        if not isinstance(gradients, Sequence) or not gradients:
            return []

        flat_shape, flat = flatten_gradients(gradients)
        self._flat_shape = flat_shape
        handle = register_tensor(flat)
        self._active_handle = handle
        features = flat.shape[1]
        metadata = getattr(context, "metadata", None) or {}
        pool_size = int(metadata.get("pool_size") or 0)
        chunk = select_adaptive_chunk_size(features, self.chunk_size, pool_size=pool_size)

        def _iter_subtasks() -> Iterable[SubTask]:
            chunk_id = 0
            for start in range(0, features, chunk):
                end = min(features, start + chunk)
                yield SubTask(
                    fn=_median_chunk,
                    args=(handle, start, end),
                    kwargs={},
                    name=f"median_chunk_{chunk_id}",
                )
                chunk_id += 1

        return _iter_subtasks()

    def reduce_subtasks(self, partials, inputs, *, context):  # type: ignore[override]
        if not partials:
            return super().compute(inputs, context=context)

        like = inputs[self.input_key][0]
        if self._flat_shape is None:
            raise RuntimeError("CoordinateWiseMedian reduce_subtasks missing shape state.")
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
            self._flat_shape = None


def _median_chunk(handle: SharedTensorHandle, start: int, end: int) -> tuple[int, np.ndarray]:
    with open_tensor(handle) as flat:
        view = np.array(flat, copy=False)
        chunk = view[:, start:end]
        chunk_copy = np.array(chunk, copy=True)
        n = chunk_copy.shape[0]
        mid = n // 2
        if n % 2 == 1:
            med = np.partition(chunk_copy, mid, axis=0)[mid, :]
        else:
            med = np.partition(chunk_copy, mid - 1, axis=0)[mid - 1, :]
    return start, med


def _to_like(arr: np.ndarray, like: Any) -> Any:
    if _HAS_TORCH and isinstance(like, torch.Tensor):  # type: ignore[arg-type]
        return torch.from_numpy(arr).to(dtype=like.dtype)
    be = get_backend()
    return be.asarray(arr, like=like)
