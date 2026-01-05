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

try:
    import torch

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    _HAS_TORCH = False


class CoordinateWiseTrimmedMean(Aggregator):
    """
    Coordinate-wise Trimmed Mean (CwTM) aggregator.

    This aggregator computes a trimmed mean independently for each coordinate.
    For each dimension, it sorts the values, removes the f smallest and f
    largest values, and averages the remaining (n - 2f) values.

    This makes it robust to up to f Byzantine nodes on each coordinate when
    honest gradients are well-separated from outliers.

    Parameters
    ----------
    f : int
        Number of extreme values to trim from each end. Must satisfy
        0 <= 2f < n where n is the number of gradients.
    chunk_size : int, optional
        Size of chunks for parallel processing. Default is 4096.

    Examples
    --------
    >>> aggregator = CoordinateWiseTrimmedMean(f=2, chunk_size=2048)
    >>> gradients = [torch.randn(1000) for _ in range(10)]
    >>> result = aggregator.aggregate(gradients)
    >>> assert result.shape == (1000,)

    Notes
    -----
    - Robust to up to f Byzantine nodes per coordinate.
    - Time complexity: O(n * d * log(n)) for sorting, O(n * d) for averaging.
      With subtasks: O(n * d * log(n) / workers).
    - Memory complexity: O(n * d) for stacking and sorting gradients.

    References
    ----------
    .. [1] Yin, D., Chen, Y., Kannan, R., & Bartlett, P. (2018). Byzantine-robust
       distributed learning: Towards optimal statistical rates. ICML.
    """

    name = "coordinate-wise-trimmed-mean"
    supports_subtasks = True
    max_subtasks_inflight = 0

    def __init__(self, f: int, *, chunk_size: int = 4096) -> None:
        if f < 0:
            raise ValueError("f must be >= 0")
        self.f = f
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        self.chunk_size = int(chunk_size)
        self._active_handle: SharedTensorHandle | None = None
        self._flat_shape: tuple[int, ...] | None = None

    def aggregate(self, gradients: Sequence[Any]) -> Any:
        """
        Compute coordinate-wise trimmed mean of gradients.

        Parameters
        ----------
        gradients : Sequence[Any]
            Non-empty sequence of gradient tensors. All must have the same
            shape and backend. The sequence length n must satisfy 2f < n.

        Returns
        -------
        Any
            Aggregated gradient tensor with the same shape and backend as
            inputs. Each coordinate is the mean of the middle (n-2f) values
            after sorting.

        Raises
        ------
        ValueError
            If gradients is empty or if 2f >= n.
        """
        if not gradients:
            raise ValueError("gradients must be a non-empty sequence")

        n = len(gradients)
        f = self.f
        if f < 0 or 2 * f >= n:
            raise ValueError(f"trim parameter f must satisfy 0 <= 2f < n (got n={n}, f={f})")

        be = get_backend()
        like = gradients[0]
        X = be.stack([be.asarray(g, like=like) for g in gradients], axis=0)  # (n, ...)
        sorted_vals = be.sort(X, axis=0)  # (n, ...)
        inner = sorted_vals[f : n - f]  # (n-2f, ...)
        return be.mean(inner, axis=0)

    def create_subtasks(self, inputs, *, context):  # type: ignore[override]
        gradients = inputs.get(self.input_key)
        if not isinstance(gradients, Sequence) or not gradients:
            return []

        n = len(gradients)
        f = self.f
        if f < 0 or 2 * f >= n:
            raise ValueError(f"trim parameter f must satisfy 0 <= 2f < n (got n={n}, f={f})")

        flat_shape, flat = _flatten_gradients(gradients)
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
                    fn=_cw_trimmed_chunk,
                    args=(handle, n, self.f, start, end, gradients[0]),
                    kwargs={},
                    name=f"cwtm_chunk_{chunk_id}",
                )
                chunk_id += 1

        return _iter_subtasks()

    def reduce_subtasks(self, partials, inputs, *, context):  # type: ignore[override]
        if not partials:
            return super().compute(inputs, context=context)

        like = inputs[self.input_key][0]
        if self._flat_shape is None:
            raise RuntimeError("TrimmedMean reduce_subtasks missing shape state.")
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


def _cw_trimmed_chunk(
    handle: SharedTensorHandle, n: int, f: int, start: int, end: int, like_template
) -> tuple[int, np.ndarray]:
    with open_tensor(handle) as flat:
        view = flat[:, start:end]
        sorted_vals = np.sort(view, axis=0)
        inner = sorted_vals[f : n - f]
        mean = np.mean(inner, axis=0)
    return start, mean


def _flatten_gradients(gradients: Sequence[Any]) -> tuple[tuple[int, ...], np.ndarray]:
    arrays = [_to_numpy(g) for g in gradients]
    stacked = np.stack(arrays, axis=0)
    grad_shape = stacked.shape[1:]
    flat = stacked.reshape(stacked.shape[0], -1)
    return grad_shape, flat


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
        return torch.from_numpy(arr).to(dtype=like.dtype)
    be = get_backend()
    return be.asarray(arr, like=like)
