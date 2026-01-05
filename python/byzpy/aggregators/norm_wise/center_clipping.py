from __future__ import annotations

import math
from typing import Any, Sequence

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

try:  # optional torch dependency for conversion
    import torch

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    _HAS_TORCH = False


class CenteredClipping(Aggregator):
    """
    Centered Clipping (CC) aggregator.

    An iterative aggregation method that clips gradients based on their
    distance from a running center estimate. The center is updated iteratively
    by averaging clipped gradients.

    Algorithm:
        For iteration m = 1 to M:
            v_m = v_{m-1} + (1/n) * sum_i (x_i - v_{m-1}) * min(1, c_tau / ||x_i - v_{m-1}||)

    Parameters
    ----------
    c_tau : float
        Clipping radius (non-negative). Gradients beyond this distance from
        the center are clipped to this radius.
    M : int, optional
        Number of iterations. Default is 10. More iterations improve
        convergence but increase computation time.
    eps : float, optional
        Small constant to avoid division by zero. Default is 1e-12.
    init : str, optional
        Initialization strategy for v0. Options: "mean", "median", or "zero".
        Default is "mean".
    chunk_size : int, optional
        Size of chunks for parallel processing. Default is 32.

    Examples
    --------
    >>> aggregator = CenteredClipping(c_tau=1.0, M=10, init="mean")
    >>> gradients = [torch.randn(100) for _ in range(10)]
    >>> result = aggregator.aggregate(gradients)
    >>> assert result.shape == (100,)

    Notes
    -----
    - Robust to Byzantine nodes when honest gradients are concentrated.
    - Time complexity: O(M * n * d) where M is iterations, n is number of
      gradients, d is dimension. With subtasks: O(M * n * d / workers).
    - Memory complexity: O(n * d) for gradients, O(d) for center estimate.
    - Supports barriered subtasks for parallel iteration execution.

    References
    ----------
    .. [1] Karimireddy, S. P., He, L., & Jaggi, M. (2021). Learning from
       history for Byzantine robust optimization. ICML.
    """

    name = "centered-clipping"
    supports_barriered_subtasks = True
    max_subtasks_inflight = 0

    def __init__(
        self,
        *,
        c_tau: float,
        M: int = 10,
        eps: float = 1e-12,
        init: str = "mean",
        chunk_size: int = 32,
    ) -> None:
        if c_tau < 0:
            raise ValueError("c_tau must be >= 0")
        if M <= 0:
            raise ValueError("M must be >= 1")
        if eps <= 0:
            raise ValueError("eps must be > 0")
        if init not in {"mean", "median", "zero"}:
            raise ValueError("init must be one of {'mean','median','zero'}")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        self.c_tau = float(c_tau)
        self.M = int(M)
        self.eps = float(eps)
        self.init = init
        self.chunk_size = int(chunk_size)

    def aggregate(self, gradients: Sequence[Any]) -> Any:
        """
        Run M iterations of the centered clipping update.

        Parameters
        ----------
        gradients : Sequence[Any]
            Non-empty sequence of gradient tensors. All must have the same
            shape and backend.

        Returns
        -------
        Any
            Aggregated gradient tensor (the final center estimate v_M) with
            the same shape and backend as inputs.

        Raises
        ------
        ValueError
            If gradients sequence is empty.
        """
        if not gradients:
            raise ValueError("gradients must be a non-empty sequence")

        be = get_backend()
        like = gradients[0]
        X = be.stack([be.asarray(g, like=like) for g in gradients], axis=0)  # (n, ...)

        # init v0
        if self.init == "mean":
            v = be.mean(X, axis=0)
        elif self.init == "median":
            v = be.median(X, axis=0)
        else:  # "zero"
            v = X[0] * 0

        n = len(gradients)
        axes_feat = tuple(range(1, X.ndim))

        for _ in range(self.M):
            diff = X - v  # (n, ...)
            dist = be.sqrt(be.sum(diff * diff, axis=axes_feat))  # (n,)
            dist = be.maximum(dist, dist * 0 + self.eps)  # clamp

            alpha = be.minimum(dist * 0 + 1.0, (dist * 0 + self.c_tau) / dist)  # (n,)
            alpha_b = be.reshape(alpha, (alpha.shape[0],) + (1,) * (X.ndim - 1))

            v = v + (1.0 / n) * be.sum(diff * alpha_b, axis=0)

        return v

    async def run_barriered_subtasks(self, inputs, *, context, pool):  # type: ignore[override]
        gradients = inputs.get(self.input_key)
        if not isinstance(gradients, Sequence) or not gradients:
            raise ValueError("gradients must be a non-empty sequence")

        flat_shape, flat = flatten_gradients(gradients)
        like = gradients[0]
        flat_np = np.asarray(flat)
        if not flat_np.flags.c_contiguous:
            flat_np = np.ascontiguousarray(flat_np)
        n = flat_np.shape[0]

        grad_handle = register_tensor(flat_np)
        vector = _cc_init_vector(flat_np, init=self.init)
        vec_handle = register_tensor(vector)
        pool_size = getattr(pool, "size", 1)
        chunk = self._select_chunk_size(n, pool_size)
        chunk_count = max(1, math.ceil(n / chunk))
        contrib_template = np.zeros((chunk_count, flat_np.shape[1]), dtype=vector.dtype)
        contrib_handle = register_tensor(contrib_template)

        try:
            _write_handle(vec_handle, vector)
            for iteration in range(self.M):
                subtasks = []
                chunk_id = 0
                for start in range(0, n, chunk):
                    end = min(n, start + chunk)
                    subtasks.append(
                        SubTask(
                            fn=_centered_clipping_chunk,
                            args=(
                                grad_handle,
                                vec_handle,
                                contrib_handle,
                                chunk_id,
                                start,
                                end,
                                self.c_tau,
                                self.eps,
                            ),
                            kwargs={},
                            name=f"cc_chunk_{iteration}_{chunk_id}",
                        )
                    )
                    chunk_id += 1

                partials = await self._run_subtasks(pool, subtasks, self.max_subtasks_inflight, context)  # type: ignore[arg-type]
                if not partials:
                    break

                with open_tensor(contrib_handle) as contribs:
                    total = np.sum(contribs[:chunk_count], axis=0)
                vector = vector + (1.0 / n) * total
                _write_handle(vec_handle, vector)

            reshaped = vector.reshape(flat_shape)
            return _to_like(reshaped, like)
        finally:
            cleanup_tensor(grad_handle)
            cleanup_tensor(vec_handle)
            cleanup_tensor(contrib_handle)

    def _select_chunk_size(self, n: int, pool_size: int) -> int:
        return select_adaptive_chunk_size(n, self.chunk_size, pool_size=pool_size)


def _cc_init_vector(flat: np.ndarray, init: str) -> np.ndarray:
    if init == "mean":
        return flat.mean(axis=0)
    if init == "median":
        return np.median(flat, axis=0)
    return np.zeros(flat.shape[1], dtype=flat.dtype)


def _centered_clipping_chunk(
    grad_handle: SharedTensorHandle,
    vec_handle: SharedTensorHandle,
    contrib_handle: SharedTensorHandle,
    slot: int,
    start: int,
    end: int,
    c_tau: float,
    eps: float,
) -> int:
    with (
        open_tensor(grad_handle) as flat,
        open_tensor(vec_handle) as vec,
        open_tensor(contrib_handle) as contribs,
    ):
        data = np.array(flat, copy=False)
        vector = np.array(vec, copy=False)
        rows = data[start:end]
        diff = rows - vector
        dist = np.linalg.norm(diff, axis=1)
        dist = np.maximum(dist, eps)
        alpha = np.minimum(1.0, c_tau / dist)
        contrib = (alpha[:, None] * diff).sum(axis=0)
        contribs[slot, :] = contrib
    return slot


def _write_handle(handle: SharedTensorHandle, values: np.ndarray) -> None:
    with open_tensor(handle) as arr:
        arr[:] = values


def _to_like(arr: np.ndarray, like: Any) -> Any:
    if _HAS_TORCH and isinstance(like, torch.Tensor):  # type: ignore[arg-type]
        return torch.from_numpy(arr).to(dtype=like.dtype)
    be = get_backend()
    return be.asarray(arr, like=like)
