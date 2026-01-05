from __future__ import annotations

from typing import Any, Iterable, List, Optional, Sequence

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


def _to_like(arr: np.ndarray, like: Any) -> Any:
    if isinstance(like, torch.Tensor):
        return torch.from_numpy(arr).to(dtype=like.dtype, device=like.device)
    be = get_backend()
    return be.asarray(arr, like=like)


def _make_rng(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(seed)


def _noop_chunk(handle: SharedTensorHandle, start: int, end: int) -> tuple[int, None]:
    # Random vector is pre-generated; chunk tasks are no-ops that keep the pipeline uniform.
    return start, None


class GaussianAttack(Attack):
    """
    Sample each coordinate independently from N(mu, sigma^2).
    """

    uses_honest_grads = True
    supports_subtasks = True
    max_subtasks_inflight = 0

    def __init__(
        self,
        mu: float = 0.0,
        sigma: float = 1.0,
        *,
        seed: int | None = None,
        chunk_size: int = 8192,
    ) -> None:
        self.mu = float(mu)
        if sigma < 0:
            raise ValueError("sigma must be >= 0")
        self.sigma = float(sigma)
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        self.chunk_size = int(chunk_size)
        self.seed = seed
        self._handle: SharedTensorHandle | None = None
        self._flat_shape: tuple[int, ...] | None = None
        self._like_template: Any | None = None

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
            raise ValueError("GaussianAttack requires honest_grads.")
        flat_shape, flat = flatten_gradients(honest_grads)
        feature_dim = flat.shape[1]
        rng = _make_rng(self.seed)
        sampled = rng.normal(loc=self.mu, scale=self.sigma, size=feature_dim)
        reshaped = sampled.reshape(flat_shape)
        return _to_like(reshaped, honest_grads[0])

    def create_subtasks(self, inputs, *, context):  # type: ignore[override]
        grads = inputs.get("honest_grads")
        if not isinstance(grads, Sequence) or not grads:
            return []

        flat_shape, flat = flatten_gradients(grads)
        self._flat_shape = flat_shape
        self._like_template = grads[0]
        feature_dim = flat.shape[1]

        rng = _make_rng(self.seed)
        buffer = rng.normal(loc=self.mu, scale=self.sigma, size=feature_dim).astype(
            np.float64, copy=False
        )
        handle = register_tensor(buffer)
        self._handle = handle

        metadata = getattr(context, "metadata", None) or {}
        pool_size = int(metadata.get("pool_size") or 0)
        chunk = select_adaptive_chunk_size(
            feature_dim, self.chunk_size, pool_size=pool_size, allow_small_chunks=True
        )
        starts = list(range(0, feature_dim, chunk))

        def _iter() -> Iterable[SubTask]:
            for idx, start in enumerate(starts):
                end = min(feature_dim, start + chunk)
                yield SubTask(
                    fn=_noop_chunk,
                    args=(handle, start, end),
                    kwargs={},
                    name=f"gaussian_chunk_{idx}",
                )

        return _iter()

    def reduce_subtasks(self, partials, inputs, *, context):  # type: ignore[override]
        if not partials:
            return super().compute(inputs, context=context)
        if self._handle is None or self._flat_shape is None or self._like_template is None:
            raise RuntimeError("GaussianAttack missing state for reduction.")

        try:
            with open_tensor(self._handle) as flat:
                data = np.array(flat, copy=True)
            reshaped = data.reshape(self._flat_shape)
            return _to_like(reshaped, self._like_template)
        finally:
            cleanup_tensor(self._handle)
            self._handle = None
            self._flat_shape = None
            self._like_template = None


__all__ = ["GaussianAttack"]
