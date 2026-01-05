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


def _copy_chunk(src: SharedTensorHandle, dst: SharedTensorHandle, start: int, end: int):
    with open_tensor(src) as s, open_tensor(dst) as d:
        d[start:end] = s[start:end]
    return start, None


class MimicAttack(Attack):
    """
    Mimic an honest worker: return the vector of worker epsilon.
    """

    uses_honest_grads = True
    supports_subtasks = True
    max_subtasks_inflight = 0

    def __init__(self, epsilon: int = 0, *, chunk_size: int = 8192) -> None:
        if epsilon < 0:
            raise ValueError("epsilon must be >= 0")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        self.epsilon = int(epsilon)
        self.chunk_size = int(chunk_size)
        self._src_handle: SharedTensorHandle | None = None
        self._dst_handle: SharedTensorHandle | None = None
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
            raise ValueError("MimicAttack requires honest_grads.")
        if self.epsilon >= len(honest_grads):
            raise ValueError(
                f"epsilon must be < number of honest_grads (got epsilon={self.epsilon}, n={len(honest_grads)})"
            )
        g = honest_grads[self.epsilon]
        if isinstance(g, (np.ndarray, torch.Tensor)):
            return g
        be = get_backend()
        return be.asarray(g, like=g)

    def create_subtasks(self, inputs, *, context):  # type: ignore[override]
        grads = inputs.get("honest_grads")
        if not isinstance(grads, Sequence) or not grads:
            return []
        if self.epsilon >= len(grads):
            raise ValueError(
                f"epsilon must be < number of honest_grads (got epsilon={self.epsilon}, n={len(grads)})"
            )

        flat_shape, flat = flatten_gradients(grads)
        feature_dim = flat.shape[1]
        target = np.array(flat[self.epsilon], copy=True)
        src_handle = register_tensor(target)
        dst_handle = register_tensor(np.zeros_like(target))

        self._src_handle = src_handle
        self._dst_handle = dst_handle
        self._flat_shape = flat_shape
        self._like_template = grads[0]

        metadata = getattr(context, "metadata", None) or {}
        pool_size = int(metadata.get("pool_size") or 0)
        chunk = select_adaptive_chunk_size(
            feature_dim, self.chunk_size, pool_size=pool_size, allow_small_chunks=True
        )

        def _iter() -> Iterable[SubTask]:
            chunk_id = 0
            for start in range(0, feature_dim, chunk):
                end = min(feature_dim, start + chunk)
                yield SubTask(
                    fn=_copy_chunk,
                    args=(src_handle, dst_handle, start, end),
                    kwargs={},
                    name=f"mimic_chunk_{chunk_id}",
                )
                chunk_id += 1

        return _iter()

    def reduce_subtasks(self, partials, inputs, *, context):  # type: ignore[override]
        if not partials:
            return super().compute(inputs, context=context)
        if (
            self._src_handle is None
            or self._dst_handle is None
            or self._flat_shape is None
            or self._like_template is None
        ):
            raise RuntimeError("MimicAttack missing state for reduction.")

        try:
            with open_tensor(self._dst_handle) as dst:
                data = np.array(dst, copy=True)
            reshaped = data.reshape(self._flat_shape)
            return _to_like(reshaped, self._like_template)
        finally:
            cleanup_tensor(self._src_handle)
            cleanup_tensor(self._dst_handle)
            self._src_handle = None
            self._dst_handle = None
            self._flat_shape = None
            self._like_template = None


__all__ = ["MimicAttack"]
