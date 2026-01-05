from __future__ import annotations

from typing import Any, Iterable, List, Optional

import numpy as np
import torch
import torch.nn as nn

from byzpy.aggregators._chunking import select_adaptive_chunk_size
from byzpy.attacks.base import Attack
from byzpy.configs.backend import get_backend
from byzpy.engine.graph.operator import OpContext
from byzpy.engine.graph.subtask import SubTask
from byzpy.engine.storage.shared_store import (
    SharedTensorHandle,
    cleanup_tensor,
    open_tensor,
    register_tensor,
)


class SignFlipAttack(Attack):
    """g_mal = scale * base_grad (default: sign flip with scale=-1)."""

    uses_base_grad = True
    supports_subtasks = True
    max_subtasks_inflight = 0

    def __init__(self, scale: float = -1.0, *, chunk_size: int = 8192) -> None:
        self.scale = float(scale)
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        self.chunk_size = int(chunk_size)
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
        if base_grad is None:
            raise ValueError("SignFlipAttack requires base_grad.")
        be = get_backend()
        g = be.asarray(base_grad, like=base_grad)
        out = self.scale * g
        return be.copy(out)

    def create_subtasks(self, inputs, *, context: OpContext):  # type: ignore[override]
        base_grad = inputs.get("base_grad")
        if base_grad is None:
            return []

        flat_shape, flat = _flatten_single_gradient(base_grad)
        self._flat_shape = flat_shape
        self._like_template = base_grad
        handle = register_tensor(flat)
        self._handle = handle
        total = flat.shape[0]
        metadata = getattr(context, "metadata", None) or {}
        pool_size = int(metadata.get("pool_size") or 0)
        chunk = select_adaptive_chunk_size(
            total, self.chunk_size, pool_size=pool_size, allow_small_chunks=True
        )

        def _iter() -> Iterable[SubTask]:
            chunk_id = 0
            for start in range(0, total, chunk):
                end = min(total, start + chunk)
                yield SubTask(
                    fn=_signflip_chunk,
                    args=(handle, start, end, self.scale),
                    kwargs={},
                    name=f"signflip_chunk_{chunk_id}",
                )
                chunk_id += 1

        return _iter()

    def reduce_subtasks(self, partials, inputs, *, context: OpContext):  # type: ignore[override]
        if not partials:
            return super().compute(inputs, context=context)

        if self._handle is None or self._flat_shape is None or self._like_template is None:
            raise RuntimeError("SignFlipAttack missing chunk state.")

        total = int(np.prod(self._flat_shape))
        assembled = np.zeros(total, dtype=np.float64)
        for idx, part in enumerate(partials):
            try:
                start, chunk = part
            except Exception as exc:  # pragma: no cover
                raise ValueError(
                    f"SignFlipAttack received malformed partial at index {idx}: {part!r}"
                ) from exc
            end = start + chunk.shape[0]
            assembled[start:end] = chunk

        try:
            reshaped = assembled.reshape(self._flat_shape)
            return _to_like(reshaped, self._like_template)
        finally:
            cleanup_tensor(self._handle)
            self._handle = None
            self._flat_shape = None
            self._like_template = None


def _signflip_chunk(
    handle: SharedTensorHandle, start: int, end: int, scale: float
) -> tuple[int, np.ndarray]:
    with open_tensor(handle) as flat:
        view = np.array(flat, copy=False)
        chunk = np.array(view[start:end], copy=True)
        chunk *= scale
        return start, chunk


def _flatten_single_gradient(grad: Any) -> tuple[tuple[int, ...], np.ndarray]:
    if isinstance(grad, SharedTensorHandle):
        with open_tensor(grad) as arr:
            np_arr = np.array(arr, copy=True)
    elif isinstance(grad, dict) and {"name", "shape", "dtype"} <= grad.keys():
        handle = SharedTensorHandle(**grad)
        with open_tensor(handle) as arr:
            np_arr = np.array(arr, copy=True)
    elif isinstance(grad, torch.Tensor):
        np_arr = grad.detach().cpu().numpy()
    else:
        np_arr = np.asarray(grad)
    shape = np_arr.shape
    flat = np_arr.reshape(-1)
    return shape, flat


def _to_like(arr: np.ndarray, like: Any) -> Any:
    if isinstance(like, torch.Tensor):
        return torch.from_numpy(arr).to(device=like.device, dtype=like.dtype)
    be = get_backend()
    return be.asarray(arr, like=like)
