from __future__ import annotations

from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from byzpy.aggregators._chunking import select_adaptive_chunk_size
from byzpy.aggregators.coordinate_wise._tiling import flatten_gradients
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


class EmpireAttack(Attack):
    """
    Empire attack: scaled mean of honest gradients.

    This attack computes the mean of honest gradients and scales it by a
    factor (typically negative). It requires access to honest gradients from
    other nodes.

    The attack strategy is: g_malicious = scale * mean(honest_grads)

    Parameters
    ----------
    scale : float, optional
        Scaling factor for the mean. Default is -1.0 (inverted mean).
        Negative values create adversarial gradients that point in the
        opposite direction of the honest consensus.
    chunk_size : int, optional
        Size of chunks for parallel mean computation. Default is 8.

    Examples
    --------
    >>> attack = EmpireAttack(scale=-1.0)
    >>> honest_grads = [torch.randn(100) for _ in range(5)]
    >>> malicious = attack.apply(honest_grads=honest_grads)
    >>> assert malicious.shape == (100,)
    >>> # malicious is approximately -mean(honest_grads)

    Notes
    -----
    - Requires ``uses_honest_grads=True`` (needs access to honest gradients).
    - Supports parallel execution via subtasks for large gradients.
    - Time complexity: O(n * d) where n is number of honest gradients and
      d is dimension. With subtasks: O(n * d / workers).
    - Memory complexity: O(n * d) for stacking gradients.

    References
    ----------
    .. [1] Baruch, M., Baruch, G., & Goldberg, Y. (2019). A little is enough:
       Circumventing defenses for distributed learning. NeurIPS.
    """

    uses_honest_grads = True
    supports_subtasks = True

    def __init__(self, scale: float = -1.0, *, chunk_size: int = 8) -> None:
        self.scale = float(scale)
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        self.chunk_size = int(chunk_size)
        self._active_handle: SharedTensorHandle | None = None
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
            raise ValueError("EmpireAttack requires honest_grads.")
        be = get_backend()
        like = honest_grads[0]
        X = be.stack([be.asarray(g, like=like) for g in honest_grads], axis=0)
        mu = be.mean(X, axis=0)
        out = self.scale * mu
        return be.copy(out)

    def create_subtasks(self, inputs, *, context: OpContext):  # type: ignore[override]
        grads_list = self._require_honest_grads(inputs)
        if not grads_list:
            return []

        reference = grads_list[0]
        flat_shape, flat = flatten_gradients(grads_list)
        self._flat_shape = flat_shape
        self._like_template = reference
        handle = register_tensor(np.asarray(flat, dtype=np.float64))
        self._active_handle = handle
        grads = flat.shape[0]
        metadata = getattr(context, "metadata", None) or {}
        pool_size = int(metadata.get("pool_size") or 0)
        chunk = select_adaptive_chunk_size(grads, self.chunk_size, pool_size=pool_size)

        def _iter() -> Iterable[SubTask]:
            chunk_id = 0
            for start in range(0, grads, chunk):
                end = min(grads, start + chunk)
                yield SubTask(
                    fn=_empire_partial_sum,
                    args=(handle, start, end),
                    kwargs={},
                    name=f"empire_chunk_{chunk_id}",
                )
                chunk_id += 1

        return _iter()

    def reduce_subtasks(self, partials, inputs, *, context: OpContext):  # type: ignore[override]
        if not partials:
            return super().compute(inputs, context=context)

        if self._active_handle is None or self._flat_shape is None or self._like_template is None:
            raise RuntimeError("EmpireAttack missing handle state for reduction.")

        total = None
        count = 0

        for idx, item in enumerate(partials):
            try:
                part_sum, part_count = item
            except Exception as exc:  # pragma: no cover
                raise ValueError(
                    f"EmpireAttack received malformed partial at index {idx}: {item!r}"
                ) from exc
            if part_count <= 0:
                continue
            part_arr = np.asarray(part_sum, dtype=np.float64)
            if total is None:
                total = np.array(part_arr, copy=True)
            else:
                total += part_arr
            count += int(part_count)

        if count == 0:
            raise ValueError("EmpireAttack received zero total count across subtasks.")

        try:
            mean = total / float(count)
            reshaped = mean.reshape(self._flat_shape)
            scaled = float(self.scale) * reshaped
            return _to_like(scaled, self._like_template)
        finally:
            cleanup_tensor(self._active_handle)
            self._active_handle = None
            self._flat_shape = None
            self._like_template = None

    @staticmethod
    def _require_honest_grads(inputs: Mapping[str, Any]) -> List[torch.Tensor]:
        grads = inputs.get("honest_grads")
        if grads is None:
            raise KeyError("EmpireAttack requires 'honest_grads'")
        return list(grads)


def _empire_partial_sum(handle: SharedTensorHandle, start: int, end: int) -> Tuple[np.ndarray, int]:
    if start >= end:
        raise ValueError("EmpireAttack subtasks require at least one gradient.")
    with open_tensor(handle) as flat:
        chunk = flat[start:end]
        return np.sum(chunk, axis=0), int(chunk.shape[0])


def _to_like(arr: np.ndarray, like: Any) -> Any:
    if isinstance(like, torch.Tensor):
        return torch.from_numpy(arr).to(device=like.device, dtype=like.dtype)
    be = get_backend()
    return be.asarray(arr, like=like)


__all__ = ["EmpireAttack"]
