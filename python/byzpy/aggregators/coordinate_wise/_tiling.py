from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ...engine.storage.shared_store import SharedTensorHandle, open_tensor

try:  # optional torch dependency
    import torch

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    _HAS_TORCH = False


def flatten_gradients(gradients: Sequence[Any]) -> tuple[tuple[int, ...], np.ndarray]:
    arrays = [_to_numpy(g) for g in gradients]
    stacked = np.stack(arrays, axis=0)
    shape = stacked.shape[1:]
    flat = stacked.reshape(stacked.shape[0], -1)
    return shape, flat


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


__all__ = ["flatten_gradients"]
