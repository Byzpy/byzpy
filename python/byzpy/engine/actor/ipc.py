from __future__ import annotations

from typing import Any

import numpy as np

try:  # torch optional for some deployments
    import torch

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    _HAS_TORCH = False

from ..storage.shared_store import SharedTensorHandle, cleanup_tensor, open_tensor, register_tensor

_SHM_MARK = "__BYZ_SHARED_TENSOR__"


def wrap_payload(obj: Any) -> Any:
    if _HAS_TORCH and isinstance(obj, torch.Tensor):  # type: ignore[arg-type]
        return _wrap_numpy(obj.detach().cpu().numpy())
    if isinstance(obj, np.ndarray):
        return _wrap_numpy(obj)
    if isinstance(obj, (list, tuple)):
        typ = type(obj)
        return typ(wrap_payload(x) for x in obj)
    if isinstance(obj, dict):
        return {k: wrap_payload(v) for k, v in obj.items()}
    return obj


def unwrap_payload(obj: Any) -> Any:
    if isinstance(obj, tuple) and len(obj) == 2 and obj[0] == _SHM_MARK:
        return _tensor_from_handle(obj[1])
    if isinstance(obj, list):
        return [unwrap_payload(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(unwrap_payload(x) for x in obj)
    if isinstance(obj, dict):
        return {k: unwrap_payload(v) for k, v in obj.items()}
    return obj


def _wrap_numpy(arr: np.ndarray) -> tuple[str, SharedTensorHandle]:
    handle = register_tensor(np.array(arr, copy=True))
    return (_SHM_MARK, handle)


def _tensor_from_handle(handle: SharedTensorHandle) -> Any:
    with open_tensor(handle) as arr:
        np_arr = np.array(arr, copy=True)
    cleanup_tensor(handle)
    if _HAS_TORCH:
        return torch.from_numpy(np_arr)  # type: ignore[return-value]
    return np_arr


__all__ = ["wrap_payload", "unwrap_payload"]
