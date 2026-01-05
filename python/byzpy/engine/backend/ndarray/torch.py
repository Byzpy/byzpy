from __future__ import annotations

from typing import Any, Sequence

import torch

from .base import _Backend


class _TorchBackend(_Backend[torch.Tensor]):
    name = "torch"

    def asarray(self, x: torch.Tensor, like: torch.Tensor | None = None) -> Any:
        kwargs = {}
        if like is not None:
            if hasattr(like, "dtype"):
                kwargs["dtype"] = like.dtype
            if hasattr(like, "device"):
                kwargs["device"] = like.device
        return torch.as_tensor(x, **kwargs)

    def stack(self, xs: Sequence[torch.Tensor], axis: int = 0) -> torch.Tensor:
        return torch.stack(list(xs), dim=axis)

    def median(self, x: torch.Tensor, axis: int = 0) -> torch.Tensor:
        return torch.median(x, dim=axis).values

    def sort(self, x: torch.Tensor, axis: int = 0) -> torch.Tensor:
        return torch.sort(x, dim=axis).values

    def mean(self, x: torch.Tensor, axis: int = 0) -> torch.Tensor:
        return torch.mean(x, dim=axis)

    def abs(self, x: torch.Tensor) -> torch.Tensor:
        return torch.abs(x)

    def argsort(self, x: torch.Tensor, axis: int = 0) -> torch.Tensor:
        return torch.argsort(x, dim=axis)

    def take_along_axis(
        self, a: torch.Tensor, indices: torch.Tensor, axis: int = 0
    ) -> torch.Tensor:
        if hasattr(torch, "take_along_dim"):
            return torch.take_along_dim(a, indices, dim=axis)
        return torch.gather(a, dim=axis, index=indices)

    def sum(self, x: torch.Tensor, axis: int | tuple[int, ...] | None = None) -> torch.Tensor:
        return torch.sum(x, dim=axis) if axis is not None else torch.sum(x)

    def index_select(self, a: torch.Tensor, axis: int, indices: torch.Tensor) -> torch.Tensor:
        if indices.dtype != torch.long:
            indices = indices.to(torch.long)
        return torch.index_select(a, dim=axis, index=indices)

    def sqrt(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(x)

    def maximum(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.maximum(a, b)

    def reshape(self, x: torch.Tensor, newshape: tuple[int, ...]) -> torch.Tensor:
        return x.reshape(newshape)

    def max(self, x: torch.Tensor, axis: int | tuple[int, ...] | None = None) -> torch.Tensor:
        return torch.amax(x, dim=axis) if axis is not None else torch.amax(x)

    def minimum(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.minimum(a, b)

    def copy(self, x: torch.Tensor) -> torch.Tensor:
        return x.clone()
