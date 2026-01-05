from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from byzpy.attacks.base import Attack


def _mirror_labels(y: torch.Tensor, K: int) -> torch.Tensor:
    return (K - 1 - y).to(dtype=torch.long, device=y.device)


def _apply_mapping(y: torch.Tensor, mapping: Dict[int, int]) -> torch.Tensor:
    out = y.clone()
    for src, dst in mapping.items():
        m = y == int(src)
        if m.any():
            out[m] = int(dst)
    return out


def _flatten_grads(model: nn.Module) -> torch.Tensor:
    parts = []
    device = next(model.parameters()).device
    for p in model.parameters():
        if not p.requires_grad:
            continue
        g = p.grad if p.grad is not None else torch.zeros_like(p)
        parts.append(g.reshape(-1))
    return torch.cat(parts) if parts else torch.tensor([], device=device)


class LabelFlipAttack(Attack):
    """
    Compute ∇_θ L(model(x), flip(y)) and return the flattened vector.
    """

    uses_model_batch = True

    def __init__(
        self,
        *,
        num_classes: Optional[int] = None,
        mapping: Optional[Dict[int, int]] = None,
        loss_fn: Optional[nn.Module] = None,
        scale: float = 1.0,
    ) -> None:
        if mapping is None and num_classes is None:
            raise ValueError("Provide either `mapping` or `num_classes`.")
        self.num_classes = num_classes
        self.mapping = mapping
        self.loss_fn = loss_fn or nn.CrossEntropyLoss(reduction="mean")
        self.scale = float(scale)

    def apply(
        self,
        *,
        model: Optional[nn.Module] = None,
        x: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        honest_grads: Optional[List[torch.Tensor]] = None,
        base_grad: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if model is None or x is None or y is None:
            raise ValueError("LabelFlipAttack requires model, x, y.")
        y = y.to(dtype=torch.long, device=x.device)
        if self.mapping is not None:
            y_flip = _apply_mapping(y, self.mapping)
        else:
            assert self.num_classes is not None
            y_flip = _mirror_labels(y, int(self.num_classes))

        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()
        model.train(True)
        logits = model(x)
        loss = self.loss_fn(logits, y_flip)
        loss.backward()

        g_vec = _flatten_grads(model).detach()
        g_vec = self.scale * g_vec

        for p in model.parameters():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

        return g_vec
