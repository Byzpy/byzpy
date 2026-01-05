from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from ...aggregators import Aggregator
from ...attacks import Attack
from ...pre_aggregators import PreAggregator

Tensor = torch.Tensor


def _flatten_params(model: nn.Module) -> Tensor:
    return torch.cat([p.detach().view(-1) for p in model.parameters()])


def _write_params_(model: nn.Module, vec: Tensor) -> None:
    off = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(vec[off : off + n].view_as(p).to(p.device))
        off += n


class P2PHonestMixin:
    """
    Requires the concrete node to actually define & initialize these in __init__:
      - model: nn.Module
      - device: torch.device
      - criterion: nn.Module (loss)
      - optimizer: torch.optim.Optimizer
      - next_batch(): -> tuple[Tensor, Tensor]
      - p2p_agg: Aggregator
      - p2p_pre: Optional[PreAggregator]
    These attributes are *declared* here for type checkers.
    """

    # ---- attribute declarations for type checkers ----
    model: nn.Module
    device: torch.device
    criterion: nn.Module
    optimizer: torch.optim.Optimizer

    def next_batch(self) -> Tuple[Tensor, Tensor]: ...  # provided by the concrete node

    p2p_agg: Aggregator
    p2p_pre: Optional[PreAggregator] = None

    # ---- helpers ----
    def get_param_vector(self) -> Tensor:
        return _flatten_params(self.model).to(self.device)

    def set_param_vector(self, vec: Tensor) -> None:
        _write_params_(self.model, vec.to(self.device))

    # ---- P2P steps ----
    def p2p_half_step(self, lr: float) -> Tensor:
        x, y = self.next_batch()
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        loss = self.criterion(logits, y)
        loss.backward()
        with torch.no_grad():
            for p in self.model.parameters():
                if p.grad is not None:
                    p.add_(p.grad, alpha=-lr)
        return self.get_param_vector()

    def p2p_aggregate_and_set(
        self,
        self_theta_half: Tensor,
        neighbor_vectors: List[Tensor],
    ) -> None:
        vecs = [self_theta_half] + neighbor_vectors
        if self.p2p_pre is not None:
            vecs = self.p2p_pre.pre_aggregate(vecs)
        new_vec = self.p2p_agg.aggregate(vecs)
        self.set_param_vector(new_vec)


class P2PByzantineMixin:
    """
    Concrete byzantine node must define:
      - device: torch.device
      - attack: Attack
    """

    device: torch.device
    attack: Attack

    def p2p_broadcast_vector(
        self,
        *,
        neighbor_vectors: Optional[List[Tensor]] = None,
        like: Optional[Tensor] = None,
    ) -> Tensor:
        mal = self.attack.apply(
            model=None, x=None, y=None, honest_grads=neighbor_vectors, base_grad=None
        )
        out = torch.as_tensor(mal)
        if like is not None:
            out = out.to(device=like.device, dtype=like.dtype)
        return out
