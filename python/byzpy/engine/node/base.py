from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import torch


class Node(ABC):
    @abstractmethod
    def next_batch(self) -> Tuple[torch.Tensor, torch.Tensor]: ...
    @abstractmethod
    def apply_server_gradient(self, grad_vec: torch.Tensor) -> None: ...


class HonestNode(Node, ABC):
    @abstractmethod
    def honest_gradient(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor: ...

    def honest_gradient_for_next_batch(self) -> torch.Tensor:
        x, y = self.next_batch()
        return self.honest_gradient(x, y).detach()


class ByzantineNode(Node, ABC):
    @abstractmethod
    def byzantine_gradient(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        honest_grads: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor: ...

    def byzantine_gradient_for_next_batch(
        self, honest_grads: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        x = torch.empty(0)
        y = torch.empty(0, dtype=torch.long)
        return self.byzantine_gradient(x, y, honest_grads=honest_grads).detach()
