from __future__ import annotations

"""
Backwards-compatible wrapper that uses the process-based ParameterServerRunner.
"""
from typing import Callable, List, Optional, Sequence

import torch

from ..node.actors import ByzantineNodeActor, HonestNodeActor
from .runner import ParameterServerRunner


class DecentralizedParameterServer:
    """
    Minimal PS orchestrator using per-node processes.
    Honest nodes supply gradients via their actor API; byzantine nodes are ignored for now.
    """

    def __init__(
        self,
        honest_nodes: List[HonestNodeActor],
        byzantine_nodes: Optional[List[ByzantineNodeActor]],
        aggregator: Callable[[Sequence[torch.Tensor]], torch.Tensor],
    ) -> None:
        self._honest = honest_nodes
        self._byz = byzantine_nodes or []
        grad_fns = [lambda h=h: h.grad for h in self._honest]
        self._runner = ParameterServerRunner(worker_grad_fns=grad_fns, aggregator=aggregator)

    async def bootstrap(self) -> None:
        self._runner.start()

    async def round(self) -> torch.Tensor:
        return self._runner.run_round()

    async def shutdown(self) -> None:
        self._runner.stop()
