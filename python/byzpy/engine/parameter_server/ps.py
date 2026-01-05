from __future__ import annotations

import asyncio
from typing import AsyncIterator, Iterable, List, Optional, Sequence

import numpy as np
import torch

from ...aggregators import Aggregator
from ...engine.graph.ops import make_single_operator_graph
from ...engine.graph.pool import ActorPool
from ...engine.graph.scheduler import NodeScheduler
from ...engine.storage.shared_store import SharedTensorHandle, cleanup_tensor, register_tensor
from ...pre_aggregators import PreAggregator
from ..node.actors import ByzantineNodeActor, HonestNodeActor


class ParameterServer:
    """
    Parameter server for synchronous distributed training.

    This class orchestrates a parameter server training loop where:
    1. All honest nodes compute gradients in parallel
    2. Byzantine nodes generate malicious gradients (if present)
    3. Gradients are optionally pre-aggregated
    4. An aggregator combines all gradients into a single update
    5. The aggregated gradient is sent back to all nodes

    Parameters
    ----------
    honest_nodes : List[HonestNodeActor]
        List of honest node actors that compute legitimate gradients.
    byzantine_nodes : List[ByzantineNodeActor]
        List of Byzantine node actors that generate malicious gradients.
    aggregator : Aggregator
        Aggregator instance to combine gradients (e.g., MultiKrum, Median).
    pre_aggregator : Optional[PreAggregator], optional
        Optional pre-aggregator to transform gradients before aggregation
        (e.g., Bucketing, Clipping).
    update_byzantines : bool, optional
        If True, Byzantine nodes also receive the aggregated gradient update.
        Default is False.
    actor_pool : ActorPool | None, optional
        Optional actor pool for parallel aggregation. If None, aggregation
        runs on the main thread.
    scheduler_metadata : Optional[dict], optional
        Additional metadata to pass to the graph scheduler.

    Examples
    --------
    >>> from byzpy.aggregators.geometric_wise.krum import MultiKrum
    >>> aggregator = MultiKrum(f=2, q=5)
    >>> ps = ParameterServer(
    ...     honest_nodes=honest_actors,
    ...     byzantine_nodes=byz_actors,
    ...     aggregator=aggregator
    ... )
    >>> for _ in range(100):
    ...     await ps.round()
    >>> await ps.shutdown()
    """

    def __init__(
        self,
        honest_nodes: List[HonestNodeActor],
        byzantine_nodes: List[ByzantineNodeActor],
        aggregator: Aggregator,
        pre_aggregator: Optional[PreAggregator] = None,
        update_byzantines: bool = False,
        *,
        actor_pool: ActorPool | None = None,
        scheduler_metadata: Optional[dict] = None,
    ):
        self.hon = honest_nodes
        self.byz = byzantine_nodes
        self.agg = aggregator
        self.pre = pre_aggregator
        self.update_byz = update_byzantines
        self.pool = actor_pool
        self.scheduler: NodeScheduler | None = None
        if actor_pool is not None:
            graph = make_single_operator_graph(
                node_name="agg",
                operator=self.agg,
                input_keys=("gradients",),
            )
            self.scheduler = NodeScheduler(graph, pool=actor_pool, metadata=scheduler_metadata)

    async def _stream_honest(self) -> AsyncIterator[torch.Tensor]:
        coros = [h.honest_gradient_for_next_batch() for h in self.hon]
        for fut in asyncio.as_completed(coros):
            yield await fut

    async def _stream_byz(
        self, honest_grads: Sequence[torch.Tensor]
    ) -> AsyncIterator[torch.Tensor]:
        if not self.byz:
            return
        coros = [b.byzantine_gradient_for_next_batch(honest_grads) for b in self.byz]
        for fut in asyncio.as_completed(coros):
            yield await fut

    async def round(self) -> torch.Tensor:
        """
        Execute one training round.

        This method:
        1. Collects gradients from all honest nodes
        2. Generates malicious gradients from Byzantine nodes
        3. Optionally applies pre-aggregation
        4. Aggregates all gradients
        5. Sends the aggregated gradient to all nodes

        Returns
        -------
        torch.Tensor
            The aggregated gradient vector.
        """
        gradients: list[torch.Tensor] = []

        async for grad in self._stream_honest():
            gradients.append(grad)

        async for grad in self._stream_byz(tuple(gradients)):
            gradients.append(grad)

        if self.pre:
            gradients = list(self.pre.pre_aggregate(gradients))

        handles = self._register_shared_gradients(gradients)
        try:
            if self.scheduler is not None:
                result = await self.scheduler.run({"gradients": handles})
                g = result["agg"]
            else:
                g = self.agg.aggregate(handles)
        finally:
            self._cleanup_shared_gradients(handles)

        await asyncio.gather(
            *[n.apply_server_gradient(g) for n in self.hon]
            + ([n.apply_server_gradient(g) for n in self.byz] if self.update_byz else [])
        )
        return g

    async def shutdown(self):
        await asyncio.gather(*[n._ref._backend.close() for n in (self.hon + self.byz)])

    def _register_shared_gradients(self, grads: Sequence[torch.Tensor]) -> list[SharedTensorHandle]:
        handles: list[SharedTensorHandle] = []
        for grad in grads:
            arr = grad.detach().cpu().numpy()
            handles.append(register_tensor(np.array(arr, copy=True)))
        return handles

    def _cleanup_shared_gradients(self, handles: Sequence[SharedTensorHandle]) -> None:
        for handle in handles:
            cleanup_tensor(handle)
