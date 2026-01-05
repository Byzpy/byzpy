from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from ..graph.graph import ComputationGraph
from ..graph.pool import ActorPool, ActorPoolConfig
from ..graph.scheduler import NodeScheduler


@dataclass(frozen=True)
class NodePipeline:
    """
    Declarative description of a computation pipeline bound to a node.
    """

    graph: ComputationGraph
    metadata: Mapping[str, Any] | None = None


class NodeApplication:
    """
    Application-layer runtime for a single node.

    This class manages an actor pool, one or more computation graphs
    ("pipelines"), and provides helpers to run them through a NodeScheduler.
    It serves as the bridge between the application layer (aggregators,
    attacks, etc.) and the scheduling layer (graphs, actors).

    A node application can register multiple pipelines (computation graphs)
    and execute them on demand. Each pipeline can use the shared actor pool
    for parallel execution.

    Parameters
    ----------
    name : str
        Unique name for this node application.
    actor_pool : ActorPool | Sequence[ActorPoolConfig]
        Either an existing ActorPool instance or a sequence of ActorPoolConfig
        objects to create a new pool.
    metadata : Optional[Mapping[str, Any]], optional
        Base metadata to include in all pipeline executions.

    Examples
    --------
    >>> from byzpy.engine.graph.pool import ActorPoolConfig
    >>> app = NodeApplication(
    ...     name="node0",
    ...     actor_pool=[ActorPoolConfig(backend="thread", count=4)]
    ... )
    >>> # Register and run pipelines...
    """

    def __init__(
        self,
        *,
        name: str,
        actor_pool: ActorPool | Sequence[ActorPoolConfig],
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.name = name
        if isinstance(actor_pool, ActorPool):
            self._pool = actor_pool
        else:
            self._pool = ActorPool(actor_pool)
        self._pipelines: Dict[str, NodePipeline] = {}
        self._base_metadata: Dict[str, Any] = dict(metadata or {})

    @property
    def pool(self) -> ActorPool:
        return self._pool

    def register_pipeline(
        self,
        name: str,
        graph: ComputationGraph,
        *,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if name in self._pipelines:
            raise ValueError(f"Pipeline {name!r} already registered for node {self.name!r}.")
        self._pipelines[name] = NodePipeline(graph=graph, metadata=dict(metadata or {}))

    def has_pipeline(self, name: str) -> bool:
        return name in self._pipelines

    def list_pipelines(self) -> Iterable[str]:
        return self._pipelines.keys()

    async def run_pipeline(
        self,
        name: str,
        inputs: Mapping[str, Any],
        *,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        if name not in self._pipelines:
            raise KeyError(f"Unknown pipeline {name!r} for node {self.name!r}.")
        pipeline = self._pipelines[name]
        merged_meta: Dict[str, Any] = {
            "node": self.name,
            "pipeline": name,
        }
        merged_meta.update(self._base_metadata)
        if pipeline.metadata:
            merged_meta.update(pipeline.metadata)
        if metadata:
            merged_meta.update(metadata)

        scheduler = NodeScheduler(pipeline.graph, pool=self._pool, metadata=merged_meta)
        return await scheduler.run(inputs)

    def run_pipeline_sync(
        self,
        name: str,
        inputs: Mapping[str, Any],
        *,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Synchronous helper that runs ``run_pipeline`` by attaching to the current
        running loop when possible, or spinning up a temporary event loop.
        """

        async def _runner():
            return await self.run_pipeline(name, inputs, metadata=metadata)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(_runner())
        else:
            # Running inside an event loop: synchronous execution would deadlock.
            raise RuntimeError(
                "run_pipeline_sync() cannot be called from an async context; "
                "use the async pipeline APIs instead."
            )

    async def shutdown(self) -> None:
        await self._pool.shutdown()


class HonestNodeApplication(NodeApplication):
    """
    Application runtime for honest nodes.

    This is a convenience wrapper that reserves special pipeline names for
    honest node behaviors:
    - "aggregate": Aggregation pipeline for combining gradients
    - "honest_gradient": Pipeline for computing honest gradients

    It provides helper methods like :meth:`aggregate` and :meth:`honest_gradient`
    that automatically use the registered pipelines.

    Examples
    --------
    >>> app = HonestNodeApplication(
    ...     name="honest_node0",
    ...     actor_pool=[ActorPoolConfig(backend="thread", count=2)]
    ... )
    >>> # Register aggregation pipeline...
    >>> result = await app.aggregate(gradients=[grad1, grad2, grad3])
    """

    AGGREGATION_PIPELINE = "aggregate"
    GRADIENT_PIPELINE = "honest_gradient"

    async def aggregate(
        self, *, gradients: Sequence[Any], metadata: Optional[Mapping[str, Any]] = None
    ) -> Any:
        if not self.has_pipeline(self.AGGREGATION_PIPELINE):
            raise KeyError(f"No aggregation pipeline registered on node {self.name!r}.")
        result = await self.run_pipeline(
            self.AGGREGATION_PIPELINE,
            {"gradients": gradients},
            metadata=metadata,
        )
        # Return the first (and typically only) output value.
        return next(iter(result.values()))

    def aggregate_sync(
        self, *, gradients: Sequence[Any], metadata: Optional[Mapping[str, Any]] = None
    ) -> Any:
        if not self.has_pipeline(self.AGGREGATION_PIPELINE):
            raise KeyError(f"No aggregation pipeline registered on node {self.name!r}.")
        result = self.run_pipeline_sync(
            self.AGGREGATION_PIPELINE,
            {"gradients": gradients},
            metadata=metadata,
        )
        return next(iter(result.values()))

    async def honest_gradient(
        self, inputs: Mapping[str, Any], *, metadata: Optional[Mapping[str, Any]] = None
    ) -> Any:
        if not self.has_pipeline(self.GRADIENT_PIPELINE):
            raise KeyError(f"No honest gradient pipeline registered on node {self.name!r}.")
        result = await self.run_pipeline(
            self.GRADIENT_PIPELINE,
            inputs,
            metadata=metadata,
        )
        return next(iter(result.values()))

    def honest_gradient_sync(
        self, inputs: Mapping[str, Any], *, metadata: Optional[Mapping[str, Any]] = None
    ) -> Any:
        if not self.has_pipeline(self.GRADIENT_PIPELINE):
            raise KeyError(f"No honest gradient pipeline registered on node {self.name!r}.")
        result = self.run_pipeline_sync(
            self.GRADIENT_PIPELINE,
            inputs,
            metadata=metadata,
        )
        return next(iter(result.values()))


class ByzantineNodeApplication(NodeApplication):
    """
    Application runtime for Byzantine nodes.

    This is a convenience wrapper for Byzantine nodes that primarily execute
    attack pipelines. It reserves the "attack" pipeline name and provides
    the :meth:`run_attack` helper method.

    Examples
    --------
    >>> app = ByzantineNodeApplication(
    ...     name="byz_node0",
    ...     actor_pool=[ActorPoolConfig(backend="thread", count=1)]
    ... )
    >>> # Register attack pipeline...
    >>> malicious_grad = await app.run_attack(inputs={"honest_grads": grads})
    """

    ATTACK_PIPELINE = "attack"

    async def run_attack(
        self, *, inputs: Mapping[str, Any], metadata: Optional[Mapping[str, Any]] = None
    ) -> Any:
        if not self.has_pipeline(self.ATTACK_PIPELINE):
            raise KeyError(f"No attack pipeline registered on node {self.name!r}.")
        result = await self.run_pipeline(
            self.ATTACK_PIPELINE,
            inputs,
            metadata=metadata,
        )
        return next(iter(result.values()))

    def run_attack_sync(
        self, *, inputs: Mapping[str, Any], metadata: Optional[Mapping[str, Any]] = None
    ) -> Any:
        if not self.has_pipeline(self.ATTACK_PIPELINE):
            raise KeyError(f"No attack pipeline registered on node {self.name!r}.")
        result = self.run_pipeline_sync(
            self.ATTACK_PIPELINE,
            inputs,
            metadata=metadata,
        )
        return next(iter(result.values()))


__all__ = [
    "NodeApplication",
    "NodePipeline",
    "HonestNodeApplication",
    "ByzantineNodeApplication",
]
