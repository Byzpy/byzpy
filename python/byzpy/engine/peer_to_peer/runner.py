"""
Refactored P2P runner using DecentralizedNode infrastructure.

This replaces the prototype NodeRunner-based implementation with a production
implementation using DecentralizedCluster and DecentralizedNode.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch

from ...aggregators.coordinate_wise import CoordinateWiseMedian
from ..graph.graph import ComputationGraph, GraphInput, GraphNode
from ..graph.ops import CallableOp, make_single_operator_graph
from ..graph.pool import ActorPoolConfig
from ..node.actors import ByzantineNodeActor, HonestNodeActor
from ..node.application import ByzantineNodeApplication, HonestNodeApplication
from ..node.cluster import DecentralizedCluster
from ..node.context import NodeContext, ProcessContext
from ..node.decentralized import DecentralizedNode
from ..peer_to_peer.topology import Topology


class _PicklableActorCall:
    """Wrapper that allows calling actor methods without capturing actor in closure."""

    def __init__(self, actor_key: str):
        self.actor_key = actor_key

    async def p2p_half_step(self, lr: float, actor_registry: Dict[str, Any]):
        actor = actor_registry.get(self.actor_key)
        if actor is None:
            raise RuntimeError(f"Actor {self.actor_key} not found in registry")
        return await actor.p2p_half_step(lr)

    async def p2p_broadcast_vector(
        self, neighbor_vectors=None, like=None, actor_registry: Dict[str, Any] = None
    ):
        actor = actor_registry.get(self.actor_key) if actor_registry else None
        if actor is None:
            raise RuntimeError(f"Actor {self.actor_key} not found in registry")
        return await actor.p2p_broadcast_vector(neighbor_vectors=neighbor_vectors, like=like)


_NODE_OBJECT_REGISTRY: Dict[str, Any] = {}
_ACTOR_REGISTRY: Dict[str, Any] = {}


def _create_honest_node_application(
    actor: HonestNodeActor,
    node_id: str,
    lr: float,
) -> HonestNodeApplication:
    """Create a NodeApplication for an honest node with P2P pipelines."""
    app = HonestNodeApplication(
        name=f"honest_{node_id}",
        actor_pool=[ActorPoolConfig(backend="thread", count=1)],
    )

    # Extract underlying node object from actor backend (this can be pickled)
    # For ThreadActorBackend, the node object is stored in _obj
    node_key = f"honest_{node_id}"
    try:
        if hasattr(actor, "_ref") and hasattr(actor._ref, "_backend"):
            backend = actor._ref._backend
            if hasattr(backend, "_obj") and backend._obj is not None:
                # Store the actual node object (picklable)
                _NODE_OBJECT_REGISTRY[node_key] = backend._obj
    except Exception:
        pass

    # Also store actor for in-process direct calls
    _ACTOR_REGISTRY[node_key] = actor

    # Half-step pipeline - import registry inside function to avoid closure capture
    async def half_step(lr: float):
        # Import registry inside function (not captured in closure)
        from byzpy.engine.peer_to_peer.runner import _ACTOR_REGISTRY, _NODE_OBJECT_REGISTRY

        # Try node object registry (works in subprocess after unpickling)
        node_obj = _NODE_OBJECT_REGISTRY.get(node_key)
        if node_obj is not None:
            # p2p_half_step is synchronous, returns Tensor directly
            result = node_obj.p2p_half_step(lr)
            # If it's a coroutine (from ActorRef), await it; otherwise return directly
            if hasattr(result, "__await__"):
                return await result
            return result
        # Fallback to actor (in-process only) - ActorRef makes it async
        actor_ref = _ACTOR_REGISTRY.get(node_key)
        if actor_ref is not None:
            return await actor_ref.p2p_half_step(lr)
        raise RuntimeError(f"Node object {node_key} not found in registry")

    half_step_op = CallableOp(half_step, input_mapping={"lr": "lr"})
    half_step_graph = make_single_operator_graph(
        node_name="half_step",
        operator=half_step_op,
        input_keys=("lr",),
    )
    app.register_pipeline("half_step", half_step_graph)

    # Aggregation pipeline
    # Use the aggregator directly (it's an Operator)
    aggregator = CoordinateWiseMedian()
    aggregate_graph = make_single_operator_graph(
        node_name="aggregate",
        operator=aggregator,
        input_keys=("gradients",),
    )
    app.register_pipeline("aggregate", aggregate_graph)

    # Note: We don't store actor reference in app to avoid pickling issues
    # The actor is accessed via closure in the pipeline functions
    return app


def _create_byzantine_node_application(
    actor: ByzantineNodeActor,
    node_id: str,
) -> ByzantineNodeApplication:
    """Create a NodeApplication for a byzantine node with P2P pipelines."""
    app = ByzantineNodeApplication(
        name=f"byz_{node_id}",
        actor_pool=[ActorPoolConfig(backend="thread", count=1)],
    )

    # Extract underlying node object from actor backend (this can be pickled)
    node_key = f"byz_{node_id}"
    try:
        if hasattr(actor, "_ref") and hasattr(actor._ref, "_backend"):
            backend = actor._ref._backend
            if hasattr(backend, "_obj") and backend._obj is not None:
                # Store the actual node object (picklable)
                _NODE_OBJECT_REGISTRY[node_key] = backend._obj
    except Exception:
        pass

    # Also store actor for in-process direct calls
    _ACTOR_REGISTRY[node_key] = actor

    # Broadcast pipeline - import registry inside function to avoid closure capture
    async def broadcast_vector(
        neighbor_vectors: Optional[List[torch.Tensor]] = None,
        like: Optional[torch.Tensor] = None,
    ):
        # Import registry inside function (not captured in closure)
        from byzpy.engine.peer_to_peer.runner import _ACTOR_REGISTRY, _NODE_OBJECT_REGISTRY

        # Try node object registry (works in subprocess after unpickling)
        node_obj = _NODE_OBJECT_REGISTRY.get(node_key)
        if node_obj is not None:
            # p2p_broadcast_vector is synchronous, returns Tensor directly
            result = node_obj.p2p_broadcast_vector(neighbor_vectors=neighbor_vectors, like=like)
            # If it's a coroutine (from ActorRef), await it; otherwise return directly
            if hasattr(result, "__await__"):
                return await result
            return result
        # Fallback to actor (in-process only) - ActorRef makes it async
        actor_ref = _ACTOR_REGISTRY.get(node_key)
        if actor_ref is not None:
            return await actor_ref.p2p_broadcast_vector(
                neighbor_vectors=neighbor_vectors, like=like
            )
        raise RuntimeError(f"Node object {node_key} not found in registry")

    broadcast_op = CallableOp(
        broadcast_vector,
        input_mapping={"neighbor_vectors": "neighbor_vectors", "like": "like"},
    )
    broadcast_graph = make_single_operator_graph(
        node_name="broadcast",
        operator=broadcast_op,
        input_keys=("neighbor_vectors", "like"),
    )
    app.register_pipeline("broadcast", broadcast_graph)

    return app


class DecentralizedPeerToPeer:
    """P2P runner using DecentralizedCluster and DecentralizedNode."""

    def __init__(
        self,
        honest_nodes: List[HonestNodeActor],
        byzantine_nodes: List[ByzantineNodeActor],
        topology: Topology,
        *,
        lr: float = 0.05,
        context_factory: Optional[Callable[[str, int], NodeContext]] = None,
    ) -> None:
        self._cluster = DecentralizedCluster()
        self.topology = topology
        self.honest = honest_nodes
        self.byz = byzantine_nodes
        self.lr = lr
        self.context_factory = context_factory
        self._node_applications: Dict[str, Any] = {}
        self._decentralized_nodes: Dict[str, DecentralizedNode] = {}
        self._gradient_cache: Dict[str, List[torch.Tensor]] = (
            {}
        )  # Cache gradients received per node

    @property
    def cluster(self):
        """Access the underlying DecentralizedCluster instance."""
        return self._cluster

    async def start(self) -> None:
        """Start all nodes in the cluster."""
        n = len(self.honest) + len(self.byz)

        for idx in range(n):
            node_id = str(idx)

            # Create context
            if self.context_factory:
                context = self.context_factory(node_id, idx)
            else:
                context = ProcessContext()

            if idx < len(self.honest):
                # Honest node
                actor = self.honest[idx]
                actor.lr = self.lr
                app = _create_honest_node_application(actor, node_id, self.lr)
                self._node_applications[node_id] = app

                node = await self._cluster.add_node(
                    node_id=node_id,
                    application=app,
                    topology=self.topology,
                    context=context,
                )
                self._decentralized_nodes[node_id] = node

                # Register message handler for gradients
                def make_gradient_handler(nid, n):
                    async def on_gradient(from_id, payload):
                        if nid not in self._gradient_cache:
                            self._gradient_cache[nid] = []
                        self._gradient_cache[nid].append(payload.get("vector"))

                    return on_gradient

                node.register_message_handler("gradient", make_gradient_handler(node_id, node))
            else:
                # Byzantine node
                byz_idx = idx - len(self.honest)
                actor = self.byz[byz_idx]
                app = _create_byzantine_node_application(actor, node_id)
                self._node_applications[node_id] = app

                node = await self._cluster.add_node(
                    node_id=node_id,
                    application=app,
                    topology=self.topology,
                    context=context,
                )
                self._decentralized_nodes[node_id] = node

                # Register message handler for byzantine node
                def make_byz_handler(nid, n):
                    async def on_gradient(from_id, payload):
                        if nid not in self._gradient_cache:
                            self._gradient_cache[nid] = []
                        self._gradient_cache[nid].append(payload.get("vector"))

                    return on_gradient

                node.register_message_handler("gradient", make_byz_handler(node_id, node))

        # Start all nodes
        await self._cluster.start_all()

    async def stop(self) -> None:
        """Stop all nodes in the cluster."""
        await self._cluster.shutdown_all()

    async def run_round_async(self) -> None:
        """Run one training round asynchronously."""
        n = len(self.honest) + len(self.byz)

        # Step 1: Each node performs half-step (honest) or prepares broadcast (byzantine)
        half_step_results: Dict[str, torch.Tensor] = {}

        for i in range(n):
            node_id = str(i)
            node = self._decentralized_nodes[node_id]

            if i < len(self.honest):
                # Honest node: perform half-step
                result = await node.execute_pipeline("half_step", {"lr": self.lr})
                half_step_results[node_id] = result["half_step"]
            else:
                # Byzantine node: prepare broadcast (will be done in step 2)
                pass

        # Step 2: Broadcast vectors to neighbors
        template: Optional[torch.Tensor] = None
        if self.honest and "0" in half_step_results:
            template = half_step_results["0"]

        for i in range(n):
            node_id = str(i)
            node = self._decentralized_nodes[node_id]

            if i < len(self.honest):
                # Honest node: broadcast half-step result
                payload = half_step_results[node_id]
                await node.broadcast_message("gradient", {"vector": payload})
            else:
                # Byzantine node: generate and broadcast malicious vector
                byz_idx = i - len(self.honest)
                actor = self.byz[byz_idx]

                # Wait a bit for messages to propagate (needed for process-based execution)
                await asyncio.sleep(0.1)

                # Get cached neighbor vectors
                neighbor_vecs = self._gradient_cache.get(node_id, [])
                if template is None and neighbor_vecs:
                    template = neighbor_vecs[0]

                # Only generate malicious vector if we have neighbor vectors or a template
                if neighbor_vecs or template is not None:
                    # Use template if no neighbor vectors yet (first round)
                    if not neighbor_vecs and template is not None:
                        # For first round, use template as a placeholder
                        # Some attacks may still fail, but this handles the common case
                        neighbor_vecs = [template]

                    # Generate malicious vector
                    # Get node object from registry (works across processes)
                    from byzpy.engine.peer_to_peer.runner import (
                        _ACTOR_REGISTRY,
                        _NODE_OBJECT_REGISTRY,
                    )

                    node_key = f"byz_{node_id}"
                    node_obj = _NODE_OBJECT_REGISTRY.get(node_key)
                    if node_obj is not None:
                        malicious = node_obj.p2p_broadcast_vector(
                            neighbor_vectors=neighbor_vecs if neighbor_vecs else None,
                            like=template,
                        )
                        # Check if result is awaitable
                        if hasattr(malicious, "__await__"):
                            malicious = await malicious
                    else:
                        actor_ref = _ACTOR_REGISTRY.get(node_key)
                        if actor_ref is not None:
                            malicious = await actor_ref.p2p_broadcast_vector(
                                neighbor_vectors=(neighbor_vecs if neighbor_vecs else None),
                                like=template,
                            )
                        else:
                            # Fallback: use actor directly
                            malicious = await actor.p2p_broadcast_vector(
                                neighbor_vectors=(neighbor_vecs if neighbor_vecs else None),
                                like=template,
                            )
                    await node.broadcast_message("gradient", {"vector": malicious})
                # If no template and no neighbor vectors, skip byzantine broadcast for this round

        # Step 3: Wait for messages to propagate
        await asyncio.sleep(0.1)

        # Step 4: Each honest node aggregates received gradients
        for i in range(len(self.honest)):
            node_id = str(i)
            node = self._decentralized_nodes[node_id]

            # Get cached gradients
            received_gradients = self._gradient_cache.get(node_id, [])
            if received_gradients:
                # Include own half-step result
                all_gradients = [half_step_results[node_id]] + received_gradients

                # Trigger aggregation pipeline with gradients
                await node.execute_pipeline("aggregate", {"gradients": all_gradients})

                # Clear cache for next round
                self._gradient_cache[node_id] = []

    def run_round(self) -> None:
        """Synchronous helper for demos/tests."""
        asyncio.run(self.run_round_async())
