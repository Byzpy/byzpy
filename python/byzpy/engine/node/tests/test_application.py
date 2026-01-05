from __future__ import annotations

import asyncio
import threading
from typing import Sequence

import pytest
import torch

from byzpy.aggregators.base import Aggregator
from byzpy.attacks.empire import EmpireAttack
from byzpy.engine.graph.graph import ComputationGraph, GraphNode, graph_input
from byzpy.engine.graph.pool import ActorPoolConfig
from byzpy.engine.node.application import ByzantineNodeApplication, HonestNodeApplication
from byzpy.engine.node.distributed import DistributedByzantineNode, DistributedHonestNode

EXECUTION_RECORD: list[int] = []


class _SumAggregator(Aggregator):
    name = "sum"

    def aggregate(self, gradients: Sequence[torch.Tensor]) -> torch.Tensor:
        if not gradients:
            raise ValueError("No gradients supplied.")
        # Handle SharedTensorHandle objects if present
        import numpy as np

        from byzpy.engine.storage.shared_store import SharedTensorHandle, open_tensor

        # Convert first gradient to tensor to get shape/device
        first_grad = gradients[0]
        if isinstance(first_grad, SharedTensorHandle):
            with open_tensor(first_grad) as arr:
                first_tensor = torch.from_numpy(np.array(arr, copy=True))
        else:
            first_tensor = (
                first_grad if isinstance(first_grad, torch.Tensor) else torch.tensor(first_grad)
            )

        total = torch.zeros_like(first_tensor)
        for grad in gradients:
            if isinstance(grad, SharedTensorHandle):
                with open_tensor(grad) as arr:
                    grad_tensor = torch.from_numpy(np.array(arr, copy=True))
            else:
                grad_tensor = grad if isinstance(grad, torch.Tensor) else torch.tensor(grad)
            total = total + grad_tensor.to(total.device)
        return total


def _build_single_node_graph(name: str, op, *, input_key: str) -> ComputationGraph:
    return ComputationGraph(
        nodes=[
            GraphNode(
                name=name,
                op=op,
                inputs={input_key: graph_input(input_key)},
            )
        ],
        outputs=[name],
    )


@pytest.mark.asyncio
async def test_honest_node_application_runs_aggregation_graph():
    app = HonestNodeApplication(
        name="node-honest",
        actor_pool=[ActorPoolConfig(backend="thread", count=2)],
        metadata={"role": "honest"},
    )

    agg_graph = _build_single_node_graph(
        "aggregate",
        _SumAggregator(),
        input_key="gradients",
    )
    app.register_pipeline(app.AGGREGATION_PIPELINE, agg_graph)

    grads = [
        torch.arange(4, dtype=torch.float32),
        torch.arange(4, dtype=torch.float32) * 2.0,
    ]
    result = await app.aggregate(gradients=grads)
    expected = grads[0] + grads[1]
    assert torch.equal(result, expected)

    await app.shutdown()


class _ToyHonestNode(DistributedHonestNode):
    def __init__(self):
        super().__init__(
            actor_pool=[ActorPoolConfig(backend="thread", count=2)],
            aggregator=_SumAggregator(),
            name="toy-honest",
        )
        self._batch = (
            torch.arange(4, dtype=torch.float32),
            torch.arange(4, dtype=torch.long),
        )
        self.last_update = None

    def next_batch(self):
        return self._batch

    def apply_server_gradient(self, grad_vec: torch.Tensor) -> None:
        self.last_update = grad_vec

    def local_honest_gradient(self, *, x, y):
        # Simple deterministic gradient for testing.
        return x.float() + y.float()


class _ToyByzantineNode(DistributedByzantineNode):
    def __init__(self):
        super().__init__(
            actor_pool=[ActorPoolConfig(backend="thread", count=2)],
            attack=EmpireAttack(scale=-2.0),
            name="toy-byz",
        )

    def next_batch(self):
        return torch.empty(0), torch.empty(0, dtype=torch.long)

    def apply_server_gradient(self, grad_vec: torch.Tensor) -> None:
        pass


def test_distributed_honest_node_sync_interfaces():
    node = _ToyHonestNode()

    # Synchronous gradient should run through the pipeline automatically.
    x = torch.ones(4, dtype=torch.float32)
    y = torch.arange(4, dtype=torch.float32)
    grad = node.honest_gradient(x, y)
    assert torch.equal(grad, x + y)

    grads = [torch.ones(4, dtype=torch.float32), torch.ones(4, dtype=torch.float32)]
    agg = node.aggregate_sync(grads)
    assert torch.equal(agg, grads[0] + grads[1])

    import asyncio as _asyncio

    _asyncio.run(node.shutdown_distributed())


@pytest.mark.asyncio
async def test_distributed_honest_node_async_aggregate():
    node = _ToyHonestNode()
    grads = [torch.ones(4, dtype=torch.float32), torch.ones(4, dtype=torch.float32)]
    agg = await node.aggregate(grads)
    assert torch.equal(agg, grads[0] + grads[1])
    await node.shutdown_distributed()


def test_distributed_byzantine_node_sync_attack():
    node = _ToyByzantineNode()
    grads = [torch.arange(5, dtype=torch.float32) + i for i in range(3)]

    # Synchronous API
    sync_vec = node.byzantine_gradient(
        torch.empty(0), torch.empty(0, dtype=torch.long), honest_grads=grads
    )
    manual = -2.0 * torch.stack(grads, dim=0).mean(dim=0)
    assert torch.allclose(sync_vec, manual, atol=1e-6)

    import asyncio as _asyncio

    _asyncio.run(node.shutdown_distributed())


@pytest.mark.asyncio
async def test_distributed_byzantine_node_async_attack():
    node = _ToyByzantineNode()
    grads = [torch.arange(5, dtype=torch.float32) + i for i in range(3)]
    manual = -2.0 * torch.stack(grads, dim=0).mean(dim=0)

    # Async convenience method
    async_vec = await node.byzantine_gradient_async(honest_grads=grads)
    assert torch.allclose(async_vec, manual, atol=1e-6)

    await node.shutdown_distributed()


class _OverrideByzantineNode(DistributedByzantineNode):
    def __init__(self):
        super().__init__(
            actor_pool=[ActorPoolConfig(backend="thread", count=2)],
            attack=None,
            name="override-byz",
        )

    def next_batch(self):
        return torch.empty(0), torch.empty(0, dtype=torch.long)

    def apply_server_gradient(self, grad_vec: torch.Tensor) -> None:
        pass

    def byzantine_gradient(self, honest_grads=None):
        if not honest_grads:
            raise ValueError("Need honest gradients.")
        EXECUTION_RECORD.append(threading.get_ident())
        stacked = torch.stack(list(honest_grads), dim=0)
        return -stacked.mean(dim=0)


def test_custom_byzantine_gradient_override_runs_via_pool():
    node = _OverrideByzantineNode()
    grads = [torch.arange(6, dtype=torch.float32) + 2 * i for i in range(4)]
    expected = -torch.stack(grads, dim=0).mean(dim=0)

    EXECUTION_RECORD.clear()
    out_sync = node.byzantine_gradient(None, None, honest_grads=grads)
    assert torch.allclose(out_sync, expected, atol=1e-6)
    assert len(EXECUTION_RECORD) == 1
    main_thread = threading.get_ident()
    assert EXECUTION_RECORD[-1] != main_thread

    asyncio.run(node.shutdown_distributed())

    node_async = _OverrideByzantineNode()
    async_out = asyncio.run(node_async.byzantine_gradient_async(honest_grads=grads))
    assert torch.allclose(async_out, expected, atol=1e-6)
    assert len(EXECUTION_RECORD) == 2
    assert EXECUTION_RECORD[-1] != main_thread

    asyncio.run(node_async.shutdown_distributed())


@pytest.mark.asyncio
async def test_empire_attack_operator_distributes_mean():
    app = ByzantineNodeApplication(
        name="node-byz",
        actor_pool=[ActorPoolConfig(backend="thread", count=3)],
        metadata={"role": "byzantine"},
    )

    attack_op = EmpireAttack(scale=-1.5, chunk_size=2)
    attack_graph = _build_single_node_graph(
        "attack",
        attack_op,
        input_key="honest_grads",
    )

    app.register_pipeline(app.ATTACK_PIPELINE, attack_graph)

    grads = [torch.randn(16) for _ in range(6)]
    manual = -1.5 * torch.stack(grads, dim=0).mean(dim=0)

    result = await app.run_attack(inputs={"honest_grads": grads})
    assert torch.allclose(result, manual, atol=1e-6)

    await app.shutdown()
