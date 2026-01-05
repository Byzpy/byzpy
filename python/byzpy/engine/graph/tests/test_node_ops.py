from __future__ import annotations

import asyncio
import threading
from typing import Sequence

import pytest
import torch

from byzpy.aggregators.base import Aggregator
from byzpy.engine.graph.graph import ComputationGraph
from byzpy.engine.graph.ops import RemoteCallableOp, make_single_operator_graph
from byzpy.engine.graph.pool import ActorPool, ActorPoolConfig
from byzpy.engine.graph.scheduler import NodeScheduler


class _SumAggregator(Aggregator):
    def aggregate(self, gradients: Sequence[torch.Tensor]) -> torch.Tensor:
        total = torch.zeros_like(gradients[0])
        for grad in gradients:
            total = total + grad
        return total


@pytest.mark.asyncio
async def test_aggregator_op_single_node_graph():
    op = _SumAggregator()
    graph = make_single_operator_graph(
        node_name="agg",
        operator=op,
        input_keys=("gradients",),
    )

    pool = ActorPool([ActorPoolConfig(backend="thread", count=2)])
    scheduler = NodeScheduler(graph, pool=pool)

    grads = [torch.arange(4, dtype=torch.float32), torch.arange(4, dtype=torch.float32)]
    out = await scheduler.run({"gradients": grads})
    assert torch.equal(out["agg"], grads[0] + grads[1])

    await pool.shutdown()


@pytest.mark.asyncio
async def test_remote_callable_op_runs_on_pool_thread():
    thread_ids: list[int] = []

    def _fn(value: torch.Tensor) -> torch.Tensor:
        thread_ids.append(threading.get_ident())
        return value * 2

    op = RemoteCallableOp(_fn, input_mapping={"value": "value"})
    graph: ComputationGraph = make_single_operator_graph(
        node_name="remote",
        operator=op,
        input_keys=("value",),
    )

    pool = ActorPool([ActorPoolConfig(backend="thread", count=2)])
    scheduler = NodeScheduler(graph, pool=pool)

    main_thread = threading.get_ident()
    value = torch.ones(3, dtype=torch.float32)
    out = await scheduler.run({"value": value})

    assert torch.equal(out["remote"], value * 2)
    assert thread_ids, "remote callable should execute on pool worker"
    assert thread_ids[0] != main_thread

    await pool.shutdown()
