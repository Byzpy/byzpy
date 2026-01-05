from __future__ import annotations

from typing import cast

import pytest

from byzpy.engine.graph.graph import ComputationGraph, GraphNode, graph_input
from byzpy.engine.graph.operator import Operator
from byzpy.engine.graph.scheduler import NodeScheduler


class _AddOp(Operator):
    def compute(self, inputs, *, context):  # type: ignore[override]
        return inputs["lhs"] + inputs["rhs"]


class _RecordMetadataOp(Operator):
    def __init__(self):
        self.last_metadata = None

    def compute(self, inputs, *, context):  # type: ignore[override]
        self.last_metadata = context.metadata
        return inputs["value"] * 2


def _make_graph():
    data = graph_input("data")
    bias = graph_input("bias")
    doubler = GraphNode(name="double", op=_RecordMetadataOp(), inputs={"value": data})
    summer = GraphNode(
        name="sum",
        op=_AddOp(),
        inputs={"lhs": "double", "rhs": bias},
    )
    graph = ComputationGraph([doubler, summer], outputs=["sum", "double"])
    return graph, doubler, summer


@pytest.mark.asyncio
async def test_node_scheduler_runs_graph_and_returns_outputs():
    graph, _, _ = _make_graph()
    scheduler = NodeScheduler(graph, metadata={"phase": "train"})

    outputs = await scheduler.run({"data": 5, "bias": 3})

    assert outputs == {"sum": 13, "double": 10}


@pytest.mark.asyncio
async def test_node_scheduler_raises_for_missing_inputs():
    graph, _, _ = _make_graph()
    scheduler = NodeScheduler(graph)

    with pytest.raises(ValueError, match="Missing graph inputs"):
        await scheduler.run({"data": 1})


@pytest.mark.asyncio
async def test_node_scheduler_passes_metadata_into_context():
    graph, double_node, _ = _make_graph()
    doubler = cast(_RecordMetadataOp, double_node.op)
    scheduler = NodeScheduler(graph, metadata={"stage": "eval"})

    await scheduler.run({"data": 4, "bias": 0})

    assert doubler.last_metadata == {"stage": "eval"}


def test_resolve_inputs_detects_missing_dependency():
    graph, _, summer = _make_graph()
    scheduler = NodeScheduler(graph)

    with pytest.raises(KeyError, match="has not been computed"):
        scheduler._resolve_inputs(summer, cache={})
