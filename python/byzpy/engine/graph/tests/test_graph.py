from __future__ import annotations

import pytest

from byzpy.engine.graph.graph import ComputationGraph, GraphNode, graph_input
from byzpy.engine.graph.operator import Operator


class _CollectOperator(Operator):
    """Simple operator used to create graph nodes for tests."""

    name = "collect"

    def compute(self, inputs, *, context):  # type: ignore[override]
        return inputs


_OP = _CollectOperator()


def _node(name: str, inputs: dict | None = None) -> GraphNode:
    return GraphNode(name=name, op=_OP, inputs=inputs or {})


def test_graph_topological_order_and_required_inputs():
    data = graph_input("dataset")
    preprocess = _node("preprocess", {"source": data})
    train = _node("train", {"preprocessed": "preprocess"})
    evaluate = _node("evaluate", {"model": "train", "raw": data})

    graph = ComputationGraph([evaluate, preprocess, train])

    assert [node.name for node in graph.nodes_in_order()] == [
        "preprocess",
        "train",
        "evaluate",
    ]
    assert graph.outputs == ["evaluate"]
    assert graph.required_inputs == frozenset({"dataset"})


def test_graph_accepts_explicit_outputs_subset():
    src = graph_input("src")
    preprocess = _node("preprocess", {"source": src})
    train = _node("train", {"preprocessed": "preprocess"})

    graph = ComputationGraph([preprocess, train], outputs=["preprocess"])

    assert graph.outputs == ["preprocess"]


def test_graph_rejects_unknown_outputs():
    node = _node("only")
    with pytest.raises(ValueError, match="Unknown output node"):
        ComputationGraph([node], outputs=["missing"])


def test_graph_requires_non_empty_node_list():
    with pytest.raises(ValueError, match="requires at least one node"):
        ComputationGraph([])


def test_graph_rejects_duplicate_node_names():
    node_a = _node("shared")
    node_b = _node("shared")

    with pytest.raises(ValueError, match="Duplicate graph node"):
        ComputationGraph([node_a, node_b])


def test_graph_detects_missing_dependencies_as_cycle():
    node = _node("trainer", {"upstream": "missing"})

    with pytest.raises(ValueError, match="contains a cycle"):
        ComputationGraph([node])


def test_graph_detects_cycles():
    node_a = _node("node_a", {"node_b": "node_b"})
    node_b = _node("node_b", {"node_a": "node_a"})

    with pytest.raises(ValueError, match="contains a cycle"):
        ComputationGraph([node_a, node_b])
