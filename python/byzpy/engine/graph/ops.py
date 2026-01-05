from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Mapping, Sequence

from .graph import ComputationGraph, GraphNode, graph_input
from .operator import OpContext, Operator
from .subtask import SubTask


class CallableOp(Operator):
    """
    Lightweight operator that forwards its inputs to a Python callable.
    The callable is expected to be serializable via ``cloudpickle`` so it can
    be shipped to worker actors when executed through an actor pool.
    """

    name = "callable"

    def __init__(self, fn: Callable[..., Any], *, input_mapping: Mapping[str, str]) -> None:
        self.fn = fn
        self.input_mapping = dict(input_mapping)

    def compute(self, inputs: Mapping[str, Any], *, context: OpContext) -> Any:  # type: ignore[override]
        kwargs: Dict[str, Any] = {}
        for param, source in self.input_mapping.items():
            if source not in inputs:
                raise KeyError(
                    f"CallableOp missing required input {source!r} for parameter {param!r}"
                )
            kwargs[param] = inputs[source]
        return self.fn(**kwargs)


class RemoteCallableOp(Operator):
    """
    Operator that pushes the callable execution onto the actor pool through a
    single subtask. Used for user-defined overrides that must run remotely.
    """

    name = "remote_callable"
    supports_subtasks = True

    def __init__(self, fn: Callable[..., Any], *, input_mapping: Mapping[str, str]) -> None:
        self.fn = fn
        self.input_mapping = dict(input_mapping)

    def compute(self, inputs: Mapping[str, Any], *, context: OpContext) -> Any:  # type: ignore[override]
        kwargs: Dict[str, Any] = {}
        for param, source in self.input_mapping.items():
            if source not in inputs:
                raise KeyError(f"RemoteCallableOp missing required input {source!r}")
            kwargs[param] = inputs[source]
        return self.fn(**kwargs)

    def create_subtasks(self, inputs: Mapping[str, Any], *, context: OpContext) -> Iterable[SubTask]:  # type: ignore[override]
        kwargs: Dict[str, Any] = {}
        for param, source in self.input_mapping.items():
            if source not in inputs:
                raise KeyError(f"RemoteCallableOp missing required input {source!r}")
            kwargs[param] = inputs[source]
        return [SubTask(fn=_invoke_remote_callable, args=(self.fn, kwargs), kwargs={})]

    def reduce_subtasks(
        self,
        partials: Sequence[Any],
        inputs: Mapping[str, Any],
        *,
        context: OpContext,
    ) -> Any:
        if not partials:
            raise RuntimeError("RemoteCallableOp expected exactly one partial result.")
        return partials[0]


def _invoke_remote_callable(fn: Callable[..., Any], kwargs: Mapping[str, Any]) -> Any:
    return fn(**kwargs)


def make_single_operator_graph(
    *,
    node_name: str,
    operator: Operator,
    input_keys: Sequence[str],
) -> ComputationGraph:
    """
    Helper to create a single-node computation graph that wires the provided
    ``operator`` to GraphInputs named after ``input_keys``.
    """

    inputs = {key: graph_input(key) for key in input_keys}
    node = GraphNode(name=node_name, op=operator, inputs=inputs)
    return ComputationGraph(nodes=[node], outputs=[node_name])


__all__ = [
    "CallableOp",
    "RemoteCallableOp",
    "make_single_operator_graph",
]
