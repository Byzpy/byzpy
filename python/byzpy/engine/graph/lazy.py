"""
Lazy evaluation API for building computation graphs.

This module provides a fluent API for constructing computation graphs
without executing them. Execution is deferred until explicitly requested,
enabling graph optimization and parallel scheduling.

Classes
-------
GraphBuilder
    Builder class for constructing computation graphs with a fluent API.
LazyNode
    A node in the lazy computation graph that supports chained operations.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence, Union

from .graph import ComputationGraph, GraphInput, GraphNode
from .operator import Operator


class GraphBuilder:
    """
    Fluent API for building computation graphs with lazy evaluation.

    This builder allows constructing computation graphs by defining inputs
    and chaining operator applications. No computation happens during
    graph construction - execution is deferred until the graph is passed
    to a scheduler.

    Examples
    --------
    >>> from byzpy.engine.graph.lazy import GraphBuilder
    >>> from byzpy.engine.graph.parallel_scheduler import ParallelScheduler
    >>>
    >>> builder = GraphBuilder()
    >>> gradients = builder.input("gradients")
    >>> doubled = gradients.apply(double_op)
    >>> result = doubled.apply(sum_op)
    >>>
    >>> graph = builder.build(outputs=[result.key])
    >>> scheduler = ParallelScheduler(graph)
    >>> outputs = await scheduler.run({"gradients": data})
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, GraphNode] = {}
        self._inputs: Dict[str, GraphInput] = {}
        self._node_counter: int = 0

    def input(self, name: str) -> "LazyNode":
        """
        Create a lazy input node representing external data.

        Multiple calls with the same name return nodes referencing
        the same underlying GraphInput.

        Parameters
        ----------
        name : str
            Name of the input. This name will be used when providing
            data to the scheduler.

        Returns
        -------
        LazyNode
            A lazy node representing this input that can be chained
            with operator applications.
        """
        if name not in self._inputs:
            self._inputs[name] = GraphInput(name)
        return LazyNode(builder=self, key=name, is_input=True)

    def build(self, outputs: Sequence[str]) -> ComputationGraph:
        """
        Build the computation graph from the accumulated nodes.

        Parameters
        ----------
        outputs : Sequence[str]
            Names of the output nodes to include in the graph outputs.

        Returns
        -------
        ComputationGraph
            The built computation graph ready for execution.

        Raises
        ------
        ValueError
            If no nodes have been added (empty graph).
            If any output name is not found in the graph.
        """
        # Validate at least one node exists
        if not self._nodes:
            raise ValueError("GraphBuilder requires at least one node")

        # Validate all requested outputs exist
        missing = [name for name in outputs if name not in self._nodes]
        if missing:
            raise ValueError(f"Unknown output node: {missing[0]}")

        nodes = list(self._nodes.values())
        return ComputationGraph(nodes, outputs=list(outputs))

    def _generate_node_name(self, operator: Operator) -> str:
        """Generate a unique node name for an operator."""
        name = f"{operator.name}_{self._node_counter}"
        self._node_counter += 1
        return name


class LazyNode:
    """
    A node in the lazy computation graph.

    Operations on this node build the graph structure rather than
    executing computations. Each `apply()` call returns a new LazyNode
    representing the output of applying an operator.

    This class should not be instantiated directly - use
    `GraphBuilder.input()` to create input nodes, then chain
    `apply()` calls.

    Parameters
    ----------
    builder : GraphBuilder
        The parent builder managing the graph construction.
    key : str
        The unique key identifying this node.
    is_input : bool, optional
        Whether this node represents a graph input (default False).
    """

    def __init__(
        self,
        builder: GraphBuilder,
        key: str,
        is_input: bool = False,
    ) -> None:
        self._builder = builder
        self._key = key
        self._is_input = is_input

    @property
    def key(self) -> str:
        """The unique key identifying this node."""
        return self._key

    def apply(
        self,
        operator: Operator,
        *,
        input_key: str | None = None,
        extra_inputs: Dict[str, Union["LazyNode", str]] | None = None,
        name: str | None = None,
    ) -> "LazyNode":
        """
        Apply an operator to this node, returning a new lazy node.

        This method does NOT execute the operator - it only records
        the operation in the graph structure. Execution happens when
        the built graph is passed to a scheduler.

        Parameters
        ----------
        operator : Operator
            The operator to apply to this node's output.
        input_key : str | None, optional
            Name of the input parameter for the operator. If None,
            auto-detected from `operator.input_key` or defaults to "vectors".
        extra_inputs : Dict[str, LazyNode | str] | None, optional
            Additional input dependencies. Keys are parameter names,
            values can be LazyNode instances or string node names.
        name : str | None, optional
            Custom name for the output node. If None, auto-generated
            from the operator name.

        Returns
        -------
        LazyNode
            A new lazy node representing the operator's output.

        Raises
        ------
        TypeError
            If `operator` is not an Operator instance.
        """
        if not isinstance(operator, Operator):
            raise TypeError(f"operator must be an Operator instance, got {type(operator).__name__}")

        # Auto-detect input key from operator attribute or use default
        if input_key is None:
            input_key = getattr(operator, "input_key", "vectors")

        # Generate output name or use provided
        output_name = name if name is not None else self._builder._generate_node_name(operator)

        # Build inputs mapping
        inputs: Dict[str, Union[str, GraphInput]] = {}

        # Main input - either a GraphInput reference or a node name string
        if self._is_input:
            inputs[input_key] = self._builder._inputs[self._key]
        else:
            inputs[input_key] = self._key

        # Extra inputs
        if extra_inputs:
            for arg, dep in extra_inputs.items():
                if isinstance(dep, LazyNode):
                    if dep._is_input:
                        inputs[arg] = self._builder._inputs[dep._key]
                    else:
                        inputs[arg] = dep._key
                else:
                    # String reference to another node
                    inputs[arg] = dep

        # Create and register the graph node
        node = GraphNode(name=output_name, op=operator, inputs=inputs)
        self._builder._nodes[output_name] = node

        return LazyNode(builder=self._builder, key=output_name)


__all__ = ["GraphBuilder", "LazyNode"]
