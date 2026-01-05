"""Shared abstractions for gradient aggregators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Sequence

from byzpy.engine.graph.operator import OpContext, Operator


class Aggregator(Operator, ABC):
    """
    Base class for every gradient aggregator.

    Aggregators combine multiple gradient vectors from different nodes into a
    single aggregated gradient. This is the core mechanism for Byzantine-robust
    distributed learning, as aggregators must be resilient to malicious or
    corrupted gradients.

    Subclasses must implement :meth:`aggregate` to define the aggregation
    strategy. The base class exposes the Operator interface so aggregators can
    be scheduled inside computation graphs just like any other operator.

    Aggregators can optionally support parallel execution via subtasks by
    setting ``supports_subtasks = True`` and implementing
    :meth:`create_subtasks` and :meth:`reduce_subtasks`.

    Examples
    --------
    >>> from byzpy.aggregators.coordinate_wise.median import CoordinateWiseMedian
    >>> aggregator = CoordinateWiseMedian()
    >>> gradients = [torch.randn(100) for _ in range(10)]
    >>> result = aggregator.aggregate(gradients)
    >>> result.shape
    torch.Size([100])
    """

    name = "aggregator"
    input_key = "gradients"

    def compute(self, inputs: Mapping[str, Any], *, context: OpContext) -> Any:  # type: ignore[override]
        """
        Compute the aggregated gradient from input gradients.

        This method is called by the computation graph scheduler. It extracts
        gradients from the inputs dictionary and delegates to :meth:`aggregate`.

        Parameters
        ----------
        inputs : Mapping[str, Any]
            Input dictionary containing gradients under the key specified by
            :attr:`input_key` (default: "gradients").
        context : OpContext
            Runtime context with node metadata and pool information.

        Returns
        -------
        Any
            Aggregated gradient tensor with the same shape and backend as the
            input gradients.

        Raises
        ------
        KeyError
            If the expected input key is missing.
        TypeError
            If the input is not a sequence.
        """
        if self.input_key not in inputs:
            raise KeyError(f"{self.name} expects input key {self.input_key!r}")
        gradients = inputs[self.input_key]
        if not isinstance(gradients, Sequence):
            raise TypeError(f"{self.name} expects a sequence at {self.input_key!r}")
        return self.aggregate(gradients)

    @abstractmethod
    def aggregate(self, gradients: Sequence[Any]) -> Any:
        """
        Reduce a sequence of gradient tensors into a single aggregated tensor.

        This is the core method that subclasses must implement. It receives a
        sequence of gradient vectors (one per node) and returns a single
        aggregated gradient vector.

        Parameters
        ----------
        gradients : Sequence[Any]
            Ordered sequence of gradient tensors (or tensor-like objects).
            All gradients must have the same shape and be from the same
            backend (NumPy, PyTorch, etc.). The sequence should be non-empty.

        Returns
        -------
        Any
            A tensor with the same shape, dtype, and device/backend as the
            input gradients representing the aggregated gradient.

        Raises
        ------
        ValueError
            If the gradients sequence is empty or invalid.
        """
        ...


__all__ = ["Aggregator"]
