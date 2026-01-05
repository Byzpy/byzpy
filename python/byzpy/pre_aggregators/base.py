from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Mapping, Sequence

from byzpy.engine.graph.operator import OpContext, Operator


class PreAggregator(Operator, ABC):
    """
    Base class for pre-aggregation operations.

    Pre-aggregators transform a sequence of vectors before aggregation. They
    can reshape, filter, or combine vectors in various ways. Unlike aggregators
    which reduce to a single vector, pre-aggregators return a list of vectors
    (possibly of different length than the input).

    Common use cases include:
    - Bucketing: Group vectors into buckets and average within each bucket.
    - Clipping: Clip vectors to a maximum norm.
    - Nearest-neighbor mixing: Combine vectors with their nearest neighbors.

    Subclasses must implement :meth:`pre_aggregate` to define the
    transformation strategy.

    Examples
    --------
    >>> from byzpy.pre_aggregators.bucketing import Bucketing
    >>> preagg = Bucketing(bucket_size=4)
    >>> vectors = [torch.randn(100) for _ in range(10)]
    >>> result = preagg.pre_aggregate(vectors)
    >>> len(result)  # 10 vectors -> 3 buckets (ceil(10/4))
    3
    """

    name = "pre_aggregator"
    input_key = "vectors"

    def compute(self, inputs: Mapping[str, Any], *, context: OpContext) -> List[Any]:  # type: ignore[override]
        """
        Compute the pre-aggregated vectors from input vectors.

        This method is called by the computation graph scheduler. It extracts
        vectors from the inputs dictionary and delegates to :meth:`pre_aggregate`.

        Parameters
        ----------
        inputs : Mapping[str, Any]
            Input dictionary containing vectors under the key specified by
            :attr:`input_key` (default: "vectors").
        context : OpContext
            Runtime context with node metadata and pool information.

        Returns
        -------
        List[Any]
            List of transformed vectors. The length may differ from the input.

        Raises
        ------
        KeyError
            If the expected input key is missing.
        TypeError
            If the input is not a sequence.
        """
        if self.input_key not in inputs:
            raise KeyError(f"{self.name} expects input key {self.input_key!r}")
        values = inputs[self.input_key]
        if not isinstance(values, Sequence):
            raise TypeError(f"{self.name} expects a sequence at {self.input_key!r}")
        return self.pre_aggregate(values)

    @abstractmethod
    def pre_aggregate(self, xs: Sequence[Any]) -> List[Any]:
        """
        Transform a sequence of vectors into a list of pre-aggregated vectors.

        Parameters
        ----------
        xs : Sequence[Any]
            Input sequence of vectors. All vectors must have the same shape
            and backend. The sequence should be non-empty.

        Returns
        -------
        List[Any]
            List of transformed vectors. Each vector has the same shape and
            backend as inputs, but the list length may differ from the input
            length.

        Raises
        ------
        ValueError
            If the input sequence is empty or invalid.
        """
        ...
