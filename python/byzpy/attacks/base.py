from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Mapping, Optional

import torch
import torch.nn as nn

from byzpy.engine.graph.operator import OpContext, Operator


class Attack(Operator, ABC):
    """
    Base class for Byzantine attack implementations.

    Attacks simulate malicious behavior in distributed learning by generating
    adversarial gradient vectors. Subclasses implement different attack
    strategies by setting input requirement flags and implementing :meth:`apply`.

    Input Requirements
    ------------------
    Subclasses declare what information they need by setting class attributes:

    - ``uses_base_grad``: If True, the attack needs the node's own honest
      gradient vector.
    - ``uses_model_batch``: If True, the attack needs the model, input batch
      (x), and labels (y) for gradient computation.
    - ``uses_honest_grads``: If True, the attack needs a list of other honest
      nodes' gradient vectors.

    Examples
    --------
    >>> from byzpy.attacks.empire import EmpireAttack
    >>> attack = EmpireAttack(scale=-1.0)
    >>> malicious_grad = attack.apply(honest_grads=[grad1, grad2, grad3])
    >>> assert malicious_grad.shape == grad1.shape

    Notes
    -----
    - All attacks must return a single gradient vector with the same shape
      and backend as the inputs.
    - Attacks are used in simulations to test Byzantine-robust aggregators.
    - The attack's ``apply`` method is called by the computation graph
      scheduler with the requested inputs.
    """

    uses_base_grad: bool = False
    uses_model_batch: bool = False
    uses_honest_grads: bool = False

    name = "attack"
    supports_subtasks = False

    def compute(self, inputs: Mapping[str, Any], *, context: OpContext) -> Any:  # type: ignore[override]
        kwargs = self._collect_inputs(inputs)
        return self.apply(**kwargs)

    @abstractmethod
    def apply(
        self,
        *,
        model: Optional[nn.Module] = None,
        x: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        honest_grads: Optional[List[Any]] = None,
        base_grad: Optional[Any] = None,
    ) -> Any:
        """
        Generate and return a malicious gradient vector.

        This method implements the attack strategy. It receives the requested
        inputs (based on the class's ``uses_*`` flags) and returns a single
        malicious gradient vector.

        Parameters
        ----------
        model : Optional[nn.Module]
            PyTorch model (required if ``uses_model_batch=True``).
        x : Optional[torch.Tensor]
            Input batch tensor (required if ``uses_model_batch=True``).
        y : Optional[torch.Tensor]
            Label tensor (required if ``uses_model_batch=True``).
        honest_grads : Optional[List[Any]]
            List of honest nodes' gradient vectors (required if
            ``uses_honest_grads=True``).
        base_grad : Optional[Any]
            The node's own honest gradient vector (required if
            ``uses_base_grad=True``).

        Returns
        -------
        Any
            A single malicious gradient vector with the same shape, dtype, and
            device/backend as the input gradients.

        Raises
        ------
        ValueError
            If required inputs are missing or invalid.
        """
        ...

    def _collect_inputs(self, inputs: Mapping[str, Any]) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}

        if self.uses_model_batch:
            for key in ("model", "x", "y"):
                if key not in inputs:
                    raise KeyError(f"Attack requires input {key!r}")
            kwargs["model"] = inputs["model"]
            kwargs["x"] = inputs["x"]
            kwargs["y"] = inputs["y"]

        if self.uses_honest_grads:
            if "honest_grads" not in inputs:
                raise KeyError("Attack requires 'honest_grads'")
            kwargs["honest_grads"] = inputs["honest_grads"]

        if self.uses_base_grad:
            if "base_grad" not in inputs:
                raise KeyError("Attack requires 'base_grad'")
            kwargs["base_grad"] = inputs["base_grad"]

        return kwargs


__all__ = ["Attack"]
