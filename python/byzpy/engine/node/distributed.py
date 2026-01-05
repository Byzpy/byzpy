from __future__ import annotations

import inspect
from typing import Mapping, Optional, Sequence

import numpy as np
import torch

from byzpy.aggregators.base import Aggregator
from byzpy.attacks.base import Attack
from byzpy.engine.graph.ops import CallableOp, RemoteCallableOp, make_single_operator_graph
from byzpy.engine.graph.pool import ActorPool, ActorPoolConfig
from byzpy.engine.storage.shared_store import SharedTensorHandle, cleanup_tensor, register_tensor

from .application import ByzantineNodeApplication, HonestNodeApplication, NodeApplication
from .base import ByzantineNode, HonestNode


class _DistributedNodeBase:
    """
    Shared helper that wires a ``NodeApplication`` to a node instance. Subclasses
    decide which pipelines to register and expose convenience wrappers around the
    application runtime.
    """

    def __init__(
        self,
        *,
        app_cls: type[NodeApplication],
        name: Optional[str],
        actor_pool: ActorPool | Sequence[ActorPoolConfig],
        metadata: Optional[Mapping[str, object]] = None,
    ) -> None:
        self._app = app_cls(
            name=name or self.__class__.__name__,
            actor_pool=actor_pool,
            metadata=metadata,
        )

    @property
    def application(self) -> NodeApplication:
        return self._app

    @property
    def pool(self) -> ActorPool:
        return self._app.pool

    async def shutdown_distributed(self) -> None:
        await self._app.shutdown()


class DistributedHonestNode(_DistributedNodeBase, HonestNode):
    """
    Honest node whose heavy-weight computations are executed through an actor
    pool. Users only implement the *local* gradient computation; the framework
    wraps it in a computation graph automatically and dispatches it via the
    scheduler when the node's APIs are called.
    """

    def __init__(
        self,
        *,
        actor_pool: ActorPool | Sequence[ActorPoolConfig],
        aggregator: Aggregator,
        metadata: Optional[Mapping[str, object]] = None,
        name: Optional[str] = None,
    ) -> None:
        self._aggregator = aggregator
        super().__init__(
            app_cls=HonestNodeApplication,
            name=name,
            actor_pool=actor_pool,
            metadata=metadata,
        )
        self._register_aggregation_pipeline()
        self._register_gradient_pipeline()

    # ---- hooks for subclasses -------------------------------------------------
    def local_honest_gradient(self, *, x, y):
        """
        Subclasses override this to provide the *local* gradient computation. The
        base ``honest_gradient`` API will run it through the scheduler.
        """
        raise NotImplementedError("local_honest_gradient() must be implemented by subclasses.")

    # ---- pipeline registration ------------------------------------------------
    def _register_aggregation_pipeline(self) -> None:
        graph = make_single_operator_graph(
            node_name="aggregate",
            operator=self._aggregator,
            input_keys=("gradients",),
        )
        self._app.register_pipeline(self._app.AGGREGATION_PIPELINE, graph)

    def _register_gradient_pipeline(self) -> None:
        op = CallableOp(self._gradient_callable, input_mapping={"x": "x", "y": "y"})
        graph = make_single_operator_graph(
            node_name="honest_gradient",
            operator=op,
            input_keys=("x", "y"),
        )
        self._app.register_pipeline(self._app.GRADIENT_PIPELINE, graph)

    def _gradient_callable(self, *, x, y):
        return self.local_honest_gradient(x=x, y=y)

    # ---- public API -----------------------------------------------------------
    async def aggregate(self, gradients) -> torch.Tensor:
        handles = self._register_shared_gradients(gradients)
        try:
            return await self._app.aggregate(gradients=handles)
        finally:
            self._cleanup_shared_gradients(handles)

    def aggregate_sync(self, gradients):
        handles = self._register_shared_gradients(gradients)
        try:
            return self._app.aggregate_sync(gradients=handles)
        finally:
            self._cleanup_shared_gradients(handles)

    def honest_gradient(self, x, y):
        return self._app.honest_gradient_sync({"x": x, "y": y})

    def _register_shared_gradients(self, grads: Sequence[torch.Tensor]) -> list[SharedTensorHandle]:
        handles: list[SharedTensorHandle] = []
        for grad in grads:
            arr = grad.detach().cpu().numpy()
            handles.append(register_tensor(np.array(arr, copy=True)))
        return handles

    def _cleanup_shared_gradients(self, handles: Sequence[SharedTensorHandle]) -> None:
        for handle in handles:
            cleanup_tensor(handle)


class DistributedByzantineNode(_DistributedNodeBase, ByzantineNode):
    _distributed_user_bz = None  # type: ignore[assignment]

    def __init_subclass__(cls, **kwargs):
        """Initialize a subclass with distributed node configuration."""
        super().__init_subclass__(**kwargs)
        inherited_impl = getattr(cls, "_distributed_user_bz", None)
        user_impl = cls.__dict__.get("byzantine_gradient")
        if user_impl is None or user_impl is DistributedByzantineNode.byzantine_gradient:
            cls._distributed_user_bz = inherited_impl
        else:
            cls._distributed_user_bz = user_impl
            cls.byzantine_gradient = DistributedByzantineNode.byzantine_gradient  # type: ignore[assignment]

    """
    Byzantine node that automatically wraps its attack into a computation graph.
    """

    def __init__(
        self,
        *,
        actor_pool: ActorPool | Sequence[ActorPoolConfig],
        attack: Attack | None = None,
        metadata: Optional[Mapping[str, object]] = None,
        name: Optional[str] = None,
    ) -> None:
        self.attack = attack
        super().__init__(
            app_cls=ByzantineNodeApplication,
            name=name,
            actor_pool=actor_pool,
            metadata=metadata,
        )
        self._custom_bz_callable = None
        self._custom_input_keys: tuple[str, ...] = ()
        self._custom_required_keys: tuple[str, ...] = ()
        user_impl = getattr(type(self), "_distributed_user_bz", None)
        if user_impl is not None:
            self._custom_bz_callable = user_impl.__get__(self, type(self))
            self._register_custom_byzantine_pipeline()
        else:
            if self.attack is None:
                raise ValueError(
                    "DistributedByzantineNode requires an Attack instance when byzantine_gradient is not overridden."
                )
            self._register_attack_pipeline()

    def _register_attack_pipeline(self) -> None:
        op = self.attack

        input_keys = []
        if getattr(self.attack, "uses_model_batch", False):
            input_keys.extend(("model", "x", "y"))
        if getattr(self.attack, "uses_honest_grads", False):
            input_keys.append("honest_grads")
        if getattr(self.attack, "uses_base_grad", False):
            input_keys.append("base_grad")

        graph = make_single_operator_graph(
            node_name="attack",
            operator=op,
            input_keys=input_keys,
        )
        self._app.register_pipeline(self._app.ATTACK_PIPELINE, graph)

    def _register_custom_byzantine_pipeline(self) -> None:
        assert self._custom_bz_callable is not None
        sig = inspect.signature(self._custom_bz_callable)
        mapping: dict[str, str] = {}
        input_keys: list[str] = []
        required: list[str] = []
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            mapping[name] = name
            input_keys.append(name)
            if param.default is inspect.Signature.empty:
                required.append(name)
        self._custom_input_keys = tuple(input_keys)
        self._custom_required_keys = tuple(required)
        op = RemoteCallableOp(self._custom_bz_callable, input_mapping=mapping)
        graph = make_single_operator_graph(
            node_name="attack",
            operator=op,
            input_keys=input_keys,
        )
        self._app.register_pipeline(self._app.ATTACK_PIPELINE, graph)

    # Allow subclasses to customise default attack inputs
    def prepare_attack_inputs(
        self, *, x=None, y=None, honest_grads=None, base_grad=None, model=None
    ) -> Mapping[str, object]:
        if self._custom_bz_callable is not None:
            raise RuntimeError(
                "prepare_attack_inputs should not be used when byzantine_gradient is overridden."
            )
        inputs = {}
        if getattr(self.attack, "uses_model_batch", False):
            if model is None:
                raise ValueError("Attack requires 'model' but none was supplied.")
            if x is None or y is None:
                raise ValueError("Attack requires batch inputs 'x' and 'y'.")
            inputs["model"] = model
            inputs["x"] = x
            inputs["y"] = y
        if getattr(self.attack, "uses_honest_grads", False):
            if honest_grads is None:
                raise ValueError("Attack requires 'honest_grads'.")
            inputs["honest_grads"] = honest_grads
        if getattr(self.attack, "uses_base_grad", False):
            if base_grad is None:
                raise ValueError("Attack requires 'base_grad'.")
            inputs["base_grad"] = base_grad
        return inputs

    def _build_custom_inputs(
        self,
        *,
        x=None,
        y=None,
        honest_grads=None,
        base_grad=None,
        model=None,
    ) -> Mapping[str, object]:
        data = {
            "x": x,
            "y": y,
            "honest_grads": honest_grads,
            "base_grad": base_grad,
            "model": model,
        }
        inputs: dict[str, object] = {}
        for key in self._custom_input_keys:
            value = data.get(key)
            if key in self._custom_required_keys and value is None:
                raise ValueError(f"Custom byzantine_gradient requires argument {key!r}.")
            if value is not None:
                inputs[key] = value
        return inputs

    async def run_attack(self, *, inputs: Mapping[str, object]) -> torch.Tensor:
        return await self._app.run_attack(inputs=inputs)

    def byzantine_gradient(self, x, y, honest_grads=None):
        if self._custom_bz_callable is not None:
            inputs = self._build_custom_inputs(
                x=x,
                y=y,
                honest_grads=honest_grads,
            )
            return self._app.run_attack_sync(inputs=inputs)
        inputs = self.prepare_attack_inputs(
            x=x,
            y=y,
            honest_grads=honest_grads,
        )
        return self._app.run_attack_sync(inputs=inputs)

    async def byzantine_gradient_async(
        self, *, x=None, y=None, honest_grads=None, base_grad=None, model=None
    ):
        if self._custom_bz_callable is not None:
            inputs = self._build_custom_inputs(
                x=x,
                y=y,
                honest_grads=honest_grads,
                base_grad=base_grad,
                model=model,
            )
            return await self._app.run_attack(inputs=inputs)
        inputs = self.prepare_attack_inputs(
            x=x,
            y=y,
            honest_grads=honest_grads,
            base_grad=base_grad,
            model=model,
        )
        return await self._app.run_attack(inputs=inputs)


__all__ = [
    "DistributedHonestNode",
    "DistributedByzantineNode",
]
