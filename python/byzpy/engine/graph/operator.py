from __future__ import annotations

import asyncio
import dataclasses
import inspect
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence

from .pool import ActorPool
from .subtask import SubTask


@dataclass(frozen=True)
class OpContext:
    """
    Runtime metadata passed to each operator invocation.
    """

    node_name: str
    metadata: Mapping[str, Any] | None = None


class Operator:
    """
    Base operator that can participate in a computation graph.
    """

    name: str = "operator"
    supports_subtasks: bool = False
    max_subtasks_inflight: int | None = None
    supports_barriered_subtasks: bool = False

    def compute(self, inputs: Mapping[str, Any], *, context: OpContext) -> Any:
        raise NotImplementedError

    def create_subtasks(
        self, inputs: Mapping[str, Any], *, context: OpContext
    ) -> Iterable[SubTask]:
        return []

    def reduce_subtasks(
        self,
        partials: Sequence[Any],
        inputs: Mapping[str, Any],
        *,
        context: OpContext,
    ) -> Any:
        raise RuntimeError(f"Operator {self.name} does not implement reduce_subtasks().")

    async def run_barriered_subtasks(
        self, inputs: Mapping[str, Any], *, context: OpContext, pool: ActorPool
    ) -> Any:
        raise RuntimeError(f"Operator {self.name} does not implement barriered subtasks.")

    async def run(
        self, inputs: Mapping[str, Any], *, context: OpContext, pool: ActorPool | None
    ) -> Any:
        can_barriered = self.supports_barriered_subtasks and pool is not None
        if can_barriered:
            return await _maybe_await(self.run_barriered_subtasks(inputs, context=context, pool=pool))  # type: ignore[arg-type]

        can_parallel = self.supports_subtasks and pool is not None and pool.size > 1

        if can_parallel:
            subtasks_iter = self.create_subtasks(inputs, context=context)
            partials = await self._run_subtasks(pool, subtasks_iter, self.max_subtasks_inflight, context)  # type: ignore[arg-type]
            if partials:
                return await _maybe_await(self.reduce_subtasks(partials, inputs, context=context))

        return await _maybe_await(self.compute(inputs, context=context))

    async def _run_subtasks(
        self,
        pool: ActorPool,
        subtasks: Iterable[SubTask],
        limit: int | None,
        context: OpContext,
    ) -> list[Any]:
        metadata = context.metadata or {}
        worker_affinities = metadata.get("worker_affinities")
        if worker_affinities:
            subtasks = _assign_worker_affinities(subtasks, worker_affinities)
        return await _run_subtasks_windowed(pool, subtasks, limit)


async def _maybe_await(val: Any) -> Any:
    if inspect.isawaitable(val):
        return await val
    return val


async def _run_subtasks_windowed(
    pool: ActorPool, subtasks: Iterable[SubTask], limit: int | None
) -> list[Any]:
    iterator = iter(subtasks)

    if limit is None or limit == 0:
        limit = max(1, pool.size * 8)
    elif limit < 0:
        limit = max(1, pool.size * abs(limit))

    pending: set[asyncio.Task[tuple[int, Any]]] = set()
    results: dict[int, Any] = {}
    total = 0

    async def _run_indexed(subtask: SubTask, order: int) -> tuple[int, Any]:
        res = await pool.run_subtask(subtask)
        return order, res

    async def _schedule_next() -> bool:
        nonlocal total
        try:
            st = next(iterator)
        except StopIteration:
            return False
        pending.add(asyncio.create_task(_run_indexed(st, total)))
        total += 1
        return True

    for _ in range(limit):
        if not await _schedule_next():
            break

    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            idx, value = task.result()
            results[idx] = value
            await _schedule_next()
    return [results[i] for i in range(total)]


def _assign_worker_affinities(
    subtasks: Iterable[SubTask], worker_affinities: Sequence[str]
) -> Iterable[SubTask]:
    if not worker_affinities:
        return subtasks

    def _iter():
        for idx, st in enumerate(subtasks):
            if st.affinity is not None:
                yield st
            else:
                hint = worker_affinities[idx % len(worker_affinities)]
                yield dataclasses.replace(st, affinity=hint)

    return _iter()


class MessageTriggerOp(Operator):
    """Operator that waits for a message before proceeding."""

    def __init__(self, message_type: str, timeout: Optional[float] = None):
        if not message_type:
            raise ValueError("message_type cannot be empty")
        self.message_type = message_type
        self.timeout = timeout
        self.name = f"message_trigger_{message_type}"

    async def run(
        self, inputs: Mapping[str, Any], *, context: OpContext, pool: ActorPool | None
    ) -> Any:
        scheduler = context.metadata.get("scheduler")
        if scheduler is None:
            raise RuntimeError("MessageTriggerOp requires scheduler in context metadata")

        message = await scheduler.wait_for_message(self.message_type, timeout=self.timeout)
        return message


__all__ = ["OpContext", "Operator", "MessageTriggerOp"]
