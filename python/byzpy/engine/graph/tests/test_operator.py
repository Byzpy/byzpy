from __future__ import annotations

import pytest

from byzpy.engine.graph.operator import OpContext, Operator
from byzpy.engine.graph.subtask import SubTask


class _RecordingOp(Operator):
    def __init__(self):
        self.calls = 0

    def compute(self, inputs, *, context):  # type: ignore[override]
        self.calls += 1
        return inputs["value"] + 1


class _AwaitableOp(Operator):
    def compute(self, inputs, *, context):  # type: ignore[override]
        async def _inner():
            return inputs["value"] * 2

        return _inner()


def _scale(value: int, factor: int) -> int:
    return value * factor


class _ParallelOp(Operator):
    supports_subtasks = True

    def __init__(self):
        self.compute_calls = 0
        self.reduced = None

    def compute(self, inputs, *, context):  # type: ignore[override]
        self.compute_calls += 1
        factor = inputs["factor"]
        offset = inputs.get("offset", 0)
        return sum(val * factor for val in inputs["parts"]) + offset

    def create_subtasks(self, inputs, *, context):
        factor = inputs["factor"]
        return [
            SubTask(fn=_scale, args=(value,), kwargs={"factor": factor}, name=f"part-{idx}")
            for idx, value in enumerate(inputs["parts"])
        ]

    def reduce_subtasks(self, partials, inputs, *, context):
        async def _inner():
            self.reduced = list(partials)
            return sum(partials) + inputs.get("offset", 0)

        return _inner()


class _FakePool:
    def __init__(self, *, size: int):
        self.size = size
        self.seen = None
        self.executed = None
        self.calls = 0
        self.subtask_calls = 0
        self.subtasks = []

    async def run_many(self, subtasks):
        self.seen = list(subtasks)
        self.calls += 1
        results = []
        for task in self.seen:
            results.append(task.fn(*task.args, **task.kwargs))
        self.executed = results
        return results

    async def run_subtask(self, subtask: SubTask):
        self.subtask_calls += 1
        self.subtasks.append(subtask)
        return subtask.fn(*subtask.args, **subtask.kwargs)


class _BarrieredOp(Operator):
    supports_barriered_subtasks = True

    def __init__(self):
        self.rounds = 0
        self.partial_history: list[list[int]] = []

    async def run_barriered_subtasks(self, inputs, *, context, pool):  # type: ignore[override]
        total = 0
        for _ in range(inputs["rounds"]):
            self.rounds += 1
            subtasks = [
                SubTask(
                    fn=_scale,
                    args=(value,),
                    kwargs={"factor": inputs["factor"]},
                    name=f"round-{self.rounds}-{idx}",
                )
                for idx, value in enumerate(inputs["parts"])
            ]
            ctx = OpContext(node_name="test")
            partials = await self._run_subtasks(pool, subtasks, None, ctx)
            self.partial_history.append(list(partials))
            total += sum(partials)
        return total


@pytest.mark.asyncio
async def test_operator_run_invokes_compute_without_pool():
    op = _RecordingOp()
    ctx = OpContext(node_name="node")

    out = await op.run({"value": 2}, context=ctx, pool=None)

    assert out == 3
    assert op.calls == 1


@pytest.mark.asyncio
async def test_operator_run_awaits_coroutines():
    op = _AwaitableOp()
    ctx = OpContext(node_name="node")

    out = await op.run({"value": 5}, context=ctx, pool=None)

    assert out == 10


@pytest.mark.asyncio
async def test_operator_run_uses_pool_for_subtasks():
    op = _ParallelOp()
    pool = _FakePool(size=2)
    ctx = OpContext(node_name="node")

    out = await op.run({"parts": [1, 2, 3], "factor": 2, "offset": 5}, context=ctx, pool=pool)

    assert out == 17
    assert pool.subtask_calls == 3
    assert [task.fn(*task.args, **task.kwargs) for task in pool.subtasks] == [2, 4, 6]
    assert op.reduced == [2, 4, 6]
    assert op.compute_calls == 0


@pytest.mark.asyncio
async def test_operator_run_falls_back_to_compute_when_no_parallelism():
    op = _ParallelOp()
    pool = _FakePool(size=1)
    ctx = OpContext(node_name="node")

    out = await op.run({"parts": [1, 1], "factor": 3, "offset": 4}, context=ctx, pool=pool)

    assert out == 10
    assert op.compute_calls == 1
    assert pool.seen is None


@pytest.mark.asyncio
async def test_operator_run_falls_back_when_no_subtasks_created():
    class _EmptyParallel(_ParallelOp):
        def create_subtasks(self, inputs, *, context):
            return []

    op = _EmptyParallel()
    pool = _FakePool(size=3)
    ctx = OpContext(node_name="node")

    out = await op.run({"parts": [2], "factor": 5, "offset": 1}, context=ctx, pool=pool)

    assert out == 11
    assert op.compute_calls == 1
    assert pool.seen is None


@pytest.mark.asyncio
async def test_operator_run_supports_barriered_mode():
    op = _BarrieredOp()
    pool = _FakePool(size=2)
    ctx = OpContext(node_name="node")

    out = await op.run({"parts": [1, 3], "factor": 2, "rounds": 2}, context=ctx, pool=pool)

    assert out == (1 + 3) * 2 * 2
    assert op.rounds == 2
    assert len(op.partial_history) == 2
    assert pool.subtask_calls == len([1, 3]) * 2
