import pytest
import torch

from byzpy.aggregators.norm_wise import CenteredClipping
from byzpy.engine.graph.operator import OpContext


class _StubPool:
    def __init__(self, size: int = 4):
        self.size = size
        self.calls = 0

    async def run_many(self, subtasks):
        self.calls += 1
        results = []
        for task in subtasks:
            results.append(task.fn(*task.args, **task.kwargs))
        return results

    async def run_subtask(self, subtask):
        self.calls += 1
        return subtask.fn(*subtask.args, **subtask.kwargs)


def test_cc_torch_basic_outlier_limited():
    honest = [
        torch.tensor([0.0, 0.0]),
        torch.tensor([0.1, -0.1]),
        torch.tensor([-0.1, 0.1]),
        torch.tensor([0.05, 0.0]),
    ]
    grads = honest + [torch.tensor([50.0, 50.0])]

    agg = CenteredClipping(c_tau=0.2, M=15, init="median")
    out = agg.aggregate(grads)

    assert torch.norm(out - torch.tensor([0.0, 0.0])).item() < 0.3


@pytest.mark.asyncio
async def test_cc_barriered_matches_direct():
    grads = [
        torch.tensor([0.0, 0.0]),
        torch.tensor([0.1, -0.1]),
        torch.tensor([-0.1, 0.1]),
        torch.tensor([0.05, 0.0]),
        torch.tensor([1.0, 1.0]),
        torch.tensor([-0.5, -0.25]),
    ]
    agg = CenteredClipping(c_tau=0.5, M=5, init="mean", chunk_size=2)
    ctx = OpContext(node_name="cc")
    pool = _StubPool()

    direct = agg.aggregate(grads)
    barriered = await agg.run_barriered_subtasks({"gradients": grads}, context=ctx, pool=pool)

    assert torch.allclose(barriered, direct, atol=1e-7)
    assert pool.calls > 0
