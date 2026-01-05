import pytest
import torch

from byzpy.aggregators.geometric_wise.geometric_median import GeometricMedian
from byzpy.engine.graph.operator import OpContext


class _StubPool:
    def __init__(self, size: int = 4):
        self.size = size
        self.calls = 0

    async def run_subtask(self, subtask):
        self.calls += 1
        return subtask.fn(*subtask.args, **subtask.kwargs)


def test_gm_torch_1d_matches_numpy_median():
    grads = [
        torch.tensor([0.0]),
        torch.tensor([1.0]),
        torch.tensor([100.0]),
        torch.tensor([1.5]),
    ]
    agg = GeometricMedian(tol=1e-8, max_iter=512, init="median")
    out = agg.aggregate(grads)
    torch_med = torch.median(torch.stack(grads), dim=0)
    assert torch.allclose(
        out, torch.tensor([torch_med], dtype=out.dtype, device=out.device), atol=1e-6
    )


def test_gm_torch_outlier_robust():
    grads = [
        torch.tensor([0.0, 0.0]),
        torch.tensor([0.1, -0.1]),
        torch.tensor([-0.1, 0.1]),
        torch.tensor([0.05, 0.0]),
        torch.tensor([100.0, 100.0]),
    ]
    agg = GeometricMedian(tol=1e-8, max_iter=512, init="median")
    out = agg.aggregate(grads)
    assert (
        torch.norm(out - torch.tensor([0.0, 0.0], dtype=out.dtype, device=out.device)).item() < 0.2
    )


@pytest.mark.asyncio
async def test_gm_barriered_matches_direct():
    grads = [
        torch.tensor([0.0, 0.0]),
        torch.tensor([0.5, -0.4]),
        torch.tensor([-0.25, 0.2]),
        torch.tensor([0.1, 0.05]),
        torch.tensor([100.0, 100.0]),
    ]
    agg = GeometricMedian(tol=1e-8, max_iter=128, init="median", chunk_size=2)
    ctx = OpContext(node_name="gm")
    pool = _StubPool(size=4)

    direct = agg.aggregate(grads)
    barriered = await agg.run_barriered_subtasks({"gradients": grads}, context=ctx, pool=pool)

    assert torch.allclose(barriered, direct, atol=1e-6)
    assert pool.calls > 0
