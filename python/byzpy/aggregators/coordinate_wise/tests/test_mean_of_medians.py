import numpy as np
import pytest
import torch

from byzpy.aggregators.coordinate_wise.mean_of_medians import MeanOfMedians
from byzpy.engine.graph.operator import OpContext
from byzpy.engine.storage.shared_store import cleanup_tensor


def test_meamed_torch_n3_f1():
    grads = [
        torch.tensor([0.0, 0.0]),
        torch.tensor([1.0, 100.0]),
        torch.tensor([100.0, 1.0]),
    ]
    agg = MeanOfMedians(f=1)
    out = agg.aggregate(grads)
    assert torch.allclose(out, torch.tensor([0.5, 0.5]), atol=1e-7)


def test_meamed_torch_n5_f2():
    grads = [
        torch.tensor([-10.0, 0.0]),
        torch.tensor([1.0, 100.0]),
        torch.tensor([2.0, 2.0]),
        torch.tensor([3.0, -100.0]),
        torch.tensor([100.0, 1.0]),
    ]
    agg = MeanOfMedians(f=2)
    out = agg.aggregate(grads)
    assert torch.allclose(out, torch.tensor([2.0, 1.0]))


def test_meamed_empty_or_bad_f_raises():
    agg = MeanOfMedians(f=1)
    with pytest.raises(ValueError):
        _ = agg.aggregate([])
    # f >= n
    agg_bad = MeanOfMedians(f=1)
    with pytest.raises(ValueError):
        _ = agg_bad.aggregate([np.array([1.0])])


def test_meamed_subtasks_scale_with_pool():
    grads = [torch.arange(0, 16384, dtype=torch.float32) for _ in range(8)]
    inputs = {"gradients": grads}

    def _count(pool_size: int) -> int:
        agg = MeanOfMedians(f=1, chunk_size=1024)
        ctx = OpContext(node_name="node", metadata={"pool_size": pool_size})
        subtasks = list(agg.create_subtasks(inputs, context=ctx))
        handle = agg._active_handle  # type: ignore[attr-defined]
        if handle is not None:
            cleanup_tensor(handle)
            agg._active_handle = None  # type: ignore[attr-defined]
        agg._flat_shape = None  # type: ignore[attr-defined]
        return len(subtasks)

    assert _count(8) > _count(1)


def test_meamed_chunk_matches_direct():
    grads = [torch.randn(4096) for _ in range(32)]
    agg = MeanOfMedians(f=5, chunk_size=1024)
    direct = agg.aggregate(grads)

    inputs = {"gradients": grads}
    ctx = OpContext(node_name="node", metadata={"pool_size": 4})
    subtasks = list(agg.create_subtasks(inputs, context=ctx))
    partials = [task.fn(*task.args, **task.kwargs) for task in subtasks]
    chunked = agg.reduce_subtasks(partials, inputs, context=ctx)

    assert torch.allclose(direct, chunked, atol=1e-6)
