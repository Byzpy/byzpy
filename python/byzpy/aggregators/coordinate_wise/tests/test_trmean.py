import torch

from byzpy.aggregators.coordinate_wise.trimmed_mean import CoordinateWiseTrimmedMean
from byzpy.engine.graph.operator import OpContext
from byzpy.engine.storage.shared_store import cleanup_tensor


def test_trimmed_mean_torch_f1_equals_median_when_n3():
    grads = [
        torch.tensor([1.0, 5.0, 3.0]),
        torch.tensor([2.0, 3.0, 40.0]),
        torch.tensor([10.0, 20.0, 30.0]),
    ]
    agg = CoordinateWiseTrimmedMean(f=1)
    out = agg.aggregate(grads)
    assert torch.allclose(out, torch.tensor([2.0, 5.0, 30.0]))


def test_trimmed_mean_torch_general():
    grads = [
        torch.tensor([0.0, 0.0, 0.0]),
        torch.tensor([10.0, 10.0, 10.0]),
        torch.tensor([20.0, 20.0, 20.0]),
        torch.tensor([30.0, 30.0, 30.0]),
        torch.tensor([40.0, 40.0, 40.0]),
    ]
    agg = CoordinateWiseTrimmedMean(f=1)
    out = agg.aggregate(grads)
    assert torch.allclose(out, torch.tensor([20.0, 20.0, 20.0]))


def test_trimmed_mean_subtasks_increase_with_pool_size():
    grads = [torch.arange(0, 1024, dtype=torch.float32) for _ in range(8)]
    agg = CoordinateWiseTrimmedMean(f=1, chunk_size=64)
    inputs = {"gradients": grads}

    ctx_single = OpContext(node_name="test", metadata={"pool_size": 1})
    ctx_many = OpContext(node_name="test", metadata={"pool_size": 8})

    def _subtasks(ctx):
        sts = list(agg.create_subtasks(inputs, context=ctx))
        handle = agg._active_handle  # type: ignore[attr-defined]
        if handle is not None:
            cleanup_tensor(handle)
            agg._active_handle = None  # type: ignore[attr-defined]
        return sts

    single = _subtasks(ctx_single)
    many = _subtasks(ctx_many)
    assert len(many) > len(single)
