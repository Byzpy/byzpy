import torch

from byzpy.aggregators.coordinate_wise.median import CoordinateWiseMedian
from byzpy.engine.graph.operator import OpContext


def test_median_torch_backend():
    grads = [
        torch.tensor([1.0, 5.0, 3.0]),
        torch.tensor([2.0, 3.0, 40.0]),
        torch.tensor([10.0, 20.0, 30.0]),
    ]
    agg = CoordinateWiseMedian()
    result = agg.aggregate(grads)
    assert isinstance(result, torch.Tensor)
    assert torch.allclose(result, torch.tensor([2.0, 5.0, 30.0]))


def test_median_chunk_matches_direct():
    grads = [torch.randn(4096) for _ in range(48)]
    agg = CoordinateWiseMedian(chunk_size=512)
    direct = agg.aggregate(grads)

    inputs = {"gradients": grads}
    ctx = OpContext(node_name="median", metadata={"pool_size": 4})
    subtasks = list(agg.create_subtasks(inputs, context=ctx))
    partials = [task.fn(*task.args, **task.kwargs) for task in subtasks]
    chunked = agg.reduce_subtasks(partials, inputs, context=ctx)

    assert torch.allclose(direct, chunked, atol=1e-6)
