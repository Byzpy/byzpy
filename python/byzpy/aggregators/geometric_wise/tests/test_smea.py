import torch

from byzpy.aggregators.geometric_wise import SMEA
from byzpy.engine.graph.operator import OpContext


def test_smea_filters_outlier():
    honest = [
        torch.tensor([0.0, 0.0]),
        torch.tensor([1.0, 0.0]),
        torch.tensor([0.0, 1.0]),
    ]
    grads = honest + [torch.tensor([6.0, 6.0])]
    agg = SMEA(f=1)
    out = agg.aggregate(grads)
    expected = torch.mean(torch.stack(honest, dim=0), dim=0)
    assert torch.allclose(out, expected, atol=1e-6)


def test_smea_chunk_matches_direct():
    grads = [torch.randn(16) for _ in range(7)]

    direct = SMEA(f=2, chunk_size=3).aggregate(grads)

    chunked = SMEA(f=2, chunk_size=3)
    inputs = {"gradients": grads}
    ctx = OpContext(node_name="smea", metadata={"pool_size": 4})
    subtasks = list(chunked.create_subtasks(inputs, context=ctx))
    partials = [task.fn(*task.args, **task.kwargs) for task in subtasks]
    reduced = chunked.reduce_subtasks(
        partials, inputs, context=OpContext(node_name="smea", metadata={})
    )

    assert torch.allclose(direct, reduced, atol=1e-6)
