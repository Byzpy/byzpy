import torch

from byzpy.engine.graph.operator import OpContext
from byzpy.pre_aggregators import ARC


def test_arc_expected_output_matches_reference():
    xs = [
        torch.tensor([1.0, 2.0, 3.0]),
        torch.tensor([4.0, 5.0, 6.0]),
        torch.tensor([7.0, 8.0, 9.0]),
    ]
    agg = ARC(f=1)
    out = agg.pre_aggregate(xs)
    stacked = torch.stack(
        [v if isinstance(v, torch.Tensor) else torch.as_tensor(v) for v in out], dim=0
    )
    expected = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [4.41004009, 5.04004582, 5.67005155],
        ]
    )
    assert torch.allclose(stacked, expected, atol=1e-6)


def test_arc_chunk_matches_direct():
    xs = [torch.randn(256) for _ in range(64)]

    direct = ARC(f=4, chunk_size=8).pre_aggregate(xs)

    chunked = ARC(f=4, chunk_size=8)
    inputs = {"vectors": xs}
    subtasks = list(
        chunked.create_subtasks(
            inputs, context=OpContext(node_name="arc", metadata={"pool_size": 4})
        )
    )
    partials = [task.fn(*task.args, **task.kwargs) for task in subtasks]
    reduced = chunked.reduce_subtasks(
        partials, inputs, context=OpContext(node_name="arc", metadata={})
    )

    def _stack(vals):
        return torch.stack(
            [v if isinstance(v, torch.Tensor) else torch.as_tensor(v) for v in vals],
            dim=0,
        )

    assert torch.allclose(_stack(direct), _stack(reduced), atol=1e-6)
