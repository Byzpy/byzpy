from __future__ import annotations

import torch

from byzpy.aggregators.geometric_wise.monna import MoNNA
from byzpy.engine.graph.operator import OpContext


def test_monna_basic_average() -> None:
    agg = MoNNA(f=1, reference_index=0)
    grads = [
        torch.tensor([0.0, 0.0]),
        torch.tensor([10.0, 0.0]),
        torch.tensor([0.0, 10.0]),
    ]
    out = agg.aggregate(grads)
    assert torch.allclose(out, torch.tensor([5.0, 0.0]))


def test_monna_reference_choice() -> None:
    agg = MoNNA(f=1, reference_index=1)
    grads = [
        torch.tensor([0.0, 0.0]),
        torch.tensor([10.0, 0.0]),
        torch.tensor([0.0, 10.0]),
    ]
    out = agg.aggregate(grads)
    assert torch.allclose(out, torch.tensor([5.0, 0.0]))


def test_monna_subtasks() -> None:
    agg = MoNNA(f=1, reference_index=2, chunk_size=1)
    grads = [torch.randn(4) for _ in range(4)]
    subtasks = list(
        agg.create_subtasks(
            {"gradients": grads},
            context=type("ctx", (), {"metadata": {"pool_size": 2}}),
        )
    )
    assert subtasks, "MoNNA should generate subtasks for chunking"


def test_monna_chunk_matches_direct() -> None:
    grads = [torch.randn(128) for _ in range(16)]
    args = {"f": 3, "reference_index": 5, "chunk_size": 2}
    direct = MoNNA(**args).aggregate(grads)

    chunked = MoNNA(**args)
    inputs = {"gradients": grads}
    ctx = OpContext(node_name="monna", metadata={"pool_size": 4})
    subtasks = list(chunked.create_subtasks(inputs, context=ctx))
    partials = [task.fn(*task.args, **task.kwargs) for task in subtasks]
    reduced = chunked.reduce_subtasks(partials, inputs, context=ctx)

    assert torch.allclose(direct, reduced, atol=1e-6)
