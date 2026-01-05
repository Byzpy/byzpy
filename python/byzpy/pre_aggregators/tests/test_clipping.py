from __future__ import annotations

import torch

from byzpy.engine.graph.operator import OpContext
from byzpy.pre_aggregators.clipping import Clipping


def test_clipping_basic_scaling() -> None:
    pre = Clipping(threshold=1.0)
    vecs = [torch.tensor([3.0, 0.0]), torch.tensor([0.0, 0.5])]
    out = pre.pre_aggregate(vecs)
    assert torch.allclose(out[0], torch.tensor([1.0, 0.0]))
    assert torch.allclose(out[1], torch.tensor([0.0, 0.5]))


def test_clipping_chunk_matches_direct() -> None:
    vecs = [torch.randn(128) for _ in range(10)]
    args = {"threshold": 0.5, "chunk_size": 3}
    direct = Clipping(**args).pre_aggregate(vecs)

    chunked = Clipping(**args)
    ctx = OpContext(node_name="clip", metadata={"pool_size": 4})
    inputs = {"vectors": vecs}
    subtasks = list(chunked.create_subtasks(inputs, context=ctx))
    partials = [task.fn(*task.args, **task.kwargs) for task in subtasks]
    reduced = chunked.reduce_subtasks(partials, inputs, context=ctx)
    assert all(torch.allclose(a, b) for a, b in zip(direct, reduced))
