from __future__ import annotations

import torch

from byzpy.aggregators.norm_wise.caf import CAF
from byzpy.engine.graph.operator import OpContext


def test_caf_returns_mean_for_symmetric_data() -> None:
    grads = [
        torch.tensor([1.0, 0.0]),
        torch.tensor([-1.0, 0.0]),
        torch.tensor([0.0, 1.0]),
        torch.tensor([0.0, -1.0]),
    ]
    agg = CAF(f=1)
    out = agg.aggregate(grads)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


def test_caf_invalid_f_raises() -> None:
    grads = [torch.tensor([0.0]) for _ in range(2)]
    agg = CAF(f=1)
    try:
        agg.aggregate(grads)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("CAF should raise when 2f >= n")


def test_caf_chunk_matches_direct() -> None:
    grads = [torch.randn(256) for _ in range(32)]
    args = {"f": 4, "chunk_size": 4, "power_iters": 2}
    direct = CAF(**args).aggregate(grads)

    chunked = CAF(**args)
    inputs = {"gradients": grads}
    ctx = OpContext(node_name="caf", metadata={"pool_size": 4})
    subtasks = list(chunked.create_subtasks(inputs, context=ctx))
    partials = [task.fn(*task.args, **task.kwargs) for task in subtasks]
    reduced = chunked.reduce_subtasks(partials, inputs, context=ctx)
    assert torch.allclose(direct, reduced, atol=1e-5)
