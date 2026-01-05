import torch

from byzpy.aggregators.norm_wise.comparative_gradient_elimination import (
    ComparativeGradientElimination,
)
from byzpy.engine.graph.operator import OpContext
from byzpy.engine.storage.shared_store import cleanup_tensor


def test_cge_direct_matches_chunked():
    grads = [torch.randn(1024) for _ in range(16)]
    direct = ComparativeGradientElimination(f=3).aggregate(grads)

    chunked = ComparativeGradientElimination(f=3, chunk_size=128)
    inputs = {"gradients": grads}
    subtasks = list(chunked.create_subtasks(inputs, context=None))
    partials = [task.fn(*task.args, **task.kwargs) for task in subtasks]
    reduced = chunked.reduce_subtasks(partials, inputs, context=None)

    assert torch.allclose(reduced, direct, atol=1e-6)


def test_cge_subtasks_scale_with_pool():
    grads = [torch.arange(0, 65536, dtype=torch.float32) for _ in range(8)]
    inputs = {"gradients": grads}

    def _count(pool_size: int) -> int:
        agg = ComparativeGradientElimination(f=1, chunk_size=4096)
        ctx = OpContext(node_name="cge", metadata={"pool_size": pool_size})
        subtasks = list(agg.create_subtasks(inputs, context=ctx))
        handle = agg._active_handle  # type: ignore[attr-defined]
        if handle is not None:
            cleanup_tensor(handle)
            agg._active_handle = None  # type: ignore[attr-defined]
        agg._flat_shape = None  # type: ignore[attr-defined]
        agg._n = None  # type: ignore[attr-defined]
        return len(subtasks)

    assert _count(8) > _count(1)
