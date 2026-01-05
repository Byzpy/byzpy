import torch

from byzpy.engine.graph.operator import OpContext
from byzpy.engine.storage.shared_store import cleanup_tensor
from byzpy.pre_aggregators import NearestNeighborMixing


def test_nnm_simple_line():
    agg = NearestNeighborMixing(f=1)
    xs = [torch.tensor([0.0, 0.0]), torch.tensor([1.0, 0.0]), torch.tensor([10.0, 0.0])]
    out = agg.pre_aggregate(xs)
    got = torch.stack(out, dim=0)
    exp = torch.tensor([[0.5, 0.0], [0.5, 0.0], [5.5, 0.0]])
    assert torch.allclose(got, exp, atol=1e-7)


def test_nnm_chunk_matches_direct():
    xs = [torch.randn(128) for _ in range(16)]
    direct = NearestNeighborMixing(f=3).pre_aggregate(xs)

    chunked = NearestNeighborMixing(f=3, feature_chunk_size=32)
    inputs = {"vectors": xs}
    subtasks = list(chunked.create_subtasks(inputs, context=None))
    partials = [task.fn(*task.args, **task.kwargs) for task in subtasks]
    reduced = chunked.reduce_subtasks(partials, inputs, context=None)

    stacked_direct = torch.stack(direct)
    stacked_chunk = torch.stack(reduced)
    assert torch.allclose(stacked_direct, stacked_chunk, atol=1e-6)


def test_nnm_subtasks_scale_with_pool():
    xs = [torch.randn(16384) for _ in range(32)]
    inputs = {"vectors": xs}

    def _count(pool_size: int) -> int:
        op = NearestNeighborMixing(f=4, feature_chunk_size=1024)
        ctx = OpContext(node_name="nnm", metadata={"pool_size": pool_size})
        subtasks = list(op.create_subtasks(inputs, context=ctx))
        handle = op._active_handle  # type: ignore[attr-defined]
        if handle is not None:
            cleanup_tensor(handle)
            op._active_handle = None  # type: ignore[attr-defined]
        op._flat_shape = None  # type: ignore[attr-defined]
        op._like_template = None  # type: ignore[attr-defined]
        op._n = None  # type: ignore[attr-defined]
        return len(subtasks)

    assert _count(8) > _count(1)
