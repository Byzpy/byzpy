import torch

from byzpy.engine.graph.operator import OpContext
from byzpy.engine.storage.shared_store import cleanup_tensor
from byzpy.pre_aggregators import Bucketing


def test_bucketing_fixed_perm():
    xs = [torch.tensor([v], dtype=torch.float32) for v in [1, 2, 3, 4, 5]]

    # permutation: [2,0,4,1,3] -> buckets size 2: [3,1], [5,2], last [4]
    perm = [2, 0, 4, 1, 3]
    agg = Bucketing(bucket_size=2, perm=perm)
    out = agg.pre_aggregate(xs)

    # expected means: [2, 3.5, 4]
    got = torch.stack(out, dim=0).reshape(-1)
    assert torch.allclose(got, torch.tensor([2.0, 3.5, 4.0]))


def test_bucketing_chunk_matches_direct():
    xs = [torch.randn(1024) for _ in range(96)]
    perm = list(range(len(xs)))

    agg_direct = Bucketing(bucket_size=8, feature_chunk_size=512, perm=perm)
    direct = agg_direct.pre_aggregate(xs)

    agg_chunk = Bucketing(bucket_size=8, feature_chunk_size=512, perm=perm)
    inputs = {"vectors": xs}
    subtasks = list(
        agg_chunk.create_subtasks(
            inputs, context=OpContext(node_name="bkt", metadata={"pool_size": 4})
        )
    )
    partials = [task.fn(*task.args, **task.kwargs) for task in subtasks]
    reduced = agg_chunk.reduce_subtasks(
        partials, inputs, context=OpContext(node_name="bkt", metadata={})
    )

    def _stack(vals):
        return torch.stack([v if isinstance(v, torch.Tensor) else torch.as_tensor(v) for v in vals])

    assert torch.allclose(_stack(direct), _stack(reduced), atol=1e-6)


def test_bucketing_subtasks_scale_with_pool():
    xs = [torch.randn(65536) for _ in range(128)]
    inputs = {"vectors": xs}

    def _count(pool_size: int) -> int:
        agg = Bucketing(bucket_size=16, feature_chunk_size=4096)
        ctx = OpContext(node_name="bkt", metadata={"pool_size": pool_size})
        subtasks = list(agg.create_subtasks(inputs, context=ctx))
        handle = agg._active_handle  # type: ignore[attr-defined]
        if handle is not None:
            cleanup_tensor(handle)
            agg._active_handle = None  # type: ignore[attr-defined]
        agg._flat_shape = None  # type: ignore[attr-defined]
        agg._bucket_slices = None  # type: ignore[attr-defined]
        agg._bucket_count = None  # type: ignore[attr-defined]
        agg._like_template = None  # type: ignore[attr-defined]
        return len(subtasks)

    assert _count(8) > _count(1)
