import random

import numpy as np
import pytest
import torch

from byzpy.aggregators.coordinate_wise._tiling import as_flat_batch
from byzpy.engine.graph.batch import FlatTensorBatch
from byzpy.engine.graph.operator import OpContext
from byzpy.engine.storage.shared_store import SharedTensorHandle, cleanup_tensor
from byzpy.pre_aggregators import Bucketing
from byzpy.pre_aggregators.bucketing import _bucketing_bucket_chunk, _to_like


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

    try:
        if isinstance(reduced, FlatTensorBatch):
            reduced_list = reduced.materialize_list()
        else:
            reduced_list = list(reduced)

        def _stack(vals):
            return torch.stack(
                [v if isinstance(v, torch.Tensor) else torch.as_tensor(v) for v in vals]
            )

        assert torch.allclose(_stack(direct), _stack(reduced_list), atol=1e-6)
    finally:
        if isinstance(reduced, FlatTensorBatch):
            reduced.release()


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


def _stack(vals):
    return torch.stack([v if isinstance(v, torch.Tensor) else torch.as_tensor(v) for v in vals])


def test_bucketing_rejects_invalid_init_args():
    with pytest.raises(ValueError, match="bucket_size must be >= 1"):
        Bucketing(bucket_size=0)
    with pytest.raises(ValueError, match="feature_chunk_size must be > 0"):
        Bucketing(bucket_size=1, feature_chunk_size=0)


def test_bucketing_pre_aggregate_empty_raises():
    with pytest.raises(ValueError, match="non-empty sequence"):
        Bucketing(bucket_size=2).pre_aggregate([])


def test_bucketing_invalid_perm_raises():
    xs = [torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0])]
    with pytest.raises(ValueError, match="permutation of range"):
        Bucketing(bucket_size=1, perm=[0, 1]).pre_aggregate(xs)
    with pytest.raises(ValueError, match="permutation of range"):
        Bucketing(bucket_size=1, perm=[0, 0, 1]).pre_aggregate(xs)


def test_bucketing_seeded_rng_is_deterministic():
    xs = [torch.tensor([float(v)]) for v in range(6)]
    first = Bucketing(bucket_size=2, rng=random.Random(7)).pre_aggregate(xs)
    second = Bucketing(bucket_size=2, rng=random.Random(7)).pre_aggregate(xs)
    assert torch.allclose(_stack(first), _stack(second))


def test_bucketing_pre_aggregate_flat_tensor_batch():
    xs = [torch.tensor([v], dtype=torch.float32) for v in [1.0, 2.0, 3.0, 4.0]]
    perm = [0, 1, 2, 3]
    batch = as_flat_batch(xs)
    try:
        out = Bucketing(bucket_size=2, perm=perm).pre_aggregate(batch)
        got = _stack(out).reshape(-1)
        assert torch.allclose(got, torch.tensor([1.5, 3.5]))
    finally:
        batch.release()


def test_bucketing_create_subtasks_skips_empty_inputs():
    agg = Bucketing(bucket_size=2)
    ctx = OpContext(node_name="bkt")
    assert list(agg.create_subtasks({}, context=ctx)) == []
    assert list(agg.create_subtasks({"vectors": []}, context=ctx)) == []
    assert list(agg.create_subtasks({"vectors": object()}, context=ctx)) == []

    empty_batch = FlatTensorBatch(
        handle=SharedTensorHandle(name="empty-bkt", shape=(0, 3), dtype="float32"),
        flat_shape=(3,),
        like=torch.zeros(3),
        owns_handle=False,
    )
    assert list(agg.create_subtasks({"vectors": empty_batch}, context=ctx)) == []


def test_bucketing_chunk_reuses_flat_tensor_batch_handle():
    xs = [torch.randn(8) for _ in range(6)]
    perm = list(range(len(xs)))
    batch = as_flat_batch(xs)
    try:
        agg = Bucketing(bucket_size=2, feature_chunk_size=2, perm=perm)
        ctx = OpContext(node_name="bkt", metadata={"pool_size": 2})
        subtasks = list(agg.create_subtasks({"vectors": batch}, context=ctx))
        assert agg._active_owned is False
        partials = [task.fn(*task.args, **task.kwargs) for task in subtasks]
        reduced = agg.reduce_subtasks(partials, {"vectors": batch}, context=ctx)
        try:
            reduced_list = (
                reduced.materialize_list()
                if isinstance(reduced, FlatTensorBatch)
                else list(reduced)
            )
            direct = Bucketing(bucket_size=2, perm=perm).pre_aggregate(xs)
            assert torch.allclose(_stack(direct), _stack(reduced_list), atol=1e-6)
            assert batch._released is False
        finally:
            if isinstance(reduced, FlatTensorBatch):
                reduced.release()
    finally:
        batch.release()


def test_bucketing_reduce_empty_partials_falls_back_to_compute():
    xs = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
    perm = [0, 1]
    agg = Bucketing(bucket_size=2, perm=perm)
    ctx = OpContext(node_name="bkt")
    inputs = {"vectors": xs}
    list(agg.create_subtasks(inputs, context=ctx))
    out = agg.reduce_subtasks([], inputs, context=ctx)
    assert torch.allclose(_stack(out), torch.tensor([[2.0, 3.0]]))
    assert agg._active_handle is None


def test_bucketing_reduce_empty_partials_without_state():
    xs = [torch.tensor([2.0]), torch.tensor([4.0])]
    agg = Bucketing(bucket_size=2, perm=[0, 1])
    out = agg.reduce_subtasks([], {"vectors": xs}, context=OpContext(node_name="bkt"))
    assert torch.allclose(_stack(out), torch.tensor([[3.0]]))


def test_bucketing_reduce_missing_state_raises():
    dummy = SharedTensorHandle(name="unused", shape=(1, 2), dtype="float32")
    ctx = OpContext(node_name="bkt")
    partials = [(0, np.zeros((1, 2)))]
    inputs = {"vectors": [torch.zeros(2)]}

    def _raise_with(**fields):
        agg = Bucketing(bucket_size=1)
        for name, value in fields.items():
            setattr(agg, name, value)
        with pytest.raises(RuntimeError, match="missing reduction state"):
            agg.reduce_subtasks(partials, inputs, context=ctx)

    _raise_with()
    _raise_with(_active_handle=dummy)
    _raise_with(_active_handle=dummy, _flat_shape=(2,))
    _raise_with(_active_handle=dummy, _flat_shape=(2,), _bucket_count=1)


def test_bucketing_reduce_casts_mismatched_partial_dtype():
    xs = [torch.tensor([1.0, 2.0], dtype=torch.float32), torch.tensor([3.0, 4.0])]
    perm = [0, 1]
    agg = Bucketing(bucket_size=2, perm=perm)
    ctx = OpContext(node_name="bkt")
    inputs = {"vectors": xs}
    subtasks = list(agg.create_subtasks(inputs, context=ctx))
    partials = [
        (offset, np.asarray(chunk, dtype=np.float64))
        for offset, chunk in (task.fn(*task.args, **task.kwargs) for task in subtasks)
    ]
    reduced = agg.reduce_subtasks(partials, inputs, context=ctx)
    try:
        reduced_list = (
            reduced.materialize_list() if isinstance(reduced, FlatTensorBatch) else list(reduced)
        )
        assert torch.allclose(_stack(reduced_list), torch.tensor([[2.0, 3.0]]))
    finally:
        if isinstance(reduced, FlatTensorBatch):
            reduced.release()


def test_bucketing_reduce_list_when_flat_batch_disabled(monkeypatch):
    monkeypatch.setenv("BYZPY_DISABLE_FLAT_BATCH", "1")
    xs = [torch.tensor([1.0, 3.0]), torch.tensor([5.0, 7.0])]
    perm = [0, 1]
    agg = Bucketing(bucket_size=2, perm=perm)
    ctx = OpContext(node_name="bkt")
    inputs = {"vectors": xs}
    subtasks = list(agg.create_subtasks(inputs, context=ctx))
    partials = [task.fn(*task.args, **task.kwargs) for task in subtasks]
    reduced = agg.reduce_subtasks(partials, inputs, context=ctx)
    assert not isinstance(reduced, FlatTensorBatch)
    assert torch.allclose(_stack(reduced), torch.tensor([[3.0, 5.0]]))


def test_bucketing_reduce_list_vectors_when_flat_batch_disabled(monkeypatch):
    monkeypatch.setenv("BYZPY_DISABLE_FLAT_BATCH", "1")
    xs = [[1.0, 2.0], [3.0, 4.0]]
    perm = [0, 1]
    agg = Bucketing(bucket_size=2, perm=perm)
    ctx = OpContext(node_name="bkt")
    inputs = {"vectors": xs}
    subtasks = list(agg.create_subtasks(inputs, context=ctx))
    partials = [task.fn(*task.args, **task.kwargs) for task in subtasks]
    reduced = agg.reduce_subtasks(partials, inputs, context=ctx)
    stacked = np.stack([np.asarray(v) for v in reduced])
    np.testing.assert_allclose(stacked, np.array([[2.0, 3.0]]))


def test_bucketing_create_subtasks_without_pool_metadata():
    xs = [torch.randn(4) for _ in range(4)]
    agg = Bucketing(bucket_size=2, feature_chunk_size=1, perm=list(range(4)))

    class _Bare:
        pass

    subtasks = list(agg.create_subtasks({"vectors": xs}, context=_Bare()))
    assert len(subtasks) >= 1
    if agg._active_handle is not None:
        cleanup_tensor(agg._active_handle)
        agg._active_handle = None
        agg._active_owned = False


def test_bucketing_bucket_chunk_skips_empty_indices():
    xs = [torch.tensor([1.0, 2.0], dtype=torch.float32), torch.tensor([3.0, 4.0])]
    batch = as_flat_batch(xs)
    try:
        offset, out = _bucketing_bucket_chunk(
            batch.handle, (np.array([], dtype=np.int64), np.array([0, 1], dtype=np.int64)), 3
        )
        assert offset == 3
        assert out.shape == (2, 2)
        np.testing.assert_allclose(out[0], np.zeros(2))
        np.testing.assert_allclose(out[1], np.array([2.0, 3.0]))
    finally:
        batch.release()


def test_bucketing_to_like_non_torch_like():
    arr = np.array([1.5, 2.5], dtype=np.float64)
    converted = _to_like(arr, [0.0, 0.0])
    assert torch.allclose(
        torch.as_tensor(converted, dtype=torch.float64),
        torch.tensor([1.5, 2.5], dtype=torch.float64),
    )


def test_bucketing_cleanup_state_owned_without_handle():
    agg = Bucketing(bucket_size=1)
    agg._active_owned = True
    agg._active_handle = None
    agg._flat_shape = (1,)
    agg._bucket_count = 1
    agg._like_template = torch.zeros(1)
    agg._cleanup_state()
    assert agg._active_owned is False
    assert agg._flat_shape is None
    assert agg._bucket_count is None
    assert agg._like_template is None
