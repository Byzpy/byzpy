import torch

from byzpy.aggregators.geometric_wise.krum import Krum, MultiKrum
from byzpy.engine.graph.operator import OpContext
from byzpy.engine.storage.shared_store import cleanup_tensor


def test_krum_torch_basic():
    grads = [
        torch.tensor([0.0, 0.0]),
        torch.tensor([0.0, 0.0]),
        torch.tensor([0.0, 0.0]),
        torch.tensor([0.0, 0.0]),
        torch.tensor([100.0, 100.0]),
    ]
    agg = Krum(f=1)
    out = agg.aggregate(grads)
    assert torch.allclose(out, torch.tensor([0.0, 0.0]))


def test_multi_krum_torch_q3():
    grads = [
        torch.tensor([0.0, 0.0]),
        torch.tensor([0.1, 0.0]),
        torch.tensor([-0.1, 0.0]),
        torch.tensor([0.0, 0.1]),
        torch.tensor([100.0, 100.0]),
    ]
    agg = MultiKrum(f=1, q=3)
    out = agg.aggregate(grads)
    assert torch.allclose(out, torch.tensor([0.0, 0.03333333]), atol=5e-2)


def test_multi_krum_chunking_matches_direct():
    grads = [
        torch.tensor([0.0, 0.0]),
        torch.tensor([0.1, 0.0]),
        torch.tensor([-0.1, 0.0]),
        torch.tensor([0.0, 0.1]),
        torch.tensor([0.05, -0.05]),
        torch.tensor([100.0, 100.0]),
    ]
    agg = MultiKrum(f=1, q=2, chunk_size=2)
    direct = agg.aggregate(grads)

    inputs = {"gradients": grads}
    subtasks = list(agg.create_subtasks(inputs, context=None))
    partials = [task.fn(*task.args, **task.kwargs) for task in subtasks]
    chunked = agg.reduce_subtasks(partials, inputs, context=None)

    assert isinstance(chunked, torch.Tensor)
    assert torch.allclose(chunked, direct)


def test_krum_chunking_matches_direct():
    grads = [
        torch.tensor([0.0, 0.0]),
        torch.tensor([0.1, -0.2]),
        torch.tensor([-0.1, 0.15]),
        torch.tensor([0.0, 0.05]),
        torch.tensor([100.0, 100.0]),
    ]
    agg = Krum(f=1, chunk_size=2)
    direct = agg.aggregate(grads)

    inputs = {"gradients": grads}
    ctx = OpContext(node_name="krum", metadata={"pool_size": 2})
    subtasks = list(agg.create_subtasks(inputs, context=ctx))
    partials = [task.fn(*task.args, **task.kwargs) for task in subtasks]
    chunked = agg.reduce_subtasks(partials, inputs, context=ctx)

    assert torch.allclose(chunked, direct)


def test_multi_krum_subtasks_scale_with_pool_size():
    grads = [torch.arange(0, 16, dtype=torch.float32) for _ in range(640)]
    agg = MultiKrum(f=32, q=16, chunk_size=64)
    inputs = {"gradients": grads}

    ctx_single = OpContext(node_name="node", metadata={"pool_size": 1})
    ctx_many = OpContext(node_name="node", metadata={"pool_size": 8})

    def _collect(ctx):
        sts = list(agg.create_subtasks(inputs, context=ctx))
        handle = agg._active_handle  # type: ignore[attr-defined]
        if handle is not None:
            cleanup_tensor(handle)
            agg._active_handle = None  # type: ignore[attr-defined]
        return sts

    single = _collect(ctx_single)
    many = _collect(ctx_many)

    assert len(many) > len(single)
