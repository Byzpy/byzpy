import torch

from byzpy.attacks.little import LittleAttack
from byzpy.engine.graph.operator import OpContext
from byzpy.engine.storage.shared_store import cleanup_tensor


def test_little_chunk_matches_direct():
    grads = [torch.randn(4096) for _ in range(32)]
    direct = LittleAttack(f=3).apply(honest_grads=grads)

    chunked = LittleAttack(f=3, chunk_size=2048)
    inputs = {"honest_grads": grads}
    subtasks = list(chunked.create_subtasks(inputs, context=None))
    partials = [task.fn(*task.args, **task.kwargs) for task in subtasks]
    reduced = chunked.reduce_subtasks(partials, inputs, context=None)

    assert torch.allclose(reduced, direct, atol=1e-6)


def test_little_subtasks_scale_with_pool():
    grads = [torch.randn(8192) for _ in range(32)]
    inputs = {"honest_grads": grads}

    def _count(pool_size: int) -> int:
        attack = LittleAttack(f=3, chunk_size=512)
        ctx = OpContext(node_name="little", metadata={"pool_size": pool_size})
        subtasks = list(attack.create_subtasks(inputs, context=ctx))
        handle = attack._active_handle  # type: ignore[attr-defined]
        if handle is not None:
            cleanup_tensor(handle)
            attack._active_handle = None  # type: ignore[attr-defined]
        attack._flat_shape = None  # type: ignore[attr-defined]
        attack._n = None  # type: ignore[attr-defined]
        attack._like = None  # type: ignore[attr-defined]
        return len(subtasks)

    assert _count(8) > _count(1)
