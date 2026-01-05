import torch

from byzpy.attacks.empire import EmpireAttack
from byzpy.engine.graph.operator import OpContext
from byzpy.engine.storage.shared_store import cleanup_tensor


def test_empire_torch_single_gradient_and_scale():
    hon = [torch.tensor([1.0, 5.0]), torch.tensor([2.0, 3.0]), torch.tensor([3.0, 1.0])]
    # mean = [2, 3], scale=-2 -> [-4, -6]
    attack = EmpireAttack(scale=-2.0)
    g = attack.apply(honest_grads=hon)
    assert isinstance(g, torch.Tensor)
    assert torch.allclose(g, torch.tensor([-4.0, -6.0]), atol=1e-7)

    # independence check
    g_mod = g.clone()
    g_mod[0].add_(50.0)
    g2 = attack.apply(honest_grads=hon)
    assert torch.allclose(g2, torch.tensor([-4.0, -6.0]), atol=1e-7)
    assert not torch.allclose(g_mod, g2)


def test_empire_chunked_matches_direct():
    hon = [torch.randn(2048) for _ in range(17)]
    direct = EmpireAttack(scale=1.0).apply(honest_grads=hon)

    chunked = EmpireAttack(scale=1.0, chunk_size=3)
    inputs = {"honest_grads": hon}
    subtasks = list(chunked.create_subtasks(inputs, context=None))
    partials = [task.fn(*task.args, **task.kwargs) for task in subtasks]
    reduced = chunked.reduce_subtasks(partials, inputs, context=None)

    assert torch.allclose(reduced, direct, atol=1e-6)


def test_empire_subtasks_scale_with_pool():
    hon = [torch.randn(256) for _ in range(256)]
    inputs = {"honest_grads": hon}

    def _count(pool_size: int) -> int:
        attack = EmpireAttack(scale=1.0, chunk_size=16)
        ctx = OpContext(node_name="empire", metadata={"pool_size": pool_size})
        subtasks = list(attack.create_subtasks(inputs, context=ctx))
        handle = attack._active_handle  # type: ignore[attr-defined]
        if handle is not None:
            cleanup_tensor(handle)
            attack._active_handle = None  # type: ignore[attr-defined]
        attack._flat_shape = None  # type: ignore[attr-defined]
        attack._like_template = None  # type: ignore[attr-defined]
        return len(subtasks)

    assert _count(8) > _count(1)
