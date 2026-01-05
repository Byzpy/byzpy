import torch

from byzpy.attacks.sign_flip import SignFlipAttack
from byzpy.engine.graph.operator import OpContext


def test_sign_flip_basic_and_independence():
    g = torch.tensor([1.0, -2.0, 3.0])
    atk = SignFlipAttack(scale=-1.0)
    out = atk.apply(base_grad=g)

    assert isinstance(out, torch.Tensor)
    assert torch.allclose(out, -1.0 * g, rtol=0, atol=1e-12)
    assert out.data_ptr() != g.data_ptr()  # not the same storage

    out_mod = out.clone()
    out_mod[0].add_(50.0)
    out2 = atk.apply(base_grad=g)
    assert torch.allclose(out2, -1.0 * g, rtol=0, atol=1e-12)
    assert not torch.allclose(out_mod, out2)


def test_sign_flip_chunk_matches_direct():
    grad = torch.randn(65536)
    direct = SignFlipAttack(scale=0.5).apply(base_grad=grad)

    chunked = SignFlipAttack(scale=0.5, chunk_size=2048)
    inputs = {"base_grad": grad}
    ctx = OpContext(node_name="signflip", metadata={"pool_size": 4})
    subtasks = list(chunked.create_subtasks(inputs, context=ctx))
    partials = [task.fn(*task.args, **task.kwargs) for task in subtasks]
    reduced = chunked.reduce_subtasks(partials, inputs, context=ctx)

    assert torch.allclose(reduced, direct, atol=1e-6)
