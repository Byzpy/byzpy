import torch

from byzpy.attacks.inf import InfAttack
from byzpy.engine.graph.operator import OpContext


def test_inf_direct_output():
    grads = [torch.randn(3), torch.randn(3)]
    attack = InfAttack()
    out = attack.apply(honest_grads=grads)
    assert torch.isinf(out).all()


def test_inf_chunk_matches_direct():
    grads = [torch.randn(64) for _ in range(8)]
    direct = InfAttack(chunk_size=16).apply(honest_grads=grads)

    chunked = InfAttack(chunk_size=16)
    inputs = {"honest_grads": grads}
    ctx = OpContext(node_name="inf", metadata={"pool_size": 4})
    subtasks = list(chunked.create_subtasks(inputs, context=ctx))
    partials = [t.fn(*t.args, **t.kwargs) for t in subtasks]
    reduced = chunked.reduce_subtasks(
        partials, inputs, context=OpContext(node_name="inf", metadata={})
    )

    assert torch.allclose(direct, reduced)
