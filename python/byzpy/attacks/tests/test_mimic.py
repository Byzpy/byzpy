import torch

from byzpy.attacks.mimic import MimicAttack
from byzpy.engine.graph.operator import OpContext


def test_mimic_direct_expected():
    grads = [
        torch.tensor([1.0, 2.0]),
        torch.tensor([3.0, 4.0]),
        torch.tensor([5.0, 6.0]),
    ]
    attack = MimicAttack(epsilon=1)
    out = attack.apply(honest_grads=grads)
    assert torch.allclose(out, grads[1])


def test_mimic_chunk_matches_direct():
    grads = [torch.randn(128) for _ in range(5)]
    direct = MimicAttack(epsilon=3, chunk_size=32).apply(honest_grads=grads)

    chunked = MimicAttack(epsilon=3, chunk_size=32)
    inputs = {"honest_grads": grads}
    ctx = OpContext(node_name="mimic", metadata={"pool_size": 4})
    subtasks = list(chunked.create_subtasks(inputs, context=ctx))
    partials = [t.fn(*t.args, **t.kwargs) for t in subtasks]
    reduced = chunked.reduce_subtasks(
        partials, inputs, context=OpContext(node_name="mimic", metadata={})
    )

    assert torch.allclose(direct, reduced, atol=1e-7)
