import numpy as np
import torch

from byzpy.attacks.gaussian import GaussianAttack
from byzpy.engine.graph.operator import OpContext


def test_gaussian_direct_matches_rng():
    grads = [torch.zeros(3), torch.ones(3)]
    attack = GaussianAttack(mu=0.0, sigma=1.0, seed=0)
    out = attack.apply(honest_grads=grads)
    expected = np.random.default_rng(0).normal(loc=0.0, scale=1.0, size=3)
    assert torch.allclose(out, torch.as_tensor(expected, dtype=out.dtype, device=out.device))


def test_gaussian_chunk_matches_direct():
    grads = [torch.randn(128) for _ in range(4)]
    direct = GaussianAttack(mu=0.5, sigma=2.0, seed=42, chunk_size=16).apply(honest_grads=grads)

    chunked = GaussianAttack(mu=0.5, sigma=2.0, seed=42, chunk_size=16)
    inputs = {"honest_grads": grads}
    ctx = OpContext(node_name="gauss", metadata={"pool_size": 4})
    subtasks = list(chunked.create_subtasks(inputs, context=ctx))
    partials = [t.fn(*t.args, **t.kwargs) for t in subtasks]
    reduced = chunked.reduce_subtasks(
        partials, inputs, context=OpContext(node_name="gauss", metadata={})
    )

    assert torch.allclose(direct, reduced, atol=1e-7)
