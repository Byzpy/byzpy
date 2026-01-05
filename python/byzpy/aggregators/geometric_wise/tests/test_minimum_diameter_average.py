import torch

from byzpy.aggregators.geometric_wise import MinimumDiameterAveraging


def test_mda_torch_simple():
    honest = [
        torch.tensor([0.0, 0.0]),
        torch.tensor([0.1, 0.0]),
        torch.tensor([-0.1, 0.0]),
        torch.tensor([0.0, 0.1]),
    ]
    grads = honest + [torch.tensor([100.0, 100.0])]
    agg = MinimumDiameterAveraging(f=1)
    out = agg.aggregate(grads)
    exp = torch.mean(torch.stack(honest, dim=0), dim=0)
    assert torch.allclose(out, exp, atol=1e-7)
