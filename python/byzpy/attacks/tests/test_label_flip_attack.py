import pytest
import torch
import torch.nn as nn

from byzpy.attacks.label_flip import LabelFlipAttack


class TinyLinear(nn.Module):
    def __init__(self, d_in=8, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(d_in, num_classes)

    def forward(self, x):
        return self.fc(x)


def honest_grad_vec(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Return flattened gradient vector for standard CE loss on (x, y)."""
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()
    logits = model(x)
    loss = nn.CrossEntropyLoss()(logits, y)
    loss.backward()

    parts = []
    device = next(model.parameters()).device
    for p in model.parameters():
        if not p.requires_grad:
            continue
        g = p.grad if p.grad is not None else torch.zeros_like(p)
        parts.append(g.reshape(-1))
    vec = torch.cat(parts) if parts else torch.tensor([], device=device)

    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()
    return vec.detach()


def test_label_flipping_vector_changes_vs_honest():
    torch.manual_seed(0)
    B, D, K = 6, 8, 10
    x = torch.randn(B, D)
    y = torch.randint(0, K, (B,), dtype=torch.long)

    m = TinyLinear(D, K)
    g_honest = honest_grad_vec(m, x, y)

    attack = LabelFlipAttack(num_classes=K, scale=1.0)
    g_mal = attack.apply(model=m, x=x, y=y)

    assert isinstance(g_mal, torch.Tensor)
    assert g_mal.ndim == 1
    assert g_mal.shape == g_honest.shape
    assert not torch.allclose(g_honest, g_mal)


def test_label_flipping_mirror_equals_explicit_mapping():
    torch.manual_seed(1)
    B, D, K = 4, 5, 10
    x = torch.randn(B, D)
    y = torch.randint(0, K, (B,), dtype=torch.long)

    m1 = TinyLinear(D, K)
    m2 = TinyLinear(D, K)
    m2.load_state_dict(m1.state_dict())

    # mirror rule
    mirror_attack = LabelFlipAttack(num_classes=K)
    g_mirror = mirror_attack.apply(model=m1, x=x, y=y)

    # explicit mapping (should equal mirror)
    mapping = {i: (K - 1 - i) for i in range(K)}
    map_attack = LabelFlipAttack(mapping=mapping)
    g_map = map_attack.apply(model=m2, x=x, y=y)

    assert g_mirror.shape == g_map.shape
    assert torch.allclose(g_mirror, g_map, atol=1e-7, rtol=1e-7)


def test_label_flipping_scale_applies_linearly():
    torch.manual_seed(2)
    B, D, K = 3, 6, 10
    x = torch.randn(B, D)
    y = torch.randint(0, K, (B,), dtype=torch.long)

    m = TinyLinear(D, K)

    a1 = LabelFlipAttack(num_classes=K, scale=1.0)
    g1 = a1.apply(model=m, x=x, y=y)

    a3 = LabelFlipAttack(num_classes=K, scale=3.0)
    g3 = a3.apply(model=m, x=x, y=y)

    assert g1.shape == g3.shape
    assert torch.allclose(g3, 3.0 * g1, atol=1e-7, rtol=1e-7)


def test_label_flipping_ctor_errors_when_no_rule_provided():
    with pytest.raises(ValueError):
        _ = LabelFlipAttack()  # must pass num_classes or mapping
