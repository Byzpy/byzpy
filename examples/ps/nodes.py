from __future__ import annotations

from typing import Sequence, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import datasets, transforms

from byzpy.aggregators.coordinate_wise import CoordinateWiseMedian
from byzpy.attacks import EmpireAttack
from byzpy.engine.graph.pool import ActorPoolConfig
from byzpy.engine.node.distributed import DistributedByzantineNode, DistributedHonestNode


def _flatten_grads(model: nn.Module) -> torch.Tensor:
    parts = []
    for p in model.parameters():
        parts.append((torch.zeros_like(p) if p.grad is None else p.grad).view(-1))
    return torch.cat(parts)


def _write_vector_into_grads_(model: nn.Module, vec: torch.Tensor) -> None:
    offset = 0
    for p in model.parameters():
        numel = p.numel()
        chunk = vec[offset : offset + numel].view_as(p).to(p.device)
        if p.grad is None:
            p.grad = chunk.clone()
        else:
            p.grad.copy_(chunk)
        offset += numel


def select_pool_backend(spec: str) -> str:
    if spec == "process":
        return "thread"
    if spec.startswith("ucx://"):
        return "gpu"
    if spec.startswith("tcp://"):
        return "thread"
    return spec


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class DistributedPSHonestNode(DistributedHonestNode):
    def __init__(
        self,
        *,
        indices: Sequence[int],
        batch_size: int = 64,
        shuffle: bool = True,
        lr: float = 0.05,
        momentum: float = 0.9,
        device: str = "cpu",
        data_root: str = "./data",
        pool_backend: str = "thread",
        model_cls: Type[nn.Module] = SmallCNN,
    ) -> None:
        dev = torch.device(device)
        agg = CoordinateWiseMedian()
        super().__init__(
            actor_pool=[ActorPoolConfig(backend=pool_backend, count=1, name="worker")],
            aggregator=agg,
            metadata={"pool_backend": pool_backend},
            name=f"honest-{pool_backend}",
        )

        tfm = transforms.Compose([transforms.ToTensor()])
        full = datasets.MNIST(root=data_root, train=True, download=True, transform=tfm)
        subset = data.Subset(full, list(indices))
        self.loader = data.DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=True if shuffle else False,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
        )
        self._it = iter(self.loader)

        self.model = model_cls().to(dev)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        self.criterion = nn.CrossEntropyLoss()
        self.device = dev

    def next_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            x, y = next(self._it)
        except StopIteration:
            self._it = iter(self.loader)
            x, y = next(self._it)
        return x.to(self.device), y.to(self.device)

    def local_honest_gradient(self, *, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        y = y.to(self.device)
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        loss = self.criterion(logits, y)
        loss.backward()
        return _flatten_grads(self.model)

    def apply_server_gradient(self, aggregated_grad: torch.Tensor) -> None:
        _write_vector_into_grads_(self.model, aggregated_grad.to(self.device))
        self.optimizer.step()

    def dump_state_dict(self):
        return {k: v.detach().cpu() for k, v in self.model.state_dict().items()}


class DistributedPSByzNode(DistributedByzantineNode):
    def __init__(
        self,
        *,
        device: str = "cpu",
        scale: float = -1.0,
        pool_backend: str = "thread",
    ) -> None:
        self.device = torch.device(device)
        attack = EmpireAttack(scale=scale)
        super().__init__(
            actor_pool=[ActorPoolConfig(backend=pool_backend, count=1, name="worker")],
            attack=attack,
            metadata={"pool_backend": pool_backend},
            name=f"byz-{pool_backend}",
        )
        self.attack = attack

    def next_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.empty(0), torch.empty(0, dtype=torch.long)

    def apply_server_gradient(self, aggregated_grad: torch.Tensor) -> None:
        pass


__all__ = [
    "SmallCNN",
    "DistributedPSHonestNode",
    "DistributedPSByzNode",
    "select_pool_backend",
]
