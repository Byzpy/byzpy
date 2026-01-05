#!/usr/bin/env python
"""
Benchmark ByzFL ParameterServer training with MultiKrum aggregator.

This benchmark performs actual MNIST training using ByzFL's ParameterServer
implementation with MultiKrum aggregation, following the ByzFL example pattern.

We report both total training time and per-round average time.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass

import torch
import torch.utils.data as data
from torchvision import datasets, transforms


@dataclass(frozen=True)
class BenchmarkRun:
    label: str
    total_seconds: float
    iterations: int

    @property
    def total_ms(self) -> float:
        return self.total_seconds * 1_000.0

    @property
    def avg_ms(self) -> float:
        return self.total_ms / max(1, self.iterations)


def _require_byzfl():
    """Import ByzFL components, raising SystemExit if not available."""
    try:
        from byzfl import ByzantineClient, Client, DataDistributor, Server
        from byzfl.utils.misc import set_random_seed
        from torch import Tensor

        return Client, Server, ByzantineClient, DataDistributor, set_random_seed, Tensor
    except ImportError as exc:
        raise SystemExit(
            "The byzfl package is required for this benchmark (pip install byzfl or add it to your environment)."
        ) from exc


def _train_byzfl(
    *,
    num_honest: int,
    num_byz: int,
    rounds: int,
    batch_size: int,
    lr: float,
    f: int,
    seed: int,
    data_root: str,
) -> BenchmarkRun:
    """Train using ByzFL ParameterServer with MultiKrum."""
    Client, Server, ByzantineClient, DataDistributor, set_random_seed, Tensor = _require_byzfl()

    set_random_seed(seed)

    # Data preparation
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_dataset = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    train_dataset.targets = Tensor(train_dataset.targets).long()
    train_loader = data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

    # Distribute data among clients using non-IID Dirichlet distribution
    data_distributor = DataDistributor(
        {
            "data_distribution_name": "dirichlet_niid",
            "distribution_parameter": 0.5,
            "nb_honest": num_honest,
            "data_loader": train_loader,
            "batch_size": batch_size,
        }
    )
    client_dataloaders = data_distributor.split_data()

    # Initialize honest clients
    honest_clients = [
        Client(
            {
                "model_name": "cnn_mnist",
                "device": "cpu",
                "optimizer_name": "SGD",
                "learning_rate": lr,
                "loss_name": "NLLLoss",
                "weight_decay": 0.0001,
                "milestones": [rounds],
                "learning_rate_decay": 0.25,
                "LabelFlipping": False,
                "training_dataloader": client_dataloaders[i],
                "momentum": 0.9,
                "nb_labels": 10,
                "store_per_client_metrics": False,
            }
        )
        for i in range(num_honest)
    ]

    # Prepare test dataset
    test_dataset = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)
    test_dataset.targets = Tensor(test_dataset.targets).long()
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Server setup with MultiKrum aggregator
    server = Server(
        {
            "device": "cpu",
            "model_name": "cnn_mnist",
            "test_loader": test_loader,
            "optimizer_name": "SGD",
            "learning_rate": lr,
            "weight_decay": 0.0001,
            "milestones": [rounds],
            "learning_rate_decay": 0.25,
            "aggregator_info": {"name": "MultiKrum", "parameters": {"f": f}},
            "pre_agg_list": [],
        }
    )

    # Byzantine client setup
    attack = {
        "name": "InnerProductManipulation",
        "f": num_byz,
        "parameters": {"tau": 3.0},
    }
    byz_client = ByzantineClient(attack)

    # Send initial model to all clients
    new_model = server.get_dict_parameters()
    for client in honest_clients:
        client.set_model_state(new_model)

    # Training loop
    start = time.perf_counter()
    for training_step in range(rounds):
        # Honest clients compute gradients
        for client in honest_clients:
            client.compute_gradients()

        # Aggregate honest gradients
        honest_gradients = [client.get_flat_gradients_with_momentum() for client in honest_clients]

        # Apply Byzantine attack
        byz_vector = byz_client.apply_attack(honest_gradients)

        # Combine honest and Byzantine gradients
        gradients = honest_gradients + byz_vector

        # Update global model with aggregated gradients
        # ByzFL Server aggregates and updates in one step
        server.update_model_with_gradients(gradients)

        # Send updated model to clients
        new_model = server.get_dict_parameters()
        for client in honest_clients:
            client.set_model_state(new_model)
    elapsed = time.perf_counter() - start

    return BenchmarkRun("ByzFL ParameterServer (MultiKrum)", elapsed, rounds)


def _print_results(run: BenchmarkRun) -> None:
    """Print benchmark results."""
    print("\nParameterServer Training (ByzFL MultiKrum)")
    print("=" * 70)
    print(f"{'Mode':50s} {'Total ms':>12s} {'Avg ms/round':>15s}")
    print("-" * 70)
    print(f"{run.label:50s} {run.total_ms:12.2f} {run.avg_ms:15.2f}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark ByzFL ParameterServer training with MultiKrum aggregator."
    )
    parser.add_argument("--num-honest", type=int, default=10, help="Number of honest clients.")
    parser.add_argument("--num-byz", type=int, default=3, help="Number of Byzantine clients.")
    parser.add_argument("--rounds", type=int, default=50, help="Number of training rounds.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size per client.")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate.")
    parser.add_argument("--f", type=int, default=3, help="MultiKrum fault tolerance parameter.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--data-root", type=str, default="./data", help="MNIST data directory.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run = _train_byzfl(
        num_honest=args.num_honest,
        num_byz=args.num_byz,
        rounds=args.rounds,
        batch_size=args.batch_size,
        lr=args.lr,
        f=args.f,
        seed=args.seed,
        data_root=args.data_root,
    )
    _print_results(run)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Benchmark cancelled.", file=sys.stderr)
        raise
