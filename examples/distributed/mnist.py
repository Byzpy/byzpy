#!/usr/bin/env python
"""
Distributed ByzPy ParameterServer Training Across Multiple Machines

This example demonstrates how to run ByzPy training with nodes distributed across
multiple machines. Each node runs on a separate machine and communicates via TCP.

Setup:
    1. Start a remote actor server on each machine:
       python examples/distributed/server.py --host 0.0.0.0 --port 29000

    2. Run this client script with the list of remote hosts:
       python examples/distributed/mnist.py \
           --remote-hosts tcp://machine1:29000,tcp://machine2:29000,tcp://machine3:29000 \
           --num-honest 3 --num-byz 1 --rounds 50

Architecture:
    - Each node runs on a separate machine via RemoteActorBackend
    - Nodes communicate via TCP for gradient exchange
    - ParameterServer orchestrates training from the client machine
    - Aggregation happens on the client machine

Requirements:
    - ByzPy installed on all machines
    - Network connectivity between machines
    - Same Python version on all machines (recommended)
    - MNIST dataset accessible (will download if needed)
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import List, Tuple

import torch
import torch.utils.data as data
from torchvision import datasets, transforms

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.ps.nodes import (
    DistributedPSByzNode,
    DistributedPSHonestNode,
    SmallCNN,
    select_pool_backend,
)

from byzpy.aggregators.geometric_wise.krum import MultiKrum
from byzpy.configs.actor import set_actor
from byzpy.engine.node.actors import ByzantineNodeActor, HonestNodeActor
from byzpy.engine.parameter_server.ps import ParameterServer


def shard_indices(n_items: int, n_shards: int) -> List[List[int]]:
    """Split dataset indices into shards for each node."""
    return [list(range(i, n_items, n_shards)) for i in range(n_shards)]


def make_test_loader(batch_size: int = 512) -> data.DataLoader:
    """Create test data loader for evaluation."""
    tfm = transforms.Compose([transforms.ToTensor()])
    test = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)
    return data.DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=0)


def evaluate(model: torch.nn.Module, device: torch.device) -> Tuple[float, float]:
    """Evaluate model on test set."""
    loader = make_test_loader()
    ce = torch.nn.CrossEntropyLoss(reduction="sum")
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum += ce(logits, y).item()
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    model.train()
    return loss_sum / total, correct / total


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Distributed ByzPy ParameterServer training across multiple machines."
    )
    parser.add_argument(
        "--remote-hosts",
        type=str,
        required=True,
        help="Comma-separated list of remote actor server addresses (e.g., tcp://192.168.1.10:29000,tcp://192.168.1.11:29000)",
    )
    parser.add_argument("--num-honest", type=int, default=3, help="Number of honest nodes.")
    parser.add_argument("--num-byz", type=int, default=1, help="Number of Byzantine nodes.")
    parser.add_argument("--rounds", type=int, default=50, help="Number of training rounds.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size per node.")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate.")
    parser.add_argument("--f", type=int, default=1, help="MultiKrum fault tolerance parameter.")
    parser.add_argument(
        "--q",
        type=int,
        default=None,
        help="MultiKrum parameter q (defaults to n - f - 1).",
    )
    parser.add_argument("--chunk-size", type=int, default=32, help="MultiKrum chunk size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--data-root", type=str, default="./data", help="MNIST data directory.")
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=10,
        help="Evaluate model every N rounds (0 to disable evaluation).",
    )
    args = parser.parse_args()

    remote_hosts = [h.strip() for h in args.remote_hosts.split(",") if h.strip()]
    if not remote_hosts:
        parser.error("--remote-hosts must specify at least one remote host")

    total_nodes = args.num_honest + args.num_byz
    if len(remote_hosts) < total_nodes:
        print(
            f"Warning: Only {len(remote_hosts)} remote hosts provided but {total_nodes} nodes needed. "
            f"Nodes will be distributed across available hosts (round-robin).",
            file=sys.stderr,
        )

    if args.q is None:
        args.q = max(1, args.num_honest - args.f - 1)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_localhost = all("127.0.0.1" in h or "localhost" in h for h in remote_hosts)

    print("=" * 70)
    print("Distributed ByzPy ParameterServer Training")
    if all_localhost:
        print("(Local Testing Mode - validates distributed architecture)")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  - Honest nodes: {args.num_honest}")
    print(f"  - Byzantine nodes: {args.num_byz}")
    print(f"  - Total nodes: {total_nodes}")
    print(f"  - Remote hosts: {len(remote_hosts)}")
    for i, host in enumerate(remote_hosts):
        print(f"    [{i}] {host}")
    print(f"  - Training rounds: {args.rounds}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Learning rate: {args.lr}")
    print(f"  - MultiKrum: f={args.f}, q={args.q}")
    print(f"  - Device: {device}")
    print("=" * 70)

    tfm = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root=args.data_root, train=True, download=True, transform=tfm)
    shards = shard_indices(len(train_dataset), args.num_honest)

    def pick_remote_host(node_index: int) -> str:
        """Round-robin assignment of nodes to remote hosts."""
        return remote_hosts[node_index % len(remote_hosts)]

    # Create honest nodes - each on a different remote machine
    print("\nCreating honest node actors on remote machines...")
    honest_actors: List[HonestNodeActor] = []
    for i in range(args.num_honest):
        remote_host = pick_remote_host(i)
        print(f"  Node {i}: {remote_host}")
        pool_backend = select_pool_backend(remote_host)
        h = await HonestNodeActor.spawn(
            DistributedPSHonestNode,
            backend=set_actor(remote_host),
            kwargs=dict(
                indices=shards[i],
                batch_size=args.batch_size,
                shuffle=True,
                lr=args.lr,
                momentum=0.9,
                device="cpu",  # Use CPU on remote machines (can be changed to "cuda" if GPUs available)
                data_root=args.data_root,
                pool_backend=pool_backend,
            ),
        )
        honest_actors.append(h)
        print(f"    ✓ Connected to {remote_host}")

    # Create byzantine nodes - each on a different remote machine
    print("\nCreating byzantine node actors on remote machines...")
    byz_actors: List[ByzantineNodeActor] = []
    for j in range(args.num_byz):
        remote_host = pick_remote_host(args.num_honest + j)
        print(f"  Byzantine node {j}: {remote_host}")
        pool_backend = select_pool_backend(remote_host)
        b = await ByzantineNodeActor.spawn(
            DistributedPSByzNode,
            backend=set_actor(remote_host),
            kwargs=dict(
                device="cpu",  # Use CPU on remote machines
                scale=-1.0,
                pool_backend=pool_backend,
            ),
        )
        byz_actors.append(b)
        print(f"    ✓ Connected to {remote_host}")

    # Create MultiKrum aggregator
    aggregator = MultiKrum(f=args.f, q=args.q, chunk_size=args.chunk_size)

    # Create ParameterServer - aggregates gradients from all remote nodes
    ps = ParameterServer(
        honest_nodes=honest_actors,
        byzantine_nodes=byz_actors,
        aggregator=aggregator,
        update_byzantines=False,
    )

    # Create evaluation model
    eval_model = SmallCNN().to(device)

    print("\n" + "=" * 70)
    print("Starting Distributed Training")
    print("=" * 70)
    print("Nodes are distributed across multiple machines.")
    print("Gradients are collected via TCP and aggregated on the client machine.")
    print("=" * 70)

    start_time = time.perf_counter()
    for r in range(1, args.rounds + 1):
        await ps.round()

        if args.eval_interval > 0 and r % args.eval_interval == 0:
            sd = await honest_actors[0].dump_state_dict()
            eval_model.load_state_dict(sd, strict=True)
            loss, acc = await asyncio.to_thread(evaluate, eval_model, device)
            elapsed = time.perf_counter() - start_time
            print(f"[round {r:04d}] test loss={loss:.4f}  acc={acc:.4f}  elapsed={elapsed:.2f}s")

    total_time = time.perf_counter() - start_time

    print("\n" + "=" * 70)
    print("Training Complete - Final Evaluations")
    print("=" * 70)

    # Final evaluation for each honest node
    print("\nFinal honest-node evaluations:")
    for i, h in enumerate(honest_actors):
        sd = await h.dump_state_dict()
        eval_model.load_state_dict(sd, strict=True)
        loss, acc = await asyncio.to_thread(evaluate, eval_model, device)
        print(f"  Node {i} ({pick_remote_host(i)}): loss={loss:.4f}  acc={acc:.4f}")

    print(f"\nTotal training time: {total_time:.2f}s")
    print(f"Average time per round: {total_time / args.rounds:.2f}s")

    # Cleanup
    print("\nShutting down all remote nodes...")
    await ps.shutdown()
    print("  ✓ All nodes shut down successfully")

    print("\n" + "=" * 70)
    print("Distributed Training Complete!")
    print("=" * 70)
    print("\nKey Features Demonstrated:")
    if all_localhost:
        print("  ✓ Distributed architecture validated (localhost testing)")
        print("  ✓ Nodes communicate via TCP (RemoteActorBackend)")
        print("  ✓ Ready to extend to multiple physical machines")
    else:
        print("  ✓ Nodes distributed across multiple machines")
        print("  ✓ Communication via TCP (RemoteActorBackend)")
    print("  ✓ Centralized aggregation on client machine")
    print("  ✓ Fault-tolerant training with MultiKrum")
    print("=" * 70)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nTraining cancelled.", file=sys.stderr)
        sys.exit(1)
