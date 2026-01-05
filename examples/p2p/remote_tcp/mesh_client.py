"""
Fully Distributed Mesh P2P Training with MNIST.

Each node runs on its own server and connects directly to all other nodes.
No central server required - fully peer-to-peer mesh communication.

Architecture:
    ┌───────────────┐         ┌───────────────┐
    │   Node 0      │◄───────►│   Node 1      │
    │ (Machine A)   │   TCP   │ (Machine B)   │
    └───────┬───────┘         └───────┬───────┘
            │                         │
            │         TCP             │
            │                         │
    ┌───────┴─────────────────────────┴───────┐
    │                                         │
    ▼                                         ▼
┌───────────────┐                     ┌───────────────┐
│   Node 2      │◄───────────────────►│   Node 3      │
│ (Machine C)   │         TCP         │ (Machine D)   │
└───────────────┘                     └───────────────┘

Usage:
    # Create a config file (nodes.yaml or nodes.json)
    # Then run on each machine:

    # Machine A (Node 0 - honest):
    python mesh_client.py --config nodes.yaml --node-id 0 --node-type honest

    # Machine B (Node 1 - honest):
    python mesh_client.py --config nodes.yaml --node-id 1 --node-type honest

    # Machine C (Node 2 - honest):
    python mesh_client.py --config nodes.yaml --node-id 2 --node-type honest

    # Machine D (Node 3 - byzantine):
    python mesh_client.py --config nodes.yaml --node-id 3 --node-type byzantine
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import datasets, transforms

SCRIPT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from byzpy.aggregators.coordinate_wise.median import CoordinateWiseMedian
from byzpy.attacks.empire import EmpireAttack
from byzpy.engine.graph.ops import CallableOp, make_single_operator_graph
from byzpy.engine.graph.pool import ActorPoolConfig
from byzpy.engine.node.application import ByzantineNodeApplication, HonestNodeApplication
from byzpy.engine.node.context import MeshRemoteContext
from byzpy.engine.node.decentralized import DecentralizedNode
from byzpy.engine.peer_to_peer.topology import Topology


class SmallCNN(nn.Module):
    """Simple CNN for MNIST."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load node configuration from YAML or JSON file."""
    if config_path.endswith(".yaml") or config_path.endswith(".yml"):
        try:
            import yaml

            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except ImportError:
            raise ImportError(
                "PyYAML is required for YAML config files. Install with: pip install pyyaml"
            )
    else:
        with open(config_path, "r") as f:
            return json.load(f)


def shard_indices(n_items: int, n_shards: int) -> List[List[int]]:
    """Split dataset indices into shards for each node."""
    return [list(range(i, n_items, n_shards)) for i in range(n_shards)]


def make_test_loader(batch_size: int = 512, data_root: str = "./data") -> data.DataLoader:
    """Create test data loader for evaluation."""
    tfm = transforms.Compose([transforms.ToTensor()])
    test = datasets.MNIST(root=data_root, train=False, download=True, transform=tfm)
    return data.DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=0)


def evaluate(model: torch.nn.Module, device: torch.device, data_root: str) -> Tuple[float, float]:
    """Evaluate model on test set."""
    loader = make_test_loader(data_root=data_root)
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


def _flatten_params(model: nn.Module) -> torch.Tensor:
    """Flatten model parameters into a single vector."""
    params = []
    for p in model.parameters():
        params.append(p.data.flatten())
    return torch.cat(params) if params else torch.tensor([])


def _write_params(model: nn.Module, vec: torch.Tensor) -> None:
    """Write parameter vector back into model."""
    offset = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(vec[offset : offset + n].view_as(p).to(p.device))
        offset += n


async def run_honest_node(
    node: DecentralizedNode,
    node_id: str,
    indices: List[int],
    lr: float,
    max_rounds: int,
    batch_size: int,
    data_root: str,
    device: torch.device,
):
    """Run an honest node's training loop."""
    print(f"[Node {node_id}] Initializing honest node...")

    model = SmallCNN().to(device)
    criterion = nn.CrossEntropyLoss()

    tfm = transforms.Compose([transforms.ToTensor()])
    full = datasets.MNIST(root=data_root, train=True, download=True, transform=tfm)
    subset = data.Subset(full, indices)
    data_loader = data.DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=False,
    )
    data_iter_ref = [iter(data_loader)]

    print(f"[Node {node_id}] Model and data loader created")

    node_state = {
        "model": model,
        "criterion": criterion,
        "data_loader": data_loader,
        "data_iter": data_iter_ref,
        "device": device,
    }

    # Register half-step pipeline
    async def half_step(lr: float):
        state = node_state
        try:
            x, y = next(state["data_iter"][0])
        except StopIteration:
            state["data_iter"][0] = iter(state["data_loader"])
            x, y = next(state["data_iter"][0])

        x, y = x.to(state["device"]), y.to(state["device"])
        state["model"].zero_grad(set_to_none=True)
        logits = state["model"](x)
        loss = state["criterion"](logits, y)
        loss.backward()

        with torch.no_grad():
            for p in state["model"].parameters():
                if p.grad is not None:
                    p.add_(p.grad, alpha=-lr)

        return _flatten_params(state["model"])

    half_step_op = CallableOp(half_step, input_mapping={"lr": "lr"})
    half_step_graph = make_single_operator_graph(
        node_name="half_step", operator=half_step_op, input_keys=("lr",)
    )
    node.application.register_pipeline("half_step", half_step_graph)

    # Register aggregation pipeline
    aggregator = CoordinateWiseMedian()
    aggregate_graph = make_single_operator_graph(
        node_name="aggregate", operator=aggregator, input_keys=("gradients",)
    )
    node.application.register_pipeline("aggregate", aggregate_graph)

    # Register update model pipeline
    async def update_model(param_vector: torch.Tensor):
        _write_params(node_state["model"], param_vector)

    update_model_op = CallableOp(update_model, input_mapping={"param_vector": "param_vector"})
    update_model_graph = make_single_operator_graph(
        node_name="update_model", operator=update_model_op, input_keys=("param_vector",)
    )
    node.application.register_pipeline("update_model", update_model_graph)

    # Training state
    gradient_cache: List[torch.Tensor] = []
    self_half_step_result: torch.Tensor = None
    rounds_completed = 0

    # Register gradient message handler
    async def on_gradient(from_id: str, payload: dict):
        nonlocal gradient_cache, self_half_step_result, rounds_completed

        if rounds_completed >= max_rounds:
            return

        gradient = payload.get("vector")
        if gradient is not None:
            gradient_cache.append(gradient)

        if self_half_step_result is not None and len(gradient_cache) >= 1:
            if rounds_completed >= max_rounds:
                return

            try:
                num_grads = len(gradient_cache) + 1
                result = await node.execute_pipeline(
                    "aggregate", {"gradients": [self_half_step_result] + gradient_cache}
                )
                aggregated = result.get("aggregate")

                if aggregated is not None:
                    await node.execute_pipeline("update_model", {"param_vector": aggregated})

                gradient_cache.clear()
                rounds_completed += 1
                print(
                    f"[Node {node_id}] Round {rounds_completed}/{max_rounds} "
                    f"(aggregated {num_grads} gradients)"
                )

                if rounds_completed % 10 == 0:
                    loss, acc = evaluate(node_state["model"], device, data_root)
                    print(f"[Node {node_id}] Eval: loss={loss:.4f}, acc={acc:.4f}")

            except Exception as e:
                print(f"[Node {node_id}] Aggregation error: {e}")

    node.register_message_handler("gradient", on_gradient)

    # Autonomous training loop
    async def training_loop():
        nonlocal self_half_step_result, rounds_completed

        # Wait for all peers to connect
        print(f"[Node {node_id}] Waiting for peer connections...")
        await asyncio.sleep(3.0)

        connected = node.context.get_connected_peers()
        print(f"[Node {node_id}] Connected to {len(connected)} peers: {connected}")
        print(f"[Node {node_id}] Starting training loop...")

        while node._running and rounds_completed < max_rounds:
            if rounds_completed >= max_rounds:
                break

            try:
                result = await node.execute_pipeline("half_step", {"lr": lr})
                half_step_vector = result.get("half_step")

                if half_step_vector is not None and rounds_completed < max_rounds:
                    self_half_step_result = half_step_vector
                    await node.broadcast_message("gradient", {"vector": half_step_vector})

                await asyncio.sleep(0.2)
            except Exception as e:
                print(f"[Node {node_id}] Training step error: {e}")
                await asyncio.sleep(0.5)

        print(f"[Node {node_id}] Training completed ({rounds_completed} rounds)")

        loss, acc = evaluate(node_state["model"], device, data_root)
        print(f"[Node {node_id}] Final: loss={loss:.4f}, acc={acc:.4f}")

    await node.start_autonomous_task(training_loop(), "training")


async def run_byzantine_node(
    node: DecentralizedNode,
    node_id: str,
    max_rounds: int,
    scale: float,
):
    """Run a byzantine node's attack loop."""
    print(f"[Node {node_id}] Initializing byzantine node...")

    attack = EmpireAttack(scale=scale)
    node_state = {"attack": attack}

    print(f"[Node {node_id}] Attack created (scale={scale})")

    # Register broadcast pipeline
    async def broadcast(neighbor_vectors: List[torch.Tensor], like: torch.Tensor):
        if not neighbor_vectors:
            return like
        return node_state["attack"].apply(honest_grads=neighbor_vectors)

    broadcast_op = CallableOp(
        broadcast, input_mapping={"neighbor_vectors": "neighbor_vectors", "like": "like"}
    )
    broadcast_graph = make_single_operator_graph(
        node_name="broadcast", operator=broadcast_op, input_keys=("neighbor_vectors", "like")
    )
    node.application.register_pipeline("broadcast", broadcast_graph)

    gradient_cache: List[torch.Tensor] = []
    rounds_completed = 0

    async def on_gradient(from_id: str, payload: dict):
        nonlocal gradient_cache
        gradient = payload.get("vector")
        if gradient is not None:
            gradient_cache.append(gradient)

    node.register_message_handler("gradient", on_gradient)

    async def attack_loop():
        nonlocal gradient_cache, rounds_completed

        print(f"[Node {node_id}] Waiting for peer connections...")
        await asyncio.sleep(3.0)

        connected = node.context.get_connected_peers()
        print(f"[Node {node_id}] Connected to {len(connected)} peers: {connected}")
        print(f"[Node {node_id}] Starting attack loop...")

        while node._running and rounds_completed < max_rounds:
            if rounds_completed >= max_rounds:
                break

            if len(gradient_cache) > 0:
                try:
                    template = gradient_cache[0]
                    result = await node.execute_pipeline(
                        "broadcast", {"neighbor_vectors": gradient_cache, "like": template}
                    )
                    malicious = result.get("broadcast")

                    if malicious is not None and rounds_completed < max_rounds:
                        await node.broadcast_message("gradient", {"vector": malicious})
                        gradient_cache.clear()
                        rounds_completed += 1
                        print(f"[Node {node_id}] Attack round {rounds_completed}/{max_rounds}")
                except Exception as e:
                    print(f"[Node {node_id}] Attack error: {e}")

            await asyncio.sleep(0.2)

        print(f"[Node {node_id}] Attack completed ({rounds_completed} rounds)")

    await node.start_autonomous_task(attack_loop(), "attack")


async def main(args):
    """Main entry point for running a distributed mesh node."""
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load configuration
    config = load_config(args.config)
    nodes_config = config.get("nodes", [])

    if not nodes_config:
        raise ValueError("Config must contain 'nodes' list")

    # Build peer addresses from config
    peer_addresses = {}
    local_host = None
    local_port = None
    total_honest = 0

    for node_cfg in nodes_config:
        nid = str(node_cfg["id"])
        host = node_cfg["host"]
        port = node_cfg["port"]
        peer_addresses[nid] = (host, port)

        if nid == str(args.node_id):
            local_host = node_cfg.get("bind_host", "0.0.0.0")
            local_port = port

        if node_cfg.get("type", "honest") == "honest":
            total_honest += 1

    if local_port is None:
        raise ValueError(f"Node ID {args.node_id} not found in config")

    total_nodes = len(nodes_config)

    print("=" * 70)
    print(f"Fully Distributed Mesh Node {args.node_id}")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  - Config file: {args.config}")
    print(f"  - Node ID: {args.node_id}")
    print(f"  - Node type: {args.node_type}")
    print(f"  - Local server: {local_host}:{local_port}")
    print(f"  - Total nodes: {total_nodes}")
    print(f"  - Honest nodes: {total_honest}")
    print(f"  - Peer addresses: {peer_addresses}")
    print(f"  - Rounds: {args.rounds}")
    print(f"  - Device: {device}")
    print("=" * 70)

    # Create topology
    topology = Topology.ring(n=total_nodes, k=1)

    # Build node_id_map
    node_id_map = {i: str(nodes_config[i]["id"]) for i in range(total_nodes)}

    # Create MeshRemoteContext
    context = MeshRemoteContext(
        local_host=local_host,
        local_port=local_port,
        peer_addresses=peer_addresses,
        connect_timeout=10.0,
        reconnect_interval=5.0,
    )

    # Create application and node
    node_id = str(args.node_id)
    if args.node_type == "honest":
        app = HonestNodeApplication(
            name=f"honest_{node_id}",
            actor_pool=[ActorPoolConfig(backend="thread", count=1)],
        )
    else:
        app = ByzantineNodeApplication(
            name=f"byz_{node_id}",
            actor_pool=[ActorPoolConfig(backend="thread", count=1)],
        )

    node = DecentralizedNode(
        node_id=node_id,
        application=app,
        context=context,
        topology=topology,
        node_id_map=node_id_map,
    )

    print(f"[Node {node_id}] Starting node and connecting to peers...")
    await node.start()
    print(f"[Node {node_id}] Node started!")

    # Run the appropriate training/attack loop
    if args.node_type == "honest":
        # Find this node's position among honest nodes for data sharding
        honest_index = 0
        for node_cfg in nodes_config:
            if str(node_cfg["id"]) == node_id:
                break
            if node_cfg.get("type", "honest") == "honest":
                honest_index += 1

        tfm = transforms.Compose([transforms.ToTensor()])
        _tmp_train = datasets.MNIST(root=args.data_root, train=True, download=True, transform=tfm)
        shards = shard_indices(len(_tmp_train), total_honest)
        indices = shards[honest_index % len(shards)]

        await run_honest_node(
            node=node,
            node_id=node_id,
            indices=indices,
            lr=args.lr,
            max_rounds=args.rounds,
            batch_size=args.batch_size,
            data_root=args.data_root,
            device=device,
        )
    else:
        await run_byzantine_node(
            node=node,
            node_id=node_id,
            max_rounds=args.rounds,
            scale=args.byz_scale,
        )

    # Wait for training to complete
    wait_time = args.rounds * 0.3 + 15
    print(f"[Node {node_id}] Running for {wait_time:.0f} seconds...")
    await asyncio.sleep(wait_time)

    print(f"[Node {node_id}] Shutting down...")
    await node.shutdown()
    print(f"[Node {node_id}] Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fully Distributed Mesh P2P Training")

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to nodes configuration file (YAML or JSON)",
    )
    parser.add_argument(
        "--node-id",
        type=int,
        required=True,
        help="This node's ID (must match an entry in config)",
    )
    parser.add_argument(
        "--node-type",
        type=str,
        choices=["honest", "byzantine"],
        default="honest",
        help="Node type (default: honest)",
    )

    # Training configuration
    parser.add_argument("--rounds", type=int, default=50, help="Training rounds (default: 50)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate (default: 0.05)")
    parser.add_argument(
        "--data-root",
        type=str,
        default="./data",
        help="Data root directory (default: ./data)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    parser.add_argument(
        "--byz-scale",
        type=float,
        default=-1.0,
        help="Byzantine attack scale (default: -1.0)",
    )

    args = parser.parse_args()
    asyncio.run(main(args))
