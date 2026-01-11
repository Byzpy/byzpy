"""
Fully Distributed Mesh P2P Training with MNIST.

Each node runs on its own server and connects directly to all other nodes.
No central server required - fully peer-to-peer mesh communication.

Training Algorithm: Decentralized SGD with Robust Gradient Aggregation
    Each round:
    1. Each node computes gradients on its local minibatch (forward + backward pass)
    2. Nodes broadcast their GRADIENT vectors to neighbors
    3. Each node aggregates received gradients via CoordinateWiseMedian (robust aggregation)
    4. Each node applies the aggregated gradient to update its model: θ ← θ - lr * g_agg

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


def _flatten_grads(model: nn.Module) -> torch.Tensor:
    """Flatten model gradients into a single vector."""
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.flatten())
        else:
            grads.append(torch.zeros(p.numel(), device=p.device))
    return torch.cat(grads) if grads else torch.tensor([])


def _apply_gradient(model: nn.Module, grad_vec: torch.Tensor, lr: float) -> None:
    """Apply gradient vector to model parameters: θ ← θ - lr * grad."""
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            grad_chunk = grad_vec[offset : offset + n].view_as(p).to(p.device)
            p.add_(grad_chunk, alpha=-lr)
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
        "lr": lr,  # Store learning rate for apply_gradient
    }

    # Register compute_gradient pipeline: computes gradient but does NOT apply it
    async def compute_gradient():
        """Compute gradient on local minibatch, return gradient vector (not parameters)."""
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

        # Return the gradient vector (NOT updated parameters)
        return _flatten_grads(state["model"])

    compute_gradient_op = CallableOp(compute_gradient, input_mapping={})
    compute_gradient_graph = make_single_operator_graph(
        node_name="compute_gradient", operator=compute_gradient_op, input_keys=()
    )
    node.application.register_pipeline("compute_gradient", compute_gradient_graph)

    # Register aggregation pipeline for gradients
    aggregator = CoordinateWiseMedian()
    aggregate_graph = make_single_operator_graph(
        node_name="aggregate", operator=aggregator, input_keys=("gradients",)
    )
    node.application.register_pipeline("aggregate", aggregate_graph)

    # Register apply_gradient pipeline: applies aggregated gradient to model
    async def apply_gradient(grad_vector: torch.Tensor):
        """Apply aggregated gradient: θ ← θ - lr * grad_vector."""
        _apply_gradient(node_state["model"], grad_vector, node_state["lr"])

    apply_gradient_op = CallableOp(apply_gradient, input_mapping={"grad_vector": "grad_vector"})
    apply_gradient_graph = make_single_operator_graph(
        node_name="apply_gradient", operator=apply_gradient_op, input_keys=("grad_vector",)
    )
    node.application.register_pipeline("apply_gradient", apply_gradient_graph)

    # Training state with round-based synchronization
    gradient_cache: Dict[int, List[torch.Tensor]] = {}  # round -> neighbor gradients
    local_gradient: torch.Tensor = None  # this node's gradient for current round
    current_round = 0
    rounds_completed = 0
    round_complete_event = asyncio.Event()

    # Register gradient message handler
    async def on_gradient(from_id: str, payload: dict):
        nonlocal gradient_cache, local_gradient, current_round, rounds_completed

        if rounds_completed >= max_rounds:
            return

        gradient = payload.get("vector")
        round_num = payload.get("round", 0)

        if gradient is not None:
            if round_num not in gradient_cache:
                gradient_cache[round_num] = []
            gradient_cache[round_num].append(gradient)

        # Only aggregate if we have gradients for the current round
        if (
            local_gradient is not None
            and current_round in gradient_cache
            and len(gradient_cache[current_round]) >= 1
        ):
            if rounds_completed >= max_rounds:
                return

            try:
                neighbor_grads = gradient_cache[current_round]
                num_grads = len(neighbor_grads) + 1

                # Aggregate gradients (local + neighbors) via robust aggregation
                result = await node.execute_pipeline(
                    "aggregate", {"gradients": [local_gradient] + neighbor_grads}
                )
                aggregated_grad = result.get("aggregate")

                # Apply aggregated gradient: θ ← θ - lr * aggregated_grad
                if aggregated_grad is not None:
                    await node.execute_pipeline("apply_gradient", {"grad_vector": aggregated_grad})

                # Clean up old rounds
                gradient_cache.pop(current_round, None)
                rounds_completed += 1
                print(
                    f"[Node {node_id}] Round {rounds_completed}/{max_rounds} "
                    f"(aggregated {num_grads} gradients)"
                )

                # Signal that this round is complete
                round_complete_event.set()

                if rounds_completed % 10 == 0:
                    loss, acc = evaluate(node_state["model"], device, data_root)
                    print(f"[Node {node_id}] Eval: loss={loss:.4f}, acc={acc:.4f}")

            except Exception as e:
                print(f"[Node {node_id}] Aggregation error: {e}")

    node.register_message_handler("gradient", on_gradient)

    # Autonomous training loop with round synchronization (Decentralized SGD)
    async def training_loop():
        nonlocal local_gradient, current_round, rounds_completed

        # Wait for all peers to connect
        print(f"[Node {node_id}] Waiting for peer connections...")
        await asyncio.sleep(3.0)

        connected = node.context.get_connected_peers()
        print(f"[Node {node_id}] Connected to {len(connected)} peers: {connected}")
        print(f"[Node {node_id}] Starting decentralized SGD training loop...")

        while node._running and rounds_completed < max_rounds:
            try:
                # Clear event for this round
                round_complete_event.clear()

                # Step 1: Compute local gradient (forward + backward, but don't apply yet)
                result = await node.execute_pipeline("compute_gradient", {})
                grad_vector = result.get("compute_gradient")

                if grad_vector is not None and rounds_completed < max_rounds:
                    local_gradient = grad_vector
                    # Step 2: Broadcast gradient to neighbors
                    await node.broadcast_message(
                        "gradient", {"vector": grad_vector, "round": current_round}
                    )

                # Step 3 & 4 happen in on_gradient handler:
                #   - Aggregate gradients from neighbors
                #   - Apply aggregated gradient to model

                # Wait for this round's aggregation to complete (with timeout)
                try:
                    await asyncio.wait_for(round_complete_event.wait(), timeout=10.0)
                except asyncio.TimeoutError:
                    print(f"[Node {node_id}] Round {current_round} timeout, retrying...")
                    continue

                # Move to next round
                current_round = rounds_completed

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

    gradient_cache: Dict[int, List[torch.Tensor]] = {}  # round -> gradients
    current_round = 0
    rounds_completed = 0

    async def on_gradient(from_id: str, payload: dict):
        nonlocal gradient_cache
        gradient = payload.get("vector")
        round_num = payload.get("round", 0)
        if gradient is not None:
            if round_num not in gradient_cache:
                gradient_cache[round_num] = []
            gradient_cache[round_num].append(gradient)

    node.register_message_handler("gradient", on_gradient)

    async def attack_loop():
        nonlocal gradient_cache, current_round, rounds_completed

        print(f"[Node {node_id}] Waiting for peer connections...")
        await asyncio.sleep(3.0)

        connected = node.context.get_connected_peers()
        print(f"[Node {node_id}] Connected to {len(connected)} peers: {connected}")
        print(f"[Node {node_id}] Starting attack loop...")

        while node._running and rounds_completed < max_rounds:
            if rounds_completed >= max_rounds:
                break

            # Check if we have gradients for the current round
            if current_round in gradient_cache and len(gradient_cache[current_round]) > 0:
                try:
                    grads_for_round = gradient_cache[current_round]
                    template = grads_for_round[0]
                    result = await node.execute_pipeline(
                        "broadcast", {"neighbor_vectors": grads_for_round, "like": template}
                    )
                    malicious = result.get("broadcast")

                    if malicious is not None and rounds_completed < max_rounds:
                        await node.broadcast_message(
                            "gradient", {"vector": malicious, "round": current_round}
                        )
                        gradient_cache.pop(current_round, None)
                        rounds_completed += 1
                        current_round = rounds_completed
                        print(f"[Node {node_id}] Attack round {rounds_completed}/{max_rounds}")
                except Exception as e:
                    print(f"[Node {node_id}] Attack error: {e}")

            await asyncio.sleep(0.1)

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
