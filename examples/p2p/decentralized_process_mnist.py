#!/usr/bin/env python3
"""
Fully Decentralized Process-Based P2P Training with MNIST

This example demonstrates fully decentralized P2P training where each node runs
in its own OS process with its own scheduler, communicating via message-driven
topology. This showcases the new DecentralizedNode infrastructure with ProcessContext.

Key Features:
- Each node runs in a separate OS process (ProcessContext)
- Nodes communicate via topology-aware message routing
- Message-driven graph execution
- Fully decoupled nodes with independent schedulers
- Process-level isolation for true parallelism

Usage:
    python examples/p2p/decentralized_process_mnist.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import List, Tuple

import torch
import torch.utils.data as data
from torchvision import datasets, transforms

# Add project root to path
SCRIPT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from examples.p2p.nodes import (
    DistributedP2PByzNode,
    DistributedP2PHonestNode,
    SmallCNN,
    select_pool_backend,
)

from byzpy.configs.actor import set_actor
from byzpy.engine.node.actors import ByzantineNodeActor, HonestNodeActor
from byzpy.engine.node.context import ProcessContext
from byzpy.engine.node.decentralized import DecentralizedNode
from byzpy.engine.peer_to_peer.runner import DecentralizedPeerToPeer
from byzpy.engine.peer_to_peer.topology import Topology


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


async def main():
    """
    Main function demonstrating fully decentralized process-based P2P training.

    Each node runs in its own OS process with:
    - Independent scheduler
    - Own asyncio event loop
    - Process-level isolation
    - Topology-based message communication
    """
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Configuration
    n_honest = 4
    n_byz = 1
    rounds = 10  # Reduced for quick testing (can be increased for full training)
    batch_size = 64
    lr = 0.05
    momentum = 0.9

    print("=" * 70)
    print("Fully Decentralized Process-Based P2P Training with MNIST")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  - Honest nodes: {n_honest}")
    print(f"  - Byzantine nodes: {n_byz}")
    print(f"  - Training rounds: {rounds}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Learning rate: {lr}")
    print(f"  - Device: {device}")
    print(f"  - Each node runs in separate OS process (ProcessContext)")
    print("=" * 70)

    tfm = transforms.Compose([transforms.ToTensor()])
    _tmp_train = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    shards = shard_indices(len(_tmp_train), n_honest)

    actor_backend = "thread"
    pool_backend = select_pool_backend(actor_backend)

    print("\nCreating honest node actors...")
    honest_actors: List[HonestNodeActor] = []
    for i in range(n_honest):
        h = await HonestNodeActor.spawn(
            DistributedP2PHonestNode,
            backend=set_actor(actor_backend),
            kwargs=dict(
                indices=shards[i],
                batch_size=batch_size,
                shuffle=True,
                lr=lr,
                momentum=momentum,
                device=str(device),
                data_root="./data",
                pool_backend=pool_backend,
            ),
        )
        honest_actors.append(h)
        print(f"  ✓ Created honest node {i} with {len(shards[i])} training samples")

    print("\nCreating byzantine node actors...")
    byz_actors: List[ByzantineNodeActor] = []
    for i in range(n_byz):
        b = await ByzantineNodeActor.spawn(
            DistributedP2PByzNode,
            backend=set_actor(actor_backend),
            kwargs=dict(
                device=str(device),
                scale=-1.0,  # Byzantine attack scale
                pool_backend=pool_backend,
            ),
        )
        byz_actors.append(b)
        print(f"  ✓ Created byzantine node {i}")

    # Create ring topology for P2P communication
    # Each node communicates with k neighbors (k=1 means immediate neighbors only)
    topo = Topology.ring(n=len(honest_actors) + len(byz_actors), k=1)
    print(f"\nTopology: Ring with {len(honest_actors) + len(byz_actors)} nodes (k=1)")

    # Create DecentralizedPeerToPeer runner with ProcessContext factory
    # This ensures each DecentralizedNode runs in its own OS process
    def context_factory(node_id: str, node_index: int) -> ProcessContext:
        """Factory function that creates ProcessContext for each node."""
        return ProcessContext()

    print("\nInitializing P2P training runner...")
    p2p = DecentralizedPeerToPeer(
        honest_nodes=honest_actors,
        byzantine_nodes=byz_actors,
        topology=topo,
        lr=lr,
        context_factory=context_factory,  # Each node in separate process
    )

    print("Starting all nodes (each in separate OS process)...")
    await p2p.start()
    print("  ✓ All nodes started successfully")

    nodes: dict[str, DecentralizedNode] = p2p._decentralized_nodes

    eval_model = SmallCNN().to(device)

    print("\n" + "=" * 70)
    print("Starting Fully Asynchronous P2P Training")
    print("=" * 70)
    print("Nodes progress independently based on neighbor messages:")
    print("  - Each node triggers its own half-step pipeline")
    print("  - Nodes broadcast gradients to neighbors via topology")
    print("  - When gradients are received, nodes automatically aggregate")
    print("  - No synchronous round() API - fully decentralized!")
    print("=" * 70)

    # Track training state per node
    training_rounds: dict[str, int] = {node_id: 0 for node_id in nodes.keys()}
    node_half_step_results: dict[str, torch.Tensor] = {}  # Track each node's last half-step result

    # Set up message handlers that trigger aggregation when gradients are received
    async def setup_async_training():
        """Set up message handlers for fully asynchronous training."""
        for node_id, node in nodes.items():
            node_idx = int(node_id)
            is_honest = node_idx < len(honest_actors)

            def make_async_handler(nid: str, n: DecentralizedNode, is_h: bool):
                async def on_gradient(from_id: str, payload: dict):
                    """When gradient received, trigger aggregation if we have self's half-step."""
                    if not is_h:
                        return

                    if nid not in p2p._gradient_cache:
                        p2p._gradient_cache[nid] = []
                    p2p._gradient_cache[nid].append(payload.get("vector"))

                    if nid not in node_half_step_results:
                        return

                    neighbor_vecs = p2p._gradient_cache.get(nid, [])
                    self_theta_half = node_half_step_results[nid]

                    from byzpy.engine.peer_to_peer.runner import _NODE_OBJECT_REGISTRY

                    node_key = f"honest_{nid}"
                    node_obj = _NODE_OBJECT_REGISTRY.get(node_key)

                    if node_obj is not None and len(neighbor_vecs) >= 1:
                        try:
                            node_obj.p2p_aggregate_and_set(
                                self_theta_half=self_theta_half,
                                neighbor_vectors=neighbor_vecs,
                            )
                            training_rounds[nid] += 1
                            p2p._gradient_cache[nid] = []
                        except Exception as e:
                            pass

                return on_gradient

            node.register_message_handler("gradient", make_async_handler(node_id, node, is_honest))

    await setup_async_training()

    async def node_training_loop(node_id: str, node: DecentralizedNode, is_honest: bool):
        """Independent training loop for each node."""
        node_idx = int(node_id)
        max_rounds = rounds

        for r in range(1, max_rounds + 1):
            if not is_honest:
                await asyncio.sleep(0.15)

                neighbor_vecs = p2p._gradient_cache.get(node_id, [])
                if neighbor_vecs:
                    from byzpy.engine.peer_to_peer.runner import _NODE_OBJECT_REGISTRY

                    node_key = f"byz_{node_id}"
                    node_obj = _NODE_OBJECT_REGISTRY.get(node_key)

                    if node_obj is not None:
                        template = neighbor_vecs[0] if neighbor_vecs else None
                        if template is not None:
                            malicious = node_obj.p2p_broadcast_vector(
                                neighbor_vectors=(neighbor_vecs if neighbor_vecs else None),
                                like=template,
                            )
                            await node.broadcast_message("gradient", {"vector": malicious})
            else:
                try:
                    result = await node.execute_pipeline("half_step", {"lr": lr})
                    half_step_vector = result.get("half_step")
                    if half_step_vector is not None:
                        node_half_step_results[node_id] = half_step_vector
                        await node.broadcast_message("gradient", {"vector": half_step_vector})
                except Exception as e:
                    pass

            await asyncio.sleep(0.3)

    training_tasks = [
        asyncio.create_task(node_training_loop(node_id, node, int(node_id) < len(honest_actors)))
        for node_id, node in nodes.items()
    ]

    # Wait for training to progress and evaluate periodically
    evaluation_interval = 5
    for eval_round in range(evaluation_interval, rounds + 1, evaluation_interval):
        # Wait for nodes to reach this round
        await asyncio.sleep(1.0)  # Give nodes time to progress

        # Check if we should evaluate
        if any(rounds >= eval_round for rounds in training_rounds.values()):
            # Get model state from first honest node
            sd = await honest_actors[0].dump_state_dict()
            eval_model.load_state_dict(sd, strict=True)
            loss, acc = await asyncio.to_thread(evaluate, eval_model, device)
            print(f"[round {eval_round:04d}] test loss={loss:.4f}  acc={acc:.4f}")

    # Wait for all training tasks to complete
    await asyncio.gather(*training_tasks, return_exceptions=True)

    print("\n" + "=" * 70)
    print("Training Complete - Final Evaluations")
    print("=" * 70)

    # Final evaluation for each honest node
    print("\nFinal honest-node evaluations:")
    for i, h in enumerate(honest_actors):
        sd = await h.dump_state_dict()
        eval_model.load_state_dict(sd, strict=True)
        loss, acc = await asyncio.to_thread(evaluate, eval_model, device)
        print(f"  Node {i}: loss={loss:.4f}  acc={acc:.4f}")

    print("\nShutting down all nodes...")
    await p2p.stop()
    print("  ✓ All nodes shut down successfully")

    print("\n" + "=" * 70)
    print("Decentralized P2P Training Complete!")
    print("=" * 70)
    print("\nKey Features Demonstrated:")
    print("  ✓ Each node ran in separate OS process (ProcessContext)")
    print("  ✓ Nodes communicated via topology-aware message routing")
    print("  ✓ Message-driven graph execution")
    print("  ✓ Fully decoupled nodes with independent schedulers")
    print("  ✓ Process-level isolation for true parallelism")
    print("  ✓ Fully asynchronous training - no synchronous round() API")
    print("  ✓ Nodes progress independently based on neighbor messages")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
