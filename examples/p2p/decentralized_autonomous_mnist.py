from __future__ import annotations

import asyncio
import os
import sys
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import datasets, transforms

# Add project root to path
SCRIPT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from examples.p2p.nodes import SmallCNN, select_pool_backend

from byzpy.aggregators.coordinate_wise.median import CoordinateWiseMedian
from byzpy.attacks.empire import EmpireAttack
from byzpy.configs.actor import set_actor
from byzpy.engine.graph.ops import CallableOp, make_single_operator_graph
from byzpy.engine.graph.pool import ActorPoolConfig
from byzpy.engine.node.actors import ByzantineNodeActor, HonestNodeActor
from byzpy.engine.node.application import ByzantineNodeApplication, HonestNodeApplication
from byzpy.engine.node.cluster import DecentralizedCluster
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


async def main():
    """
    Main function demonstrating fully autonomous decentralized P2P training.

    Architecture:
    - Each node runs in separate OS process with ProcessContext
    - Models, optimizers, data loaders are created IN EACH SUBPROCESS
    - Initialization callbacks set up node-local state and pipelines in subprocess
    - Nodes coordinate purely through messages via topology
    - No centralized coordination - nodes operate autonomously
    - No registry access - all state is node-local
    """
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Configuration
    n_honest = 4
    n_byz = 1
    rounds = 50  # Training rounds
    batch_size = 64
    lr = 0.05
    momentum = 0.9
    data_root = "./data"

    print("=" * 70)
    print("Fully Autonomous Decentralized P2P Training with MNIST")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  - Honest nodes: {n_honest}")
    print(f"  - Byzantine nodes: {n_byz}")
    print(f"  - Training rounds: {rounds}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Learning rate: {lr}")
    print(f"  - Device: {device}")
    print(f"  - Coordination: Fully autonomous (nodes run training loops in subprocess)")
    print(f"  - State: Node-local (models created in subprocess, not pickled)")
    print("=" * 70)

    tfm = transforms.Compose([transforms.ToTensor()])
    _tmp_train = datasets.MNIST(root=data_root, train=True, download=True, transform=tfm)
    shards = shard_indices(len(_tmp_train), n_honest)

    def context_factory(node_id: str, node_index: int) -> ProcessContext:
        """Factory function that creates ProcessContext for each node."""
        return ProcessContext()

    # Create initialization callbacks that create models IN THE SUBPROCESS
    def make_honest_init_callback(
        node_id: str,
        indices: List[int],
        lr: float,
        max_rounds: int,
        batch_size: int,
        data_root: str,
        device_str: str,
    ):
        """Create initialization callback for honest nodes."""

        async def init_callback(node: DecentralizedNode):
            # Get process ID to demonstrate process-level isolation
            pid = os.getpid()
            print(f"[Node {node_id}, PID {pid}] Initializing honest node in subprocess...")

            # Create model, data loader, etc. IN THIS SUBPROCESS (node-local)
            dev = torch.device(device_str)
            model = SmallCNN().to(dev)
            criterion = nn.CrossEntropyLoss()

            # Create data loader in subprocess
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
                persistent_workers=False,
            )
            data_iter_ref = [iter(data_loader)]  # Mutable reference for iterator

            print(f"[Node {node_id}, PID {pid}] Model and data loader created (node-local)")

            # Store node-local state in a dict accessible to pipelines
            # We'll pass this to pipelines via application metadata or closure
            node_state = {
                "model": model,
                "criterion": criterion,
                "data_loader": data_loader,
                "data_iter": data_iter_ref,
                "device": dev,
            }

            # Register pipelines that use node-local state
            # Half-step pipeline
            async def half_step(lr: float):
                """Perform half-step of gradient descent on node-local model."""
                state = node_state  # Access node-local state
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
                node_name="half_step",
                operator=half_step_op,
                input_keys=("lr",),
            )
            node.application.register_pipeline("half_step", half_step_graph)

            # Aggregation pipeline
            aggregator = CoordinateWiseMedian()
            aggregate_graph = make_single_operator_graph(
                node_name="aggregate",
                operator=aggregator,
                input_keys=("gradients",),
            )
            node.application.register_pipeline("aggregate", aggregate_graph)

            # Update model pipeline
            async def update_model(param_vector: torch.Tensor):
                """Update node-local model parameters."""
                state = node_state
                _write_params(state["model"], param_vector)

            update_model_op = CallableOp(
                update_model, input_mapping={"param_vector": "param_vector"}
            )
            update_model_graph = make_single_operator_graph(
                node_name="update_model",
                operator=update_model_op,
                input_keys=("param_vector",),
            )
            node.application.register_pipeline("update_model", update_model_graph)

            # Local state for training loop
            gradient_cache: List[torch.Tensor] = []
            self_half_step_result: torch.Tensor = None
            rounds_completed = 0

            # Register message handler for gradients
            async def on_gradient(from_id: str, payload: dict):
                nonlocal gradient_cache, self_half_step_result, rounds_completed
                pid = os.getpid()

                # Skip processing if we've already completed max_rounds
                if rounds_completed >= max_rounds:
                    return

                gradient = payload.get("vector")
                if gradient is not None:
                    gradient_cache.append(gradient)

                # Trigger aggregation when we have at least one neighbor gradient
                if self_half_step_result is not None and len(gradient_cache) >= 1:
                    # Double-check we haven't exceeded max_rounds
                    if rounds_completed >= max_rounds:
                        return

                    try:
                        num_grads = len(gradient_cache) + 1
                        # Aggregate using pipeline
                        result = await node.execute_pipeline(
                            "aggregate",
                            {"gradients": [self_half_step_result] + gradient_cache},
                        )
                        aggregated = result.get("aggregate")

                        if aggregated is not None:
                            # Update node-local model using pipeline
                            await node.execute_pipeline(
                                "update_model", {"param_vector": aggregated}
                            )

                        gradient_cache.clear()
                        rounds_completed += 1
                        print(
                            f"[Node {node_id}, PID {pid}] Completed round {rounds_completed}/{max_rounds} (aggregated {num_grads} gradients)"
                        )
                    except Exception as e:
                        pass  # Continue even if aggregation fails

            node.register_message_handler("gradient", on_gradient)

            # Start autonomous training loop
            async def training_loop():
                nonlocal self_half_step_result, rounds_completed
                pid = os.getpid()

                # Small delay to let all nodes start
                await asyncio.sleep(0.5)
                print(f"[Node {node_id}, PID {pid}] Starting autonomous training loop...")

                step_count = 0
                while node._running and rounds_completed < max_rounds:
                    # Check condition again before each iteration (in case it changed during sleep)
                    if rounds_completed >= max_rounds:
                        break

                    try:
                        # Perform half-step using pipeline (on node-local model)
                        result = await node.execute_pipeline("half_step", {"lr": lr})
                        half_step_vector = result.get("half_step")

                        if half_step_vector is not None:
                            # Only broadcast if we haven't exceeded max_rounds
                            if rounds_completed < max_rounds:
                                self_half_step_result = half_step_vector
                                step_count += 1

                                # Broadcast gradient to neighbors
                                await node.broadcast_message(
                                    "gradient", {"vector": half_step_vector}
                                )
                                if step_count % 5 == 0:  # Print every 5 steps
                                    print(
                                        f"[Node {node_id}, PID {pid}] Step {step_count}: half-step complete, broadcast gradient (rounds completed: {rounds_completed}/{max_rounds})"
                                    )
                            else:
                                break  # Exit loop if we've reached max_rounds

                        # Wait before next step
                        await asyncio.sleep(0.2)
                    except Exception as e:
                        # Continue even if step fails
                        await asyncio.sleep(0.1)

                print(
                    f"[Node {node_id}, PID {pid}] Training loop completed ({rounds_completed} rounds)"
                )

            await node.start_autonomous_task(training_loop(), "training")

        return init_callback

    def make_byzantine_init_callback(node_id: str, max_rounds: int, device_str: str, scale: float):
        """Create initialization callback for byzantine nodes."""

        async def init_callback(node: DecentralizedNode):
            # Get process ID to demonstrate process-level isolation
            pid = os.getpid()
            print(f"[Node {node_id}, PID {pid}] Initializing byzantine node in subprocess...")

            # Create attack IN THIS SUBPROCESS (node-local)
            attack = EmpireAttack(scale=scale)
            node_state = {"attack": attack}

            print(f"[Node {node_id}, PID {pid}] Attack created (node-local)")

            # Register broadcast pipeline
            async def broadcast(neighbor_vectors: List[torch.Tensor], like: torch.Tensor):
                """Generate malicious vector using node-local attack."""
                if not neighbor_vectors:
                    return like
                state = node_state
                result = state["attack"].apply(honest_grads=neighbor_vectors)
                return result

            broadcast_op = CallableOp(
                broadcast,
                input_mapping={"neighbor_vectors": "neighbor_vectors", "like": "like"},
            )
            broadcast_graph = make_single_operator_graph(
                node_name="broadcast",
                operator=broadcast_op,
                input_keys=("neighbor_vectors", "like"),
            )
            node.application.register_pipeline("broadcast", broadcast_graph)

            # Local state for attack loop
            gradient_cache: List[torch.Tensor] = []
            rounds_completed = 0

            # Register handler to collect gradients
            async def on_gradient(from_id: str, payload: dict):
                nonlocal gradient_cache
                gradient = payload.get("vector")
                if gradient is not None:
                    gradient_cache.append(gradient)

            node.register_message_handler("gradient", on_gradient)

            # Start autonomous attack loop
            async def attack_loop():
                nonlocal gradient_cache, rounds_completed
                pid = os.getpid()

                await asyncio.sleep(0.5)
                print(f"[Node {node_id}, PID {pid}] Starting autonomous attack loop...")

                while node._running and rounds_completed < max_rounds:
                    # Check condition again before each iteration
                    if rounds_completed >= max_rounds:
                        break

                    if len(gradient_cache) > 0:
                        # Double-check we haven't exceeded max_rounds
                        if rounds_completed >= max_rounds:
                            break

                        try:
                            # Use broadcast pipeline (with node-local attack)
                            template = gradient_cache[0]
                            result = await node.execute_pipeline(
                                "broadcast",
                                {"neighbor_vectors": gradient_cache, "like": template},
                            )
                            malicious = result.get("broadcast")

                            if malicious is not None:
                                # Only broadcast if we haven't exceeded max_rounds
                                if rounds_completed < max_rounds:
                                    await node.broadcast_message("gradient", {"vector": malicious})
                                    gradient_cache.clear()
                                    rounds_completed += 1
                                    print(
                                        f"[Node {node_id}, PID {pid}] Attack round {rounds_completed}/{max_rounds} complete (broadcast malicious vector)"
                                    )
                                else:
                                    break
                        except Exception as e:
                            pass
                    await asyncio.sleep(0.2)

                print(
                    f"[Node {node_id}, PID {pid}] Attack loop completed ({rounds_completed} rounds)"
                )

            await node.start_autonomous_task(attack_loop(), "attack")

        return init_callback

    print("\nInitializing P2P training runner with autonomous nodes...")

    # Create applications (pipelines will be registered in subprocess)
    cluster = DecentralizedCluster()
    n = n_honest + n_byz

    # Create nodes with empty applications (pipelines registered in init_callback)
    for idx in range(n):
        node_id = str(idx)
        context = context_factory(node_id, idx)

        if idx < n_honest:
            # Honest node - create empty application (pipelines registered in subprocess)
            app = HonestNodeApplication(
                name=f"honest_{node_id}",
                actor_pool=[ActorPoolConfig(backend="thread", count=1)],
            )

            node = await cluster.add_node(
                node_id=node_id,
                application=app,
                topology=Topology.ring(n=n, k=1),
                context=context,
            )

            # Set initialization callback (executed in subprocess)
            node._init_callback = make_honest_init_callback(
                node_id, shards[idx], lr, rounds, batch_size, data_root, str(device)
            )
        else:
            # Byzantine node
            app = ByzantineNodeApplication(
                name=f"byz_{node_id}",
                actor_pool=[ActorPoolConfig(backend="thread", count=1)],
            )

            node = await cluster.add_node(
                node_id=node_id,
                application=app,
                topology=Topology.ring(n=n, k=1),
                context=context,
            )

            # Set initialization callback (executed in subprocess)
            node._init_callback = make_byzantine_init_callback(node_id, rounds, str(device), -1.0)

    print("Starting all nodes (each in separate OS process with autonomous training loops)...")
    await cluster.start_all()
    # Wait for training to complete
    print(f"\nWaiting for {rounds} training rounds to complete...")
    await asyncio.sleep(rounds * 0.3 + 5.0)  # Approximate time for rounds + buffer

    print("\nShutting down all nodes...")
    await cluster.shutdown_all()
    print("  ✓ All nodes shut down successfully")


if __name__ == "__main__":
    asyncio.run(main())
