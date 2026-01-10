"""
Remote TCP Parameter Server Training with MNIST.

Each node (server or worker) runs on its own machine and communicates via TCP.
One node acts as the Parameter Server (aggregator), the rest are workers.

Architecture:
                    ┌─────────────────────────────────────────┐
                    │         Parameter Server                │
                    │   - Receives gradients from workers     │
                    │   - Aggregates using robust aggregator  │
                    │   - Broadcasts aggregated gradient back │
                    └─────────────────────────────────────────┘
                               ▲       ▲       ▲
                               │       │       │
                    TCP ───────┘       │       └─────── TCP
                                       │
                                  TCP ─┘
                               ▼       ▼       ▼
                ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
                │  Worker 0    │ │  Worker 1    │ │  Worker 2    │
                │ (honest)     │ │ (honest)     │ │ (byzantine)  │
                │  Machine A   │ │  Machine B   │ │  Machine C   │
                └──────────────┘ └──────────────┘ └──────────────┘

Usage:
    # Create a config file (nodes.yaml) with server and worker addresses
    # Then run on each machine:

    # On Server machine:
    python ps_node.py --config nodes.yaml --role server

    # On Worker machine A (honest):
    python ps_node.py --config nodes.yaml --role worker --worker-id 0

    # On Worker machine B (honest):
    python ps_node.py --config nodes.yaml --role worker --worker-id 1

    # On Worker machine C (byzantine):
    python ps_node.py --config nodes.yaml --role worker --worker-id 2 --worker-type byzantine
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import hmac
import json
import os
import pickle
import struct
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import datasets, transforms

SCRIPT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from byzpy.aggregators.coordinate_wise.median import CoordinateWiseMedian
from byzpy.aggregators.geometric_wise.krum import MultiKrum
from byzpy.attacks.empire import EmpireAttack

# =============================================================================
# Model Definition
# =============================================================================


class SmallCNN(nn.Module):
    """Simple CNN for MNIST classification."""

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


# =============================================================================
# TCP Message Protocol
# =============================================================================

# Maximum message size (256 MB) - prevents DoS via memory exhaustion
MAX_MSG_BYTES = 256 * 1024 * 1024

# Shared secret for HMAC authentication (in production, use secure key exchange)
# Set via environment variable or leave None to disable HMAC
_HMAC_SECRET: Optional[bytes] = os.environ.get("BYZPY_HMAC_SECRET", "").encode() or None


def _compute_hmac(data: bytes) -> bytes:
    """Compute HMAC-SHA256 for data."""
    if _HMAC_SECRET is None:
        return b""
    return hmac.new(_HMAC_SECRET, data, hashlib.sha256).digest()


def _verify_hmac(data: bytes, tag: bytes) -> bool:
    """Verify HMAC-SHA256 tag for data."""
    if _HMAC_SECRET is None:
        return True  # HMAC disabled
    expected = hmac.new(_HMAC_SECRET, data, hashlib.sha256).digest()
    return hmac.compare_digest(expected, tag)


async def send_message(writer: asyncio.StreamWriter, msg: Dict[str, Any]) -> None:
    """Send a message with length prefix and optional HMAC authentication.

    Wire format:
        [4 bytes: length (big-endian)] [32 bytes: HMAC or zeros] [payload]
    """
    payload = pickle.dumps(msg)
    if len(payload) > MAX_MSG_BYTES:
        raise ValueError(f"Message too large: {len(payload)} bytes > {MAX_MSG_BYTES}")
    tag = _compute_hmac(payload)
    # Pad HMAC to 32 bytes (or zeros if disabled)
    tag_padded = tag.ljust(32, b"\x00") if tag else b"\x00" * 32
    length = struct.pack(">I", len(payload))
    writer.write(length + tag_padded + payload)
    await writer.drain()


async def recv_message(reader: asyncio.StreamReader) -> Optional[Dict[str, Any]]:
    """Receive a message with length prefix and HMAC verification.

    Returns None on connection close, verification failure, or size limit exceeded.
    """
    try:
        length_bytes = await reader.readexactly(4)
        length = struct.unpack(">I", length_bytes)[0]

        # Enforce maximum message size
        if length > MAX_MSG_BYTES:
            print(f"[Security] Rejecting oversized message: {length} bytes")
            return None

        tag = await reader.readexactly(32)
        payload = await reader.readexactly(length)

        # Verify HMAC if enabled
        if not _verify_hmac(payload, tag.rstrip(b"\x00")):
            print("[Security] HMAC verification failed - rejecting message")
            return None

        return pickle.loads(payload)
    except asyncio.IncompleteReadError:
        return None
    except Exception as e:
        print(f"[Security] Error receiving message: {e}")
        return None


# =============================================================================
# Configuration Loading
# =============================================================================


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file."""
    if config_path.endswith(".yaml") or config_path.endswith(".yml"):
        try:
            import yaml

            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except ImportError as err:
            raise ImportError(
                "PyYAML is required for YAML config files. Install with: pip install pyyaml"
            ) from err
    else:
        with open(config_path, "r") as f:
            return json.load(f)


# =============================================================================
# Utility Functions
# =============================================================================


def shard_indices(n_items: int, n_shards: int) -> List[List[int]]:
    """Split dataset indices into shards."""
    return [list(range(i, n_items, n_shards)) for i in range(n_shards)]


def _flatten_grads(model: nn.Module) -> torch.Tensor:
    """Flatten model gradients into a single vector."""
    parts = []
    for p in model.parameters():
        parts.append((torch.zeros_like(p) if p.grad is None else p.grad).view(-1))
    return torch.cat(parts)


def _write_vector_into_grads(model: nn.Module, vec: torch.Tensor) -> None:
    """Write a gradient vector into model parameters."""
    # Validate vector length matches total model parameters
    total_params = sum(p.numel() for p in model.parameters())
    vec_numel = vec.numel()
    if vec_numel != total_params:
        raise ValueError(
            f"Vector size mismatch for {model.__class__.__name__}: "
            f"expected {total_params} elements, got {vec_numel}"
        )

    offset = 0
    for p in model.parameters():
        numel = p.numel()
        chunk = vec[offset : offset + numel].view_as(p).to(p.device)
        if p.grad is None:
            p.grad = chunk.clone()
        else:
            p.grad.copy_(chunk)
        offset += numel


def make_test_loader(batch_size: int = 512, data_root: str = "./data") -> data.DataLoader:
    """Create test data loader."""
    tfm = transforms.Compose([transforms.ToTensor()])
    test = datasets.MNIST(root=data_root, train=False, download=True, transform=tfm)
    return data.DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=0)


def evaluate(model: nn.Module, device: torch.device, data_root: str) -> Tuple[float, float]:
    """Evaluate model on test set."""
    loader = make_test_loader(data_root=data_root)
    ce = nn.CrossEntropyLoss(reduction="sum")
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


# =============================================================================
# Parameter Server
# =============================================================================


class ParameterServerNode:
    """
    Parameter Server that coordinates distributed training.

    Receives gradients from workers, aggregates them, and broadcasts back.
    """

    def __init__(
        self,
        host: str,
        port: int,
        num_workers: int,
        num_honest: int,
        aggregator: Any,
        max_rounds: int,
        data_root: str = "./data",
    ):
        self.host = host
        self.port = port
        self.num_workers = num_workers
        self.num_honest = num_honest
        self.aggregator = aggregator
        self.max_rounds = max_rounds
        self.data_root = data_root

        # State
        self._running = False
        self._server: Optional[asyncio.Server] = None
        self._workers: Dict[int, asyncio.StreamWriter] = {}
        self._worker_readers: Dict[int, asyncio.StreamReader] = {}
        self._gradients: Dict[int, torch.Tensor] = {}
        self._round = 0
        self._model = SmallCNN()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)

        # Synchronization
        self._workers_ready = asyncio.Event()
        self._round_complete = asyncio.Event()

    async def start(self) -> None:
        """Start the parameter server."""
        self._running = True
        self._server = await asyncio.start_server(
            self._handle_worker_connection,
            self.host,
            self.port,
        )
        print(f"[PS] Server started on {self.host}:{self.port}")
        print(f"[PS] Waiting for {self.num_workers} workers to connect...")

    async def _handle_worker_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Handle incoming worker connection."""
        peer = writer.get_extra_info("peername")
        worker_id: Optional[int] = None

        try:
            while self._running:
                msg = await recv_message(reader)
                if msg is None:
                    break

                msg_type = msg.get("type")

                if msg_type == "register":
                    worker_id = msg["worker_id"]
                    self._workers[worker_id] = writer
                    self._worker_readers[worker_id] = reader
                    print(f"[PS] Worker {worker_id} connected from {peer}")

                    # Send initial model state
                    await send_message(
                        writer,
                        {
                            "type": "init_model",
                            "state_dict": {k: v.cpu() for k, v in self._model.state_dict().items()},
                        },
                    )

                    # Check if all workers connected
                    if len(self._workers) >= self.num_workers:
                        self._workers_ready.set()

                elif msg_type == "gradient":
                    if worker_id is not None:
                        gradient = msg["gradient"]
                        self._gradients[worker_id] = gradient
                        # print(f"[PS] Received gradient from worker {worker_id} (round {self._round})")

                        # Check if all gradients received
                        if len(self._gradients) >= self.num_workers:
                            self._round_complete.set()

        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[PS] Error handling worker {worker_id}: {e}")
        finally:
            if worker_id is not None and worker_id in self._workers:
                del self._workers[worker_id]
                if worker_id in self._worker_readers:
                    del self._worker_readers[worker_id]
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def run_training(self) -> None:
        """Run the training loop."""
        print("[PS] Waiting for all workers...")
        await self._workers_ready.wait()
        print(f"[PS] All {self.num_workers} workers connected. Starting training...")

        for r in range(1, self.max_rounds + 1):
            self._round = r
            self._gradients.clear()
            self._round_complete.clear()

            # Signal workers to compute gradients
            await self._broadcast(
                {
                    "type": "compute_gradient",
                    "round": r,
                }
            )

            # Wait for all gradients
            try:
                await asyncio.wait_for(self._round_complete.wait(), timeout=60.0)
            except asyncio.TimeoutError:
                print(f"[PS] Round {r} timeout - only received {len(self._gradients)} gradients")
                continue

            print(f"[PS] Round {r}: received {len(self._gradients)} gradients")

            # Aggregate gradients
            gradients_list = [self._gradients[i] for i in sorted(self._gradients.keys())]
            aggregated = self.aggregator.aggregate(gradients_list)

            # Apply to model
            _write_vector_into_grads(self._model, aggregated.to(self._device))
            # Simple SGD update
            with torch.no_grad():
                for p in self._model.parameters():
                    if p.grad is not None:
                        p.add_(p.grad, alpha=-0.05)

            # Broadcast aggregated gradient to workers
            await self._broadcast(
                {
                    "type": "aggregated_gradient",
                    "gradient": aggregated.cpu(),
                    "round": r,
                }
            )

            # Evaluate periodically
            if r % 10 == 0:
                loss, acc = evaluate(self._model, self._device, self.data_root)
                print(f"[PS] Round {r}/{self.max_rounds} - loss={loss:.4f}, acc={acc:.4f}")

        # Final evaluation
        loss, acc = evaluate(self._model, self._device, self.data_root)
        print(f"[PS] Final - loss={loss:.4f}, acc={acc:.4f}")

        # Signal workers to stop
        await self._broadcast({"type": "stop"})

    async def _broadcast(self, msg: Dict[str, Any]) -> None:
        """Broadcast a message to all connected workers."""
        disconnected = []
        for worker_id, writer in list(self._workers.items()):
            try:
                await send_message(writer, msg)
            except Exception as e:
                print(f"[PS] Failed to send to worker {worker_id}: {e}")
                disconnected.append(worker_id)

        for worker_id in disconnected:
            self._workers.pop(worker_id, None)

    async def shutdown(self) -> None:
        """Shutdown the server."""
        self._running = False
        if self._server:
            self._server.close()
            await self._server.wait_closed()


# =============================================================================
# Worker Node (Honest)
# =============================================================================


class HonestWorkerNode:
    """
    Honest worker that computes legitimate gradients.
    """

    def __init__(
        self,
        worker_id: int,
        server_host: str,
        server_port: int,
        indices: Sequence[int],
        batch_size: int,
        lr: float,
        momentum: float,
        data_root: str = "./data",
    ):
        self.worker_id = worker_id
        self.server_host = server_host
        self.server_port = server_port
        self.indices = indices
        self.batch_size = batch_size
        self.lr = lr
        self.momentum = momentum
        self.data_root = data_root

        # State
        self._running = False
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._model = SmallCNN()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)
        self._optimizer = torch.optim.SGD(self._model.parameters(), lr=lr, momentum=momentum)
        self._criterion = nn.CrossEntropyLoss()
        self._round = 0  # Track current training round

        # Data
        tfm = transforms.Compose([transforms.ToTensor()])
        full = datasets.MNIST(root=data_root, train=True, download=True, transform=tfm)
        subset = data.Subset(full, list(indices))
        self._loader = data.DataLoader(
            subset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0
        )
        self._data_iter = iter(self._loader)

    def _next_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get next batch, cycling if needed."""
        try:
            x, y = next(self._data_iter)
        except StopIteration:
            self._data_iter = iter(self._loader)
            x, y = next(self._data_iter)
        return x.to(self._device), y.to(self._device)

    def _compute_gradient(self) -> torch.Tensor:
        """Compute gradient for current batch."""
        x, y = self._next_batch()
        self._model.zero_grad(set_to_none=True)
        logits = self._model(x)
        loss = self._criterion(logits, y)
        loss.backward()
        return _flatten_grads(self._model)

    async def connect(self) -> None:
        """Connect to parameter server."""
        print(
            f"[Worker {self.worker_id}] Connecting to server {self.server_host}:{self.server_port}..."
        )
        self._reader, self._writer = await asyncio.open_connection(
            self.server_host, self.server_port
        )
        self._running = True

        # Register with server
        await send_message(
            self._writer,
            {
                "type": "register",
                "worker_id": self.worker_id,
            },
        )
        print(f"[Worker {self.worker_id}] Connected and registered")

    async def run(self) -> None:
        """Run the worker loop."""
        while self._running:
            msg = await recv_message(self._reader)
            if msg is None:
                break

            msg_type = msg.get("type")

            if msg_type == "init_model":
                # Initialize model with server's state
                state_dict = msg["state_dict"]
                self._model.load_state_dict({k: v.to(self._device) for k, v in state_dict.items()})
                print(f"[Worker {self.worker_id}] Model initialized from server")

            elif msg_type == "compute_gradient":
                round_num = msg["round"]
                self._round = round_num  # Update current round
                gradient = self._compute_gradient()
                await send_message(
                    self._writer,
                    {
                        "type": "gradient",
                        "gradient": gradient.cpu(),
                        "round": round_num,
                    },
                )
                print(f"[Worker {self.worker_id}] Sent gradient for round {round_num}")

            elif msg_type == "aggregated_gradient":
                round_num = msg["round"]
                # Validate round number matches current round
                if round_num != self._round:
                    print(
                        f"[Worker {self.worker_id}] Warning: Ignoring stale gradient "
                        f"(received round {round_num}, expected {self._round})"
                    )
                    continue

                aggregated = msg["gradient"].to(self._device)
                _write_vector_into_grads(self._model, aggregated)
                self._optimizer.step()

                if round_num % 10 == 0:
                    loss, acc = evaluate(self._model, self._device, self.data_root)
                    print(
                        f"[Worker {self.worker_id}] Round {round_num} - loss={loss:.4f}, acc={acc:.4f}"
                    )

            elif msg_type == "stop":
                print(f"[Worker {self.worker_id}] Received stop signal")
                break

        print(f"[Worker {self.worker_id}] Shutting down")
        self._running = False

    async def shutdown(self) -> None:
        """Shutdown the worker."""
        self._running = False
        if self._writer:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                pass


# =============================================================================
# Worker Node (Byzantine)
# =============================================================================


class ByzantineWorkerNode:
    """
    Byzantine worker that generates malicious gradients.
    """

    def __init__(
        self,
        worker_id: int,
        server_host: str,
        server_port: int,
        attack_scale: float = -1.0,
    ):
        self.worker_id = worker_id
        self.server_host = server_host
        self.server_port = server_port
        self.attack = EmpireAttack(scale=attack_scale)

        # State
        self._running = False
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._last_aggregated: Optional[torch.Tensor] = None

    async def connect(self) -> None:
        """Connect to parameter server."""
        print(
            f"[ByzWorker {self.worker_id}] Connecting to server {self.server_host}:{self.server_port}..."
        )
        self._reader, self._writer = await asyncio.open_connection(
            self.server_host, self.server_port
        )
        self._running = True

        # Register with server
        await send_message(
            self._writer,
            {
                "type": "register",
                "worker_id": self.worker_id,
            },
        )
        print(f"[ByzWorker {self.worker_id}] Connected and registered")

    async def run(self) -> None:
        """Run the byzantine worker loop."""
        while self._running:
            msg = await recv_message(self._reader)
            if msg is None:
                break

            msg_type = msg.get("type")

            if msg_type == "init_model":
                print(f"[ByzWorker {self.worker_id}] Ignoring model init (byzantine)")

            elif msg_type == "compute_gradient":
                round_num = msg["round"]
                # Generate malicious gradient based on last aggregated
                if self._last_aggregated is not None:
                    malicious = self.attack.apply(honest_grads=[self._last_aggregated])
                else:
                    # First round - send zero gradient
                    # Get gradient shape from model
                    model = SmallCNN()
                    total_params = sum(p.numel() for p in model.parameters())
                    malicious = torch.zeros(total_params)

                await send_message(
                    self._writer,
                    {
                        "type": "gradient",
                        "gradient": malicious.cpu(),
                        "round": round_num,
                    },
                )
                print(f"[ByzWorker {self.worker_id}] Sent malicious gradient for round {round_num}")

            elif msg_type == "aggregated_gradient":
                round_num = msg["round"]
                self._last_aggregated = msg["gradient"]

            elif msg_type == "stop":
                print(f"[ByzWorker {self.worker_id}] Received stop signal")
                break

        print(f"[ByzWorker {self.worker_id}] Shutting down")
        self._running = False

    async def shutdown(self) -> None:
        """Shutdown the worker."""
        self._running = False
        if self._writer:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                pass


# =============================================================================
# Main Entry Point
# =============================================================================


async def run_server(config: Dict[str, Any], args) -> None:
    """Run the parameter server."""
    server_cfg = config["server"]
    workers_cfg = config.get("workers", [])
    training_cfg = config.get("training", {})
    agg_cfg = config.get("aggregation", {})

    host = server_cfg.get("bind_host", "0.0.0.0")
    port = server_cfg["port"]
    num_workers = len(workers_cfg)
    num_honest = sum(1 for w in workers_cfg if w.get("type", "honest") == "honest")
    max_rounds = args.rounds or training_cfg.get("rounds", 50)

    # Create aggregator
    agg_method = agg_cfg.get("method", "coordinate_wise_median")
    if agg_method == "multi_krum":
        f = agg_cfg.get("krum_f", 1)
        q = agg_cfg.get("krum_q", num_workers - f)
        aggregator = MultiKrum(f=f, q=q)
        print(f"[PS] Using MultiKrum aggregator (f={f}, q={q})")
    else:
        aggregator = CoordinateWiseMedian()
        print(f"[PS] Using CoordinateWiseMedian aggregator")

    ps = ParameterServerNode(
        host=host,
        port=port,
        num_workers=num_workers,
        num_honest=num_honest,
        aggregator=aggregator,
        max_rounds=max_rounds,
        data_root=args.data_root,
    )

    print("=" * 70)
    print("Parameter Server - Remote TCP Training")
    print("=" * 70)
    print(f"  Host: {host}:{port}")
    print(f"  Workers: {num_workers} ({num_honest} honest)")
    print(f"  Rounds: {max_rounds}")
    print("=" * 70)

    await ps.start()

    # Start server coroutine (handles connections)
    server_task = asyncio.create_task(ps._server.serve_forever())

    try:
        await ps.run_training()
    except KeyboardInterrupt:
        print("\n[PS] Interrupted")
    finally:
        await ps.shutdown()
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass


async def run_worker(config: Dict[str, Any], args) -> None:
    """Run a worker node."""
    server_cfg = config["server"]
    workers_cfg = config.get("workers", [])
    training_cfg = config.get("training", {})

    # Find this worker's config
    worker_cfg = None
    for w in workers_cfg:
        if w["id"] == args.worker_id:
            worker_cfg = w
            break

    if worker_cfg is None:
        raise ValueError(f"Worker ID {args.worker_id} not found in config")

    server_host = server_cfg["host"]
    server_port = server_cfg["port"]
    worker_type = args.worker_type or worker_cfg.get("type", "honest")

    print("=" * 70)
    print(f"Worker {args.worker_id} - Remote TCP Training")
    print("=" * 70)
    print(f"  Server: {server_host}:{server_port}")
    print(f"  Type: {worker_type}")
    print("=" * 70)

    if worker_type == "honest":
        # Calculate data shard
        data_shard = worker_cfg.get("data_shard", args.worker_id)
        num_honest = sum(1 for w in workers_cfg if w.get("type", "honest") == "honest")

        tfm = transforms.Compose([transforms.ToTensor()])
        full_train = datasets.MNIST(root=args.data_root, train=True, download=True, transform=tfm)
        shards = shard_indices(len(full_train), num_honest)
        indices = shards[data_shard % len(shards)]

        worker = HonestWorkerNode(
            worker_id=args.worker_id,
            server_host=server_host,
            server_port=server_port,
            indices=indices,
            batch_size=training_cfg.get("batch_size", 64),
            lr=training_cfg.get("learning_rate", 0.05),
            momentum=training_cfg.get("momentum", 0.9),
            data_root=args.data_root,
        )
    else:
        attack_scale = worker_cfg.get("attack_scale", -1.0)
        worker = ByzantineWorkerNode(
            worker_id=args.worker_id,
            server_host=server_host,
            server_port=server_port,
            attack_scale=attack_scale,
        )

    try:
        await worker.connect()
        await worker.run()
    except KeyboardInterrupt:
        print(f"\n[Worker {args.worker_id}] Interrupted")
    except ConnectionRefusedError:
        print(f"[Worker {args.worker_id}] Connection refused - is the server running?")
    finally:
        await worker.shutdown()


async def main(args) -> None:
    """Main entry point."""
    config = load_config(args.config)

    if args.role == "server":
        await run_server(config, args)
    elif args.role == "worker":
        if args.worker_id is None:
            raise ValueError("--worker-id is required when --role=worker")
        await run_worker(config, args)
    else:
        raise ValueError(f"Unknown role: {args.role}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remote TCP Parameter Server Training")

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file (YAML or JSON)",
    )
    parser.add_argument(
        "--role",
        type=str,
        choices=["server", "worker"],
        required=True,
        help="Node role: 'server' (parameter server) or 'worker'",
    )
    parser.add_argument(
        "--worker-id",
        type=int,
        default=None,
        help="Worker ID (required for --role=worker)",
    )
    parser.add_argument(
        "--worker-type",
        type=str,
        choices=["honest", "byzantine"],
        default=None,
        help="Worker type (overrides config)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="Training rounds (overrides config)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="./data",
        help="Data root directory (default: ./data)",
    )

    args = parser.parse_args()
    asyncio.run(main(args))
