"""
ByzPy Actor Runtime Demo

This demo showcases ByzPy's actor runtime capabilities:

1. **Arbitrary Python Classes as Actors**: Any Python class can be wrapped as an
   actor and deployed on different backends (thread, process, remote TCP).

2. **Cross-Backend Communication**: Actors on different backends can communicate
   seamlessly via channels (thread <-> process, local <-> remote, etc.).

3. **Unified API**: Same code works regardless of where the actor runs.

Architecture:
    ┌────────────────────────────────────────────────────────────────┐
    │                       Coordinator                              │
    │  - Spawns actors on different backends                         │
    │  - Orchestrates communication between actors                   │
    └────────────────────────────────────────────────────────────────┘
              │                    │                    │
              ▼                    ▼                    ▼
    ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
    │  Thread Actor    │ │  Process Actor   │ │  Remote Actor    │
    │  (same process)  │ │  (child process) │ │  (TCP server)    │
    │                  │◄─────────────────►│◄─────────────────►│
    │  Calculator      │ │  Aggregator      │ │  Logger          │
    └──────────────────┘ └──────────────────┘ └──────────────────┘
                              via channels

Usage:
    # Basic demo (thread + process backends)
    python actor_demo.py

    # With remote actor (start server first):
    # Terminal 1: python remote_server.py
    # Terminal 2: python actor_demo.py --with-remote

    # Show cross-backend channel communication
    python actor_demo.py --channel-demo
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, List, Optional

SCRIPT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from byzpy.engine.actor.backends.process import ProcessActorBackend
from byzpy.engine.actor.backends.remote import RemoteActorBackend
from byzpy.engine.actor.backends.thread import ThreadActorBackend
from byzpy.engine.actor.base import ActorRef

# =============================================================================
# Example 1: Arbitrary Python Classes as Actors
# =============================================================================


class Calculator:
    """
    A simple calculator class.

    This demonstrates that ANY Python class can become an actor.
    The class doesn't need to inherit from anything special.
    """

    def __init__(self, name: str = "calc"):
        self.name = name
        self.history: List[str] = []
        print(f"[{self.name}] Calculator initialized")

    def add(self, a: float, b: float) -> float:
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result

    def multiply(self, a: float, b: float) -> float:
        result = a * b
        self.history.append(f"{a} * {b} = {result}")
        return result

    def get_history(self) -> List[str]:
        return self.history.copy()

    def info(self) -> str:
        return f"Calculator '{self.name}' with {len(self.history)} operations"


class DataProcessor:
    """
    A data processing class that works with lists.

    Demonstrates that actors can handle complex data types.
    """

    def __init__(self, processor_id: int):
        self.processor_id = processor_id
        self.processed_count = 0
        print(f"[Processor-{processor_id}] DataProcessor initialized")

    def process(self, data: List[float]) -> dict:
        """Process a list of numbers and return statistics."""
        if not data:
            return {"error": "empty data"}

        self.processed_count += 1
        return {
            "processor_id": self.processor_id,
            "count": len(data),
            "sum": sum(data),
            "mean": sum(data) / len(data),
            "min": min(data),
            "max": max(data),
        }

    def get_processed_count(self) -> int:
        return self.processed_count


class Aggregator:
    """
    Aggregates results from multiple sources.

    Demonstrates stateful actors that accumulate data.
    """

    def __init__(self):
        self.results: List[Any] = []
        print("[Aggregator] Initialized")

    def add_result(self, result: Any) -> int:
        """Add a result and return current count."""
        self.results.append(result)
        return len(self.results)

    def get_all_results(self) -> List[Any]:
        return self.results.copy()

    def compute_final(self) -> dict:
        """Compute final aggregated statistics."""
        if not self.results:
            return {"error": "no results"}

        # Aggregate numeric results
        all_sums = [r.get("sum", 0) for r in self.results if isinstance(r, dict)]
        all_counts = [r.get("count", 0) for r in self.results if isinstance(r, dict)]

        return {
            "total_batches": len(self.results),
            "total_sum": sum(all_sums),
            "total_count": sum(all_counts),
        }


# =============================================================================
# Example 2: Channel-Based Communication
# =============================================================================


class Producer:
    """Produces messages and sends them via channel."""

    def __init__(self, name: str):
        self.name = name
        self.produced = 0
        print(f"[Producer-{name}] Initialized")

    def produce(self) -> dict:
        """Produce a message."""
        self.produced += 1
        return {
            "producer": self.name,
            "sequence": self.produced,
            "timestamp": time.time(),
            "data": list(range(self.produced)),
        }

    def get_produced_count(self) -> int:
        return self.produced


class Consumer:
    """Consumes messages received via channel."""

    def __init__(self, name: str):
        self.name = name
        self.consumed: List[dict] = []
        print(f"[Consumer-{name}] Initialized")

    def consume(self, message: dict) -> str:
        """Consume a message and return acknowledgment."""
        self.consumed.append(message)
        return f"Consumer-{self.name} processed message #{message.get('sequence')} from {message.get('producer')}"

    def get_consumed_count(self) -> int:
        return len(self.consumed)

    def get_consumed_messages(self) -> List[dict]:
        return self.consumed.copy()


# =============================================================================
# Demo Functions
# =============================================================================


async def demo_basic_actors():
    """
    Demo 1: Basic actor functionality.

    Shows that arbitrary Python classes can be wrapped as actors and
    executed on different backends with the same API.
    """
    print("\n" + "=" * 70)
    print("Demo 1: Arbitrary Python Classes as Actors")
    print("=" * 70)

    # Create actors on different backends using async context managers
    print("\n[1.1] Spawning Calculator on THREAD backend...")
    thread_backend = ThreadActorBackend()

    print("[1.2] Spawning DataProcessor on PROCESS backend...")
    process_backend = ProcessActorBackend()

    # Use async context managers for lifecycle management
    async with ActorRef(thread_backend) as thread_actor:
        await thread_actor._backend.construct(Calculator, args=(), kwargs={"name": "thread-calc"})

        async with ActorRef(process_backend) as process_actor:
            await process_actor._backend.construct(
                DataProcessor, args=(), kwargs={"processor_id": 1}
            )

            # Call methods on actors (same API regardless of backend!)
            print("\n[1.3] Calling methods on actors...")

            # Thread actor calls
            result1 = await thread_actor.add(10, 20)
            print(f"  Thread actor: 10 + 20 = {result1}")

            result2 = await thread_actor.multiply(5, 6)
            print(f"  Thread actor: 5 * 6 = {result2}")

            history = await thread_actor.get_history()
            print(f"  Thread actor history: {history}")

            # Process actor calls
            stats = await process_actor.process([1.0, 2.0, 3.0, 4.0, 5.0])
            print(f"  Process actor stats: {stats}")

            count = await process_actor.get_processed_count()
            print(f"  Process actor processed {count} batches")

    print("\n✓ Demo 1 complete: Same API works across thread and process backends!")


async def demo_cross_backend_communication():
    """
    Demo 2: Cross-backend channel communication.

    Shows that actors on different backends can communicate via channels.
    """
    print("\n" + "=" * 70)
    print("Demo 2: Cross-Backend Channel Communication")
    print("=" * 70)

    # Create Producer on thread backend
    print("\n[2.1] Spawning Producer on THREAD backend...")
    producer_backend = ThreadActorBackend()
    producer_ref = ActorRef(producer_backend)
    await producer_ref._backend.start()
    await producer_ref._backend.construct(Producer, args=(), kwargs={"name": "A"})

    # Create Consumer on process backend
    print("[2.2] Spawning Consumer on PROCESS backend...")
    consumer_backend = ProcessActorBackend()
    consumer_ref = ActorRef(consumer_backend)
    await consumer_ref._backend.start()
    await consumer_ref._backend.construct(Consumer, args=(), kwargs={"name": "B"})

    # Open channels
    print("\n[2.3] Opening channels for cross-backend communication...")
    producer_channel = await producer_ref.open_channel("messages")
    consumer_channel = await consumer_ref.open_channel("messages")

    producer_ep = await producer_ref.endpoint()
    consumer_ep = await consumer_ref.endpoint()

    print(f"  Producer endpoint: {producer_ep}")
    print(f"  Consumer endpoint: {consumer_ep}")

    # Producer sends messages to consumer
    print("\n[2.4] Producer (thread) sending messages to Consumer (process)...")
    for i in range(3):
        # Producer creates a message
        message = await producer_ref.produce()
        print(f"  Produced: {message}")

        # Send via channel to consumer
        await producer_channel.send(consumer_ep, message)
        print(f"  Sent to consumer via channel")

        await asyncio.sleep(0.1)

    # Consumer receives and processes messages
    print("\n[2.5] Consumer (process) receiving messages from Producer (thread)...")
    for i in range(3):
        # Receive from channel
        message = await consumer_channel.recv(timeout=5.0)
        if message:
            # Process the message
            ack = await consumer_ref.consume(message)
            print(f"  {ack}")

    # Verify
    produced = await producer_ref.get_produced_count()
    consumed = await consumer_ref.get_consumed_count()
    print(f"\n  Producer produced: {produced} messages")
    print(f"  Consumer consumed: {consumed} messages")

    # Cleanup
    await producer_ref._backend.close()
    await consumer_ref._backend.close()

    print("\n✓ Demo 2 complete: Thread and Process actors communicated via channels!")


async def demo_multi_backend_pipeline():
    """
    Demo 3: Multi-backend pipeline.

    Shows a realistic pipeline where data flows through actors on different backends.
    """
    print("\n" + "=" * 70)
    print("Demo 3: Multi-Backend Data Pipeline")
    print("=" * 70)

    # Create a pipeline: Calculators (thread) -> Aggregator (process)
    print("\n[3.1] Creating pipeline...")

    # Multiple calculator actors on thread backend
    calc_actors = []
    for i in range(3):
        backend = ThreadActorBackend()
        actor = ActorRef(backend)
        await actor._backend.start()
        await actor._backend.construct(Calculator, args=(), kwargs={"name": f"calc-{i}"})
        calc_actors.append(actor)
        print(f"  Created Calculator-{i} on thread backend")

    # Aggregator on process backend
    agg_backend = ProcessActorBackend()
    agg_actor = ActorRef(agg_backend)
    await agg_actor._backend.start()
    await agg_actor._backend.construct(Aggregator, args=(), kwargs={})
    print("  Created Aggregator on process backend")

    # Run parallel computations
    print("\n[3.2] Running parallel computations...")
    tasks = []
    for i, calc in enumerate(calc_actors):
        # Each calculator does different work
        tasks.append(calc.add(i * 10, i * 10 + 5))
        tasks.append(calc.multiply(i + 1, i + 2))

    results = await asyncio.gather(*tasks)
    print(f"  Computation results: {results}")

    # Collect results into aggregator
    print("\n[3.3] Aggregating results...")
    for i, result in enumerate(results):
        count = await agg_actor.add_result({"value": result, "index": i})
        print(f"  Added result {i}: {result} (total: {count})")

    # Get final aggregation
    all_results = await agg_actor.get_all_results()
    print(f"\n  All aggregated results: {all_results}")

    # Cleanup
    for calc in calc_actors:
        await calc._backend.close()
    await agg_actor._backend.close()

    print("\n✓ Demo 3 complete: Multi-backend pipeline executed successfully!")


async def demo_with_remote(remote_host: str, remote_port: int):
    """
    Demo 4: Remote actor communication.

    Shows communication between local and remote actors via TCP.
    Requires remote_server.py to be running.
    """
    print("\n" + "=" * 70)
    print("Demo 4: Remote Actor Communication (via TCP)")
    print("=" * 70)

    print(f"\n[4.1] Connecting to remote server at {remote_host}:{remote_port}...")

    try:
        # Create remote actor
        remote_backend = RemoteActorBackend(remote_host, remote_port)
        remote_actor = ActorRef(remote_backend)
        await remote_actor._backend.start()
        await remote_actor._backend.construct(Calculator, args=(), kwargs={"name": "remote-calc"})
        print("  Connected and spawned Calculator on remote server!")

        # Create local actor
        print("\n[4.2] Creating local Calculator on thread backend...")
        local_backend = ThreadActorBackend()
        local_actor = ActorRef(local_backend)
        await local_actor._backend.start()
        await local_actor._backend.construct(Calculator, args=(), kwargs={"name": "local-calc"})

        # Both use same API!
        print("\n[4.3] Calling both actors with identical API...")

        local_result = await local_actor.add(100, 200)
        print(f"  Local actor (thread): 100 + 200 = {local_result}")

        remote_result = await remote_actor.add(100, 200)
        print(f"  Remote actor (TCP):   100 + 200 = {remote_result}")

        local_info = await local_actor.info()
        remote_info = await remote_actor.info()
        print(f"\n  Local info:  {local_info}")
        print(f"  Remote info: {remote_info}")

        # Cleanup
        await local_actor._backend.close()
        await remote_actor._backend.close()

        print("\n✓ Demo 4 complete: Local and remote actors work with same API!")

    except ConnectionRefusedError:
        print(f"  ERROR: Could not connect to {remote_host}:{remote_port}")
        print("  Make sure to start the remote server first:")
        print("    python remote_server.py")


async def demo_channel_cross_backend():
    """
    Demo 5: Detailed channel communication between all backend types.
    """
    print("\n" + "=" * 70)
    print("Demo 5: Detailed Channel Communication")
    print("=" * 70)

    # Create actors on different backends
    print("\n[5.1] Creating actors on thread and process backends...")

    # Thread actor 1
    t1_backend = ThreadActorBackend()
    t1_actor = ActorRef(t1_backend)
    await t1_actor._backend.start()
    await t1_actor._backend.construct(Producer, args=(), kwargs={"name": "Thread-1"})

    # Thread actor 2
    t2_backend = ThreadActorBackend()
    t2_actor = ActorRef(t2_backend)
    await t2_actor._backend.start()
    await t2_actor._backend.construct(Consumer, args=(), kwargs={"name": "Thread-2"})

    # Process actor
    p1_backend = ProcessActorBackend()
    p1_actor = ActorRef(p1_backend)
    await p1_actor._backend.start()
    await p1_actor._backend.construct(Consumer, args=(), kwargs={"name": "Process-1"})

    # Open channels
    t1_channel = await t1_actor.open_channel("data")
    t2_channel = await t2_actor.open_channel("data")
    p1_channel = await p1_actor.open_channel("data")

    t1_ep = await t1_actor.endpoint()
    t2_ep = await t2_actor.endpoint()
    p1_ep = await p1_actor.endpoint()

    print(f"  Thread-1 (producer): {t1_ep.scheme}://{t1_ep.actor_id[:8]}...")
    print(f"  Thread-2 (consumer): {t2_ep.scheme}://{t2_ep.actor_id[:8]}...")
    print(f"  Process-1 (consumer): {p1_ep.scheme}://{p1_ep.actor_id[:8]}...")

    # Thread -> Thread communication
    print("\n[5.2] Thread -> Thread communication...")
    msg1 = await t1_actor.produce()
    await t1_channel.send(t2_ep, msg1)
    received1 = await t2_channel.recv(timeout=5.0)
    if received1 is None:
        print("  Timeout: Thread-2 did not receive message in time")
    else:
        ack1 = await t2_actor.consume(received1)
        print(f"  {ack1}")

    # Thread -> Process communication
    print("\n[5.3] Thread -> Process communication...")
    msg2 = await t1_actor.produce()
    await t1_channel.send(p1_ep, msg2)
    received2 = await p1_channel.recv(timeout=5.0)
    if received2 is None:
        print("  Timeout: Process-1 did not receive message in time")
    else:
        ack2 = await p1_actor.consume(received2)
        print(f"  {ack2}")

    # Summary
    print("\n[5.4] Summary:")
    t1_produced = await t1_actor.get_produced_count()
    t2_consumed = await t2_actor.get_consumed_count()
    p1_consumed = await p1_actor.get_consumed_count()
    print(f"  Thread-1 produced: {t1_produced} messages")
    print(f"  Thread-2 consumed: {t2_consumed} messages")
    print(f"  Process-1 consumed: {p1_consumed} messages")

    # Cleanup
    await t1_actor._backend.close()
    await t2_actor._backend.close()
    await p1_actor._backend.close()

    print("\n✓ Demo 5 complete: Cross-backend channels work seamlessly!")


# =============================================================================
# Main Entry Point
# =============================================================================


async def main(args):
    """Run selected demos."""
    print("=" * 70)
    print("ByzPy Actor Runtime Demo")
    print("=" * 70)
    print(
        """
This demo showcases ByzPy's actor runtime:

1. ANY Python class can become an actor
2. Actors run on different backends: thread, process, remote (TCP)
3. Same API regardless of backend
4. Cross-backend communication via channels
"""
    )

    if args.demo == "all" or args.demo == "basic":
        await demo_basic_actors()

    if args.demo == "all" or args.demo == "channel":
        await demo_cross_backend_communication()

    if args.demo == "all" or args.demo == "pipeline":
        await demo_multi_backend_pipeline()

    if args.demo == "all" or args.demo == "channel-detail":
        await demo_channel_cross_backend()

    if args.with_remote or args.demo == "remote":
        await demo_with_remote(args.remote_host, args.remote_port)

    print("\n" + "=" * 70)
    print("All demos complete!")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ByzPy Actor Runtime Demo")

    parser.add_argument(
        "--demo",
        type=str,
        choices=["all", "basic", "channel", "pipeline", "channel-detail", "remote"],
        default="all",
        help="Which demo to run (default: all)",
    )
    parser.add_argument(
        "--with-remote",
        action="store_true",
        help="Include remote actor demo (requires remote_server.py running)",
    )
    parser.add_argument(
        "--remote-host",
        type=str,
        default="localhost",
        help="Remote server host (default: localhost)",
    )
    parser.add_argument(
        "--remote-port",
        type=int,
        default=29000,
        help="Remote server port (default: 29000)",
    )

    args = parser.parse_args()
    asyncio.run(main(args))
