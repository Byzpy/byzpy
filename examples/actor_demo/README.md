# ByzPy Actor Runtime Demo

This demo showcases ByzPy's **actor runtime** - a flexible system for running Python code across different execution environments with a unified API.

## Key Features Demonstrated

### 1. Arbitrary Python Classes as Actors

Any Python class can become an actor without modification:

```python
class Calculator:
    def __init__(self, name: str):
        self.name = name

    def add(self, a: float, b: float) -> float:
        return a + b

# Spawn on thread backend
backend = ThreadActorBackend()
actor = ActorRef(backend)
await actor._backend.start()
await actor._backend.construct(Calculator, args=(), kwargs={"name": "calc"})

# Call methods (async RPC)
result = await actor.add(10, 20)  # Returns 30
```

### 2. Multiple Backend Types

Same code works on any backend:

| Backend | Description | Use Case |
|---------|-------------|----------|
| `ThreadActorBackend` | Runs in thread pool | Low overhead, shared memory |
| `ProcessActorBackend` | Runs in child process | CPU isolation, true parallelism |
| `RemoteActorBackend` | Runs on remote server via TCP | Distributed systems |

### 3. Cross-Backend Communication

Actors on different backends communicate via **channels**:

```text
Thread Actor ──── channel ────► Process Actor
     │                              │
     └──────── channel ─────────────┘
                   │
                   ▼
            Remote Actor (TCP)
```

## Quick Start

### Basic Demo (Thread + Process)

```bash
cd examples/actor_demo
python actor_demo.py
```

### With Remote Actor

```bash
# Terminal 1: Start remote server
python remote_server.py --port 29000

# Terminal 2: Run demo
python actor_demo.py --with-remote
```

### Specific Demo

```bash
# Just basic actor functionality
python actor_demo.py --demo basic

# Just channel communication
python actor_demo.py --demo channel

# Multi-backend pipeline
python actor_demo.py --demo pipeline

# Detailed channel demo
python actor_demo.py --demo channel-detail
```

## Architecture

```text
┌────────────────────────────────────────────────────────────────┐
│                       Coordinator                              │
│  - Spawns actors on different backends                         │
│  - Orchestrates communication via channels                     │
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
```

## API Overview

### Spawning an Actor

```python
from byzpy.engine.actor.base import ActorRef
from byzpy.engine.actor.backends.thread import ThreadActorBackend
from byzpy.engine.actor.backends.process import ProcessActorBackend
from byzpy.engine.actor.backends.remote import RemoteActorBackend

# Thread backend
backend = ThreadActorBackend()
actor = ActorRef(backend)
await actor._backend.start()
await actor._backend.construct(MyClass, args=(arg1,), kwargs={"key": "value"})

# Process backend (same API!)
backend = ProcessActorBackend()
# ... same pattern

# Remote backend (same API!)
backend = RemoteActorBackend("192.168.1.100", 29000)
# ... same pattern
```

### Calling Methods

```python
# All method calls are async RPC
result = await actor.my_method(arg1, arg2)
info = await actor.get_info()
```

### Channel Communication

```python
# Open channels
channel_a = await actor_a.open_channel("data")
channel_b = await actor_b.open_channel("data")

# Get endpoints
ep_a = await actor_a.endpoint()
ep_b = await actor_b.endpoint()

# Send message from A to B
await channel_a.send(ep_b, {"message": "hello"})

# Receive at B
msg = await channel_b.recv(timeout=5.0)
```

## Demo Output Example

```text
======================================================================
ByzPy Actor Runtime Demo
======================================================================

This demo showcases ByzPy's actor runtime:

1. ANY Python class can become an actor
2. Actors run on different backends: thread, process, remote (TCP)
3. Same API regardless of backend
4. Cross-backend communication via channels

======================================================================
Demo 1: Arbitrary Python Classes as Actors
======================================================================

[1.1] Spawning Calculator on THREAD backend...
[thread-calc] Calculator initialized
[1.2] Spawning DataProcessor on PROCESS backend...
[Processor-1] DataProcessor initialized

[1.3] Calling methods on actors...
  Thread actor: 10 + 20 = 30
  Thread actor: 5 * 6 = 30
  Thread actor history: ['10 + 20 = 30', '5 * 6 = 30']
  Process actor stats: {'processor_id': 1, 'count': 5, 'sum': 15.0, ...}

✓ Demo 1 complete: Same API works across thread and process backends!
```

## Files

| File | Description |
|------|-------------|
| `actor_demo.py` | Main demo script with all examples |
| `remote_server.py` | TCP server for remote actors |
| `README.md` | This documentation |

## Use Cases

### When to Use Thread Backend
- Low-latency calls
- Shared memory access needed
- I/O-bound workloads

### When to Use Process Backend
- CPU-bound workloads
- Need true parallelism (bypass GIL)
- Isolation between actors

### When to Use Remote Backend
- Distributed systems
- Run on separate machines
- Scale horizontally
