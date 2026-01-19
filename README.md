# ByzPy

> **Byzantine-robust distributed learning** with a unified actor runtime, computation-graph scheduler, and batteries-included aggregators/attacks.

<p align="left">
  <a href="https://pypi.org/project/byzpy/"><img src="https://img.shields.io/pypi/v/byzpy.svg?logo=pypi&label=PyPI" alt="PyPI"></a>
  <a href="https://codecov.io/gh/Byzpy/byzpy"><img src="https://codecov.io/gh/Byzpy/byzpy/branch/main/graph/badge.svg" alt="Codecov"></a>
  <a href="https://byzpy.github.io/byzpy/"><img src="https://img.shields.io/badge/docs-sphinx-blue?logo=readthedocs" alt="Docs"></a>
  <a href="https://github.com/Byzpy/byzpy/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/byzpy.svg" alt="License"></a>
</p>

**[Documentation](https://byzpy.github.io/byzpy/)**

## Installation

ByzPy is easy to install via pip:

```bash
pip install byzpy
```

### Installation for Developers

When you want to contribute code to ByzPy, follow the instructions below to install for development:

```bash
git clone https://github.com/Byzpy/byzpy.git
cd byzpy
pip install -e "python[dev]"
```

For GPU/CUDA support:

```bash
pip install -e "python[gpu]"
```

More details about installing ByzPy can be found at the [installation](https://byzpy.github.io/byzpy/installation.html) section in the documentation.

## Architecture Overview

ByzPy is organized into three cooperating tiers:

```
+-------------------------------------------------------------+
|                    Application Layer                        |
|  Aggregators | Attacks | Pre-Aggregators | PS/P2P Helpers   |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                    Scheduling Layer                         |
|  ComputationGraph | NodeScheduler | ActorPool | Operators   |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                       Actor Layer                           |
|  Thread | Process | GPU | Remote (TCP/UCX) | Channels       |
+-------------------------------------------------------------+
```

1. **Application Layer** - User-facing APIs: aggregators, attacks, pre-aggregators, parameter-server helpers, and peer-to-peer training.
2. **Scheduling Layer** - Computation-graph primitives (`GraphInput`, `GraphNode`, `Operator`, `ComputationGraph`) plus `NodeScheduler` and `ActorPool` that orchestrate execution.
3. **Actor Layer** - Lightweight worker backends (threads, processes, GPUs, remote TCP/UCX actors) that run tasks.

## Getting Started

Starting with a simple aggregation:

```python
from byzpy.aggregators.coordinate_wise import CoordinateWiseMedian
import torch

aggregator = CoordinateWiseMedian()
gradients = [torch.randn(1000) for _ in range(10)]
result = aggregator.aggregate(gradients)
print(result.shape)  # torch.Size([1000])
```

## Byzantine-Robust Aggregation

ByzPy provides a variety of Byzantine-robust aggregators.

**Byzantine-Robust Median:**

```python
from byzpy.aggregators.coordinate_wise import CoordinateWiseMedian
import torch

aggregator = CoordinateWiseMedian()
gradients = [torch.randn(1000) for _ in range(10)]
# Robust to up to 50% Byzantine nodes
result = aggregator.aggregate(gradients)
```

Available aggregators include:

- **Coordinate-wise**: Median, Trimmed Mean, Mean of Medians
- **Geometric**: Krum, Multi-Krum, Geometric Median, MDA, MoNNA, SMEA
- **Norm-wise**: Center Clipping, CGE, CAF

## Attack Simulation

ByzPy includes attack simulators for testing aggregator robustness:

```python
from byzpy.attacks import SignFlipAttack
import torch

attack = SignFlipAttack(scale=-1.0)
base_grad = torch.randn(1000)
malicious_grad = attack.apply(base_grad=base_grad)
# malicious_grad = -1.0 * base_grad
```

Available attacks: Empire, Sign Flip, Label Flip, Little, Gaussian, Inf, Mimic.

## Pre-Aggregators

Pre-aggregators transform gradients before aggregation to improve robustness:

```python
from byzpy.pre_aggregators import Clipping, Bucketing
import torch

gradients = [torch.randn(1000) for _ in range(10)]

# Clip gradients to a maximum norm
clipping = Clipping(threshold=2.0)
clipped = clipping.pre_aggregate(gradients)

# Group gradients into buckets and average within each
bucketing = Bucketing(bucket_size=3)
bucketed = bucketing.pre_aggregate(gradients)
```

Available pre-aggregators: Clipping, Bucketing, NearestNeighborMixing (NNM), ARC.

## Parallel Compute via Actor Pool

ByzPy operators support parallel execution through an `ActorPool`. The pool manages worker threads/processes and distributes subtasks automatically:

```python
from byzpy.engine.graph.pool import ActorPool, ActorPoolConfig
from byzpy.engine.graph.scheduler import NodeScheduler
from byzpy.engine.graph.ops import make_single_operator_graph
from byzpy.aggregators.coordinate_wise import CoordinateWiseMedian
import torch

# Create a pool of 4 thread workers
pool = ActorPool([ActorPoolConfig(backend="thread", count=4)])
await pool.start()

# Build a computation graph with the aggregator
aggregator = CoordinateWiseMedian()
graph = make_single_operator_graph(
    node_name="agg",
    operator=aggregator,
    input_keys=("gradients",),
)

# Create scheduler with pool for parallel execution
scheduler = NodeScheduler(graph, pool=pool)

# Execute - subtasks are distributed across workers
gradients = [torch.randn(100000) for _ in range(10)]
results = await scheduler.run({"gradients": gradients})
aggregated = results["agg"]

await pool.shutdown()
```

Heterogeneous pools with mixed backends are also supported:

```python
pool = ActorPool([
    ActorPoolConfig(backend="thread", count=4),   # CPU workers
    ActorPoolConfig(backend="gpu", count=2),      # GPU workers
])
```

## Actor Backends

ByzPy supports multiple execution backends through a unified API:

| Backend | Description |
|---------|-------------|
| `thread` | Dedicated thread per worker. Low-latency CPU tasks. |
| `process` | Separate Python process. True parallelism, no GIL. |
| `gpu` | CUDA-aware workers for GPU-accelerated computation. |
| `tcp` | TCP-based remote actors for distributed training. |
| `ucx` | UCX/InfiniBand for high-speed GPU clusters. |

**Wrapping any Python class as an actor:**

Any Python class can become an actor without special inheritance:

```python
from byzpy.engine.actor.backends.thread import ThreadActorBackend
from byzpy.engine.actor.backends.process import ProcessActorBackend
from byzpy.engine.actor.base import ActorRef

# Define any Python class
class Calculator:
    def __init__(self, name: str):
        self.name = name
        self.history = []

    def add(self, a: float, b: float) -> float:
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result

    def get_history(self):
        return self.history

# Wrap as actor on thread backend
async with ActorRef(ThreadActorBackend()) as actor:
    await actor._backend.construct(Calculator, args=(), kwargs={"name": "calc"})

    # Call methods remotely (same API regardless of backend)
    result = await actor.add(10, 20)       # Returns 30
    history = await actor.get_history()    # Returns ["10 + 20 = 30"]

# Or use process backend for true parallelism
async with ActorRef(ProcessActorBackend()) as actor:
    await actor._backend.construct(Calculator, args=(), kwargs={"name": "calc"})
    result = await actor.add(5, 6)         # Runs in separate process
```

**Switching backends:**

```python
from byzpy.configs.actor import set_actor

# Use threads (default, low latency)
backend = set_actor("thread")

# Use processes (true parallelism)
backend = set_actor("process")

# Use GPU workers
backend = set_actor("gpu")
```

## CLI Utilities

ByzPy provides a command-line interface for common tasks:

```bash
byzpy version                # Show installed version
byzpy doctor --format json   # Environment and CUDA/UCX diagnostics
byzpy list aggregators       # Discover built-in aggregators
byzpy list attacks           # Discover built-in attacks
```

## Running Examples

```bash
# Parameter server with threaded workers
python examples/ps/thread/mnist.py

# Peer-to-peer training
python examples/p2p/thread/mnist.py

# Remote TCP demo (start server first)
python examples/p2p/remote_tcp/server.py &
python examples/p2p/remote_tcp/mesh_client.py
```

## Repository Structure

| Path | Description |
|------|-------------|
| `python/byzpy` | Core library (actors, graphs, aggregators, attacks). |
| `examples/` | Parameter-server and peer-to-peer demos. |
| `benchmarks/` | ActorPool and aggregator benchmarks. |
| `docs/` | Sphinx documentation source. |

## Quality and Testing

Run the test suite locally:

```bash
cd python
pytest --cov=byzpy --cov-report=term-missing
```

Coverage is uploaded to [Codecov](https://codecov.io/gh/Byzpy/byzpy) on every push.

## Building Documentation

Build the docs locally:

```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs/source docs/_build/html
```

Full documentation is available at **[byzpy.github.io/byzpy/](https://byzpy.github.io/byzpy/)**

## Getting Involved

We welcome contributions!

1. Fork + branch.
2. Implement your change (tests + docs).
3. Run `pytest` and `sphinx-build`.
4. Open a PR with a concise summary + test output.

- Read the [development guide](https://byzpy.github.io/byzpy/developer_guide.html).
- Report bugs by submitting a [GitHub issue](https://github.com/Byzpy/byzpy/issues).
- Submit contributions using [pull requests](https://github.com/Byzpy/byzpy/pulls).
- See [CONTRIBUTING.md](CONTRIBUTING.md) for the full contribution guidelines.
- Review [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for community standards.
- Check [SECURITY.md](SECURITY.md) for security policies.
- Releases are tracked in [CHANGELOG.md](CHANGELOG.md).

Thank you in advance for your contributions!
