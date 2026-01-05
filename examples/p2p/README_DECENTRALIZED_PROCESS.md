# Fully Decentralized Process-Based P2P Training Example

## Overview

This example (`decentralized_process_mnist.py`) demonstrates fully decentralized P2P training where each node runs in its own OS process with its own scheduler, communicating via message-driven topology.

## Key Features

- **Process-Level Isolation**: Each node runs in a separate OS process (`ProcessContext`)
- **Independent Schedulers**: Each node has its own `NodeScheduler` instance
- **Message-Driven Communication**: Nodes communicate via topology-aware message routing
- **Graph-Based Execution**: Computation graphs executed via `NodeScheduler`
- **True Parallelism**: Nodes progress independently in parallel processes

## Architecture

```
Main Process
├── Node 0 (ProcessContext) → Separate OS Process
│   ├── DecentralizedNode
│   ├── NodeScheduler
│   ├── MessageRouter
│   └── NodeApplication
├── Node 1 (ProcessContext) → Separate OS Process
│   ├── DecentralizedNode
│   ├── NodeScheduler
│   ├── MessageRouter
│   └── NodeApplication
└── ...
```

Each node:
- Runs in its own OS process
- Has its own asyncio event loop
- Communicates via multiprocessing queues
- Executes independently based on local state and messages

## Usage

```bash
# Run the example
python examples/p2p/decentralized_process_mnist.py
```

## Configuration

You can modify the configuration in the `main()` function:

```python
n_honest = 4      # Number of honest nodes
n_byz = 1         # Number of byzantine nodes
rounds = 200      # Training rounds
batch_size = 64   # Batch size per node
lr = 0.05         # Learning rate
```

## How It Works

1. **Node Creation**: Creates `HonestNodeActor` and `ByzantineNodeActor` instances using `ThreadActorBackend` (works with ProcessContext pickling)

2. **ProcessContext Factory**: Each node is wrapped in a `ProcessContext`, which:
   - Serializes the node configuration (including node objects)
   - Starts a separate OS process
   - Creates a full `DecentralizedNode` instance in the subprocess
   - Sets up message queues for communication

3. **Topology Setup**: Uses ring topology where each node communicates with k neighbors

4. **Training Rounds**: Each round:
   - Each node performs half-step (local gradient computation)
   - Nodes broadcast their vectors to neighbors
   - Each honest node aggregates received gradients
   - Nodes update their models

5. **Message Routing**: Messages are routed via `MessageRouter` based on topology constraints

## Differences from Old Implementation

**Old (`examples/p2p/process/mnist.py`)**:
- Actors run in processes (via `ProcessActorBackend`)
- Nodes are managed in the main process
- Limited graph-based execution

**New (`decentralized_process_mnist.py`)**:
- Nodes themselves run in separate processes (via `ProcessContext`)
- Each node has full `DecentralizedNode` infrastructure
- Complete graph-based execution with message-driven triggers
- True process-level isolation and parallelism

## Requirements

- Python 3.11+
- PyTorch
- torchvision (for MNIST dataset)
- All ByzPy dependencies

## Expected Output

```
======================================================================
Fully Decentralized Process-Based P2P Training with MNIST
======================================================================
Configuration:
  - Honest nodes: 4
  - Byzantine nodes: 1
  - Training rounds: 200
  - Batch size: 64
  - Learning rate: 0.05
  - Device: cpu
  - Each node runs in separate OS process (ProcessContext)
======================================================================

Creating honest node actors...
  ✓ Created honest node 0 with 15000 training samples
  ...

Starting all nodes (each in separate OS process)...
  ✓ All nodes started successfully

======================================================================
Starting P2P Training
======================================================================
[round 0050] test loss=0.1234  acc=0.9654
[round 0100] test loss=0.0987  acc=0.9721
...

Final honest-node evaluations:
  Node 0: loss=0.0856  acc=0.9756
  ...
```

## Troubleshooting

- **Pickling Errors**: Ensure actors use `ThreadActorBackend` (not `ProcessActorBackend`) when using `ProcessContext`
- **Process Startup Delays**: Initial process startup may take a few seconds - this is normal
- **Memory Usage**: Each process has its own memory space - monitor total memory usage with many nodes
