# Remote TCP P2P Training Examples

This directory contains examples for decentralized P2P training where nodes run on **separate servers** and communicate via **TCP**.

Two architectures are available:

| Architecture | Files | Use Case |
|-------------|-------|----------|
| **Hub-and-Spoke** | `server.py`, `client.py` | Simple setup, all nodes connect to central server |
| **Peer-to-Peer Mesh** | `mesh_client.py` | Fully distributed, no central server required |

---

## Architecture 1: Hub-and-Spoke (Simple)

All nodes connect to a central server that routes messages between them.

```
                    ┌─────────────────────────────────────────┐
                    │         Central Server                  │
                    │       (RemoteNodeServer)                │
                    │   - Routes messages between nodes       │
                    └─────────────────────────────────────────┘
                               ▲       ▲       ▲
                               │       │       │
                    TCP ───────┘       │       └─────── TCP
                                       │
                                  TCP ─┘
                               ▼       ▼       ▼
                ┌──────────┐ ┌──────────┐ ┌──────────┐
                │  Node 0  │ │  Node 1  │ │  Node 2  │
                │(Machine1)│ │(Machine2)│ │(Machine3)│
                └──────────┘ └──────────┘ └──────────┘
```

### Quick Start (Hub-and-Spoke)

```bash
# Terminal 1: Start central server
python server.py --host 0.0.0.0 --port 8888

# Terminal 2: Node 0 (honest)
python client.py --server-host localhost --server-port 8888 \
    --node-id 0 --node-type honest --total-nodes 3 --honest-nodes 2 --data-shard 0

# Terminal 3: Node 1 (honest)
python client.py --server-host localhost --server-port 8888 \
    --node-id 1 --node-type honest --total-nodes 3 --honest-nodes 2 --data-shard 1

# Terminal 4: Node 2 (byzantine)
python client.py --server-host localhost --server-port 8888 \
    --node-id 2 --node-type byzantine --total-nodes 3 --honest-nodes 2
```

---

## Architecture 2: Peer-to-Peer Mesh (Fully Distributed)

Each node runs its own TCP server and connects directly to all other nodes.
**No central server required** - fully decentralized.

```
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
┌───────────────┐                         ┌───────────────┐
│   Node 2      │◄───────────────────────►│   Node 3      │
│ (Machine C)   │           TCP           │ (Machine D)   │
└───────────────┘                         └───────────────┘
```

### Configuration File

Create a config file (`nodes.yaml` or `nodes.json`) listing all nodes:

```yaml
# nodes.yaml
nodes:
  - id: 0
    host: 192.168.1.100  # IP address for other nodes to connect
    port: 8888
    type: honest

  - id: 1
    host: 192.168.1.101
    port: 8888
    type: honest

  - id: 2
    host: 192.168.1.102
    port: 8888
    type: honest

  - id: 3
    host: 192.168.1.103
    port: 8888
    type: byzantine
```

### Quick Start (Mesh)

```bash
# On Machine A (192.168.1.100):
python mesh_client.py --config nodes.yaml --node-id 0 --node-type honest

# On Machine B (192.168.1.101):
python mesh_client.py --config nodes.yaml --node-id 1 --node-type honest

# On Machine C (192.168.1.102):
python mesh_client.py --config nodes.yaml --node-id 2 --node-type honest

# On Machine D (192.168.1.103):
python mesh_client.py --config nodes.yaml --node-id 3 --node-type byzantine
```

### Local Testing (Mesh)

For testing on a single machine, use different ports:

```bash
# Use the example config with localhost and different ports
# Terminal 1: Node 0
python mesh_client.py --config nodes_example.yaml --node-id 0 --node-type honest

# Terminal 2: Node 1
python mesh_client.py --config nodes_example.yaml --node-id 1 --node-type honest

# Terminal 3: Node 2
python mesh_client.py --config nodes_example.yaml --node-id 2 --node-type honest

# Terminal 4: Node 3
python mesh_client.py --config nodes_example.yaml --node-id 3 --node-type honest

# Terminal 5: Node 4 (byzantine)
python mesh_client.py --config nodes_example.yaml --node-id 4 --node-type byzantine
```

---

## Command Line Arguments

### server.py (Hub-and-Spoke only)

| Argument | Default | Description |
|----------|---------|-------------|
| `--host` | `0.0.0.0` | Host to bind to |
| `--port` | `8888` | Port to listen on |

### client.py (Hub-and-Spoke)

| Argument | Default | Description |
|----------|---------|-------------|
| `--server-host` | (required) | Server hostname or IP |
| `--server-port` | `8888` | Server port |
| `--node-id` | (required) | Unique node ID (0, 1, 2, ...) |
| `--node-type` | `honest` | `honest` or `byzantine` |
| `--total-nodes` | (required) | Total nodes in network |
| `--honest-nodes` | (required) | Number of honest nodes |
| `--data-shard` | `0` | Data shard index |
| `--rounds` | `50` | Training rounds |
| `--batch-size` | `64` | Batch size |
| `--lr` | `0.05` | Learning rate |
| `--byz-scale` | `-1.0` | Byzantine attack scale |

### mesh_client.py (Peer-to-Peer Mesh)

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | (required) | Path to nodes config file (YAML/JSON) |
| `--node-id` | (required) | This node's ID (must match config) |
| `--node-type` | `honest` | `honest` or `byzantine` |
| `--rounds` | `50` | Training rounds |
| `--batch-size` | `64` | Batch size |
| `--lr` | `0.05` | Learning rate |
| `--data-root` | `./data` | MNIST data directory |
| `--byz-scale` | `-1.0` | Byzantine attack scale |

---

## Config File Format

### YAML Format

```yaml
training:
  rounds: 50
  batch_size: 64
  learning_rate: 0.05

nodes:
  - id: 0
    host: 192.168.1.100    # External IP/hostname
    port: 8888
    type: honest
    bind_host: "0.0.0.0"   # Optional: local bind address

  - id: 1
    host: 192.168.1.101
    port: 8888
    type: honest

  - id: 2
    host: 192.168.1.102
    port: 8888
    type: byzantine
```

### JSON Format

```json
{
  "nodes": [
    {"id": 0, "host": "192.168.1.100", "port": 8888, "type": "honest"},
    {"id": 1, "host": "192.168.1.101", "port": 8888, "type": "honest"},
    {"id": 2, "host": "192.168.1.102", "port": 8888, "type": "byzantine"}
  ]
}
```

---

## Architecture Comparison

| Feature | Hub-and-Spoke | Peer-to-Peer Mesh |
|---------|--------------|-------------------|
| Central server | Required | Not needed |
| Single point of failure | Yes | No |
| Network complexity | O(N) connections | O(N²) connections |
| Message latency | 2 hops | 1 hop |
| NAT traversal | Easy | More complex |
| Setup complexity | Simple | More config |
| Scalability | Limited | Better |

### When to Use Each

**Hub-and-Spoke** (`server.py` + `client.py`):
- Quick prototyping and testing
- Small clusters (<20 nodes)
- When nodes are behind NAT
- When simplicity is preferred

**Peer-to-Peer Mesh** (`mesh_client.py`):
- Production deployments
- Large clusters (20+ nodes)
- When latency matters
- When fault tolerance is important
- When nodes have public IPs

---

## Network Requirements

### Hub-and-Spoke
- Central server needs public IP or all nodes on same network
- Nodes only need outbound TCP to server

### Peer-to-Peer Mesh
- Each node needs to accept inbound TCP connections
- All nodes need to reach all other nodes
- For NAT scenarios, consider:
  - Port forwarding
  - VPN (e.g., Tailscale, WireGuard)
  - Cloud deployment with public IPs

---

## Files

| File | Description |
|------|-------------|
| `server.py` | Central server for hub-and-spoke |
| `client.py` | Client node for hub-and-spoke |
| `mesh_client.py` | Fully distributed mesh node |
| `nodes_example.yaml` | Example YAML config for mesh |
| `nodes_example.json` | Example JSON config for mesh |
| `README.md` | This documentation |

---

## Troubleshooting

### Connection refused
- Check firewall allows TCP on the port
- Verify the host IP is correct
- Ensure the target node is running

### Timeout connecting to peers
- Increase `connect_timeout` in `MeshRemoteContext`
- Check network connectivity between machines
- Verify no NAT/firewall blocking

### Messages not being received
- Check topology configuration
- Verify node IDs match config
- Look for errors in node output

### YAML config not loading
```bash
pip install pyyaml
```
