# Remote TCP Parameter Server Training

This directory contains an example for **Parameter Server (PS) distributed training** where each node runs on a **separate machine** and communicates via **TCP**.

## Architecture

```text
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
```

## Training Protocol

1. **Initialization**: Workers connect to the PS and receive the initial model state.
2. **Gradient Computation**: PS signals workers to compute gradients on their local data shards.
3. **Aggregation**: PS collects gradients from all workers and applies robust aggregation (e.g., Coordinate-Wise Median, Multi-Krum) to defend against Byzantine attacks.
4. **Update**: PS broadcasts the aggregated gradient back to all workers, which update their local models.
5. **Repeat**: Steps 2-4 repeat for the specified number of rounds.

---

## Quick Start

### Local Testing (Single Machine)

```bash
# Run automated test script
chmod +x test_local.sh
./test_local.sh
```

Or manually in separate terminals:

```bash
# Terminal 1: Start Parameter Server
python ps_node.py --config nodes_example.yaml --role server

# Terminal 2: Worker 0 (honest)
python ps_node.py --config nodes_example.yaml --role worker --worker-id 0

# Terminal 3: Worker 1 (honest)
python ps_node.py --config nodes_example.yaml --role worker --worker-id 1

# Terminal 4: Worker 2 (honest)
python ps_node.py --config nodes_example.yaml --role worker --worker-id 2

# Terminal 5: Worker 3 (byzantine)
python ps_node.py --config nodes_example.yaml --role worker --worker-id 3 --worker-type byzantine
```

### Remote Deployment (Multiple Machines)

1. **Create configuration file** with actual IP addresses:

```yaml
# nodes.yaml
server:
  host: 192.168.1.100    # PS machine's IP
  port: 9000
  bind_host: "0.0.0.0"

workers:
  - id: 0
    host: 192.168.1.101  # Worker 0 machine's IP
    port: 9000
    type: honest
    data_shard: 0

  - id: 1
    host: 192.168.1.102  # Worker 1 machine's IP
    port: 9000
    type: honest
    data_shard: 1

  - id: 2
    host: 192.168.1.103  # Worker 2 machine's IP (byzantine)
    port: 9000
    type: byzantine
    attack_scale: -1.0
```

2. **Copy files** to all machines (config file + ps_node.py + ByzPy installation)

3. **Run on each machine**:

```bash
# On PS machine (192.168.1.100):
python ps_node.py --config nodes.yaml --role server

# On Worker 0 machine (192.168.1.101):
python ps_node.py --config nodes.yaml --role worker --worker-id 0

# On Worker 1 machine (192.168.1.102):
python ps_node.py --config nodes.yaml --role worker --worker-id 1

# On Worker 2 machine (192.168.1.103):
python ps_node.py --config nodes.yaml --role worker --worker-id 2 --worker-type byzantine
```

---

## Configuration File Format

### YAML Format

```yaml
# Training parameters
training:
  rounds: 50
  batch_size: 64
  learning_rate: 0.05
  momentum: 0.9

# Aggregation settings
aggregation:
  method: "coordinate_wise_median"  # or "multi_krum"
  # For multi_krum:
  # krum_f: 1   # Byzantine tolerance
  # krum_q: 3   # Gradients to select

# Parameter Server
server:
  host: 192.168.1.100    # IP for workers to connect
  port: 9000
  bind_host: "0.0.0.0"   # Bind address

# Workers
workers:
  - id: 0
    host: 192.168.1.101
    port: 9000
    type: honest
    data_shard: 0

  - id: 1
    host: 192.168.1.102
    port: 9000
    type: byzantine
    attack_scale: -1.0
```

---

## Command Line Arguments

### ps_node.py

| Argument | Description |
|----------|-------------|
| `--config` | Path to configuration file (YAML or JSON) |
| `--role` | Node role: `server` or `worker` |
| `--worker-id` | Worker ID (required for `--role=worker`) |
| `--worker-type` | Override worker type: `honest` or `byzantine` |
| `--rounds` | Training rounds (overrides config) |
| `--data-root` | MNIST data directory (default: `./data`) |

---

## Aggregation Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| `coordinate_wise_median` | Takes median of each coordinate | Good general defense |
| `multi_krum` | Selects gradients closest to others | Strong Byzantine tolerance |

---

## Network Requirements

- **Firewall**: Open TCP port 9000 (or configured port) on all machines
- **Connectivity**: All workers must be able to reach the PS
- **NAT/Cloud**: Use public IPs or VPN (Tailscale, WireGuard)

---

## Comparison with P2P Mesh

| Feature | PS Architecture | P2P Mesh |
|---------|----------------|----------|
| Central coordinator | Yes (PS) | No |
| Message complexity | O(N) per round | O(N²) per round |
| Fault tolerance | Single point of failure | Fully distributed |
| Synchronization | Synchronous (all workers) | Asynchronous |
| Scalability | Good for large N | Better for small N |

### When to Use PS

- Large number of workers
- Need strict synchronization
- Simpler network setup (workers only connect to PS)
- Easier to debug and monitor

### When to Use P2P Mesh

- Need fault tolerance (no single point of failure)
- Fully decentralized requirement
- Lower latency (direct communication)

---

## Files

| File | Description |
|------|-------------|
| `ps_node.py` | Main script (runs as server or worker) |
| `nodes_example.yaml` | Example configuration (local testing) |
| `test_local.sh` | Automated local test script |
| `README.md` | This documentation |

---

## Troubleshooting

### Connection refused
- Ensure the PS server is running before starting workers
- Check firewall allows TCP on the configured port
- Verify the server IP/port in config matches the actual server

### Workers not receiving updates
- Check all workers are connected (server logs)
- Ensure network connectivity between workers and server
- Increase timeout if workers are slow

### YAML config not loading
```bash
pip install pyyaml
```
