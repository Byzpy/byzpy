#!/bin/bash
# Local test script for fully distributed mesh P2P training
# Runs 3 nodes (2 honest + 1 byzantine) on localhost with different ports

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Set PYTHONPATH to use the local fork instead of installed package
export PYTHONPATH="$SCRIPT_DIR/../../../python:$PYTHONPATH"

# Create a simple 3-node config for testing
cat > /tmp/test_nodes.json << 'EOF'
{
  "nodes": [
    {"id": 0, "host": "localhost", "port": 9990, "type": "honest"},
    {"id": 1, "host": "localhost", "port": 9991, "type": "honest"},
    {"id": 2, "host": "localhost", "port": 9992, "type": "byzantine"}
  ]
}
EOF

echo "=============================================="
echo "Testing Fully Distributed Mesh P2P Training"
echo "=============================================="
echo "Config: 2 honest nodes + 1 byzantine node"
echo "Rounds: 10 (quick test)"
echo ""

# Pre-download MNIST data to avoid race conditions
echo "Pre-downloading MNIST data..."
python3.11 -c "
from torchvision import datasets, transforms
tfm = transforms.Compose([transforms.ToTensor()])
datasets.MNIST(root='./data', train=True, download=True, transform=tfm)
datasets.MNIST(root='./data', train=False, download=True, transform=tfm)
print('MNIST data ready.')
"
echo ""

# Function to cleanup background processes on exit
cleanup() {
    echo ""
    echo "Cleaning up..."
    jobs -p | xargs -r kill 2>/dev/null || true
    wait 2>/dev/null || true
    rm -f /tmp/test_nodes.json
    echo "Done."
}
trap cleanup EXIT

# Start nodes in background with more delay to let servers start
echo "Starting Node 0 (honest)..."
python3.11 mesh_client.py --config /tmp/test_nodes.json --node-id 0 --node-type honest --rounds 10 &
PID0=$!
sleep 2  # Give more time for server to start

echo "Starting Node 1 (honest)..."
python3.11 mesh_client.py --config /tmp/test_nodes.json --node-id 1 --node-type honest --rounds 10 &
PID1=$!
sleep 2

echo "Starting Node 2 (byzantine)..."
python3.11 mesh_client.py --config /tmp/test_nodes.json --node-id 2 --node-type byzantine --rounds 10 &
PID2=$!

echo ""
echo "All nodes started. Waiting for training to complete..."
echo "(This will take about 30 seconds)"
echo ""

# Wait for all processes
wait $PID0 $PID1 $PID2 2>/dev/null || true

echo ""
echo "=============================================="
echo "Test completed!"
echo "=============================================="
