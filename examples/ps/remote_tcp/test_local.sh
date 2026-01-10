#!/bin/bash
# Local test script for PS Remote TCP example
# Runs server and workers in separate terminals on the same machine

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$SCRIPT_DIR/nodes_example.yaml"

# Track spawned PIDs for proper cleanup
PIDS=()
EXIT_CODE=0

echo "=============================================="
echo "Parameter Server Remote TCP - Local Test"
echo "=============================================="
echo ""
echo "This script will start the PS server and workers in separate processes."
echo "Config file: $CONFIG"
echo ""

# Cleanup function that kills only spawned processes
cleanup() {
    local sig="${1:-TERM}"
    echo "Cleaning up spawned processes..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "-$sig" "$pid" 2>/dev/null
        fi
    done
    # Wait for processes to terminate
    for pid in "${PIDS[@]}"; do
        wait "$pid" 2>/dev/null
    done
    exit "$EXIT_CODE"
}

# Handle signals
handle_signal() {
    EXIT_CODE=130  # Standard exit code for SIGINT
    cleanup TERM
}
trap handle_signal SIGINT SIGTERM

# Start server in background
echo "Starting Parameter Server..."
python3.11 "$SCRIPT_DIR/ps_node.py" --config "$CONFIG" --role server &
SERVER_PID=$!
PIDS+=("$SERVER_PID")
sleep 2

# Start honest workers
echo "Starting Worker 0 (honest)..."
python3.11 "$SCRIPT_DIR/ps_node.py" --config "$CONFIG" --role worker --worker-id 0 &
WORKER0_PID=$!
PIDS+=("$WORKER0_PID")
sleep 0.5

echo "Starting Worker 1 (honest)..."
python3.11 "$SCRIPT_DIR/ps_node.py" --config "$CONFIG" --role worker --worker-id 1 &
WORKER1_PID=$!
PIDS+=("$WORKER1_PID")
sleep 0.5

echo "Starting Worker 2 (honest)..."
python3.11 "$SCRIPT_DIR/ps_node.py" --config "$CONFIG" --role worker --worker-id 2 &
WORKER2_PID=$!
PIDS+=("$WORKER2_PID")
sleep 0.5

# Start byzantine worker
echo "Starting Worker 3 (byzantine)..."
python3.11 "$SCRIPT_DIR/ps_node.py" --config "$CONFIG" --role worker --worker-id 3 --worker-type byzantine &
WORKER3_PID=$!
PIDS+=("$WORKER3_PID")

echo ""
echo "All processes started. PIDs:"
echo "  Server: $SERVER_PID"
echo "  Worker 0: $WORKER0_PID"
echo "  Worker 1: $WORKER1_PID"
echo "  Worker 2: $WORKER2_PID"
echo "  Worker 3: $WORKER3_PID"
echo ""
echo "Press Ctrl+C to stop all processes."
echo ""

# Wait for server to finish and capture exit code
wait $SERVER_PID
EXIT_CODE=$?

# Cleanup remaining processes
cleanup
