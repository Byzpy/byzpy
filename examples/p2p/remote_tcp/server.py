"""
Remote TCP Server for Decentralized P2P Training.

This server acts as a message router for decentralized nodes running on remote machines.
All nodes connect to this central server, which routes messages between them.

Architecture (Hub-and-Spoke):
    ┌─────────────────────────────────────────────────────────────────┐
    │                      Central Server                             │
    │                   (RemoteNodeServer)                            │
    │  - Routes messages between connected nodes                      │
    │  - All nodes connect via TCP                                    │
    └─────────────────────────────────────────────────────────────────┘
                   ▲              ▲              ▲
                   │ TCP          │ TCP          │ TCP
                   ▼              ▼              ▼
            ┌──────────┐   ┌──────────┐   ┌──────────┐
            │ Node 0   │   │ Node 1   │   │ Node 2   │
            │(Machine1)│   │(Machine2)│   │(Machine3)│
            └──────────┘   └──────────┘   └──────────┘

Usage:
    # On the central server machine:
    python server.py --host 0.0.0.0 --port 8888

    # On each remote machine, run client.py pointing to this server
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

SCRIPT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from byzpy.engine.node.remote_server import RemoteNodeServer


async def main(host: str, port: int):
    """
    Start the remote node server.

    This server accepts TCP connections from remote nodes and routes messages between them.
    """
    print("=" * 70)
    print("Remote TCP Server for Decentralized P2P Training")
    print("=" * 70)
    print(f"Listening on: {host}:{port}")
    print(f"Waiting for remote nodes to connect...")
    print("=" * 70)

    server = RemoteNodeServer(host=host, port=port)

    try:
        await server.serve()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        await server.shutdown()
    except asyncio.CancelledError:
        await server.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remote TCP Server for P2P Training")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0 for all interfaces)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8888,
        help="Port to listen on (default: 8888)",
    )
    args = parser.parse_args()

    asyncio.run(main(args.host, args.port))
