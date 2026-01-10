"""
Remote Actor Server for ByzPy Actor Demo.

This server hosts remote actors that can be accessed via TCP.
Any Python class can be spawned on this server and called remotely.

Usage:
    python remote_server.py [--host HOST] [--port PORT]

Example:
    # Terminal 1: Start server (default binds to localhost for security)
    python remote_server.py --port 29000

    # To allow remote connections (use with caution):
    python remote_server.py --host 0.0.0.0 --port 29000

    # Terminal 2: Run demo with remote actors
    python actor_demo.py --with-remote --remote-host localhost --remote-port 29000
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

SCRIPT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from byzpy.engine.actor.backends.remote import start_actor_server


async def main(host: str, port: int):
    """Start the remote actor server."""
    print("=" * 70)
    print("ByzPy Remote Actor Server")
    print("=" * 70)
    print(f"  Host: {host}")
    print(f"  Port: {port}")
    print("=" * 70)
    print("\nServer is ready to host remote actors.")
    print("Any Python class can be spawned here and called via TCP.")
    print("\nPress Ctrl+C to stop.\n")

    await start_actor_server(host=host, port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ByzPy Remote Actor Server")
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind (default: 127.0.0.1; use 0.0.0.0 for remote access)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=29000,
        help="Port to listen on (default: 29000)",
    )

    args = parser.parse_args()

    try:
        asyncio.run(main(args.host, args.port))
    except KeyboardInterrupt:
        print("\nServer stopped.")
