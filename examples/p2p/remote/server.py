import asyncio
import os
import sys

SCRIPT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from byzpy.engine.actor.backends.remote import start_actor_server

if __name__ == "__main__":
    # Expose on all interfaces so other hosts can connect if needed
    asyncio.run(start_actor_server(host="0.0.0.0", port=29001))
