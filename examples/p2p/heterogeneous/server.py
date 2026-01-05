import asyncio

from byzpy.engine.actor.backends.gpu import start_ucx_actor_server

if __name__ == "__main__":
    # Expose on all interfaces so other hosts can connect if needed
    asyncio.run(start_ucx_actor_server(host="0.0.0.0", port=29010))
