from __future__ import annotations

"""
Decentralized P2P entry point.

This wraps the refactored DecentralizedPeerToPeer runner to preserve the
PeerToPeer API while using DecentralizedNode infrastructure.
"""
from typing import Callable, List, Optional

from ..node.actors import ByzantineNodeActor, HonestNodeActor
from ..node.context import NodeContext
from .runner import DecentralizedPeerToPeer
from .topology import Topology


class PeerToPeer:
    """
    Peer-to-peer training orchestrator.

    This class provides a high-level API for peer-to-peer distributed training
    where nodes communicate directly with their neighbors according to a
    topology (ring, complete graph, etc.). It wraps the DecentralizedPeerToPeer
    runner for backwards compatibility.

    Parameters
    ----------
    honest_nodes : List[HonestNodeActor]
        List of honest node actors participating in P2P training.
    byzantine_nodes : Optional[List[ByzantineNodeActor]]
        Optional list of Byzantine node actors.
    topology : Topology
        Communication topology defining which nodes can communicate.
    lr : float, optional
        Learning rate for gradient updates. Default is 0.05.
    channel_name : str, optional
        Channel name for communication (kept for API compatibility).
        Default is "p2p".
    context_factory : Optional[Callable[[str, int], NodeContext]], optional
        Factory function to create node contexts. If None, uses default
        context creation.

    Examples
    --------
    >>> from byzpy.engine.peer_to_peer.topology import Topology
    >>> topology = Topology.ring(n=5, k=1)
    >>> p2p = PeerToPeer(
    ...     honest_nodes=honest_actors,
    ...     byzantine_nodes=byz_actors,
    ...     topology=topology,
    ...     lr=0.01
    ... )
    >>> await p2p.bootstrap()
    >>> for _ in range(100):
    ...     await p2p.round()
    >>> await p2p.shutdown()
    """

    def __init__(
        self,
        honest_nodes: List[HonestNodeActor],
        byzantine_nodes: Optional[List[ByzantineNodeActor]],
        topology: Topology,
        *,
        lr: float = 0.05,
        channel_name: str = "p2p",
        context_factory: Optional[Callable[[str, int], NodeContext]] = None,
    ):
        # channel_name kept for API compatibility; unused in new implementation
        self._runner = DecentralizedPeerToPeer(
            honest_nodes,
            byzantine_nodes or [],
            topology,
            lr=lr,
            context_factory=context_factory,
        )

    async def bootstrap(self) -> None:
        # Start per-node runners
        await self._runner.start()

    async def round(self) -> None:
        await self._runner.run_round_async()

    async def shutdown(self):
        await self._runner.stop()
