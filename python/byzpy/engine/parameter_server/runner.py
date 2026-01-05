"""
Prototype ParameterServer runner using NodeRunner processes.

Spawns a server runner and worker runners; orchestrates a simple round where
workers compute gradients and send them to the server, which aggregates them.
This is a scaffold toward decentralized scheduling; it does not replace the
full ParameterServer yet.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence

import torch

from ..node_cluster import NodeCluster
from ..node_runner import NodeRunner


def _make_worker_step(grad_fn: Callable[[], torch.Tensor]):
    def step(state: dict) -> dict:
        state["grad"] = grad_fn()
        return state

    def on_msg(state: dict, msg: Any) -> dict:
        # no-op for now
        return state

    return step, on_msg


def _make_server_step(agg: Callable[[Sequence[torch.Tensor]], torch.Tensor]):
    def step(state: dict) -> dict:
        inbox = state.get("in_msgs", [])
        if inbox:
            state["out"] = agg(inbox)
            state["in_msgs"] = []
        return state

    def on_msg(state: dict, msg: Any) -> dict:
        msgs = state.get("in_msgs", [])
        msgs.append(msg)
        state["in_msgs"] = msgs
        return state

    return step, on_msg


class ParameterServerRunner:
    """Minimal PS runner with per-node processes."""

    def __init__(
        self,
        worker_grad_fns: List[Callable[[], torch.Tensor]],
        aggregator: Callable[[Sequence[torch.Tensor]], torch.Tensor] | None = None,
        *,
        transport=None,
    ) -> None:
        self.cluster = NodeCluster(transport=transport)
        self.worker_ids: List[str] = []
        self.server_id = "server"
        agg = aggregator or (lambda grads: sum(grads) / len(grads))
        s_step, s_on_msg = _make_server_step(agg)
        self.cluster.add_node(self.server_id, s_step, s_on_msg, init_state={})
        for idx, fn in enumerate(worker_grad_fns):
            wid = f"w{idx}"
            w_step, w_on_msg = _make_worker_step(fn)
            self.cluster.add_node(wid, w_step, w_on_msg, init_state={})
            self.worker_ids.append(wid)

    def start(self) -> None:
        self.cluster.start_all()

    def stop(self) -> None:
        self.cluster.stop_all()

    def run_round(self) -> torch.Tensor:
        # Workers compute grads
        for wid in self.worker_ids:
            self.cluster._nodes[wid].step()
        # Send to server
        for wid in self.worker_ids:
            grad = self.cluster.state(wid).get("grad")
            self.cluster.send(self.server_id, grad)
        # Allow messages to land
        self.cluster.barrier(0.01)
        # Aggregate
        self.cluster._nodes[self.server_id].step()
        out = self.cluster.state(self.server_id).get("out")
        return out
