from __future__ import annotations

import asyncio
import time

import pytest
import torch

from byzpy.aggregators.geometric_wise import MultiKrum
from byzpy.engine.graph.ops import make_single_operator_graph
from byzpy.engine.graph.pool import ActorPool, ActorPoolConfig
from byzpy.engine.graph.scheduler import NodeScheduler
from byzpy.engine.node_cluster import NodeCluster
from byzpy.engine.node_runner import NodeRunner
from byzpy.engine.transport.local import LocalTransport


def _step_counter(state: dict) -> dict:
    count = state.get("count", 0) + 1
    state["count"] = count
    return state


def _msg_accum(state: dict, msg) -> dict:
    msgs = state.get("msgs", [])
    msgs.append(msg)
    state["msgs"] = msgs
    return state


def test_node_runner_auto_progress_independent():
    r1 = NodeRunner(_step_counter, _msg_accum, init_state={"id": 1})
    r2 = NodeRunner(_step_counter, _msg_accum, init_state={"id": 2})
    r1.start()
    r2.start()
    try:
        r1.start_auto(0.01)
        r2.start_auto(0.05)
        time.sleep(0.2)
        c1 = r1.state()["count"]
        c2 = r2.state()["count"]
        assert c1 > 0 and c2 > 0
        # r1 should have stepped more often due to faster interval
        assert c1 > c2
    finally:
        r1.stop()
        r2.stop()


def test_node_runner_message_delivery():
    r = NodeRunner(_step_counter, _msg_accum, init_state={})
    r.start()
    try:
        r.send_message("hello")
        r.send_message("world")
        r.step()  # process messages
        msgs = r.state().get("msgs", [])
        assert msgs == ["hello", "world"]
    finally:
        r.stop()


def test_cluster_two_nodes_progress_and_messages():
    cluster = NodeCluster()
    cluster.add_node("a", _step_counter, _msg_accum, init_state={})
    cluster.add_node("b", _step_counter, _msg_accum, init_state={})
    cluster.start_all()
    try:
        cluster.start_auto("a", 0.01)
        cluster.start_auto("b", 0.05)
        cluster.send("b", {"from": "a"})
        cluster.barrier(0.2)
        state_a = cluster.state("a")
        state_b = cluster.state("b")
        assert state_a["count"] > state_b["count"]
        assert {"from": "a"} in state_b.get("msgs", [])
    finally:
        cluster.stop_all()


def test_cluster_with_local_transport():
    transport = LocalTransport()
    cluster = NodeCluster(transport=transport)
    cluster.add_node("a", _step_counter, _msg_accum, init_state={})
    cluster.add_node("b", _step_counter, _msg_accum, init_state={})
    cluster.start_all()
    try:
        cluster.send("b", "ping")
        cluster._nodes["b"].step()
        msgs = cluster.state("b").get("msgs", [])
        assert msgs == ["ping"]
    finally:
        cluster.stop_all()


@pytest.mark.skip(reason="Flaky test - queue.Empty timeout issue with actor pool in NodeRunner")
def test_runner_can_host_scheduler_with_actor_pool():
    def step(state: dict) -> dict:
        if state.get("done"):
            return state
        vecs = [
            torch.tensor([1.0, 0.0]),
            torch.tensor([0.9, 0.1]),
            torch.tensor([-1.0, 0.0]),
        ]
        agg = MultiKrum(f=0, q=2, chunk_size=2)
        graph = make_single_operator_graph(node_name="agg", operator=agg, input_keys=("gradients",))
        pool = ActorPool([ActorPoolConfig(backend="thread", count=2)])

        async def _run():
            await pool.start()
            try:
                sched = NodeScheduler(graph, pool=pool)
                res = await sched.run({"gradients": vecs})
            finally:
                await pool.shutdown()
            return res["agg"]

        out = asyncio.run(_run())
        state["out"] = out
        state["done"] = True
        return state

    r = NodeRunner(step, _msg_accum, init_state={})
    r.start()
    try:
        r.step()
        out = r.state()["out"]
        assert torch.allclose(out, torch.tensor([0.95, 0.05]))
    finally:
        r.stop()
