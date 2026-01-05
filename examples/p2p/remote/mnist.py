from __future__ import annotations

import asyncio
import os
import sys
from typing import List, Tuple

import torch
import torch.utils.data as data
from torchvision import datasets, transforms

SCRIPT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from examples.p2p.nodes import (
    DistributedP2PByzNode,
    DistributedP2PHonestNode,
    SmallCNN,
    select_pool_backend,
)

from byzpy.configs.actor import set_actor
from byzpy.engine.node.actors import ByzantineNodeActor, HonestNodeActor
from byzpy.engine.peer_to_peer.topology import Topology
from byzpy.engine.peer_to_peer.train import PeerToPeer


def shard_indices(n_items: int, n_shards: int) -> List[List[int]]:
    return [list(range(i, n_items, n_shards)) for i in range(n_shards)]


def make_test_loader(batch_size: int = 512) -> data.DataLoader:
    tfm = transforms.Compose([transforms.ToTensor()])
    test = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)
    return data.DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=0)


def evaluate(model: torch.nn.Module, device: torch.device) -> Tuple[float, float]:
    loader = make_test_loader()
    ce = torch.nn.CrossEntropyLoss(reduction="sum")
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum += ce(logits, y).item()
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    model.train()
    return loss_sum / total, correct / total


async def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    remotes_env = os.environ.get("P2P_REMOTES", "tcp://127.0.0.1:29000")
    REMOTES: List[str] = [r.strip() for r in remotes_env.split(",") if r.strip()]

    n_honest = int(os.environ.get("P2P_HONEST", "4"))
    n_byz = int(os.environ.get("P2P_BYZ", "1"))
    rounds = int(os.environ.get("P2P_ROUNDS", "200"))
    batch_size = int(os.environ.get("P2P_BATCH", "64"))
    data_root = os.environ.get("P2P_DATA", "./data")
    lr = float(os.environ.get("P2P_LR", "0.05"))
    momentum = float(os.environ.get("P2P_MOM", "0.9"))
    byz_scale = float(os.environ.get("P2P_BYZ_SCALE", "-1.0"))

    tfm = transforms.Compose([transforms.ToTensor()])
    _tmp_train = datasets.MNIST(root=data_root, train=True, download=True, transform=tfm)
    shards = shard_indices(len(_tmp_train), n_honest)

    def pick_remote(i: int) -> str:
        return REMOTES[i % len(REMOTES)]

    honest_actors: List[HonestNodeActor] = []
    for i in range(n_honest):
        actor_backend = pick_remote(i)
        pool_backend = select_pool_backend(actor_backend)
        h = await HonestNodeActor.spawn(
            DistributedP2PHonestNode,
            backend=set_actor(actor_backend),
            kwargs=dict(
                indices=shards[i],
                batch_size=batch_size,
                shuffle=True,
                lr=lr,
                momentum=momentum,
                device=("cuda" if torch.cuda.is_available() else "cpu"),
                data_root=data_root,
                pool_backend=pool_backend,
            ),
        )
        honest_actors.append(h)

    byz_actors: List[ByzantineNodeActor] = []
    for j in range(n_byz):
        actor_backend = pick_remote(n_honest + j)
        pool_backend = select_pool_backend(actor_backend)
        b = await ByzantineNodeActor.spawn(
            DistributedP2PByzNode,
            backend=set_actor(actor_backend),
            kwargs=dict(
                device=("cuda" if torch.cuda.is_available() else "cpu"),
                scale=byz_scale,
                pool_backend=pool_backend,
            ),
        )
        byz_actors.append(b)

    topo = Topology.complete(n=len(honest_actors) + len(byz_actors))

    p2p = PeerToPeer(
        honest_nodes=honest_actors,
        byzantine_nodes=byz_actors,
        topology=topo,
        lr=lr,
        channel_name="p2p",
    )
    await p2p.bootstrap()

    eval_model = SmallCNN().to(device)

    print("P2P Training | remote actors with distributed nodes")
    for r in range(1, rounds + 1):
        await p2p.round()
        if r % 50 == 0:
            sd = await honest_actors[0].dump_state_dict()
            eval_model.load_state_dict(sd, strict=True)
            loss, acc = await asyncio.to_thread(evaluate, eval_model, device)
            print(f"[round {r:04d}] test loss={loss:.4f}  acc={acc:.4f}")

    print("\nFinal honest-node evaluations:")
    for i, h in enumerate(honest_actors):
        sd = await h.dump_state_dict()
        eval_model.load_state_dict(sd, strict=True)
        loss, acc = await asyncio.to_thread(evaluate, eval_model, device)
        print(f"  node {i}: loss={loss:.4f}  acc={acc:.4f}")

    await p2p.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
