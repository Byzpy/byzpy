from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class Edge:
    u: int
    v: int  # directed u -> v


class Topology:
    """
    Directed communication graph  V={0..N-1}.
    Store both Γ_out(i) and Γ_in(i).
    """

    def __init__(self, n_nodes: int, edges: Iterable[Tuple[int, int]]):
        self.n = int(n_nodes)
        self.out: Dict[int, List[int]] = {i: [] for i in range(self.n)}
        self.in_: Dict[int, List[int]] = {i: [] for i in range(self.n)}
        for u, v in edges:
            self.out[u].append(v)
            self.in_[v].append(u)

    @classmethod
    def complete(cls, n: int) -> "Topology":
        return cls(n, [(i, j) for i in range(n) for j in range(n) if i != j])

    @classmethod
    def ring(cls, n: int, k: int = 1) -> "Topology":
        es = []
        for i in range(n):
            for d in range(1, k + 1):
                es.append((i, (i + d) % n))
                es.append((i, (i - d) % n))
        return cls(n, es)
