from __future__ import annotations

import math
from itertools import combinations
from typing import Any, Iterable, Sequence

import numpy as np

from ...configs.backend import get_backend
from ...engine.graph.subtask import SubTask
from ...engine.storage.shared_store import (
    SharedTensorHandle,
    cleanup_tensor,
    open_tensor,
    register_tensor,
)
from .._chunking import select_adaptive_chunk_size
from ..base import Aggregator


def _pairwise_sq_dists(stacked: Any) -> Any:
    """
    Return (n, n) matrix of squared Euclidean distances for stacked of shape (n, ...).
    """
    be = get_backend()
    if be.name == "torch":
        import torch

        flat = stacked.reshape(stacked.shape[0], -1)
        # torch.cdist computes Euclidean distances efficiently without materializing pairwise differences.
        dists = torch.cdist(flat, flat, p=2)
        return dists * dists

    if be.name == "numpy":
        import numpy as np

        arr = np.asarray(stacked)
        flat = arr.reshape(arr.shape[0], -1)
        norms = np.sum(flat * flat, axis=1, keepdims=True)
        gram = flat @ flat.T
        D2 = norms + norms.T - 2.0 * gram
        np.maximum(D2, 0.0, out=D2)
        return D2

    diff = stacked[:, None, ...] - stacked[None, :, ...]
    sq = diff * diff
    axes = tuple(range(2, sq.ndim))
    return be.sum(sq, axis=axes)


def _to_backend_indices(idxs: Iterable[int], like: Any, be_name: str) -> Any:
    """
    Create an index array/tensor in the configured backend.
    """
    ids = list(idxs)
    if be_name == "torch":
        import torch

        dev = getattr(like, "device", None)
        return torch.as_tensor(ids, dtype=torch.long, device=dev)
    else:
        import numpy as np

        return np.asarray(ids, dtype=int)


def _to_float(x: Any) -> float:
    return x.item() if hasattr(x, "item") else float(x)


def _pairwise_sq_dists_numpy(arr: np.ndarray) -> np.ndarray:
    flat = arr.reshape(arr.shape[0], -1)
    norms = np.sum(flat * flat, axis=1, keepdims=True)
    gram = flat @ flat.T
    D2 = norms + norms.T - 2.0 * gram
    np.maximum(D2, 0.0, out=D2)
    return D2


class MinimumDiameterAveraging(Aggregator):
    """
    Minimum Diameter Averaging (MDA), exact combinatorial search.

    Chooses a subset ``S`` with ``|S| = n - f`` that minimizes the diameter
    ``max_{i,j in S} ||x_i - x_j||`` and returns the average of vectors in
    that subset.

    Constraint: ``0 <= f < n``.
    """

    name = "minimum-diameter-averaging"
    supports_subtasks = True
    max_subtasks_inflight = 0  # auto-scale with actor pool size

    def __init__(self, f: int, *, chunk_size: int = 256) -> None:
        if f < 0:
            raise ValueError("f must be >= 0")
        self.f = f
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        self.chunk_size = int(chunk_size)
        self._active_handles: tuple[SharedTensorHandle, SharedTensorHandle] | None = None
        self.seed_prefix = 2
        self.seeds_per_task = 4

    def aggregate(self, gradients: Sequence[Any]) -> Any:
        """
        Exhaustively search the (n - f)-subsets for the minimum-diameter set.

        Args:
            gradients: Sequence of tensors; length must exceed ``f``.
        """
        if not gradients:
            raise ValueError("gradients must be a non-empty sequence")

        n = len(gradients)
        f = self.f
        if f >= n:
            raise ValueError(f"f must satisfy 0 <= f < n (got n={n}, f={f})")

        like = gradients[0]
        X = _stack_numpy(gradients)
        D2 = _pairwise_sq_dists_numpy(X)
        m = n - f
        best_diam = None
        best_mean = None

        if 0 < self.seed_prefix < m:
            best_diam, best_mean = _seeded_direct_search(
                X,
                D2,
                n=n,
                m=m,
                seed_prefix=self.seed_prefix,
            )
        else:
            for combo in combinations(range(n), m):
                diam, mean = _mda_eval_combo_numpy(X, D2, combo)
                if (best_diam is None) or (diam < best_diam):
                    best_diam = diam
                    best_mean = mean

        if best_mean is None or best_diam is None:
            raise RuntimeError("MinimumDiameterAveraging failed to evaluate any subset.")
        return _convert_result(mean=best_mean, like=like)

    def create_subtasks(self, inputs, *, context: Any):  # type: ignore[override]
        gradients = inputs.get(self.input_key)
        if not isinstance(gradients, Sequence) or not gradients:
            return []

        n = len(gradients)
        f = self.f
        if f >= n:
            raise ValueError(f"f must satisfy 0 <= f < n (got n={n}, f={f})")

        X = _stack_numpy(gradients)
        D2 = _pairwise_sq_dists_numpy(X)
        m = n - f

        total_combos = math.comb(n, m)
        chunk_size = self._select_chunk_size(total_combos, context)
        handle_X = register_tensor(X)
        handle_D2 = register_tensor(D2)
        self._active_handles = (handle_X, handle_D2)

        if 0 < self.seed_prefix < m:
            return self._seeded_subtasks(handle_X, handle_D2, gradients[0], n, m)

        def _subtask_iter():
            chunk_id = 0
            chunk: list[tuple[int, ...]] = []
            for combo in combinations(range(n), m):
                chunk.append(tuple(combo))
                if len(chunk) < chunk_size:
                    continue
                yield SubTask(
                    fn=_mda_best_subset_chunk,
                    args=(handle_X, handle_D2, tuple(chunk), gradients[0]),
                    kwargs={},
                    name=f"mda_chunk_{chunk_id}",
                )
                chunk_id += 1
                chunk = []
            if chunk:
                yield SubTask(
                    fn=_mda_best_subset_chunk,
                    args=(handle_X, handle_D2, tuple(chunk), gradients[0]),
                    kwargs={},
                    name=f"mda_chunk_{chunk_id}",
                )

        return _subtask_iter()

    def _seeded_subtasks(self, handle_X, handle_D2, like, n: int, m: int):
        depth = min(self.seed_prefix, m)
        seeds = combinations(range(n), depth)
        group: list[tuple[int, ...]] = []
        task_id = 0
        max_last = n - (m - depth) - 1
        for seed in seeds:
            if seed[-1] > max_last:
                continue
            group.append(seed)
            if len(group) >= self.seeds_per_task:
                yield SubTask(
                    fn=_mda_best_subset_seeded,
                    args=(handle_X, handle_D2, tuple(group), n, m, like),
                    kwargs={},
                    name=f"mda_seed_chunk_{task_id}",
                )
                task_id += 1
                group = []
        if group:
            yield SubTask(
                fn=_mda_best_subset_seeded,
                args=(handle_X, handle_D2, tuple(group), n, m, like),
                kwargs={},
                name=f"mda_seed_chunk_{task_id}",
            )

    def reduce_subtasks(self, partials, inputs, *, context: Any):  # type: ignore[override]
        if not partials:
            return super().compute(inputs, context=context)

        be = get_backend()
        best_diam = None
        best_mean = None
        try:
            for idx, item in enumerate(partials):
                try:
                    diam, mean = item
                except Exception as exc:  # pragma: no cover
                    raise ValueError(
                        f"MinimumDiameterAveraging received malformed partial at index {idx}: {item!r}"
                    ) from exc
                if best_diam is None or diam < best_diam:
                    best_diam = float(diam)
                    best_mean = mean

            if best_mean is None:
                raise RuntimeError("MinimumDiameterAveraging subtasks did not return a result.")
            return be.copy(be.asarray(best_mean))
        finally:
            handles = self._active_handles or ()
            for handle in handles:
                cleanup_tensor(handle)
            self._active_handles = None

    def _select_chunk_size(self, total_combos: int, context: Any) -> int:
        """
        Choose a chunk size that balances scheduler overhead with available parallelism.
        """
        pool_size = 0
        if context is not None:
            meta = getattr(context, "metadata", None) or {}
            pool_size = int(meta.get("pool_size") or 0)
        return select_adaptive_chunk_size(
            total_combos,
            self.chunk_size,
            pool_size=pool_size,
            min_chunks_per_worker=4,
            max_shrink_factor=8,
        )


def _mda_eval_combo_numpy(
    X: np.ndarray, D2: np.ndarray, combo: Iterable[int]
) -> tuple[float, np.ndarray]:
    idx = np.asarray(tuple(combo), dtype=int)
    sub = D2[np.ix_(idx, idx)]
    diam2 = float(np.max(sub))
    chosen = X[idx]
    mean = np.mean(chosen, axis=0)
    return diam2, mean


def _mda_best_subset_chunk(
    handle_X: SharedTensorHandle,
    handle_D2: SharedTensorHandle,
    combos: Sequence[Iterable[int]],
    like_template,
) -> tuple[float, Any]:
    with open_tensor(handle_X) as X, open_tensor(handle_D2) as D2:
        best_diam = None
        best_mean = None
        for combo in combos:
            diam, mean = _mda_eval_combo_numpy(X, D2, combo)
            if (best_diam is None) or (diam < best_diam):
                best_diam = diam
                best_mean = mean
    if best_mean is None or best_diam is None:
        raise ValueError("MinimumDiameterAveraging subtask received no combinations.")
    return best_diam, _convert_result(mean=best_mean, like=like_template)


def _mda_best_subset_seeded(
    handle_X: SharedTensorHandle,
    handle_D2: SharedTensorHandle,
    seeds: Sequence[Iterable[int]],
    n: int,
    m: int,
    like_template,
) -> tuple[float, Any]:
    with open_tensor(handle_X) as X, open_tensor(handle_D2) as D2:
        best_diam = None
        best_mean = None

        for seed in seeds:
            prefix = list(seed)
            remaining = m - len(prefix)
            if remaining < 0:
                continue
            current_max = 0.0
            if prefix:
                sub = D2[np.ix_(prefix, prefix)]
                current_max = float(np.max(sub)) if sub.size else 0.0
            for diam, mean in _search_seed(prefix, remaining, current_max, D2, X, n):
                if best_diam is None or diam < best_diam:
                    best_diam = diam
                    best_mean = mean

        if best_mean is None or best_diam is None:
            raise ValueError("Seeded MDA subtask failed to find a result.")
        return best_diam, _convert_result(mean=best_mean, like=like_template)


def _seeded_direct_search(
    X: np.ndarray, D2: np.ndarray, *, n: int, m: int, seed_prefix: int
) -> tuple[float, np.ndarray]:
    depth = min(seed_prefix, m)
    seeds = combinations(range(n), depth)
    best_diam = None
    best_mean = None
    max_last = n - (m - depth) - 1 if depth else None

    for seed in seeds:
        if depth and seed and max_last is not None and seed[-1] > max_last:
            continue
        prefix = list(seed)
        remaining = m - len(prefix)
        if remaining < 0:
            continue
        current_max = 0.0
        if prefix:
            sub = D2[np.ix_(prefix, prefix)]
            current_max = float(np.max(sub)) if sub.size else 0.0
        for diam, mean in _search_seed(prefix, remaining, current_max, D2, X, n):
            if best_diam is None or diam < best_diam:
                best_diam = diam
                best_mean = mean

    if best_mean is None or best_diam is None:
        raise RuntimeError("MinimumDiameterAveraging seeded search failed to find a subset.")
    return best_diam, best_mean


def _search_seed(
    prefix: list[int],
    remaining: int,
    current_max: float,
    D2: np.ndarray,
    X: np.ndarray,
    n: int,
):
    best_local: list[float | None] = [None]

    def dfs(start_idx: int, remain: int, current: float, indices: list[int]):
        if remain == 0:
            arr = X[indices]
            mean = np.mean(arr, axis=0)
            if best_local[0] is None or current < best_local[0]:
                best_local[0] = current
            yield current, mean
            return
        for idx in range(start_idx, n - remain + 1):
            new_max = current
            if indices:
                new_max = max(new_max, float(np.max(D2[idx, indices])))
            if best_local[0] is not None and new_max >= best_local[0]:
                continue
            indices.append(idx)
            yield from dfs(idx + 1, remain - 1, new_max, indices)
            indices.pop()

    yield from dfs(prefix[-1] + 1 if prefix else 0, remaining, current_max, list(prefix))


def _convert_result(*, mean: np.ndarray, like) -> Any:
    try:
        import torch

        if isinstance(like, torch.Tensor):
            t = torch.from_numpy(np.array(mean, copy=False)).to(dtype=like.dtype)
            return t
    except Exception:
        pass
    return np.array(mean, copy=True)


def _stack_numpy(gradients: Sequence[Any]) -> np.ndarray:
    first = gradients[0]
    try:
        import torch

        if isinstance(first, torch.Tensor):
            stacked = torch.stack([g.detach().cpu() for g in gradients], dim=0)
            return stacked.numpy()
    except Exception:
        pass
    arrays = []
    for g in gradients:
        if _is_shared_handle(g):
            arrays.append(_shared_to_numpy(g))
        else:
            arrays.append(np.asarray(g))
    return np.stack(arrays, axis=0)


def _shared_to_numpy(handle: Any) -> np.ndarray:
    if isinstance(handle, dict):
        handle = SharedTensorHandle(**handle)
    with open_tensor(handle) as arr:
        return np.array(arr, copy=True)


def _is_shared_handle(obj: Any) -> bool:
    if isinstance(obj, SharedTensorHandle):
        return True
    if isinstance(obj, dict):
        return {"name", "shape", "dtype"} <= obj.keys()
    return False


def _mda_eval_combo(X: Any, D2: Any, combo: Iterable[int], be_name: str) -> tuple[float, Any]:
    be = get_backend()
    idx = _to_backend_indices(combo, like=X, be_name=be_name)
    sub = be.index_select(D2, axis=0, indices=idx)
    sub = be.index_select(sub, axis=1, indices=idx)
    diam2 = be.max(sub)
    diam = _to_float(diam2)
    chosen = be.index_select(X, axis=0, indices=idx)
    mean = be.mean(chosen, axis=0)
    return diam, mean
