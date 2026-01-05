from __future__ import annotations

import asyncio
import pickle
import struct
from typing import TYPE_CHECKING, Any, Dict, Tuple

if TYPE_CHECKING:
    import cupy  # noqa: F401
    import torch  # noqa: F401

__all__ = [
    "have_ucx",
    "get_endpoint",
    "create_endpoint",
    "clear_pool",
    "create_listener",
    "call",
    "send_control",
    "recv_control",
    "pack_payload",
    "send_payload",
    "recv_payload",
]

_UCX_HDR = struct.Struct("!I")  # 4-byte length prefix for CPU control/payloads

_UCX_MOD = None  # cached resolved module

# Pooled endpoints and locks (keyed by (host, port))
_UCX_EP_CACHE: Dict[Tuple[str, int], Any] = {}
_UCX_EP_LOCKS: Dict[Tuple[str, int], asyncio.Lock] = {}
_UCX_CREATE_LOCK = asyncio.Lock()  # serialize slow create_endpoint


def _ucx_mod():
    """
    Return the UCX module in use: prefer `ucxx`, fallback to `ucp`. Returns None if neither present.
    """
    global _UCX_MOD
    if _UCX_MOD is not None:
        return _UCX_MOD
    try:
        import ucxx as _m  # RAPIDS UCXX

        _UCX_MOD = _m
        return _m
    except Exception:
        try:
            import ucp as _m  # legacy ucx-py

            _UCX_MOD = _m
            return _m
        except Exception:
            _UCX_MOD = None
            return None


def have_ucx() -> bool:
    return _ucx_mod() is not None


def _ucx_key(host: str, port: int) -> Tuple[str, int]:
    return (host, int(port))


async def get_endpoint(host: str, port: int):
    m = _ucx_mod()
    if m is None:
        raise RuntimeError("UCX requested but ucxx/ucp is not available")
    key = _ucx_key(host, port)
    ep = _UCX_EP_CACHE.get(key)
    if ep is not None and not getattr(ep, "closed", False):
        return ep
    async with _UCX_CREATE_LOCK:
        ep = _UCX_EP_CACHE.get(key)
        if ep is not None and not getattr(ep, "closed", False):
            return ep
        ep = await m.create_endpoint(host, int(port))
        _UCX_EP_CACHE[key] = ep
        _UCX_EP_LOCKS.setdefault(key, asyncio.Lock())
        return ep


async def create_endpoint(host: str, port: int):
    """
    Create a dedicated UCX endpoint without touching the shared cache.
    Used by persistent clients (e.g., UCXRemoteActorBackend) that need
    a stable one-to-one connection with the server.
    """
    m = _ucx_mod()
    if m is None:
        raise RuntimeError("UCX requested but ucxx/ucp is not available")
    return await m.create_endpoint(host, int(port))


def _evict_ucx_endpoint(host: str, port: int) -> None:
    _UCX_EP_CACHE.pop(_ucx_key(host, port), None)


async def clear_pool():
    for ep in list(_UCX_EP_CACHE.values()):
        try:
            await ep.close()
        except Exception:
            pass
    _UCX_EP_CACHE.clear()


async def call(host: str, port: int, fn):
    """
    Serialize a full UCX exchange (control + payload + reply) on a pooled endpoint.
    Retries once on UCX transport errors by evicting and recreating the endpoint.
    """
    key = _ucx_key(host, port)
    lock = _UCX_EP_LOCKS.setdefault(key, asyncio.Lock())

    async def _once():
        ep = await get_endpoint(host, port)
        async with lock:
            return await fn(ep)

    try:
        return await _once()
    except Exception as e:
        # Retry on common UCX transport errors
        errname = type(e).__name__
        if any(
            s in errname for s in ("UCXXError", "UCXXConnectionResetError", "UCXXCanceledError")
        ):
            _evict_ucx_endpoint(host, port)
            return await _once()
        raise


def create_listener(cb, *, host: str, port: int):
    """
    UCX compat: ucxx.create_listener(cb, port=...)  (no 'host' kw)
                ucp.create_listener(cb, port=..., ...) sometimes had different sigs.
    We try host+port first, then fall back to port-only.
    """
    m = _ucx_mod()
    if m is None:
        raise RuntimeError("UCX requested but ucxx/ucp is not available")
    try:
        # some ucp builds may accept host; ucxx won't
        return m.create_listener(cb, host=host, port=port)
    except TypeError:
        # ucxx path: no 'host' kw
        return m.create_listener(cb, port=port)


# ----------------------------- CUDA helpers -------------------------------------

_TORCH_DTYPE_TO_NUMPY_STR = {
    # float
    "torch.float16": "float16",
    "torch.bfloat16": "bfloat16",
    "torch.float32": "float32",
    "torch.float64": "float64",
    # int
    "torch.int8": "int8",
    "torch.int16": "int16",
    "torch.int32": "int32",
    "torch.int64": "int64",
    "torch.uint8": "uint8",
    # bool
    "torch.bool": "bool",
}


def _torch_dtype_to_numpy_str(dtype) -> str:
    s = str(dtype)
    return _TORCH_DTYPE_TO_NUMPY_STR.get(s, s.split("torch.", 1)[-1])


def _is_cuda_tensor(x: Any) -> bool:
    try:
        import torch

        return isinstance(x, torch.Tensor) and x.is_cuda
    except Exception:
        return False


def _cuda_to_cupy_view(x) -> "cupy.ndarray":
    """
    Zero-copy CUDA tensor -> CuPy view via DLPack (no host sync).
    """
    import cupy as cp
    import torch

    t = x.detach().contiguous()
    dl = torch.utils.dlpack.to_dlpack(t)
    return cp.from_dlpack(dl)  # modern API; avoids deprecation


def _cupy_to_cuda_tensor(x_cp: "cupy.ndarray") -> "torch.Tensor":
    import cupy as cp
    import torch

    assert isinstance(x_cp, cp.ndarray)
    from_dl = getattr(torch.utils, "dlpack").from_dlpack
    return from_dl(x_cp.toDlpack())


# ----------------------------- UCX send/recv ------------------------------------


async def send_control(ep, obj: Dict[str, Any]) -> None:
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    await ep.send(_UCX_HDR.pack(len(data)))
    await ep.send(data)


async def recv_control(ep) -> Dict[str, Any]:
    hdr = bytearray(_UCX_HDR.size)
    await ep.recv(hdr)
    (n,) = _UCX_HDR.unpack(hdr)
    buf = bytearray(n)
    await ep.recv(buf)
    return pickle.loads(bytes(buf))


def pack_payload(obj: Any) -> Tuple[str, Any]:
    """
    Decide transport representation for a single payload.
    - For CUDA tensors (and UCX present): tag='cuda' + tensor meta (no host sync)
    - Else: tag='pickle' + pickled blob
    """
    if have_ucx() and _is_cuda_tensor(obj):
        import torch

        t = obj.detach().contiguous()
        return (
            "cuda",
            {
                "dtype": _torch_dtype_to_numpy_str(t.dtype),
                "shape": tuple(t.shape),
                "nbytes": int(t.element_size() * t.numel()),
            },
        )
    blob = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    return ("pickle", blob)


async def send_payload(ep, tag: str, desc: Any, obj: Any) -> None:
    await send_control(ep, {"ptag": tag, "desc": desc})
    if tag == "cuda":
        ca = _cuda_to_cupy_view(obj)
        await ep.send(ca)  # device-to-device
    else:
        data: bytes = desc if isinstance(desc, (bytes, bytearray)) else pickle.dumps(obj)
        await ep.send(_UCX_HDR.pack(len(data)))
        await ep.send(data)


async def recv_payload(ep):
    ctrl = await recv_control(ep)
    tag, desc = ctrl["ptag"], ctrl["desc"]
    if tag == "cuda":
        import cupy as cp
        import numpy as np

        nbytes = int(desc["nbytes"])
        buf = cp.empty((nbytes,), dtype=cp.uint8)
        await ep.recv(buf)  # device receive
        np_dtype = np.dtype(str(desc["dtype"]))  # already NumPy-friendly string
        cp_view = buf.view(cp.dtype(np_dtype)).reshape(desc["shape"])
        return _cupy_to_cuda_tensor(cp_view)
    else:
        hdr = bytearray(_UCX_HDR.size)
        await ep.recv(hdr)
        (n,) = _UCX_HDR.unpack(hdr)
        buf = bytearray(n)
        await ep.recv(buf)
        return pickle.loads(bytes(buf))
