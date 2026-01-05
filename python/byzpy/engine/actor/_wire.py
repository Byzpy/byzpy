import asyncio
import pickle
import struct

_HDR = struct.Struct("!I")


async def send_obj(w: asyncio.StreamWriter, obj) -> None:
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    w.write(_HDR.pack(len(data)) + data)
    await w.drain()


async def recv_obj(r: asyncio.StreamReader):
    hdr = await r.readexactly(_HDR.size)
    (n,) = _HDR.unpack(hdr)
    data = await r.readexactly(n)
    return pickle.loads(data)
