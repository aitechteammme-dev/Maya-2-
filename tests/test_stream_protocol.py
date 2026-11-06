"""Upstream WS passthrough + the MME incremental endpoint lifecycle (FakeEngine, no GPU)."""
import asyncio
import base64
import struct

import numpy as np

from conftest import AUTH


async def collect(ws, until=("done", "error"), limit=200):
    msgs = []
    for _ in range(limit):
        m = await asyncio.wait_for(ws.receive(), 5)
        if m.type.name == "TEXT":
            j = m.json()
            msgs.append(j)
            if j["type"] in until:
                return msgs
        elif m.type.name == "BINARY":
            msgs.append({"type": "binary", "data": m.data})
        else:
            break
    return msgs


# ------------------------------------------------------------------ upstream WS (inherited, unchanged)
async def test_upstream_ws_session_lifecycle(make):
    client, srv, tts = await make()
    async with client.ws_connect("/v1/tts/stream", headers=AUTH) as ws:
        meta = await ws.receive_json()
        assert meta["sample_rate"] == 24000 and meta["encoding"] == "pcm_s16le"
        await ws.send_json({"type": "speak", "context_id": "c1", "text": "#frames=3", "language": "hi"})
        msgs = await collect(ws, until=("end", "error"))
        audio = [m for m in msgs if m["type"] == "audio"]
        assert audio and msgs[-1]["type"] == "end" and msgs[-1]["context_id"] == "c1"
        pcm = np.frombuffer(b"".join(base64.b64decode(a["audio"]) for a in audio), np.int16).reshape(-1, 1920)
        assert pcm[:, 0].tolist() == [1, 2, 3]
        await ws.send_json({"type": "ping"})
        assert (await ws.receive_json())["type"] == "pong"
    await asyncio.sleep(0.05)
    assert srv.sockets == 0


async def test_upstream_ws_cancel_and_disconnect_cleanup(make):
    client, srv, tts = await make(engine={"frame_s": 0.02})
    async with client.ws_connect("/v1/tts/stream", headers=AUTH) as ws:
        await ws.receive_json()
        await ws.send_json({"type": "speak", "context_id": "c1", "text": "#frames=400"})
        await asyncio.sleep(0.1)
        await ws.send_json({"type": "cancel", "context_id": "c1"})
        msgs = await collect(ws, until=("end", "error"))
        assert msgs[-1]["type"] == "end" and msgs[-1]["reason"] == "cancelled"
        await ws.send_json({"type": "speak", "context_id": "c2", "text": "#frames=400"})
        await asyncio.sleep(0.1)
    for _ in range(100):                        # socket closed -> engine request cancelled
        if all(r.cancelled for r in tts.engine.requests):
            break
        await asyncio.sleep(0.02)
    assert all(r.cancelled for r in tts.engine.requests)


async def test_ws_connection_limit(make):
    client, srv, _ = await make(max_ws_connections=1)
    async with client.ws_connect("/v1/tts/stream", headers=AUTH) as ws:
        await ws.receive_json()
        r = await client.get("/v1/tts/stream", headers={**AUTH, "Connection": "Upgrade", "Upgrade": "websocket",
                                                       "Sec-WebSocket-Version": "13", "Sec-WebSocket-Key": "dGhlIHNhbXBsZSBub25jZQ=="})
        assert r.status == 503


# ------------------------------------------------------------------ incremental endpoint
async def start(ws, **kw):
    meta = await ws.receive_json()
    assert meta["protocol"] == "mme-incremental/1" and meta["sample_rate"] == 24000
    await ws.send_json({"type": "start", "language": "en", **kw})
    st = await ws.receive_json()
    assert st["type"] == "started" and st["session_id"]
    return st


async def test_incremental_two_sentences_in_order(make):
    client, srv, tts = await make(engine={"frame_s": 0.002}, inc_first_min_chars=5, inc_min_chars=5)
    async with client.ws_connect("/v1/tts/incremental", headers=AUTH) as ws:
        await start(ws, session_id="call-1")
        deltas = ("Your seat is ", "confirmed #frames=2. ", "See you ", "soon #frames=3! ", "Bye")
        for delta in deltas:
            await ws.send_json({"type": "text", "text": delta})
        await ws.send_json({"type": "end_input"})
        msgs = await collect(ws)
    assert msgs[-1] == {"type": "done", "chunks": 3, "audio_s": round(8 * 1920 / 24000, 3), "chars_in": sum(map(len, deltas))}
    starts = [m for m in msgs if m["type"] == "chunk_start"]
    assert [s["text"] for s in starts] == ["Your seat is confirmed #frames=2.", "See you soon #frames=3!", "Bye"]
    # never silently drops text: every fed character is inside a dispatched chunk
    assert " ".join(s["text"] for s in starts).replace(" ", "") == "Your seat is confirmed #frames=2. See you soon #frames=3! Bye".replace(" ", "")
    audio = [m for m in msgs if m["type"] == "audio"]
    assert [a["seq"] for a in audio] == list(range(len(audio)))             # contiguous sequence numbers
    assert [a["chunk_id"] for a in audio] == sorted(a["chunk_id"] for a in audio)   # strict chunk order
    offs = [a["t_offset_s"] for a in audio]
    assert offs == sorted(offs)
    ends = [m for m in msgs if m["type"] == "chunk_end"]
    assert [e["chunk_id"] for e in ends] == [1, 2, 3] and all(e["reason"] == "aeos" for e in ends)
    # chunk_start N+1 only after chunk_end N (sequential synthesis)
    order = [(m["type"], m.get("chunk_id")) for m in msgs if m["type"] in ("chunk_start", "chunk_end")]
    assert order == [("chunk_start", 1), ("chunk_end", 1), ("chunk_start", 2), ("chunk_end", 2),
                     ("chunk_start", 3), ("chunk_end", 3)]
    await asyncio.sleep(0.05)
    assert srv.incremental_sessions == 0


async def test_incremental_binary_frames(make):
    client, *_ = await make(engine={"frame_s": 0.001})
    async with client.ws_connect("/v1/tts/incremental", headers=AUTH) as ws:
        await start(ws, binary=True)
        await ws.send_json({"type": "text", "text": "Hello there my friend #frames=2."})
        await ws.send_json({"type": "end_input"})
        msgs = await collect(ws)
    frames = [m["data"] for m in msgs if m["type"] == "binary"]
    assert frames
    seq, cid = struct.unpack("<II", frames[0][:8])
    assert (seq, cid) == (0, 1)
    assert len(b"".join(f[8:] for f in frames)) == 2 * 1920 * 2


async def test_incremental_requires_start_and_validates(make):
    client, *_ = await make()
    async with client.ws_connect("/v1/tts/incremental", headers=AUTH) as ws:
        await ws.receive_json()
        await ws.send_json({"type": "text", "text": "hi"})
        assert (await ws.receive_json())["code"] == "invalid_request"
        await ws.send_str("not json")
        assert (await ws.receive_json())["code"] == "invalid_request"
        await ws.send_json({"type": "start", "language": "xx"})
        assert (await ws.receive_json())["code"] == "invalid_request"
        await ws.send_json({"type": "start", "voice": "nope"})
        assert (await ws.receive_json())["code"] == "invalid_request"
        await ws.send_json({"type": "start", "language": "hi", "voice": "plain"})
        assert (await ws.receive_json())["type"] == "started"
        await ws.send_json({"type": "start"})
        assert (await ws.receive_json())["code"] == "invalid_request"
        await ws.send_json({"type": "bogus"})
        assert (await ws.receive_json())["code"] == "invalid_request"


async def test_incremental_empty_session_done(make):
    client, *_ = await make()
    async with client.ws_connect("/v1/tts/incremental", headers=AUTH) as ws:
        await start(ws)
        await ws.send_json({"type": "end_input"})
        msgs = await collect(ws)
    assert msgs[-1]["type"] == "done" and msgs[-1]["chunks"] == 0


async def test_incremental_cancel_reports_dropped_text_and_session_continues(make):
    client, srv, tts = await make(engine={"frame_s": 0.02}, inc_first_min_chars=5, inc_min_chars=5)
    async with client.ws_connect("/v1/tts/incremental", headers=AUTH) as ws:
        await start(ws)
        await ws.send_json({"type": "text", "text": "A long first sentence #frames=300. Queued second sentence. Dangling words"})
        await asyncio.sleep(0.15)
        await ws.send_json({"type": "cancel"})
        msgs = await collect(ws, until=("cancelled",))
        cancelled = msgs[-1]
        assert cancelled["type"] == "cancelled"
        assert "Queued second sentence." in cancelled["dropped_text"] and "Dangling words" in cancelled["dropped_text"]
        assert cancelled["dropped_chars"] == len(cancelled["dropped_text"])
        tail = await collect(ws, until=("chunk_end",))
        assert tail[-1]["reason"] == "cancelled"
        assert tts.engine.requests[0].cancelled
        # barge-in then a fresh turn on the same session
        await ws.send_json({"type": "text", "text": "New answer #frames=2."})
        await ws.send_json({"type": "end_input"})
        msgs = await collect(ws)
        assert msgs[-1]["type"] == "done"
        assert [m["text"] for m in msgs if m["type"] == "chunk_start"] == ["New answer #frames=2."]


async def test_incremental_engine_error_reports_undelivered_text(make):
    client, *_ = await make(inc_first_min_chars=5, inc_min_chars=5)
    async with client.ws_connect("/v1/tts/incremental", headers=AUTH) as ws:
        await start(ws)
        await ws.send_json({"type": "text", "text": "Fails here #error=1 #frames=5. Never spoken sentence."})
        await ws.send_json({"type": "end_input"})
        msgs = await collect(ws)
    err = msgs[-1]
    assert err["type"] == "error" and err["code"] == "internal"
    assert "Fails here" in err["undelivered_text"] and "Never spoken sentence." in err["undelivered_text"]
    assert not any(m["type"] == "done" for m in msgs)


async def test_incremental_client_disconnect_cancels_and_cleans_up(make):
    client, srv, tts = await make(engine={"frame_s": 0.02}, inc_first_min_chars=5)
    async with client.ws_connect("/v1/tts/incremental", headers=AUTH) as ws:
        await start(ws)
        await ws.send_json({"type": "text", "text": "Long running sentence #frames=500."})
        await ws.send_json({"type": "flush"})
        await asyncio.sleep(0.1)
    for _ in range(100):
        if srv.incremental_sessions == 0 and tts.engine.requests and tts.engine.requests[0].cancelled:
            break
        await asyncio.sleep(0.02)
    assert srv.incremental_sessions == 0 and tts.engine.requests[0].cancelled
    assert len(srv._jobs) == 0 and len(srv._waiting) == 0


async def test_incremental_idle_flush_confirms_trailing_dot(make):
    client, *_ = await make(inc_idle_flush_ms=60, inc_first_min_chars=5, engine={"frame_s": 0.001})
    async with client.ws_connect("/v1/tts/incremental", headers=AUTH) as ws:
        await start(ws)
        await ws.send_json({"type": "text", "text": "Thanks for calling #frames=1."})   # '.' held at buffer end
        msgs = await collect(ws, until=("chunk_end",))                                    # no flush sent: idle timer
        assert [m["text"] for m in msgs if m["type"] == "chunk_start"] == ["Thanks for calling #frames=1."]


async def test_incremental_session_text_limit(make):
    client, *_ = await make(inc_max_session_chars=20)
    async with client.ws_connect("/v1/tts/incremental", headers=AUTH) as ws:
        await start(ws)
        await ws.send_json({"type": "text", "text": "x" * 21})
        assert (await ws.receive_json())["code"] == "session_text_limit"


async def test_incremental_rejected_while_loading(make):
    client, *_ = await make(tts=None, loader=lambda: __import__("time").sleep(2))
    r = await client.get("/v1/tts/incremental", headers={**AUTH, "Connection": "Upgrade", "Upgrade": "websocket",
                                                        "Sec-WebSocket-Version": "13", "Sec-WebSocket-Key": "dGhlIHNhbXBsZSBub25jZQ=="})
    assert r.status == 503
