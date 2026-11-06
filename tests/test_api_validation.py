import asyncio

import numpy as np
import pytest

from conftest import AUTH


async def post(client, body, **kw):
    return await client.post("/v1/audio/speech", json=body, headers=AUTH, **kw)


@pytest.mark.parametrize("body,param", [
    ({}, "input"),
    ({"input": ""}, "input"),
    ({"input": 5}, "input"),
    ({"input": "hi", "language": "xx"}, "language"),
    ({"input": "hi", "response_format": "aac"}, "response_format"),
    ({"input": "hi", "response_format": "ogg"}, "response_format"),
    ({"input": "hi", "speed": 1.5}, "speed"),
    ({"input": "hi", "stream_format": "ndjson"}, "stream_format"),
    ({"input": "hi", "voice": "no-such-voice"}, "voice"),
    ({"input": "hi", "temperature": 9}, "temperature"),
    ({"input": "hi", "max_frames": 0}, "max_frames"),
    ({"input": "x" * 4001}, "input"),
])
async def test_invalid_requests_rejected_before_synthesis(make, body, param):
    client, srv, tts = await make(max_body_bytes=64 * 1024)
    r = await post(client, body)
    assert r.status == 400
    assert (await r.json())["error"]["param"] == param
    assert tts.engine.requests == []                    # nothing reached the engine


async def test_non_json_body_rejected(make):
    client, *_ = await make()
    r = await client.post("/v1/audio/speech", data=b"not json", headers=AUTH)
    assert r.status == 400


async def test_pcm_stream_headers_and_content(make):
    client, *_ = await make()
    r = await post(client, {"input": "#frames=3", "language": "hi", "response_format": "pcm"})
    assert r.status == 200
    assert r.headers["X-Sample-Rate"] == "24000" and r.headers["X-Encoding"] == "pcm_s16le"
    assert r.headers["X-Maya3-Language"] == "hi"
    pcm = np.frombuffer(await r.read(), np.int16).reshape(-1, 1920)
    assert pcm[:, 0].tolist() == [1, 2, 3]              # frame order preserved


async def test_wav_has_header_and_pcm(make):
    client, *_ = await make()
    r = await post(client, {"input": "#frames=2"})
    data = await r.read()
    assert data[:4] == b"RIFF" and data[8:12] == b"WAVE" and len(data) == 44 + 2 * 1920 * 2
    assert int.from_bytes(data[24:28], "little") == 24000


async def test_overload_returns_503_with_retry_after(make):
    client, srv, tts = await make(engine={"max_seats": 1, "frame_s": 0.05}, max_pending=1)
    first = asyncio.create_task(post(client, {"input": "#frames=20", "response_format": "pcm"}))
    await asyncio.sleep(0.01)
    second = await post(client, {"input": "#frames=1"})
    third = await post(client, {"input": "#frames=1"})
    assert 503 in (second.status, third.status)
    r = second if second.status == 503 else third
    assert r.headers.get("Retry-After") and (await r.json())["error"]["code"] == "overloaded"
    await first


async def test_client_disconnect_cancels_engine_request(make):
    client, srv, tts = await make(engine={"frame_s": 0.02})
    r = await post(client, {"input": "#frames=500", "response_format": "pcm"})
    await r.content.readany()
    r.close()
    for _ in range(100):
        if tts.engine.requests[0].cancelled:
            break
        await asyncio.sleep(0.02)
    assert tts.engine.requests[0].cancelled


async def test_metrics_record_completion(make):
    client, srv, _ = await make()
    r = await post(client, {"input": "#frames=3", "response_format": "pcm"})
    await r.read()
    await asyncio.sleep(0.05)
    snap = await (await client.get("/metrics?json=1", headers=AUTH)).json()
    assert snap["counters"]["synth_completed"] == 1
    assert snap["summaries"]["ttfa_seconds"]["count"] == 1 and snap["summaries"]["rtf"]["count"] == 1
    assert "gpu" in snap and "upstream" in snap
    text = await (await client.get("/metrics", headers=AUTH)).text()
    assert "mme_tts_ttfa_seconds_count 1" in text
