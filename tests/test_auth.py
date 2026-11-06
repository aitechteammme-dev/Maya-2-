import aiohttp
import pytest

from conftest import ADMIN, AUTH, KEY, KEY_B
from server.auth import is_admin_route, match_token, voice_allowed
from server.config import ConfigError, Settings, parse_keys


async def test_missing_and_wrong_token_rejected(make):
    client, *_ = await make()
    for headers in ({}, {"Authorization": "Bearer nope-nope-nope-nope"}, {"Authorization": "Basic abc"},
                    {"Authorization": f"Bearer {KEY}x"}):
        r = await client.post("/v1/audio/speech", json={"input": "hi"}, headers=headers)
        assert r.status == 401, headers
        assert r.headers["WWW-Authenticate"] == "Bearer"
        assert (await r.json())["error"]["code"] == "unauthorized"


async def test_valid_bearer_and_api_key_header(make):
    client, *_ = await make()
    for headers in (AUTH, {"X-MME-Api-Key": KEY}):
        r = await client.post("/v1/audio/speech", json={"input": "#frames=2", "response_format": "pcm"}, headers=headers)
        assert r.status == 200
        assert len(await r.read()) == 2 * 1920 * 2


async def test_ping_and_health_public_but_metrics_private(make):
    client, *_ = await make()
    assert (await client.get("/ping")).status == 200
    assert (await client.get("/health")).status == 200
    assert (await client.get("/metrics")).status == 401
    r = await client.get("/metrics", headers=AUTH)
    assert r.status == 200 and "mme_tts_gpu_available" in await r.text()


async def test_no_token_leak_in_errors(make):
    client, *_ = await make()
    r = await client.get("/v1/voices", headers={"Authorization": "Bearer secret-secret-secret-1"})
    assert "secret-secret" not in await r.text()


async def test_voice_registration_disabled_by_default(make):
    client, *_ = await make()
    r = await client.post("/v1/voices", json={"voice_id": "x"}, headers={"Authorization": f"Bearer {ADMIN}"})
    assert r.status == 403
    assert "disabled" in (await r.json())["error"]["message"]


async def test_voice_registration_needs_admin_when_enabled(make):
    client, *_ = await make(allow_voice_registration=True)
    r = await client.post("/v1/voices", json={"voice_id": "acme__v"}, headers=AUTH)
    assert r.status == 403
    r = await client.delete("/v1/voices/acme__v", headers=AUTH)
    assert r.status == 403
    r = await client.delete("/v1/voices/acme__v", headers={"Authorization": f"Bearer {ADMIN}"})
    assert r.status == 404                       # got past auth: upstream says unknown voice


async def test_body_size_limit(make):
    client, *_ = await make(max_body_bytes=2048)
    r = await client.post("/v1/audio/speech", data=b"x" * 5000, headers={**AUTH, "Content-Type": "application/json"})
    assert r.status == 413


async def test_tenant_isolation(make):
    client, srv, tts = await make()
    from maya3.tests.fakes import FakeVoice
    tts.engine.registry.add(FakeVoice("acme__utkarsh", "icl_spk", "hi"))
    tts.engine.registry.add(FakeVoice("beta__other", "icl_spk", "hi"))
    ids = [v["id"] for v in (await (await client.get("/v1/voices", headers=AUTH)).json())["voices"]]
    assert "acme__utkarsh" in ids and "beta__other" not in ids
    ok = await client.post("/v1/audio/speech", json={"input": "#frames=1", "voice": "acme__utkarsh"}, headers=AUTH)
    assert ok.status == 200
    bad = await client.post("/v1/audio/speech", json={"input": "#frames=1", "voice": "beta__other"}, headers=AUTH)
    assert bad.status == 403
    # same enforcement on the upstream WebSocket and the incremental socket
    async with client.ws_connect("/v1/tts/stream", headers=AUTH) as ws:
        await ws.receive_json()
        await ws.send_json({"type": "speak", "context_id": "c", "text": "#frames=1", "voice": "beta__other"})
        msg = await ws.receive_json()
        assert msg["type"] == "error" and msg["code"] == "forbidden"
    async with client.ws_connect("/v1/tts/incremental", headers=AUTH) as ws:
        await ws.receive_json()
        await ws.send_json({"type": "start", "voice": "beta__other"})
        msg = await ws.receive_json()
        assert msg["type"] == "error" and msg["code"] == "forbidden"
    r = await client.post("/v1/audio/speech", json={"input": "#frames=1", "voice": "beta__other"},
                          headers={"Authorization": f"Bearer {KEY_B}"})
    assert r.status == 200


async def test_websocket_handshake_requires_auth(make):
    client, *_ = await make()
    with pytest.raises(aiohttp.WSServerHandshakeError) as e:
        await client.ws_connect("/v1/tts/stream")
    assert e.value.status == 401
    with pytest.raises(aiohttp.WSServerHandshakeError) as e:
        await client.ws_connect("/v1/tts/incremental")
    assert e.value.status == 401
    async with client.ws_connect("/v1/tts/stream", headers=AUTH) as ws:
        meta = await ws.receive_json()
        assert meta["type"] == "metadata" and meta["sample_rate"] == 24000


def test_match_token_and_helpers():
    table = {"a" * 20: "t1", "b" * 20: "t2"}
    assert match_token("a" * 20, table) == "t1" and match_token("b" * 20, table) == "t2"
    assert match_token("a" * 19, table) is None and match_token("", table) is None
    assert is_admin_route("POST", "/v1/voices") and is_admin_route("DELETE", "/v1/voices/x")
    assert not is_admin_route("GET", "/v1/voices") and not is_admin_route("POST", "/v1/audio/speech")
    assert voice_allowed("plain", "a", False) and voice_allowed("Ahana", "a", False)
    assert voice_allowed("a__v", "a", False) and not voice_allowed("b__v", "a", False) and voice_allowed("b__v", "a", True)


def test_config_fails_closed_without_keys_and_refuses_open_prod(tmp_path):
    with pytest.raises(ConfigError, match="no API keys"):
        Settings.from_env({"MME_PROFILE": "production"}, tmp_path)
    with pytest.raises(ConfigError, match="refused in the production"):
        Settings.from_env({"MME_PROFILE": "production", "MME_AUTH_DISABLED": "1"}, tmp_path)
    s = Settings.from_env({"MME_PROFILE": "development", "MME_AUTH_DISABLED": "1"}, tmp_path)
    assert s.auth_disabled
    with pytest.raises(ConfigError, match="16 characters"):
        parse_keys("short")
    assert parse_keys("acme=" + "k" * 16 + ",other" + "x" * 16) == {"k" * 16: "acme", "other" + "x" * 16: "default"}
