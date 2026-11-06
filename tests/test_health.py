import asyncio
import threading

from conftest import AUTH
from maya3.tests.fakes import FakeTTS


async def wait_for(pred, timeout=3.0):
    for _ in range(int(timeout / 0.01)):
        if pred():
            return True
        await asyncio.sleep(0.01)
    return pred()


async def test_ready_server_reports_200(make):
    client, srv, _ = await make()
    r = await client.get("/ping")
    assert r.status == 200
    r = await client.get("/health")
    body = await r.json()
    assert r.status == 200 and body["status"] == "ready" and body["model_ready"] is True
    assert body["capacity"]["max_seats"] == 4 and body["capacity"]["available_seats"] == 4
    assert body["sessions"]["active_synthesis"] == 0
    assert "gpu" in body and body["gpu"]["available"] in (True, False)


async def test_ping_is_204_while_loading_then_200(make):
    gate = threading.Event()

    def loader():
        gate.wait(5)
        return FakeTTS()
    client, srv, _ = await make(tts=None, loader=loader)
    r = await client.get("/ping")
    assert r.status == 204                     # initialising, NOT 200
    r = await client.get("/health")
    body = await r.json()
    assert r.status == 503 and body["status"] == "initializing" and body["model_ready"] is False
    # synthesis is refused (not queued) while loading
    r = await client.post("/v1/audio/speech", json={"input": "hi"}, headers=AUTH)
    assert r.status == 503
    gate.set()
    assert await wait_for(lambda: srv.phase == "ready")
    assert (await client.get("/ping")).status == 200
    assert srv.load_seconds is not None


async def test_load_failure_fails_closed_and_requests_exit(make):
    fatal = []

    def loader():
        raise FileNotFoundError("model.safetensors missing")
    client, srv, _ = await make(tts=None, loader=loader)
    srv.on_fatal = fatal.append
    assert await wait_for(lambda: srv.load_error is not None)
    assert (await client.get("/ping")).status == 503
    body = await (await client.get("/health")).json()
    assert body["status"] == "failed" and "model.safetensors missing" in body["initialization"]["error"]
    # NOTE: on_fatal is wired after startup here, so exit scheduling itself is covered in test_lifecycle


async def test_engine_failure_turns_unhealthy(make):
    client, srv, tts = await make()
    tts.engine.failed = "CUDA error"
    assert (await client.get("/ping")).status == 503
    assert (await (await client.get("/health")).json())["status"] == "failed"


async def test_draining_is_503(make):
    client, srv, _ = await make()
    srv.draining = True
    assert (await client.get("/ping")).status == 503
