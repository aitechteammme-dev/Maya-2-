"""Runner lifecycle: separate health port, load-failure exit, shutdown draining (no signals on Windows: uses on_fatal)."""
import asyncio
import socket

import aiohttp

from conftest import make_settings
from maya3.tests.fakes import FakeTTS
from server.api import MmeServer, serve


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


async def status(url):
    async with aiohttp.ClientSession() as s:
        async with s.get(url) as r:
            return r.status


async def wait_status(url, want, timeout=5):
    for _ in range(int(timeout / 0.05)):
        try:
            if await status(url) == want:
                return True
        except aiohttp.ClientError:
            pass
        await asyncio.sleep(0.05)
    return False


async def test_serve_health_port_and_graceful_stop():
    port, hport = free_port(), free_port()
    s = make_settings(host="127.0.0.1", port=port, port_health=hport, shutdown_timeout_s=2)
    tts = FakeTTS()
    srv = MmeServer(s, tts=tts)
    task = asyncio.create_task(serve(srv, s))
    assert await wait_status(f"http://127.0.0.1:{hport}/ping", 200)      # health on its own port, unauthenticated
    assert await status(f"http://127.0.0.1:{port}/ping") == 200
    assert await status(f"http://127.0.0.1:{hport}/metrics") == 404       # health port exposes nothing else
    srv.on_fatal(0)
    assert await asyncio.wait_for(task, 10) == 0
    assert tts.engine._stop is True                                       # engine closed on shutdown


async def test_load_failure_exits_nonzero():
    port = free_port()
    s = make_settings(host="127.0.0.1", port=port, port_health=port, exit_delay_s=1.5, shutdown_timeout_s=1)

    def loader():
        raise FileNotFoundError("weights missing")
    srv = MmeServer(s, loader=loader)
    task = asyncio.create_task(serve(srv, s))
    assert await wait_status(f"http://127.0.0.1:{port}/ping", 503)        # unhealthy, never 200
    assert await asyncio.wait_for(task, 10) == 1                           # process would exit(1)
    assert "weights missing" in srv.load_error


async def test_active_stream_cancelled_on_shutdown():
    port = free_port()
    s = make_settings(host="127.0.0.1", port=port, port_health=port, shutdown_timeout_s=1)
    tts = FakeTTS(frame_s=0.02)
    srv = MmeServer(s, tts=tts)
    task = asyncio.create_task(serve(srv, s))
    assert await wait_status(f"http://127.0.0.1:{port}/ping", 200)
    sess = aiohttp.ClientSession()
    r = await sess.post(f"http://127.0.0.1:{port}/v1/audio/speech", json={"input": "#frames=1000", "response_format": "pcm"},
                        headers={"X-MME-Api-Key": next(iter(s.api_keys))})
    await r.content.readany()
    srv.on_fatal(0)
    await asyncio.wait_for(task, 10)
    assert tts.engine.requests[0].cancelled                                # in-flight engine work was cancelled
    await sess.close()
