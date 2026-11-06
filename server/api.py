"""MME deployment server: upstream maya3.server.Maya3Server + auth, /ping, /metrics, tenant isolation, an incremental-text
endpoint, fail-closed model loading and graceful shutdown.

Run:  python -m server.api        (see scripts/entrypoint.sh)

Upstream routes are inherited unchanged (POST /v1/audio/speech, WS /v1/tts/stream, GET/POST/DELETE /v1/voices,
GET /v1/models); this module only adds middleware and the routes listed in docs/API_REFERENCE.md.
"""
from __future__ import annotations

import asyncio
import json
import logging
import signal
import sys
import time

from aiohttp import web

from maya3.server import ApiError, Maya3Server

from .auth import current_is_admin, current_tenant, make_auth_middleware, voice_allowed
from .config import ConfigError, Settings
from .errors import error_response
from .health import health_payload, ping_status
from .incremental import IncrementalSession
from .metrics import Metrics

log = logging.getLogger("mme.server")


class MmeServer(Maya3Server):
    def __init__(self, settings: Settings, tts=None, loader=None):
        super().__init__(tts, max_pending=settings.max_pending, queue_deadline_ms=settings.queue_deadline_ms,
                         ws_max_contexts=settings.ws_max_contexts, ws_max_lag_s=settings.ws_max_lag_s,
                         http_max_lag_s=settings.http_max_lag_s, loader=loader or self._real_loader,
                         normalise=settings.normalise)
        self.settings = settings
        self.metrics = Metrics()
        self.phase = "ready" if tts is not None else "starting"
        self.load_seconds: float | None = None
        self.incremental_sessions = 0
        self.draining = False
        self.on_fatal = None                    # callback(exit_code) set by the runner

    # ------------------------------------------------------------------ model loading (fail closed)
    def _real_loader(self):
        from .model_store import resolve_and_verify
        s = self.settings
        self.phase = "downloading"
        paths = resolve_and_verify(s)           # raises -> load_error -> /ping 503 -> process exits
        self.phase = "loading"
        from maya3.api import Maya3TTS
        from maya3.server import load_voices_file
        cfg = {"max_seats": s.max_seats}
        if s.capacity:
            cfg["capacity"] = s.capacity
        tts = Maya3TTS.from_pretrained(paths["ckpt"], paths["backbone_dir"], paths["mimi_dir"],
                                       ecapa_dir=paths["ecapa_dir"], device=s.device, **cfg)
        # from_pretrained(<dir>) path: named voices are only auto-loaded when backbone_dir is None, so do it here
        if paths.get("voices_json"):
            tts.load_named_voices(paths["voices_json"])
        if s.voices_file:
            load_voices_file(tts, s.voices_file)
        return tts

    async def _load(self):
        t0 = time.perf_counter()
        await super()._load()
        self.load_seconds = round(time.perf_counter() - t0, 2)
        if self.load_error is not None:
            self.phase = "failed"
            self.metrics.inc("model_load_failures")
            log.error("model load FAILED after %.1fs: %s", self.load_seconds, self.load_error)
            if self.settings.exit_on_load_failure and self.on_fatal:
                asyncio.get_running_loop().call_later(self.settings.exit_delay_s, self.on_fatal, 1)
            return
        # upstream engine.start() warms in its own thread; /ping stays 204 until engine.health()["ready"]
        self.phase = "warming"
        self.metrics.observe("model_load_seconds", self.load_seconds)
        asyncio.create_task(self._watch_ready(t0))

    async def _watch_ready(self, t0):
        while True:
            try:
                st = self.engine.health()
            except Exception:
                st = {}
            if st.get("failed"):
                self.phase = "failed"
                if self.settings.exit_on_load_failure and self.on_fatal:
                    asyncio.get_running_loop().call_later(self.settings.exit_delay_s, self.on_fatal, 1)
                return
            if st.get("ready"):
                self.phase = "ready"
                self.metrics.observe("model_ready_seconds", time.perf_counter() - t0)
                log.info("READY after %.1fs (load %.1fs)", time.perf_counter() - t0, self.load_seconds or 0)
                return
            await asyncio.sleep(0.25)

    # ------------------------------------------------------------------ routes
    def app(self):
        app = super().app()
        app.middlewares.append(make_auth_middleware(self.settings, self.metrics))
        app.router.add_get("/ping", self.ping)
        app.router.add_get("/metrics", self.metrics_route)
        if self.settings.incremental_enabled:
            app.router.add_get("/v1/tts/incremental", self.incremental)
        return app

    def health_app(self):
        """Tiny app for PORT_HEALTH when it differs from PORT (RunPod LB can poll a separate port)."""
        app = web.Application()
        app.router.add_get("/ping", self.ping)
        app.router.add_get("/health", self.health)
        return app

    async def ping(self, request):
        return web.Response(status=ping_status(self), headers={"Cache-Control": "no-store"})

    async def health(self, request):
        body, status = health_payload(self)
        return web.json_response(body, status=status, headers={"Cache-Control": "no-store"})

    async def metrics_route(self, request):
        upstream = dict(self.stats)
        try:
            if self.tts is not None:
                upstream.update({k: v for k, v in self.engine.health().items() if isinstance(v, (int, float))})
        except Exception:
            pass
        upstream.update(waiting=len(self._waiting), jobs=len(self._jobs), ws_sockets=self.sockets,
                        incremental_sessions=self.incremental_sessions)
        if "json" in request.query:
            return web.json_response(self.metrics.snapshot(upstream))
        return web.Response(text=self.metrics.render_prometheus(upstream), content_type="text/plain")

    # ------------------------------------------------------------------ tenant isolation
    def resolve_voice(self, voice):
        vid = super().resolve_voice(voice)
        if not voice_allowed(vid, current_tenant.get(), current_is_admin.get()):
            raise ApiError(403, "forbidden", f"voice {vid!r} belongs to another tenant", "voice")
        return vid

    async def list_voices(self, request):
        resp = await super().list_voices(request)
        if resp.status != 200:
            return resp
        data = json.loads(resp.body)
        tenant, admin = request.get("tenant"), current_is_admin.get()
        data["voices"] = [v for v in data["voices"] if voice_allowed(v["id"], tenant, admin)]
        return web.json_response(data)

    # ------------------------------------------------------------------ websockets
    async def ws(self, request):
        if self.sockets + self.incremental_sessions >= self.settings.max_ws_connections:
            self.metrics.inc("ws_rejected_limit")
            return error_response(503, "overloaded", "too many WebSocket connections", headers={"Retry-After": "1"})
        return await super().ws(request)

    async def incremental(self, request):
        if self.tts is None or self.load_error:
            return error_response(503, "unavailable", self.load_error or "model is loading", headers={"Retry-After": "5"})
        if self.sockets + self.incremental_sessions >= self.settings.max_ws_connections:
            self.metrics.inc("ws_rejected_limit")
            return error_response(503, "overloaded", "too many WebSocket connections", headers={"Retry-After": "1"})
        ws = web.WebSocketResponse(heartbeat=30, max_msg_size=64 * 1024)
        await ws.prepare(request)
        self.incremental_sessions += 1
        self.metrics.inc("incremental_sessions_started")
        sess = IncrementalSession(self, ws, request)
        try:
            await sess.run()
        finally:
            self.incremental_sessions -= 1
            if not ws.closed:
                await ws.close()
        return ws

    # ------------------------------------------------------------------ metrics hooks
    @staticmethod
    def _ttfa_ms(job, info):
        if info.get("ttfa_s") is not None:
            return round(info["ttfa_s"] * 1000, 1)
        return round((job.t_first - job.t_submit) * 1000, 1) if job.t_first else None

    def _log_end(self, rid, via, lang, voice, chars, job, info):
        super()._log_end(rid, via, lang, voice, chars, job, info)
        ttfa = (job.t_first - job.t_submit) if job.t_first else None
        wall = info.get("wall_s") or (time.perf_counter() - job.t_submit)
        self.metrics.record_completion(ttfa_s=info.get("ttfa_s", ttfa), audio_s=info.get("audio_s"), wall_s=wall,
                                       reason=info.get("reason"))
        self.metrics.inc(f"synth_via_{via}")


# ---------------------------------------------------------------------------------------------------- runner
async def serve(srv: MmeServer, settings: Settings) -> int:
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    exit_code = {"v": 0}

    def request_stop(code=0):
        exit_code["v"] = code
        stop.set()

    srv.on_fatal = request_stop
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, request_stop, 0)
        except NotImplementedError:                      # Windows (dev only)
            pass
    runner = web.AppRunner(srv.app(), access_log=None, handler_cancellation=True)
    await runner.setup()
    sites = [web.TCPSite(runner, settings.host, settings.port, shutdown_timeout=settings.shutdown_timeout_s)]
    health_runner = None
    if settings.port_health != settings.port:
        health_runner = web.AppRunner(srv.health_app(), access_log=None)
        await health_runner.setup()
        sites.append(web.TCPSite(health_runner, settings.host, settings.port_health, shutdown_timeout=5))
    for s in sites:
        await s.start()
    log.info("listening on %s:%d (health port %d), profile=%s, max_seats=%d, max_pending=%d", settings.host, settings.port,
             settings.port_health, settings.profile, settings.max_seats, settings.max_pending)
    await stop.wait()
    log.info("shutdown requested (exit code %d): draining up to %.0fs", exit_code["v"], settings.shutdown_timeout_s)
    srv.phase = "failed" if exit_code["v"] else "starting"   # /ping stops reporting 200 while draining
    srv.draining = True
    await runner.cleanup()                                   # closes jobs, then tts.close() (upstream _cleanup)
    if health_runner:
        await health_runner.cleanup()
    return exit_code["v"]


def main(argv=None) -> int:
    try:
        settings = Settings.from_env()
    except ConfigError as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return 2
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("aiohttp.access").setLevel(logging.WARNING)
    srv = MmeServer(settings)
    return asyncio.run(serve(srv, settings))


if __name__ == "__main__":
    sys.exit(main())
