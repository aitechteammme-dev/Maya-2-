"""Readiness/health payloads. /ping follows RunPod LB semantics: 200 ready, 204 initialising, 503 failed.

(RunPod docs: https://docs.runpod.io/serverless/load-balancing/overview - 200 healthy, 204 initializing, other
unhealthy.) "Ready" means: model loaded AND engine reports ready (warm, CUDA graphs captured, not failed).
"""
from __future__ import annotations

import time

from .metrics import gpu_info

PHASES = ("starting", "downloading", "loading", "ready", "failed")


def health_payload(srv, detail: bool = False) -> tuple[dict, int]:
    ready, failed = False, srv.load_error
    engine_state: dict = {}
    if srv.tts is not None:
        try:
            engine_state = dict(srv.engine.health())
            ready = bool(engine_state.get("ready"))
            failed = failed or engine_state.get("failed")
        except Exception as exc:
            failed = f"health: {type(exc).__name__}: {exc}"
    if failed:
        status = "failed"
    elif ready:
        status = "ready"
    else:
        status = "initializing"
    max_seats = engine_state.get("max_seats", srv.settings.max_seats)
    active = engine_state.get("active", 0)
    body = {
        "status": status,
        "model_ready": ready,
        "model_id": srv.settings.model_id,
        "initialization": {"phase": "failed" if failed else srv.phase, "load_seconds": srv.load_seconds,
                           "error": failed},
        "gpu": gpu_info() if srv.tts is not None else {"available": None, "reason": "model not loaded yet"},
        "sessions": {"active_synthesis": active, "pending": engine_state.get("pending", len(srv._waiting)),
                     "ws_sockets": srv.sockets, "incremental_sessions": srv.incremental_sessions},
        "capacity": {"max_seats": max_seats, "available_seats": max(0, max_seats - active),
                     "max_pending": srv.max_pending},
        "uptime_s": round(time.time() - srv.metrics.started, 1),
    }
    return body, (200 if ready else 503)


def ping_status(srv) -> int:
    if getattr(srv, "draining", False):
        return 503
    body, _ = health_payload(srv)
    return {"ready": 200, "initializing": 204}.get(body["status"], 503)
