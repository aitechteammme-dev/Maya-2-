"""In-process metrics: counters, latency summaries, GPU info. Rendered as Prometheus text or JSON.

Distinction (docs/PERFORMANCE_TESTING.md): everything here is ENGINE-SIDE time (request admitted -> audio produced).
Network, RunPod proxy and telephony latency are not visible to the server.
"""
from __future__ import annotations

import collections
import threading
import time


class Summary:
    """Bounded reservoir (last N samples) with count/sum and percentile snapshot."""

    def __init__(self, maxlen: int = 2048):
        self.samples = collections.deque(maxlen=maxlen)
        self.count = 0
        self.total = 0.0

    def observe(self, v: float) -> None:
        self.samples.append(v)
        self.count += 1
        self.total += v

    def snapshot(self) -> dict:
        s = sorted(self.samples)
        if not s:
            return {"count": self.count, "sum": self.total}
        pick = lambda q: s[min(len(s) - 1, int(q * len(s)))]
        return {"count": self.count, "sum": round(self.total, 6), "p50": pick(0.5), "p90": pick(0.9),
                "p99": pick(0.99), "max": s[-1]}


def gpu_info() -> dict:
    """GPU availability/memory/utilisation; never raises, never imports torch at module import time."""
    try:
        import torch
    except Exception:
        return {"available": False, "reason": "torch not importable"}
    try:
        if not torch.cuda.is_available():
            return {"available": False, "reason": "cuda not available"}
        free, total = torch.cuda.mem_get_info()
        info = {"available": True, "name": torch.cuda.get_device_name(0), "vram_total_mb": round(total / 2**20),
                "vram_used_mb": round((total - free) / 2**20),
                "torch_allocated_mb": round(torch.cuda.memory_allocated() / 2**20)}
        try:
            info["utilization_pct"] = torch.cuda.utilization()   # needs pynvml; optional
        except Exception:
            info["utilization_pct"] = None
        return info
    except Exception as exc:
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}


class Metrics:
    def __init__(self):
        self._lock = threading.Lock()
        self.started = time.time()
        self.counters = collections.Counter()
        self.summaries: dict[str, Summary] = collections.defaultdict(Summary)
        self.gauges: dict[str, float] = {}

    def inc(self, name: str, n: int = 1) -> None:
        with self._lock:
            self.counters[name] += n

    def observe(self, name: str, value: float) -> None:
        with self._lock:
            self.summaries[name].observe(value)

    def set_gauge(self, name: str, value: float) -> None:
        with self._lock:
            self.gauges[name] = value

    def record_completion(self, *, ttfa_s, audio_s, wall_s, reason, kind="utterance") -> None:
        """One finished synthesis (HTTP request, WS context or incremental chunk)."""
        self.inc("synth_completed")
        if reason == "cancelled":
            self.inc("synth_cancelled")
        if ttfa_s is not None:
            self.observe("ttfa_seconds", ttfa_s)
        if audio_s:
            self.observe("audio_seconds", audio_s)
            if wall_s:
                self.observe("rtf", wall_s / audio_s)     # <1 means faster than real time
                self.observe("synth_wall_seconds", wall_s)

    def snapshot(self, upstream_stats: dict | None = None) -> dict:
        with self._lock:
            return {"uptime_s": round(time.time() - self.started, 1), "counters": dict(self.counters),
                    "gauges": dict(self.gauges), "summaries": {k: v.snapshot() for k, v in self.summaries.items()},
                    "upstream": dict(upstream_stats or {}), "gpu": gpu_info()}

    def render_prometheus(self, upstream_stats: dict | None = None) -> str:
        snap = self.snapshot(upstream_stats)
        out = []
        for k, v in sorted(snap["counters"].items()):
            out += [f"# TYPE mme_tts_{k}_total counter", f"mme_tts_{k}_total {v}"]
        for k, v in sorted(snap["gauges"].items()):
            out += [f"# TYPE mme_tts_{k} gauge", f"mme_tts_{k} {v}"]
        for k, v in sorted((snap["upstream"] or {}).items()):
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                out += [f"# TYPE mme_tts_upstream_{k} gauge", f"mme_tts_upstream_{k} {v}"]
        for k, s in sorted(snap["summaries"].items()):
            out.append(f"# TYPE mme_tts_{k} summary")
            for q, key in (("0.5", "p50"), ("0.9", "p90"), ("0.99", "p99")):
                if key in s:
                    out.append(f'mme_tts_{k}{{quantile="{q}"}} {s[key]}')
            out += [f"mme_tts_{k}_sum {s['sum']}", f"mme_tts_{k}_count {s['count']}"]
        g = snap["gpu"]
        if g.get("available"):
            for key in ("vram_total_mb", "vram_used_mb", "torch_allocated_mb"):
                out.append(f"mme_tts_gpu_{key} {g[key]}")
            if g.get("utilization_pct") is not None:
                out.append(f"mme_tts_gpu_utilization_pct {g['utilization_pct']}")
        out.append(f"mme_tts_gpu_available {1 if g.get('available') else 0}")
        return "\n".join(out) + "\n"
