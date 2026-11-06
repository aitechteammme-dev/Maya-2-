"""Deployment settings. Everything is read from environment variables (optionally pre-seeded from a profile YAML).

Variables here are CUSTOM to this deployment wrapper (upstream maya3.server only has CLI flags, no env vars -
see docs/UPSTREAM_AUDIT.md). HF_HOME / HF_HUB_OFFLINE are standard huggingface_hub variables.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

PROFILES = ("development", "production", "benchmark")
# Pinned upstream model revision (HF commit sha observed during the audit, 2026-11-06).
DEFAULT_MODEL_REVISION = "87a83efdd6551e038246b6b37836a6264713eec3"


class ConfigError(ValueError):
    pass


def _bool(v: str | None, default: bool) -> bool:
    if v is None or v == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _int(env: Mapping[str, str], key: str, default: int, lo: int = 0, hi: int | None = None) -> int:
    raw = env.get(key)
    if raw in (None, ""):
        return default
    try:
        v = int(raw)
    except ValueError:
        raise ConfigError(f"{key}={raw!r} is not an integer")
    if v < lo or (hi is not None and v > hi):
        raise ConfigError(f"{key}={v} outside {lo}..{hi if hi is not None else 'inf'}")
    return v


def _float(env: Mapping[str, str], key: str, default: float, lo: float = 0.0) -> float:
    raw = env.get(key)
    if raw in (None, ""):
        return default
    try:
        v = float(raw)
    except ValueError:
        raise ConfigError(f"{key}={raw!r} is not a number")
    if v < lo:
        raise ConfigError(f"{key}={v} must be >= {lo}")
    return v


def parse_keys(raw: str | None) -> dict[str, str]:
    """'tenantA=tokenA,tokenB' -> {token: tenant}. A bare token belongs to tenant 'default'."""
    out: dict[str, str] = {}
    for item in (raw or "").split(","):
        item = item.strip()
        if not item:
            continue
        tenant, sep, token = item.partition("=")
        if not sep:
            tenant, token = "default", item
        tenant, token = tenant.strip(), token.strip()
        if not token or not tenant:
            raise ConfigError("malformed key entry (expected 'tenant=token' or 'token')")
        if len(token) < 16:
            raise ConfigError("API keys must be at least 16 characters")
        out[token] = tenant
    return out


def load_profile(profile: str, config_dir: Path) -> dict[str, str]:
    path = config_dir / f"{profile}.yaml"
    if not path.is_file():
        return {}
    import yaml
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    env = data.get("env", data)
    return {str(k): str(v) for k, v in env.items() if v is not None}


@dataclass
class Settings:
    profile: str = "production"
    model_id: str = "maya-research/maya2-native"
    model_revision: str | None = DEFAULT_MODEL_REVISION
    hf_home: str = "/workspace/.cache/huggingface"
    host: str = "0.0.0.0"
    port: int = 8080
    port_health: int = 8080
    max_seats: int = 1
    max_pending: int = 4
    capacity: int | None = None
    queue_deadline_ms: float = 2000.0
    ws_max_contexts: int = 8
    ws_max_lag_s: float = 20.0
    http_max_lag_s: float = 120.0
    device: str = "cuda"
    normalise: bool = True
    log_level: str = "INFO"
    voices_file: str | None = None
    # security
    api_keys: dict = field(default_factory=dict)       # token -> tenant
    admin_keys: dict = field(default_factory=dict)
    auth_disabled: bool = False
    allow_voice_registration: bool = False
    max_body_bytes: int = 64 * 1024
    max_voice_upload_bytes: int = 8 * 1024 * 1024
    max_ws_connections: int = 32
    # lifecycle
    exit_on_load_failure: bool = True
    exit_delay_s: float = 10.0
    shutdown_timeout_s: float = 30.0
    allow_download: bool = True
    # incremental text adapter (custom endpoint)
    incremental_enabled: bool = True
    inc_first_min_chars: int = 12
    inc_min_chars: int = 24
    inc_clause_min_chars: int = 60
    inc_max_chars: int = 240
    inc_idle_flush_ms: int = 350
    inc_max_session_s: int = 1800
    inc_max_session_chars: int = 20000

    @property
    def hf_hub_cache(self) -> str:
        return str(Path(self.hf_home) / "hub")

    def validate(self) -> None:
        if self.profile not in PROFILES:
            raise ConfigError(f"MME_PROFILE must be one of {PROFILES}")
        if self.auth_disabled and self.profile == "production":
            raise ConfigError("MME_AUTH_DISABLED is refused in the production profile")
        if not self.auth_disabled and not self.api_keys:
            raise ConfigError("no API keys configured: set MME_API_KEYS (or MME_AUTH_DISABLED=1 outside production)")
        if self.allow_voice_registration and not self.admin_keys:
            raise ConfigError("MME_ALLOW_VOICE_REGISTRATION=1 needs MME_ADMIN_KEYS")
        if self.max_pending < 1 or self.max_seats < 1:
            raise ConfigError("MAX_SEATS and MAX_PENDING must be >= 1")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None, config_dir: Path | None = None) -> "Settings":
        base = dict(environ if environ is not None else os.environ)
        profile = base.get("MME_PROFILE", "production")
        cdir = config_dir or Path(base.get("MME_CONFIG_DIR", Path(__file__).resolve().parent.parent / "config"))
        env = {**load_profile(profile, cdir), **base}          # real environment wins over the profile file
        port = _int(env, "PORT", 8080, 1, 65535)
        revision = env.get("MODEL_REVISION", DEFAULT_MODEL_REVISION) or None
        s = cls(
            profile=profile,
            model_id=env.get("MODEL_ID", "maya-research/maya2-native"),
            model_revision=revision,
            hf_home=env.get("HF_HOME", "/workspace/.cache/huggingface"),
            host=env.get("HOST", "0.0.0.0"),
            port=port,
            port_health=_int(env, "PORT_HEALTH", port, 1, 65535),
            max_seats=_int(env, "MAX_SEATS", 1, 1, 512),
            max_pending=_int(env, "MAX_PENDING", 4, 1, 4096),
            capacity=_int(env, "MAYA_CAPACITY", 0, 0, 8192) or None,
            queue_deadline_ms=_float(env, "QUEUE_DEADLINE_MS", 2000.0, 1.0),
            ws_max_contexts=_int(env, "WS_MAX_CONTEXTS", 8, 1, 64),
            ws_max_lag_s=_float(env, "WS_MAX_LAG_S", 20.0, 1.0),
            http_max_lag_s=_float(env, "HTTP_MAX_LAG_S", 120.0, 1.0),
            device=env.get("MAYA_DEVICE", "cuda"),
            normalise=_bool(env.get("MAYA_NORMALISE"), True),
            log_level=env.get("LOG_LEVEL", "INFO").upper(),
            voices_file=env.get("VOICES_FILE") or None,
            api_keys=parse_keys(env.get("MME_API_KEYS")),
            admin_keys=parse_keys(env.get("MME_ADMIN_KEYS")),
            auth_disabled=_bool(env.get("MME_AUTH_DISABLED"), False),
            allow_voice_registration=_bool(env.get("MME_ALLOW_VOICE_REGISTRATION"), False),
            max_body_bytes=_int(env, "MME_MAX_BODY_BYTES", 64 * 1024, 1024),
            max_voice_upload_bytes=_int(env, "MME_MAX_VOICE_UPLOAD_BYTES", 8 * 1024 * 1024, 1024),
            max_ws_connections=_int(env, "MME_MAX_WS_CONNECTIONS", 32, 1),
            exit_on_load_failure=_bool(env.get("MME_EXIT_ON_LOAD_FAILURE"), True),
            exit_delay_s=_float(env, "MME_EXIT_DELAY_S", 10.0),
            shutdown_timeout_s=_float(env, "SHUTDOWN_TIMEOUT_S", 30.0),
            allow_download=not _bool(env.get("HF_HUB_OFFLINE"), False),
            incremental_enabled=_bool(env.get("MME_INCREMENTAL"), True),
            inc_first_min_chars=_int(env, "MME_INC_FIRST_MIN_CHARS", 12, 1),
            inc_min_chars=_int(env, "MME_INC_MIN_CHARS", 24, 1),
            inc_clause_min_chars=_int(env, "MME_INC_CLAUSE_MIN_CHARS", 60, 1),
            inc_max_chars=_int(env, "MME_INC_MAX_CHARS", 240, 20, 3000),
            inc_idle_flush_ms=_int(env, "MME_INC_IDLE_FLUSH_MS", 350, 0),
            inc_max_session_s=_int(env, "MME_INC_MAX_SESSION_S", 1800, 1),
            inc_max_session_chars=_int(env, "MME_INC_MAX_SESSION_CHARS", 20000, 1),
        )
        s.validate()
        return s
