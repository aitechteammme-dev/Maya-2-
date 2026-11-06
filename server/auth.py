"""Bearer-token authentication, tenant binding and request-size limits (aiohttp middleware).

Upstream maya3.server has NO authentication (docs/UPSTREAM_AUDIT.md), so this layer is the only gate.

Credentials are accepted from `Authorization: Bearer <token>` or `X-MME-Api-Key: <token>`. The second header exists
because RunPod load-balancing endpoints use the Authorization header for the RunPod API key; whether RunPod forwards
that header to the worker is UNVERIFIED (docs/RUNPOD_DEPLOYMENT.md) - clients should send both if in doubt.
Tokens are never logged. Query-string tokens are deliberately not supported (they end up in proxy logs).
"""
from __future__ import annotations

import contextvars
import hashlib
import hmac

from aiohttp import web

from .errors import forbidden, too_large, unauthorized

PUBLIC_PATHS = frozenset({"/ping", "/health"})
ADMIN_ROUTES = (("POST", "/v1/voices"), ("DELETE", "/v1/voices/"))
current_tenant: contextvars.ContextVar[str | None] = contextvars.ContextVar("mme_tenant", default=None)
current_is_admin: contextvars.ContextVar[bool] = contextvars.ContextVar("mme_is_admin", default=False)


def _digest(s: str) -> bytes:
    return hashlib.sha256(s.encode("utf-8")).digest()


def match_token(presented: str, table: dict[str, str]) -> str | None:
    """Constant-time lookup: compares sha256 digests of every configured token (no early exit, length-independent)."""
    p = _digest(presented)
    found = None
    for token, tenant in table.items():
        if hmac.compare_digest(p, _digest(token)):
            found = tenant
    return found


def extract_token(request: web.Request) -> str | None:
    h = request.headers.get("Authorization", "")
    if h[:7].lower() == "bearer " and h[7:].strip():
        return h[7:].strip()
    k = request.headers.get("X-MME-Api-Key", "").strip()
    return k or None


def is_admin_route(method: str, path: str) -> bool:
    return any(method == m and (path == p or (p.endswith("/") and path.startswith(p))) for m, p in ADMIN_ROUTES)


def make_auth_middleware(settings, metrics=None):
    @web.middleware
    async def auth_middleware(request: web.Request, handler):
        path = request.path
        if request.method == "OPTIONS" or path in PUBLIC_PATHS:
            return await handler(request)
        # size limit first (cheap, and applies to unauthenticated junk too)
        limit = settings.max_voice_upload_bytes if (request.method == "POST" and path == "/v1/voices") \
            else settings.max_body_bytes
        if request.content_length is not None and request.content_length > limit:
            return too_large(limit)

        tenant, admin = None, False
        if settings.auth_disabled:
            tenant, admin = "default", True
        else:
            token = extract_token(request)
            if token is None:
                if metrics:
                    metrics.inc("auth_failures")
                return unauthorized()
            admin_tenant = match_token(token, settings.admin_keys)
            tenant = match_token(token, settings.api_keys)
            admin = admin_tenant is not None
            if tenant is None and admin:
                tenant = admin_tenant
            if tenant is None:
                if metrics:
                    metrics.inc("auth_failures")
                return unauthorized()
        if is_admin_route(request.method, path):
            if not settings.allow_voice_registration:
                return forbidden("runtime voice registration is disabled on this deployment")
            if not admin:
                return forbidden("voice registration requires an admin credential")
        t1 = current_tenant.set(tenant)
        t2 = current_is_admin.set(admin)
        request["tenant"] = tenant
        try:
            return await handler(request)
        finally:
            current_tenant.reset(t1)
            current_is_admin.reset(t2)
    return auth_middleware


TENANT_SEP = "__"


def voice_allowed(voice_id: str, tenant: str | None, admin: bool) -> bool:
    """Voices named '<tenant>__<name>' belong to that tenant; everything else (plain, built-in named voices) is shared."""
    if admin or TENANT_SEP not in voice_id:
        return True
    return voice_id.split(TENANT_SEP, 1)[0] == tenant
