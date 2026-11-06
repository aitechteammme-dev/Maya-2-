"""Error responses in the same JSON shape upstream maya3.server uses: {"error": {message, type, code, param}}."""
from __future__ import annotations

from aiohttp import web


def error_response(status: int, code: str, message: str, param: str | None = None, headers: dict | None = None):
    kind = "invalid_request_error" if status < 500 else "server_error"
    if status in (401, 403):
        kind = "authentication_error"
    return web.json_response({"error": {"message": message, "type": kind, "code": code, "param": param}},
                             status=status, headers=headers)


def unauthorized():
    return error_response(401, "unauthorized", "missing or invalid credentials",
                          headers={"WWW-Authenticate": "Bearer"})


def forbidden(message="not permitted for this credential"):
    return error_response(403, "forbidden", message)


def too_large(limit: int):
    return error_response(413, "payload_too_large", f"request body exceeds {limit} bytes")
