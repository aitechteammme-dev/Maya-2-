"""CPU-only fixtures. Upstream's FakeEngine (maya3.tests.fakes) stands in for the GPU engine - no inference is run.

Upstream source location: $MAYA3_SRC (default ./upstream_src/maya2-native, see README 'Clone the upstream source').
"""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = Path(os.environ.get("MAYA3_SRC", ROOT / "upstream_src" / "maya2-native"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
for p in (str(ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)
if not (SRC / "maya3" / "server.py").is_file():
    pytest.exit(f"upstream source not found at {SRC}; set MAYA3_SRC or clone it (README)", 2)

from maya3.tests.fakes import FakeTTS  # noqa: E402
from server.api import MmeServer  # noqa: E402
from server.config import Settings  # noqa: E402

KEY = "test-key-tenant-a-0123456789"
KEY_B = "test-key-tenant-b-0123456789"
ADMIN = "test-admin-key-0123456789"
AUTH = {"Authorization": f"Bearer {KEY}"}


def make_settings(**kw):
    base = dict(profile="development", api_keys={KEY: "acme", KEY_B: "beta"}, admin_keys={ADMIN: "ops"},
                normalise=False, exit_delay_s=0.01, inc_idle_flush_ms=0)
    base.update(kw)
    s = Settings(**base)
    s.validate()
    return s


@pytest.fixture
def make(aiohttp_client):
    made = []

    async def _make(settings=None, tts="fake", loader=None, engine=None, **settings_kw):
        s = settings or make_settings(**settings_kw)
        t = FakeTTS(**(engine or {})) if tts == "fake" else tts
        srv = MmeServer(s, tts=t, loader=loader)
        client = await aiohttp_client(srv.app())
        made.append(t)
        return client, srv, t
    return _make
