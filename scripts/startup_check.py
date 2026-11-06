#!/usr/bin/env python3
"""Pre-flight checks run by entrypoint.sh BEFORE the server starts. Fails closed (exit 1) with a clear message.

Checks: settings/auth configuration, cache directory writable + free space, GPU visibility (when device=cuda),
whether the model is already cached. Never loads the model, never prints credentials.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.config import ConfigError, Settings  # noqa: E402

MIN_FREE_GB = 15            # release (~4 GB bf16 weights) + Mimi + ECAPA + triton cache, with headroom (estimate)


def check_cache(settings, problems, notes):
    cache = Path(settings.hf_home)
    try:
        cache.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=cache, prefix=".wtest-"):
            pass
    except OSError as exc:
        problems.append(f"HF_HOME {cache} is not writable ({exc}); mount a writable volume there")
        return
    free_gb = shutil.disk_usage(cache).free / 2**30
    notes.append(f"cache dir {cache}: {free_gb:.1f} GB free")
    snap = Path(settings.hf_hub_cache) / ("models--" + settings.model_id.replace("/", "--"))
    cached = snap.is_dir()
    notes.append(f"model cache present: {cached}")
    if not cached:
        if not settings.allow_download:
            problems.append("HF_HUB_OFFLINE is set but the model is not in the cache")
        elif free_gb < MIN_FREE_GB:
            problems.append(f"only {free_gb:.1f} GB free in {cache}; need about {MIN_FREE_GB} GB for the first download")


def check_gpu(settings, problems, notes):
    if not settings.device.startswith("cuda"):
        notes.append(f"device={settings.device} (no GPU check)")
        return
    try:
        import torch
    except Exception as exc:
        problems.append(f"torch import failed: {exc}")
        return
    if not torch.cuda.is_available():
        problems.append("device=cuda but no CUDA GPU is visible (start the container with the NVIDIA runtime / a GPU worker)")
        return
    p = torch.cuda.get_device_properties(0)
    notes.append(f"GPU: {p.name}, {p.total_memory / 2**30:.1f} GB, torch {torch.__version__}, cuda {torch.version.cuda}")
    if p.total_memory / 2**30 < 8:
        problems.append("upstream requires >= 8 GB GPU memory")


def main() -> int:
    problems: list[str] = []
    notes: list[str] = []
    try:
        settings = Settings.from_env()
    except ConfigError as exc:
        print(f"startup_check FAILED: {exc}", file=sys.stderr)
        return 1
    notes.append(f"profile={settings.profile} model={settings.model_id}@{settings.model_revision or 'main'} "
                 f"max_seats={settings.max_seats} max_pending={settings.max_pending} port={settings.port}")
    check_cache(settings, problems, notes)
    if os.environ.get("MME_SKIP_GPU_CHECK") != "1":
        check_gpu(settings, problems, notes)
    for n in notes:
        print(f"[startup_check] {n}")
    for p in problems:
        print(f"[startup_check] PROBLEM: {p}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
