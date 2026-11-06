#!/usr/bin/env python3
"""Pre-populate (or verify) the persistent model cache WITHOUT a GPU - e.g. once on a CPU pod attached to the network
volume, so GPU workers start without downloading.

    python scripts/download_model.py            # download (resumable) + verify
    python scripts/download_model.py --verify-only   # offline verification of an existing cache

Downloads (upstream maya3.hub): maya-research/maya2-native @ MODEL_REVISION, kyutai/mimi, speechbrain/spkrec-ecapa-voxceleb.
Only HF_TOKEN from the environment is used if set (the models are public); it is never printed or stored.
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.config import ConfigError, Settings  # noqa: E402
from server.model_store import ModelVerificationError, resolve_and_verify  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-only", action="store_true", help="never touch the network")
    args = ap.parse_args()
    logging.basicConfig(level="INFO", format="%(levelname)s %(name)s: %(message)s")
    try:
        # auth keys are irrelevant here
        import os
        os.environ.setdefault("MME_AUTH_DISABLED", "1")
        os.environ.setdefault("MME_PROFILE", "development")
        settings = Settings.from_env()
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    if args.verify_only:
        settings = dataclasses.replace(settings, allow_download=False)
    try:
        paths = resolve_and_verify(settings)
    except (ModelVerificationError, FileNotFoundError, OSError) as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    print("OK:")
    for k in ("ckpt", "backbone_dir", "mimi_dir", "ecapa_dir"):
        print(f"  {k}: {paths[k]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
