"""Model asset resolution + verification (CPU-only; no torch needed).

Uses upstream maya3.hub.resolve (offline-first, resumable snapshot_download) and then verifies the files so a
truncated/partial download fails closed BEFORE the engine tries to load it.

Assets (docs/UPSTREAM_AUDIT.md section 6): the maya2-native release (2 safetensors shards + index + config +
backbone/ tokenizer files + voices/voices.json), kyutai/mimi (config.json, model.safetensors) and
speechbrain/spkrec-ecapa-voxceleb (embedding_model.ckpt).
"""
from __future__ import annotations

import json
import logging
import os
import struct
from pathlib import Path

log = logging.getLogger("mme.model_store")


class ModelVerificationError(RuntimeError):
    pass


def safetensors_expected_size(path: Path) -> int:
    """Size a well-formed .safetensors file must have, from its header. Raises ModelVerificationError if unreadable."""
    try:
        with open(path, "rb") as f:
            head = f.read(8)
            if len(head) < 8:
                raise ModelVerificationError(f"{path.name}: shorter than the 8-byte header")
            (n,) = struct.unpack("<Q", head)
            if n > 100 * 1024 * 1024:
                raise ModelVerificationError(f"{path.name}: implausible header length {n}")
            raw = f.read(n)
            if len(raw) < n:
                raise ModelVerificationError(f"{path.name}: truncated header")
        meta = json.loads(raw)
    except (OSError, ValueError) as exc:
        raise ModelVerificationError(f"{path.name}: cannot read safetensors header ({exc})")
    end = 0
    for k, v in meta.items():
        if k == "__metadata__":
            continue
        end = max(end, int(v["data_offsets"][1]))
    return 8 + n + end


def verify_safetensors(path: Path) -> None:
    expected = safetensors_expected_size(path)
    actual = path.stat().st_size
    if actual != expected:
        raise ModelVerificationError(f"{path.name}: size {actual} != {expected} expected from its header "
                                     "(truncated or corrupt download)")


def verify_release(model_dir: str | Path) -> list[str]:
    """Verify a Maya3 release dir. Returns the list of verified files; raises ModelVerificationError listing problems."""
    d = Path(model_dir)
    problems: list[str] = []
    verified: list[str] = []
    for rel in ("maya3_config.json", "backbone/config.json", "backbone/tokenizer.json"):
        if not (d / rel).is_file():
            problems.append(f"missing {rel}")
    shards: list[str] = []
    idx = d / "model.safetensors.index.json"
    if idx.is_file():
        try:
            shards = sorted(set(json.loads(idx.read_text())["weight_map"].values()))
        except (ValueError, KeyError) as exc:
            problems.append(f"model.safetensors.index.json unreadable ({exc})")
    elif (d / "model.safetensors").is_file():
        shards = ["model.safetensors"]
    else:
        problems.append("no model.safetensors / model.safetensors.index.json")
    for s in shards:
        p = d / s
        if not p.is_file():
            problems.append(f"missing shard {s}")
            continue
        try:
            verify_safetensors(p)
            verified.append(s)
        except ModelVerificationError as exc:
            problems.append(str(exc))
    if problems:
        raise ModelVerificationError(f"model at {d} failed verification: " + "; ".join(problems))
    return verified


def verify_component(path: str | Path, files: tuple[str, ...], name: str) -> None:
    missing = [f for f in files if not (Path(path) / f).is_file()]
    if missing:
        raise ModelVerificationError(f"{name} at {path} is missing {missing}")
    for f in files:
        if f.endswith(".safetensors"):
            verify_safetensors(Path(path) / f)


def resolve_and_verify(settings) -> dict:
    """Resolve (download if allowed) and verify every asset. Returns the maya3.hub.resolve dict."""
    from maya3 import hub          # upstream; stdlib + huggingface_hub only at import time

    os.environ.setdefault("HF_HOME", settings.hf_home)
    Path(settings.hf_hub_cache).mkdir(parents=True, exist_ok=True)
    local_only = None if settings.allow_download else True
    paths = hub.resolve(settings.model_id, cache_dir=settings.hf_hub_cache, revision=settings.model_revision,
                        local_files_only=local_only)
    verified = verify_release(paths["ckpt"])
    verify_component(paths["mimi_dir"], hub.MIMI_FILES, "kyutai/mimi")
    verify_component(paths["ecapa_dir"], hub.ECAPA_FILES, "speechbrain/spkrec-ecapa-voxceleb")
    log.info("model assets verified: %s shards=%s mimi=%s ecapa=%s", paths["ckpt"], verified, paths["mimi_dir"],
             paths["ecapa_dir"])
    return paths
