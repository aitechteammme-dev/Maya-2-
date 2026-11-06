#!/usr/bin/env python3
"""Repeatable reference-voice registration (e.g. MME's authorised 'Utkarsh' recording).

The recording is read from a path YOU provide at run time; it is never committed or baked into the image.

Mode A - persistent (recommended): write the clip + manifest entry into the mounted voices directory. The server loads
          the manifest at startup via VOICES_FILE (upstream --voices), so the voice survives restarts/new workers.
    python scripts/register_voice.py persist --voices-dir /workspace/voices --tenant mme --name utkarsh \
        --audio ./utkarsh_ref.wav --transcript-file ./utkarsh_ref.txt --lang hi

Mode B - runtime: POST to a running server (admin key from env MME_ADMIN_KEY; the server must run with
          MME_ALLOW_VOICE_REGISTRATION=1). In-memory only: lost when the worker restarts.
    MME_ADMIN_KEY=... python scripts/register_voice.py runtime --url https://HOST --tenant mme --name utkarsh ...

Both modes validate duration (3-10 s), format (wav/flac), sample rate (>=16 kHz), non-silence, transcript and language
first. `check` validates an existing manifest. Nothing here evaluates voice similarity.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.voice_manager import (VoiceValidationError, upsert_manifest_entry, validate_manifest,  # noqa: E402
                                  validate_reference_audio, validate_registration)


def read_transcript(a) -> str:
    if a.transcript_file:
        return Path(a.transcript_file).read_text(encoding="utf-8").strip()
    return (a.transcript or "").strip()


def common(a):
    voice_id = f"{a.tenant}__{a.name}"
    text = read_transcript(a)
    validate_registration(voice_id=voice_id, ref_text=text, lang=a.lang, mode=a.mode, tenant=a.tenant)
    data = Path(a.audio).read_bytes()
    if Path(a.audio).suffix.lower() not in (".wav", ".flac"):
        raise VoiceValidationError("reference file must be .wav or .flac")
    info = validate_reference_audio(data)
    print(f"reference ok: {info.duration_s}s @ {info.sample_rate} Hz, {info.channels}ch, rms {info.rms}")
    return voice_id, text, data


def persist(a) -> int:
    voice_id, text, data = common(a)
    vdir = Path(a.voices_dir)
    (vdir / "refs").mkdir(parents=True, exist_ok=True)
    rel = f"refs/{voice_id}{Path(a.audio).suffix.lower()}"
    (vdir / rel).write_bytes(data)
    try:
        os.chmod(vdir / rel, 0o640)
    except OSError:
        pass
    upsert_manifest_entry(vdir / "voices.json", {"voice_id": voice_id, "ref_audio": rel, "ref_text": text,
                                                 "lang": a.lang, "mode": a.mode,
                                                 "meta": {"tenant": a.tenant, "consent_confirmed": a.consent}})
    problems = validate_manifest(vdir / "voices.json")
    if problems:
        print("manifest problems:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1
    print(f"registered {voice_id} in {vdir / 'voices.json'}; set VOICES_FILE={vdir / 'voices.json'} and restart the worker")
    return 0


def multipart(fields: dict, file_field: str, filename: str, content: bytes):
    b = uuid.uuid4().hex
    parts = []
    for k, v in fields.items():
        parts.append(f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    parts.append(f'--{b}\r\nContent-Disposition: form-data; name="{file_field}"; filename="{filename}"\r\n'
                 f"Content-Type: application/octet-stream\r\n\r\n".encode() + content + b"\r\n")
    parts.append(f"--{b}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={b}"


def runtime(a) -> int:
    key = os.environ.get("MME_ADMIN_KEY")
    if not key:
        print("set MME_ADMIN_KEY in the environment (never pass it on the command line)", file=sys.stderr)
        return 2
    voice_id, text, data = common(a)
    body, ctype = multipart({"voice_id": voice_id, "ref_text": text, "lang": a.lang, "mode": a.mode,
                             "replace": "true" if a.replace else "false"}, "ref_audio", Path(a.audio).name, data)
    req = urllib.request.Request(a.url.rstrip("/") + "/v1/voices", data=body, method="POST",
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": ctype})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            print(r.status, json.loads(r.read()))
        return 0
    except urllib.error.HTTPError as exc:
        print(exc.code, exc.read().decode(errors="replace"), file=sys.stderr)
        return 1


def check(a) -> int:
    problems = validate_manifest(Path(a.manifest))
    print("OK" if not problems else "\n".join(problems))
    return 1 if problems else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("persist", "runtime"):
        p = sub.add_parser(name)
        p.add_argument("--tenant", required=True)
        p.add_argument("--name", required=True)
        p.add_argument("--audio", required=True)
        p.add_argument("--transcript")
        p.add_argument("--transcript-file")
        p.add_argument("--lang", required=True)
        p.add_argument("--mode", default="icl_spk", choices=("icl_spk", "icl", "spk"))
        if name == "persist":
            p.add_argument("--voices-dir", required=True)
            p.add_argument("--consent", action="store_true", help="confirm the speaker consented to cloning")
        else:
            p.add_argument("--url", required=True)
            p.add_argument("--replace", action="store_true")
    c = sub.add_parser("check")
    c.add_argument("--manifest", required=True)
    a = ap.parse_args()
    if a.cmd == "persist" and not a.consent:
        print("refusing: pass --consent to confirm you have the speaker's authorisation to clone this voice", file=sys.stderr)
        return 2
    try:
        return {"persist": persist, "runtime": runtime, "check": check}[a.cmd](a)
    except (VoiceValidationError, OSError) as exc:
        print(f"validation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
