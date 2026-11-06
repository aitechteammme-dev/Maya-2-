"""Reference-voice validation and manifest handling (CPU-only).

Reference recordings are NEVER stored in git or in the image: they live in a mounted directory (e.g.
/workspace/voices on a RunPod network volume). The manifest format is upstream's `--voices` JSON
({"voices": [{"voice_id","ref_audio","ref_text","lang","mode","meta"}]}), `ref_audio` relative to the manifest.

Tenant isolation: tenant voices are named '<tenant>__<name>' (see auth.voice_allowed). The server only lets a
credential bound to <tenant> synthesise with, or list, those voices.

Voice similarity to any existing voice (e.g. the Chatterbox 'Utkarsh' voice) is NOT validated by anything here - it
must be benchmarked (docs/PERFORMANCE_TESTING.md). Upstream says clones are "recognisably close but not identical".
"""
from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass
from pathlib import Path

LANGS = ("as", "bn", "en", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te")   # maya3.server.LANGS
VOICE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,63}$")                      # same as upstream
ALLOWED_SUFFIXES = (".wav", ".flac")
MODES = ("icl_spk", "icl", "spk")


class VoiceValidationError(ValueError):
    pass


@dataclass
class ReferenceInfo:
    duration_s: float
    sample_rate: int
    channels: int
    rms: float


def validate_voice_id(voice_id: str, tenant: str | None = None) -> str:
    if not isinstance(voice_id, str) or not VOICE_ID_RE.match(voice_id):
        raise VoiceValidationError("voice_id must be 1-64 chars of letters, digits, '_', '.', '-'")
    if voice_id.lower() == "plain":
        raise VoiceValidationError("'plain' is reserved")
    if tenant is not None and not voice_id.startswith(f"{tenant}__"):
        raise VoiceValidationError(f"voice_id must start with '{tenant}__' for tenant isolation")
    return voice_id


def validate_reference_audio(data: bytes, *, min_s: float = 3.0, max_s: float = 10.0, min_sr: int = 16000,
                             min_rms: float = 0.005, max_bytes: int = 8 * 1024 * 1024) -> ReferenceInfo:
    """Decode and sanity-check a reference clip (wav/flac). Upstream accepts 0.5-30 s; 3-10 s is its recommendation."""
    if len(data) > max_bytes:
        raise VoiceValidationError(f"reference file is {len(data)} bytes; limit {max_bytes}")
    try:
        import numpy as np
        import soundfile as sf
        with sf.SoundFile(io.BytesIO(data)) as f:
            if f.format not in ("WAV", "FLAC"):
                raise VoiceValidationError(f"format {f.format} not accepted; use WAV or FLAC")
            sr, ch = f.samplerate, f.channels
            audio = f.read(dtype="float32", always_2d=True)
    except VoiceValidationError:
        raise
    except Exception as exc:
        raise VoiceValidationError(f"could not decode reference audio: {exc}")
    dur = len(audio) / sr if sr else 0.0
    if sr < min_sr:
        raise VoiceValidationError(f"sample rate {sr} Hz below minimum {min_sr} Hz")
    if not (min_s <= dur <= max_s):
        raise VoiceValidationError(f"duration {dur:.2f} s outside {min_s}-{max_s} s")
    mono = audio.mean(1)
    if not np.isfinite(mono).all():
        raise VoiceValidationError("audio contains NaN/Inf")
    rms = float(np.sqrt(np.mean(mono ** 2)))
    if rms < min_rms:
        raise VoiceValidationError(f"audio is (nearly) silent (rms {rms:.5f})")
    return ReferenceInfo(round(dur, 3), int(sr), int(ch), round(rms, 5))


def validate_registration(*, voice_id, ref_text, lang, mode="icl_spk", tenant=None):
    validate_voice_id(voice_id, tenant)
    if lang not in LANGS:
        raise VoiceValidationError(f"lang must be one of {', '.join(LANGS)}")
    if mode not in MODES:
        raise VoiceValidationError(f"mode must be one of {', '.join(MODES)}")
    if mode in ("icl", "icl_spk"):
        if not isinstance(ref_text, str) or len(ref_text.strip()) < 3:
            raise VoiceValidationError("ref_text (exact transcript of the clip) is required for icl/icl_spk")


def load_manifest(path: Path) -> dict:
    if not path.is_file():
        return {"voices": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {"voices": data}


def upsert_manifest_entry(manifest_path: Path, entry: dict) -> None:
    """Add/replace one voice in the manifest (atomic write). ref_audio must be relative to the manifest."""
    ref = Path(entry["ref_audio"])
    if ref.is_absolute() or ".." in ref.parts:
        raise VoiceValidationError("ref_audio must be a relative path inside the manifest directory")
    m = load_manifest(manifest_path)
    m["voices"] = [v for v in m["voices"] if v.get("voice_id") != entry["voice_id"]] + [entry]
    tmp = manifest_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(manifest_path)


def validate_manifest(manifest_path: Path, **audio_kw) -> list[str]:
    """Validate every voice in a manifest (files exist, audio ok, fields ok). Returns problems (empty = ok)."""
    problems = []
    base = manifest_path.parent
    for v in load_manifest(manifest_path)["voices"]:
        vid = v.get("voice_id", "?")
        try:
            validate_registration(voice_id=vid, ref_text=v.get("ref_text"), lang=v.get("lang"),
                                  mode=v.get("mode", "icl_spk"))
            ref = base / v["ref_audio"]
            if ref.suffix.lower() not in ALLOWED_SUFFIXES:
                raise VoiceValidationError(f"{ref.name}: only .wav/.flac")
            validate_reference_audio(ref.read_bytes(), **audio_kw)
        except (VoiceValidationError, KeyError, OSError) as exc:
            problems.append(f"{vid}: {exc}")
    return problems
