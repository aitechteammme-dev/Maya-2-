import io
import json

import numpy as np
import pytest
import soundfile as sf

from server.voice_manager import (VoiceValidationError, upsert_manifest_entry, validate_manifest,
                                  validate_reference_audio, validate_registration, validate_voice_id)


def wav(seconds=5.0, sr=24000, amp=0.3, ch=1, fmt="WAV"):
    t = np.arange(int(seconds * sr)) / sr
    x = (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    if ch > 1:
        x = np.stack([x] * ch, 1)
    buf = io.BytesIO()
    sf.write(buf, x, sr, format=fmt, subtype="PCM_16")
    return buf.getvalue()


def test_valid_reference():
    info = validate_reference_audio(wav(5.0, 24000))
    assert info.duration_s == 5.0 and info.sample_rate == 24000 and info.channels == 1
    assert validate_reference_audio(wav(4.0, 44100, ch=2, fmt="FLAC")).channels == 2


@pytest.mark.parametrize("make_data,match", [
    (lambda: wav(1.0), "duration"), (lambda: wav(15.0), "duration"), (lambda: wav(5.0, 8000), "sample rate"),
    (lambda: wav(5.0, amp=0.0), "silent"), (lambda: b"not audio", "could not decode"),
], ids=["too-short", "too-long", "low-sr", "silent", "garbage"])
def test_invalid_reference(make_data, match):
    with pytest.raises(VoiceValidationError, match=match):
        validate_reference_audio(make_data())


def test_size_limit_and_format():
    with pytest.raises(VoiceValidationError, match="limit"):
        validate_reference_audio(wav(5.0), max_bytes=100)
    buf = io.BytesIO()
    sf.write(buf, np.zeros(24000 * 5, np.float32) + 0.2, 24000, format="OGG", subtype="VORBIS")
    with pytest.raises(VoiceValidationError, match="not accepted"):
        validate_reference_audio(buf.getvalue())


def test_voice_id_and_tenant_rules():
    assert validate_voice_id("acme__utkarsh", "acme")
    for bad in ("", "plain", "-x", "a b", "x" * 65, "../x"):
        with pytest.raises(VoiceValidationError):
            validate_voice_id(bad)
    with pytest.raises(VoiceValidationError, match="tenant isolation"):
        validate_voice_id("utkarsh", "acme")


def test_registration_requires_transcript_and_language():
    validate_registration(voice_id="a__v", ref_text="नमस्ते आप कैसे हैं", lang="hi")
    validate_registration(voice_id="a__v", ref_text=None, lang="hi", mode="spk")
    with pytest.raises(VoiceValidationError, match="ref_text"):
        validate_registration(voice_id="a__v", ref_text=" ", lang="hi")
    with pytest.raises(VoiceValidationError, match="lang"):
        validate_registration(voice_id="a__v", ref_text="hello there", lang="xx")
    with pytest.raises(VoiceValidationError, match="mode"):
        validate_registration(voice_id="a__v", ref_text="hello there", lang="en", mode="plain")


def test_manifest_roundtrip_and_validation(tmp_path):
    (tmp_path / "u.wav").write_bytes(wav(5.0))
    entry = {"voice_id": "mme__utkarsh", "ref_audio": "u.wav", "ref_text": "यह एक परीक्षण है", "lang": "hi", "mode": "icl_spk"}
    m = tmp_path / "voices.json"
    upsert_manifest_entry(m, entry)
    upsert_manifest_entry(m, {**entry, "ref_text": "यह दूसरा परीक्षण है"})          # replace, not duplicate
    data = json.loads(m.read_text(encoding="utf-8"))
    assert len(data["voices"]) == 1 and "दूसरा" in data["voices"][0]["ref_text"]
    assert validate_manifest(m) == []
    (tmp_path / "u.wav").write_bytes(wav(1.0))
    assert "duration" in validate_manifest(m)[0]
    with pytest.raises(VoiceValidationError):
        upsert_manifest_entry(m, {**entry, "ref_audio": "../evil.wav"})
