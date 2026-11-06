import json
import struct

import pytest

from server.model_store import (ModelVerificationError, safetensors_expected_size, verify_release, verify_safetensors)


def write_safetensors(path, nbytes=64):
    header = json.dumps({"w": {"dtype": "U8", "shape": [nbytes], "data_offsets": [0, nbytes]},
                         "__metadata__": {"format": "pt"}}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * nbytes)


def make_release(d, shards=("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")):
    (d / "backbone").mkdir(parents=True)
    (d / "maya3_config.json").write_text("{}")
    (d / "backbone" / "config.json").write_text("{}")
    (d / "backbone" / "tokenizer.json").write_text("{}")
    for s in shards:
        write_safetensors(d / s)
    (d / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {f"t{i}": s for i, s in enumerate(shards)}}))
    return d


def test_valid_release_verifies(tmp_path):
    assert len(verify_release(make_release(tmp_path))) == 2


def test_truncated_shard_detected(tmp_path):
    make_release(tmp_path)
    shard = tmp_path / "model-00002-of-00002.safetensors"
    shard.write_bytes(shard.read_bytes()[:-10])            # interrupted download
    with pytest.raises(ModelVerificationError, match="truncated or corrupt"):
        verify_release(tmp_path)


def test_missing_shard_and_files_listed_clearly(tmp_path):
    make_release(tmp_path)
    (tmp_path / "model-00001-of-00002.safetensors").unlink()
    (tmp_path / "backbone" / "tokenizer.json").unlink()
    with pytest.raises(ModelVerificationError) as e:
        verify_release(tmp_path)
    msg = str(e.value)
    assert "missing shard model-00001-of-00002.safetensors" in msg and "missing backbone/tokenizer.json" in msg


def test_no_weights_at_all(tmp_path):
    (tmp_path / "backbone").mkdir()
    with pytest.raises(ModelVerificationError, match="no model.safetensors"):
        verify_release(tmp_path)


def test_garbage_header_rejected(tmp_path):
    p = tmp_path / "x.safetensors"
    p.write_bytes(b"\xff" * 40)
    with pytest.raises(ModelVerificationError):
        verify_safetensors(p)
    p.write_bytes(b"abc")
    with pytest.raises(ModelVerificationError, match="shorter"):
        safetensors_expected_size(p)


def test_offline_empty_cache_fails_closed(tmp_path):
    pytest.importorskip("huggingface_hub")
    from server.config import Settings
    from server.model_store import resolve_and_verify
    s = Settings(profile="development", auth_disabled=True, hf_home=str(tmp_path), allow_download=False)
    with pytest.raises(FileNotFoundError, match="not in the local Hugging Face cache"):
        resolve_and_verify(s)
