# Upstream audit - MayaResearch/maya2-native

- Audited commit: `c6b376868e268551d065c4206193e96257bef4a1` (`git clone --depth 1`, 2026-11-06). Changelog: "1.0 (2026-10-25)".
- Model repo: `maya-research/maya2-native` @ `87a83efdd6551e038246b6b37836a6264713eec3` (HF API `sha`, same date). Public (`gated: false`), license `apache-2.0`.
- The Python package is called **`maya3`**; `maya2-native` is only the repo name (README line 11).
- Method: read the source listed below. **No inference was run** (no GPU, no weights downloaded). Everything in the "Verified in source" column is a statement about code/docs; behaviour on a GPU is untested here.

## 1. Answers to the ten verification questions

| # | Question | Answer | Evidence |
|---|---|---|---|
| 1 | CUDA 12.8? | **Yes - it is the supported target.** "NVIDIA GPU with a CUDA 12.8 driver (8 GB or more)", Python 3.10-3.12, Linux. Wheels from `https://download.pytorch.org/whl/cu128`. | `README.md` "Install"; `pyproject.toml` comment lines 11-12 |
| 2 | PyTorch/CUDA versions | `torch==2.8.0` (cu128, brings `triton==3.4.0`), `torchaudio==2.8.0`, `transformers==5.6.2`, `speechbrain==1.1.2`, `tokenizers>=0.22,<0.23`, `huggingface_hub>=1.0`. Resolved lock in `requirements-lock.txt` gives `torch 2.8.0+cu128`, `triton 3.4.0`, `nvidia-cuda-runtime-cu12 12.8.90`, `nvidia-cudnn-cu12 9.10.2.21`, `numpy 2.2.6`, `aiohttp 3.14.5`. | `pyproject.toml` `dependencies`; `docs/ENGINE.md` "Pinned runtime versions" |
| 3 | WebSocket streaming? | **Yes.** `WS /v1/tts/stream`. Messages: `metadata`, `speak`, `audio` (base64 or binary), `end`, `cancel`, `ping/pong`, `error`. Several contexts per socket (`--ws-max-contexts`, default 8). | `maya3/server.py` docstring lines 58-77, `Maya3Server.ws` 810, `_Socket` 841 |
| 4 | HTTP audio streaming? | **Yes.** `POST /v1/audio/speech` streams `wav`/`pcm`/`mp3`/`opus` (chunked) or SSE (`stream_format: "sse"`); `flac` is sent whole. `aac` -> 400. | `server.py` lines 27-56, `_speech` 585-705 |
| 5 | Checkpoint download | `Maya3TTS.from_pretrained("<repo id or dir>")` -> `maya3.hub.resolve` -> `huggingface_hub.snapshot_download` with `allow_patterns` (`*.safetensors`, index, `maya3_config.json`, `config.json`, `backbone/*`, `voices/*`, `README.md`). **Offline-first**: a cached snapshot is used with no network call; `HF_HUB_OFFLINE=1` forbids downloads. `revision=` pins the model repo; Mimi/ECAPA use cached/`main` (cannot be pinned through `resolve`). | `maya3/hub.py` `snapshot`, `resolve`, `RELEASE_PATTERNS` |
| 6 | Reference-voice registration | Three routes: Python `tts.register_voice(id, ref_audio, ref_text, lang, mode)`; HTTP `POST /v1/voices` (JSON `ref_audio_b64` or multipart `ref_audio`); server flag `--voices voices.json` (registered before ready). Registry is **in memory** - not persisted by upstream. Modes: `icl_spk` (default), `icl`, `spk` (no transcript), `plain`, `named`. Needs the ECAPA dir. Prefix KV is precomputed at registration. Clip 0.5-30 s accepted by the server, 3-10 s recommended. | `maya3/api.py` `make_voice`/`register_voice`; `server.py` `add_voice` 729, `_decode_audio` 1198, `load_voices_file` 1216 |
| 7 | Incremental **text** input? | **No. Only incremental audio output.** Each `speak`/HTTP request carries the complete text; there is no "append text to a running synthesis" message. (Engine-internal splitting of long text into ~20 s pieces, `chunk_s`, is internal to one request.) HF `AGENT_GUIDE.md` line 131 says the same: "A voice-agent application must handle text aggregation, user turns, cancellation and discarding buffered playback." | `server.py` `_speak` 932; `docs/ENGINE.md` "Text." |
| 8 | Concurrent sessions? | **Yes.** Continuous batching in one CUDA-graph engine; `max_seats` (default 16 in `EngineConfig`, 64 in the CLI) concurrent generations; admission control `--max-pending`, `--queue-deadline-ms`; up to `--ws-max-contexts` per socket. Upstream-measured (not by us): 224 streams on one H100 NVL, 48 on RTX 4090 @ capacity 2048. | `engine.py` 50-80, `server.py` 410-430, 516-535; README "Streaming performance" |
| 9 | Cancel active synthesis? | **Yes.** WS `{"type":"cancel","context_id"}` -> `{"type":"end","reason":"cancelled"}`; closing the socket or dropping the HTTP connection cancels the engine request (`Request.cancel`); slow consumers (`--ws-max-lag-s`, `--http-max-lag-s`) are cancelled. Cancelling is idempotent. | `server.py` `_Socket.handle` 921-924, `_stop` 968, `Job.close` 399; `engine.py` `Request.cancel` 106 |
| 10 | Assets to cache | (a) release `maya-research/maya2-native`: 2 safetensors shards (5,000,017,272 + 4,862,496 bytes), `model.safetensors.index.json`, `maya3_config.json`, `backbone/{config,generation_config,tokenizer,tokenizer_config}.json` (no backbone weights: architecture is built from config), `voices/voices.json`; (b) `kyutai/mimi` (`config.json`, `model.safetensors`, `preprocessor_config.json`); (c) `speechbrain/spkrec-ecapa-voxceleb` (`embedding_model.ckpt`, `hyperparams.yaml`). Plus the Triton kernel cache (`TRITON_CACHE_DIR`, otherwise written inside the installed `serve/` package). | `hub.py` constants; HF tree listing; `docs/ENGINE.md` install note |

## 2. Exact production serving command (upstream)

```bash
python -m maya3.server --ckpt maya-research/maya2-native --host 0.0.0.0 --port 8080 --max-seats 64
```
(README "HTTP server"). Full CLI (`server.py main()` 1230-1284): `--ckpt` (required), `--backbone-dir`, `--mimi-dir`, `--ecapa-dir`, `--host` (default 127.0.0.1), `--port` (8080), `--device` (cuda), `--max-seats` (64), `--capacity`, `--depth-dtype`, `--max-pending`, `--queue-deadline-ms` (2000), `--voices`, `--ws-max-contexts` (8), `--ws-max-lag-s` (20), `--http-max-lag-s` (120), `--busy-frac`, `--busy-admit`, `--no-normalise`, `--log-level`.
**There are no environment variables for the server** (only `HF_HUB_OFFLINE`, `MAYA3_MIMI_DIR`, `MAYA3_ECAPA_DIR`, `MAYA3_BACKBONE_FROM_CONFIG`, `TRITON_CACHE_DIR`, `TOKENIZERS_PARALLELISM`).

This deployment does **not** call that CLI. `server/api.py` imports `maya3.server.Maya3Server` and subclasses it, because the CLI offers no hook for authentication, `/ping`, metrics, tenant isolation or fail-closed exit. The engine and all upstream routes are reused unmodified.

## 3. What upstream does NOT provide (gaps this deployment fills or documents)

| Gap | Upstream fact | Our handling |
|---|---|---|
| **Authentication** | None. HF `AGENT_GUIDE.md` l.131: "no built-in authentication; deployment access controls belong to the application environment." | `server/auth.py` bearer / `X-MME-Api-Key`, constant-time, tenant-bound; public only `/ping`, `/health`. |
| **Voice registration is unauthenticated** and in-memory | `POST/DELETE /v1/voices` open to any caller; lost on restart. | Admin-only and off by default; persistent voices via mounted `voices.json` (`VOICES_FILE`). |
| **`/ping`** | Not present; `/health` returns 200/503 only. | Added: 200 ready / 204 initializing / 503 failed (RunPod semantics). |
| **Metrics** | Only counters inside `/health`. | `/metrics` (authenticated). |
| **Incremental text** | Not supported (Q7). | Chunk-level adapter `WS /v1/tts/incremental` (custom, documented as such). |
| **Watermarking** | **None in the code** (`grep -ri watermark` = 0 hits) and HF README: "This release does not add an audio watermark." | This image neither adds nor removes any. The MME pipeline's "Approved Watermarking" stage must remain downstream of the PCM output; nothing here defeats a protection because none exists. |
| **Fail-closed startup** | Port opens immediately; load errors are only reported by `/health` (503) and the process keeps running. | Process exits non-zero after `MME_EXIT_DELAY_S` (default 10 s) so the platform restarts it. |
| **Request-size limits** | aiohttp `client_max_size` 24 MB; text limit 4000 chars; ref audio 30 s. | Middleware limits (64 KB JSON, 8 MB voice upload). |
| **Speed control** | `speed` must be 1.0. | Documented. |
| **Pinning of Mimi / ECAPA revisions** | `resolve` pins only the main repo. | Documented residual risk; pre-populate the cache and run offline (`HF_HUB_OFFLINE=1`). |

## 4. Audio format (verified in source)

- Model output: **24 kHz, mono, signed 16-bit little-endian PCM** (`SR = 24000`, `FRAME = 1920` samples = one 80 ms frame; `server.py` lines 12, 120-121; `engine.py` `SR`). WebSocket audio is always PCM s16le 24 kHz (no format negotiation).
- HTTP encodings: `wav` (streaming RIFF header with 0xFFFFFFFF sizes), `pcm`, `mp3` (CBR ~80 kbit/s, 24 kHz), `opus` (Ogg/Opus 24 kHz), `flac` (buffered). Response headers `X-Sample-Rate`, `X-Channels`, `X-Encoding`.
- Telephony (Plivo) needs 8 kHz/16 kHz and mu-law/PCM; **no resampling is done in this image** - it belongs to the MME audio packetizer and must be tested there (see `docs/STREAMING_ARCHITECTURE.md`).

## 5. Text handling

- Bundled normaliser (`maya3/text`) is on by default: numbers, currency, dates, URLs, ALL-CAPS names. Per request `"normalize": false`.
- Languages: `as bn en gu hi kn ml mr or pa ta te`; auto-detected from script when `language` is omitted (Latin => `en`, so **Hinglish in Latin script is treated as English** unless `language` is set explicitly - test this).
- Known limitations (upstream README "Limitations"): code-mixed text has the highest error rate; 1-2 word prompts can drift in voice identity; clone similarity "recognisably close but not identical"; sampling is non-deterministic except a lone request with a fixed `seed`.
- Voices: 20 named voices incl. Hindi-trained `Abhay, Nivedita, Ishwari, Ritvik, Vrinda, Sreekar, Shaunak, Akshat, Tanushree` (README table). They are "synthetic personas".

## 6. Tests shipped upstream

`./run_tests.sh` = `CUDA_VISIBLE_DEVICES= pytest maya3/tests/test_text.py test_guards.py test_server.py` (CPU, fake engine in `maya3/tests/fakes.py`). Our CPU tests reuse that `FakeEngine`.

## 7. Items not verifiable without a GPU

Actual TTFA/RTF on RunPod hardware, Hindi/Hinglish quality, clone similarity to the Chatterbox "Utkarsh" voice, Triton JIT compile inside the container, CUDA-graph capture memory on the chosen GPU, and behaviour behind the RunPod proxy. See `docs/PERFORMANCE_TESTING.md` for the acceptance plan.
