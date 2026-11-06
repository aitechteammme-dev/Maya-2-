# API reference

"**Upstream**" = inherited unchanged from `maya3.server` (contract in the docstring of `maya3/server.py`, commit `c6b3768`).
"**MME**" = added by this repo. Everything marked MME is a custom extension, not part of Maya.

## Authentication (MME)

All routes except `/ping` and `/health` require a credential:

```
Authorization: Bearer <token>        or        X-MME-Api-Key: <token>
```
- Tokens come from `MME_API_KEYS` (`tenant=token,tenant2=token2` or bare `token` = tenant `default`; min 16 chars; several tokens allow rotation). `MME_ADMIN_KEYS` are admin tokens.
- Comparison is constant-time (SHA-256 digests, no early exit). Tokens are never logged or echoed. Query-string tokens are not accepted.
- 401 `unauthorized` (with `WWW-Authenticate: Bearer`), 403 `forbidden`, 413 `payload_too_large`.
- WebSocket handshakes are authenticated by the same headers (HTTP 401 on upgrade failure).
- **RunPod note:** RunPod load-balancing endpoints use `Authorization: Bearer <RUNPOD_API_KEY>` at the proxy. Whether the header is also forwarded to the worker is **unverified**; send the app token in `X-MME-Api-Key` as well/instead and verify during deployment (RUNPOD_DEPLOYMENT.md step 7).
- `MME_AUTH_DISABLED=1` exists for local development only and is refused when `MME_PROFILE=production`; with no keys configured the server refuses to start.

Error body (same as upstream): `{"error": {"message", "type", "code", "param"}}`.

## Health

### `GET /ping` (MME; public)
| Status | Meaning |
|---|---|
| `200` | model loaded **and** engine warm/ready, not draining |
| `204` | initialising (downloading / verifying / loading / warming) |
| `503` | load failed, engine failed, or shutting down |

Empty body. Matches RunPod's documented LB health semantics (200 healthy, 204 initializing, other unhealthy).

### `GET /health` (MME override of upstream; public, deliberately minimal)
`200` when ready, `503` otherwise. Body:
```json
{"status": "ready|initializing|failed", "model_ready": true, "model_id": "maya-research/maya2-native",
 "initialization": {"phase": "starting|downloading|loading|warming|ready|failed", "load_seconds": 41.2, "error": null},
 "gpu": {"available": true, "name": "...", "vram_total_mb": 0, "vram_used_mb": 0, "torch_allocated_mb": 0, "utilization_pct": null},
 "sessions": {"active_synthesis": 0, "pending": 0, "ws_sockets": 0, "incremental_sessions": 0},
 "capacity": {"max_seats": 1, "available_seats": 1, "max_pending": 4}, "uptime_s": 12.0}
```
It exposes the load error text; it exposes no voice names, keys or paths beyond what the error message contains - restrict at the network layer if that is a concern. (Upstream's richer `/health` is replaced.)

### `GET /metrics` (MME; authenticated)
Prometheus text; `?json=1` for JSON. See PERFORMANCE_TESTING.md for the list.

## `POST /v1/audio/speech` (Upstream; auth added)

```json
{"model": "maya3", "input": "आपका ऑर्डर कल पहुँच जाएगा।", "language": "hi", "voice": "Ahana",
 "response_format": "pcm", "stream_format": "audio", "normalize": true, "max_frames": 400,
 "seed": 7, "temperature": 0.8, "top_k": 64, "top_p": 0.95, "repetition_penalty": 1.0}
```
| Field | Rules (upstream-enforced) |
|---|---|
| `input` | non-empty string, <= 4000 chars |
| `language`/`lang` | `as bn en gu hi kn ml mr or pa ta te`, or omitted/`auto` (script detection: Devanagari->hi, Latin->**en**) |
| `voice` | registered id (default `plain`); named voices case-insensitive; OpenAI names (`alloy`...) map to `plain`; unknown -> 400. Tenant voices `<tenant>__<name>` only for that tenant (MME, 403 otherwise) |
| `response_format` | `wav` (default), `pcm`, `mp3`, `opus` streamed; `flac` buffered; `aac` -> 400 |
| `stream_format` | `audio` (chunked body) or `sse` (`speech.audio.delta` base64 events + `speech.audio.done`; not flac) |
| `speed` | must be 1.0 |
| `max_frames` | 1..1500 (80 ms frames) |
| sampling | `seed` 0..2^63-1, `temperature` 0.1..1.5, `top_k` 1..2048, `top_p` (0,1], `repetition_penalty` 0..5 |

Response headers: `X-Sample-Rate: 24000`, `X-Channels: 1`, `X-Encoding`, `X-Maya3-Language`, `X-Maya3-Voice`, `X-Maya3-Normalized`, `X-Request-Id`.
Errors before audio: 400 invalid, 503 `overloaded` (+`Retry-After`) when `max_pending` is exceeded or no audio within `QUEUE_DEADLINE_MS`, 503 `unavailable` while loading. A failure **after** audio started aborts the connection (no clean end), so truncation is never silent.
Cancellation: closing the connection cancels the engine request. **Request timeout:** there is no wall-clock timeout parameter upstream; bounds are the queue deadline, the engine's duration guard (stops at 2.2x the expected duration + 1.2 s), `max_frames`, and `HTTP_MAX_LAG_S` (slow consumer).

## `WS /v1/tts/stream` (Upstream; auth added) - full-text requests, streamed audio

```
server -> {"type":"metadata","sample_rate":24000,"encoding":"pcm_s16le","channels":1,"max_contexts":8,"languages":[...]}
client -> {"type":"speak","context_id":"c1","text":"...","language":"hi","voice":"plain","binary":false,"max_frames":400,"normalize":true,"seed":7}
server -> {"type":"audio","context_id":"c1","audio":"<base64 s16le>"}                (repeated)
          binary mode: uint32-LE len(context_id) | context_id utf-8 | PCM s16le
server -> {"type":"end","context_id":"c1","reason":"aeos","frames":52,"audio_s":4.16,"ttfa_ms":61.0}
client -> {"type":"cancel","context_id":"c1"}  ->  {"type":"end","reason":"cancelled",...}
client -> {"type":"ping"} -> {"type":"pong"}
errors -> {"type":"error","context_id":"c1"|null,"code":"invalid_request|overloaded|slow_consumer|unavailable|too_many_contexts|internal","error":"..."}
```
- Every `speak` gets exactly one terminal `end` or `error`; no audio after it. Several contexts per socket (<= `WS_MAX_CONTEXTS`).
- There is **no sequence number or timestamp** in upstream audio frames; order is guaranteed per context by TCP. Use the incremental endpoint if you need `seq`/`t_offset_s`.
- Closing the socket cancels its requests. Max inbound message 64 KB.
- MME adds a global connection cap (`MME_MAX_WS_CONNECTIONS`, counts both WS routes): 503 on upgrade when full.

## `WS /v1/tts/incremental` (MME extension) - chunk-level incremental text

Granularity is **sentence/clause chunks, not tokens**; see STREAMING_ARCHITECTURE.md. One socket = one session; chunks are synthesised strictly in order.

Client -> server:
| Message | Meaning |
|---|---|
| `{"type":"start","session_id"?,"voice"?,"language"?,"normalize"?,"binary"?,"max_frames"?,<sampling>}` | must be first; `language` omitted/`auto` detects per chunk (set `hi` explicitly for Hinglish) |
| `{"type":"text","text":"<delta>"}` | append LLM text; completed chunks are dispatched immediately |
| `{"type":"flush"}` | dispatch everything buffered now (session stays open) |
| `{"type":"end_input"}` | flush, finish all audio, send `done` |
| `{"type":"cancel"}` | barge-in: cancel current chunk, drop queued chunks and buffered text; **reports** what was dropped; session continues |
| `{"type":"ping"}` | -> `pong` |

Server -> client:
| Message | Meaning |
|---|---|
| `metadata` | `{"protocol":"mme-incremental/1","sample_rate":24000,"encoding":"pcm_s16le","channels":1,"granularity":"sentence/clause chunks",...}` |
| `started` | `{"session_id","voice","language"}` |
| `chunk_start` | `{"chunk_id","text","chars"}` - the exact text now being synthesised |
| `audio` | `{"chunk_id","seq","t_offset_s","audio":"<b64 s16le>"}`; binary mode: `<II seq, chunk_id>` + PCM. `seq` is contiguous per session; `t_offset_s` = audio position of this packet's start within the session |
| `chunk_end` | `{"chunk_id","reason","audio_s","frames","ttfa_ms"}` |
| `cancelled` | `{"dropped_chars","dropped_text"}` |
| `done` | `{"chunks","audio_s","chars_in"}` |
| `error` | `{"code","error","undelivered_text"?}`; on a synthesis failure the session ends and `undelivered_text` lists text that was **not** spoken |

Limits: `MME_INC_MAX_SESSION_S` (1800), `MME_INC_MAX_SESSION_CHARS` (20000), idle auto-confirm `MME_INC_IDLE_FLUSH_MS` (350 ms: confirms a trailing `.`/`…`, never cuts mid-sentence). Chunker tunables: `MME_INC_FIRST_MIN_CHARS` 12, `MME_INC_MIN_CHARS` 24, `MME_INC_CLAUSE_MIN_CHARS` 60, `MME_INC_MAX_CHARS` 240.

## Voices

| Route | Upstream behaviour | MME change |
|---|---|---|
| `GET /v1/voices` | list `{"voices":[{id,mode,ref_lang,...}]}` | filtered to shared + caller's tenant voices (admin sees all) |
| `POST /v1/voices` | register clone: JSON (`voice_id`, `ref_audio_b64`, `ref_text`, `lang`, `mode`, `replace`) or multipart (`ref_audio` file). 201 / 409 if exists | **admin credential + `MME_ALLOW_VOICE_REGISTRATION=1`** required, else 403. Body limit 8 MB. In-memory only |
| `DELETE /v1/voices/{id}` | remove | admin + flag |

Preferred persistent workflow: `scripts/register_voice.py persist` + `VOICES_FILE` (README / RUNPOD_DEPLOYMENT.md). `GET /v1/models` is upstream (`[{"id":"maya3"}]`).

## Environment variables

Standard (huggingface_hub / upstream): `HF_HOME`, `HF_HUB_OFFLINE`, `HF_TOKEN` (optional, never baked in), `MAYA3_MIMI_DIR`, `MAYA3_ECAPA_DIR`, `TRITON_CACHE_DIR`.
**Custom to this repo** (upstream has no env vars - it uses CLI flags; these map to them):

| Variable | Default | Maps to |
|---|---|---|
| `MODEL_ID` | `maya-research/maya2-native` | `--ckpt` |
| `MODEL_REVISION` | pinned sha `87a83efd...` (empty = `main`) | `hub.resolve(revision=)` |
| `PORT` / `HOST` | `8080` / `0.0.0.0` | `--port` / `--host` |
| `PORT_HEALTH` | = `PORT` | RunPod health port; if different a second listener serves only `/ping` + `/health` |
| `MAX_SEATS` | `1` | `--max-seats` |
| `MAX_PENDING` | `4` | `--max-pending` |
| `MAYA_CAPACITY` | engine default 2048 | `--capacity` |
| `QUEUE_DEADLINE_MS` / `WS_MAX_CONTEXTS` / `WS_MAX_LAG_S` / `HTTP_MAX_LAG_S` | 2000 / 8 / 20 / 120 | same-named flags |
| `MAYA_DEVICE` / `MAYA_NORMALISE` | `cuda` / `1` | `--device` / `--no-normalise` |
| `VOICES_FILE` | unset | `--voices` |
| `LOG_LEVEL` | `INFO` | `--log-level` |
| `MME_PROFILE` | `production` | loads `config/<profile>.yaml` defaults (env wins) |
| `MME_API_KEYS`, `MME_ADMIN_KEYS` | required / unset | MME auth |
| `MME_AUTH_DISABLED`, `MME_ALLOW_VOICE_REGISTRATION` | 0 / 0 | MME |
| `MME_MAX_BODY_BYTES`, `MME_MAX_VOICE_UPLOAD_BYTES`, `MME_MAX_WS_CONNECTIONS` | 65536 / 8388608 / 32 | MME limits |
| `MME_EXIT_ON_LOAD_FAILURE`, `MME_EXIT_DELAY_S`, `SHUTDOWN_TIMEOUT_S` | 1 / 10 / 30 | lifecycle |
| `MME_INCREMENTAL`, `MME_INC_*` | 1 / see above | incremental endpoint |
| `MME_RUN_AS_ROOT`, `MME_SKIP_STARTUP_CHECK`, `MME_SKIP_GPU_CHECK`, `MME_UID` | 0 / 0 / 0 / 10001 | entrypoint |
