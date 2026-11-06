# Performance instrumentation and GPU acceptance plan

**Nothing below has been run.** No GPU was used while building this repo. Numbers quoted from upstream are upstream's own measurements on H100/RTX 4090 and are not evidence for our deployment.

## Metrics exposed (`GET /metrics`, authenticated; `?json=1` for JSON)

| Requirement | Metric |
|---|---|
| model load duration | `mme_tts_model_load_seconds` (summary), `/health` `initialization.load_seconds` |
| model readiness time | `mme_tts_model_ready_seconds` (process start of load -> engine warm) |
| time to first audio | `mme_tts_ttfa_seconds` p50/p90/p99 (engine: submit -> first PCM; also logged per request) |
| total synthesis time, RTF | `mme_tts_synth_wall_seconds`, `mme_tts_rtf` (wall/audio; <1 = faster than real time), `mme_tts_audio_seconds` |
| first playable packet | same as TTFA at the server; **client-side** TTFA is measured by the load client |
| chunks/s | derive from `mme_tts_synth_via_inc_total` rate (incremental) and `incremental_sessions_started_total` |
| GPU util / VRAM | `mme_tts_gpu_utilization_pct` (needs `pynvml`; may be absent), `gpu_vram_used_mb`, `gpu_vram_total_mb`, `gpu_torch_allocated_mb` |
| queue wait | `mme_tts_upstream_waiting`, `..._pending`; per-request queue wait is inside TTFA |
| active sessions | `mme_tts_upstream_active`, `ws_sockets`, `incremental_sessions` |
| failed / cancelled | `mme_tts_synth_cancelled_total`, upstream counters `rejected_pending`, `rejected_deadline`, `aborted_midstream`, `slow_consumer`, `client_gone`; `mme_tts_auth_failures_total`, `model_load_failures_total` |

Engine latency vs network latency: these numbers are **server-side only**. RunPod proxy, internet, Plivo/WebRTC and jitter-buffer delay are invisible to the server; measure them at the client (test G5) and report separately. Per-request log lines (`INFO maya3.server: <rid> ws lang=.. ttfa_ms=.. audio_s=..`) carry the same engine data.

## GPU acceptance test plan (run later, on a rented GPU, with explicit approval)

Use `MME_PROFILE=benchmark`. Record GPU model, driver, image tag, `MAX_SEATS`, `MAYA_CAPACITY`.

| ID | Test | Pass criteria |
|---|---|---|
| G1 | Cold start from empty volume, then warm restart | `/ping` 204->200; record download time, load time, ready time; second start has no download; no healthy-while-loading |
| G2 | Fail-closed: remove a shard on the volume with `HF_HUB_OFFLINE=1`; corrupt a shard (truncate) | `/ping` 503, clear error in `/health`, process exits 1 |
| G3 | Smoke synthesis: Hindi (Devanagari), English, **Hinglish (Latin script with `language=hi` vs `en`)**, numbers/OTP/currency, each over HTTP pcm/wav and both WS routes | audio decodes; no truncation; durations plausible; human listening notes. Record that Latin-script Hinglish is detected as `en` when `language` is omitted |
| G4 | Voice cloning: register Utkarsh via `register_voice.py persist`; synthesise the same 20 Hindi/Hinglish MME sentences with Chatterbox-Utkarsh and Maya-clone | blind A/B listening by MME staff + a speaker-similarity score (any agreed tool, e.g. ECAPA cosine) - **record the result, do not assume parity** |
| G5 | Real-time delivery through RunPod proxy (Path A) vs direct (Path B): client plays `/v1/tts/incremental` audio from a scripted Qwen3-like token stream (10-30 tokens/s) | client TTFA p50/p95; inter-packet gap p99; underruns at an 80-200 ms prebuffer; connection survives a 10-minute session; behaviour on worker scale-in |
| G6 | Chunk-seam quality: same paragraph as one request vs chunked by the adapter | listening test; note audible seams/prosody resets; tune `MME_INC_*` |
| G7 | Concurrency ladder: `python -m maya3.bench.load_ws --url ws://HOST/v1/tts/stream --texts <jsonl> --lanes N --duration 120 --binary --out run_N.json` for N = 1,2,4,8,16... with matching `MAX_SEATS` (note: this bench script talks to the upstream route; add an `X-MME-Api-Key` header by running it through a trusted tunnel/auth-disabled dev profile, since the stock client has no header option - **verify**) | TTFA p99, stalls = 0, RTF p99 < 1 at the SLA; find the knee; record VRAM at each N |
| G8 | Admission control: exceed `MAX_PENDING` and `QUEUE_DEADLINE_MS` | 503/`overloaded` + `Retry-After`, no hangs, recovery after load drops |
| G9 | Cancellation: barge-in 100 times mid-chunk on `/v1/tts/incremental` | audio stops within ~1 frame of server processing; `cancelled` reports dropped text; seats freed (`active` returns to 0); no VRAM growth |
| G10 | Disconnect storm: kill 50 client sockets mid-synthesis | `active`, `ws_sockets`, `incremental_sessions` return to 0; no leaked seats |
| G11 | Soak: 60 min at 50% of the G7 knee | no stalls, flat VRAM, no error growth |
| G12 | Graceful shutdown: SIGTERM (RunPod scale-in) with active sessions | `/ping` 503 immediately, sessions end <= `SHUTDOWN_TIMEOUT_S`, exit 0 |
| G13 | Security: calls without/with wrong token, oversized body, cross-tenant voice | 401/413/403 as in API_REFERENCE.md through the proxy (also proves which header reaches the worker) |
| G14 | Triton/CUDA: first-request compile time in the container as non-root; cache reuse after restart | record; no permission errors |

Do **not** report concurrency claims until G7/G11 are done on the target GPU. No profile guarantees any call count.

## CPU tests already passing (no GPU)
`pytest` in this repo (see BUILD_REPORT.md): auth, validation, health states, fail-closed loading, WebSocket lifecycle (both routes), chunker properties, voice/audio validation, model-file verification, lifecycle (health port, shutdown, load-failure exit). They use upstream's `FakeEngine` and prove protocol/lifecycle behaviour only - **not** inference, latency or quality.
