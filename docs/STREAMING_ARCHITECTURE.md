# Streaming architecture

## The three things people conflate

| Capability | Upstream Maya | Status here |
|---|---|---|
| **(1) Streaming audio output for a complete text request** | **Supported.** `POST /v1/audio/speech` (chunked/SSE) and `WS /v1/tts/stream` deliver PCM as it is generated (80 ms frames). Upstream-reported B=1 TTFA ~14 ms on H100 (not measured by us). | Inherited unchanged. |
| **(2) Accepting incremental text while synthesis is running** | **Not supported.** A request carries its whole text; there is no append/continue message (`server.py` `_speak`; HF AGENT_GUIDE l.131). | **Not available.** Emulated at chunk level only (below). Token-level streaming is **not** claimed anywhere. |
| **(3) Multiple concurrent independent utterances** | **Supported.** Continuous batching, `max_seats`, several contexts per WebSocket. | Inherited; capped by `MAX_SEATS`, `MAX_PENDING`, `MME_MAX_WS_CONNECTIONS`. Actual concurrency per GPU is **unmeasured** for our workload. |

A WebSocket existing does not imply (2). The `/v1/tts/stream` socket is a multiplexed *output* stream.

## Target flow and where each piece lives

```
Qwen3-30B-A3B ──SSE tokens──▶ MME Streaming Adapter ──text deltas──▶ [this service] WS /v1/tts/incremental
                                                                         │ IncrementalChunker (sentence/clause)
                                                                         │ sequential chunk synthesis (1 engine request at a time per session)
                                                                         ▼
                                                              Maya engine (upstream, continuous batching)
                                                                         │ PCM s16le 24 kHz, 80 ms frames
 Plivo/WebRTC ◀── watermark (MME, approved) ◀── packetizer/resampler (MME) ◀── audio frames (seq, t_offset_s)
```
Only the dashed-in box is delivered by this repo. The packetizer, any 24 kHz -> 8/16 kHz resampling, mu-law, and the "approved watermarking" stage are **outside** this image (upstream has no watermark; see UPSTREAM_AUDIT.md section 3). They are untested here.

## How the chunk-level adapter works (`server/incremental.py`, `server/text_chunker.py`)

1. Client sends `start`, then `text` deltas as SSE tokens arrive.
2. `IncrementalChunker` appends deltas to a buffer and emits a chunk when it sees a sentence boundary: `. ? ! …`, newline, Devanagari danda `।` / double danda `॥`; optionally a clause boundary at `, ; :` once the chunk is long enough (28 chars for the first chunk, 60 afterwards).
3. A `.` is **not** a boundary inside decimals (`3.5`), URLs/e-mails, initials (`A. Kumar`), abbreviations (`Dr. Mr. Mrs. Ms. Prof. vs. etc. a.m. p.m. e.g. i.e. ...`) or before a lowercase continuation. A `.`/`…` at the very end of the buffer is held (it could be `3.` + `5`) until the next character, an idle timeout (`MME_INC_IDLE_FLUSH_MS`), `flush` or `end_input`. Danda, `?`, `!` are unambiguous and are emitted at once.
4. Short sentences are merged forward (default min 24 chars; first chunk 12) because upstream notes that 1-2 word prompts can drift in voice identity. Trade-off: a very short first utterance ("Yes.") waits for more text or a flush. Tune `MME_INC_FIRST_MIN_CHARS`, or send `flush` after short acknowledgements.
5. Whitespace-only splitting: Latin words inside Devanagari sentences (code-switching) are never cut mid-word. Overlong runs without punctuation are split at spaces (`MME_INC_MAX_CHARS`, default 240); an unbroken blob longer than 2x that is hard-split as a last resort.
6. Each chunk goes through the upstream normaliser (digits, currency, dates...) **per chunk** with the session language, then becomes one upstream engine request. Chunks are synthesised **one at a time, in order**, so audio order is guaranteed and a session occupies at most one seat. Audio is forwarded with a session-wide contiguous `seq` and `t_offset_s`.
7. **Nothing is silently discarded.** Property test: whitespace-normalised concatenation of emitted chunks equals everything fed (200 random chunkings). `chunk_start` reports the exact text spoken; `cancel` reports `dropped_text`; a failed chunk reports `undelivered_text`.

### Latency model (an estimate, not a measurement)
`time_to_first_audio ≈ LLM time to produce the first chunk + (normaliser + queue + engine TTFA)`.
Chunking therefore trades first-audio latency against prosody/identity stability. Engine TTFA is the only part Maya controls; everything else must be measured end to end (PERFORMANCE_TESTING.md).
Between chunks there is a gap equal to the next chunk's engine TTFA unless the next chunk is already queued; because generation is much faster than real time (upstream RTF ~0.13 at B=1, unverified here) the next chunk is normally ready while the previous plays, **if** the LLM has produced it. Chunk-boundary prosody (intonation resetting between independently synthesised chunks, audible seams) is a known cost of this approach and has **not been evaluated** - it needs a listening test, especially for Hindi.

### Pipelining not implemented
We do not prefetch chunk N+1 while N is streaming (it would use a second seat per call and complicate ordering). If measured inter-chunk gaps are audible, prefetch depth 1 is the next optimisation.

## Backpressure, disconnects, cancellation

- Per-session sends are awaited with a timeout (`ws_send_timeout_s`, 15 s upstream default); a blocked client is disconnected and its engine request cancelled.
- Slow consumers: if more than `WS_MAX_LAG_S` (20 s) of generated audio is unsent, the chunk is cancelled -> `error: slow_consumer` with `undelivered_text`.
- Client disconnect: engine request cancelled, queue dropped, session slot freed (tested with the fake engine).
- Barge-in (`cancel`): the in-flight engine request is cancelled immediately; the client is also responsible for discarding audio it already buffered/packetised downstream.
- Overload: when `MAX_PENDING` waiting requests is exceeded or no first audio within `QUEUE_DEADLINE_MS`, the chunk fails with `overloaded` (session ends, text reported undelivered). MME should treat that as "fall back to the existing TTS".
- Shutdown (SIGTERM): `/ping` -> 503, active jobs cancelled, engine closed, within `SHUTDOWN_TIMEOUT_S`.

## Audio timing fields
- `seq`: contiguous per session (incremental endpoint only).
- `t_offset_s`: seconds of audio emitted before this packet in the session = `samples/24000`. It is *media time*, not wall-clock.
- Upstream `/v1/tts/stream` has neither.

## What would be required for true incremental text
Engine-level support for extending a running sequence's text prompt (new tokens appended to the backbone KV while frames are still being generated), plus stop-control changes. That is an upstream modification and is out of scope; do not assume it.
