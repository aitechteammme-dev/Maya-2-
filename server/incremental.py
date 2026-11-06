"""WS /v1/tts/incremental - MME extension (NOT part of upstream Maya; protocol documented in docs/API_REFERENCE.md).

One socket = one session. The client streams LLM text deltas; the server cuts them into sentence/clause chunks
(text_chunker) and synthesises the chunks STRICTLY IN ORDER through the upstream engine (one engine request at a time
per session, so audio order is guaranteed). Each chunk is a complete upstream synthesis request: this is chunk-level
streaming, not token-level; the first audio can only start once the first chunk boundary has been seen.

Client -> server (JSON text frames)
  {"type":"start","session_id"?, "voice"?, "language"?, "normalize"?, "binary"?, "seed"/"temperature"/..., "max_frames"?}
  {"type":"text","text":"<delta>"}        append text
  {"type":"flush"}                        emit all buffered text now (e.g. end of a sentence group); session stays open
  {"type":"end_input"}                    flush, finish all audio, send "done", close
  {"type":"cancel"}                       barge-in: stop current audio, drop queued chunks and buffered text (reported)
  {"type":"ping"}
Server -> client
  {"type":"metadata",...}  {"type":"started","session_id"}
  {"type":"chunk_start","chunk_id","text","chars"}     the exact text being synthesised (nothing is dropped silently)
  {"type":"audio","chunk_id","seq","t_offset_s","audio":"<b64 s16le>"}   or binary: <II seq chunk_id> + PCM s16le
  {"type":"chunk_end","chunk_id","reason","audio_s","frames","ttfa_ms"}
  {"type":"cancelled","dropped_chars","dropped_text"}
  {"type":"done","chunks","audio_s","chars_in"}  |  {"type":"error","code","error","undelivered_text"?}
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import struct
import time
import uuid

from aiohttp import WSMsgType, web

from maya3.server import (FRAME, LANGS, SR, ApiError, Overloaded, bad, parse_lang, parse_max_frames, parse_normalize,
                          parse_sampling)

from .text_chunker import ChunkerConfig, IncrementalChunker

log = logging.getLogger("mme.incremental")
_END = object()


def binary_frame(seq: int, chunk_id: int, pcm: bytes) -> bytes:
    return struct.pack("<II", seq, chunk_id) + pcm


class IncrementalSession:
    def __init__(self, srv, ws: web.WebSocketResponse, request: web.Request):
        self.srv, self.ws, self.request = srv, ws, request
        s = srv.settings
        self.chunker = IncrementalChunker(ChunkerConfig(
            first_min_chars=s.inc_first_min_chars, min_chars=s.inc_min_chars,
            clause_min_chars=s.inc_clause_min_chars, max_chars=s.inc_max_chars))
        self.q: asyncio.Queue = asyncio.Queue()
        self.started = False
        self.opts: dict = {}
        self.session_id = ""
        self.gen = 0                      # bumped on cancel; stale queued chunks are dropped (and reported)
        self.job = None
        self.seq = 0
        self.chunk_id = 0
        self.audio_samples = 0
        self.chunks_done = 0
        self.last_text_t = 0.0
        self.t0 = time.monotonic()
        self.failed = False
        self.input_ended = False

    # ---- sending
    async def send(self, msg) -> bool:
        if self.ws.closed:
            return False
        try:
            if isinstance(msg, bytes):
                await asyncio.wait_for(self.ws.send_bytes(msg), self.srv.ws_send_timeout_s)
            else:
                await asyncio.wait_for(self.ws.send_str(json.dumps(msg, ensure_ascii=False)), self.srv.ws_send_timeout_s)
            return True
        except (asyncio.TimeoutError, ConnectionResetError, RuntimeError):
            log.warning("incremental %s: send failed/blocked, closing", self.session_id)
            self.failed = True
            self._cancel_job()
            if self.request.transport is not None:
                self.request.transport.abort()
            return False

    async def error(self, code, message, **extra):
        self.srv.metrics.inc("incremental_errors")
        await self.send({"type": "error", "code": code, "error": message, **extra})

    # ---- control
    def _cancel_job(self):
        if self.job is not None:
            self.job.close()
            self.job.interrupt()

    def _enqueue(self, chunks):
        for c in chunks:
            self.q.put_nowait((self.gen, c))

    async def _start(self, msg):
        if self.started:
            raise bad("session already started")
        lang = msg.get("language", msg.get("lang"))
        self.opts = {
            "voice": self.srv.resolve_voice(msg.get("voice")),
            "lang": None if lang in (None, "auto") else parse_lang(msg, "x"),
            "normalize": parse_normalize(msg),
            "max_frames": parse_max_frames(msg.get("max_frames")),
            "sampling": parse_sampling(msg),
            "binary": bool(msg.get("binary", False)),
        }
        sid = msg.get("session_id")
        if sid is not None and (not isinstance(sid, str) or not 1 <= len(sid) <= 128):
            raise bad("session_id must be a string of 1-128 characters", "session_id")
        self.session_id = sid or f"inc-{uuid.uuid4().hex[:12]}"
        self.started = True
        self.srv.ready()
        await self.send({"type": "started", "session_id": self.session_id, "voice": self.opts["voice"],
                         "language": self.opts["lang"] or "auto"})

    async def _handle(self, data: str) -> bool:
        """Returns False when the reader loop should stop."""
        try:
            msg = json.loads(data)
            if not isinstance(msg, dict):
                raise bad("message must be a JSON object")
            kind = msg.get("type")
            if kind == "ping":
                await self.send({"type": "pong"})
                return True
            if kind == "start":
                await self._start(msg)
                return True
            if not self.started:
                raise bad("send {'type':'start'} first", "type")
            if self.input_ended and kind in ("text", "flush"):
                raise bad("input already ended", "type")
            if kind == "text":
                t = msg.get("text")
                if not isinstance(t, str):
                    raise bad("text must be a string", "text")
                if self.chunker.chars_in + len(t) > self.srv.settings.inc_max_session_chars:
                    raise ApiError(413, "session_text_limit", "session text limit reached")
                self.last_text_t = time.monotonic()
                self._enqueue(self.chunker.feed(t))
            elif kind == "flush":
                self._enqueue(self.chunker.flush())
            elif kind == "end_input":
                self._enqueue(self.chunker.flush())
                self.input_ended = True
                self.q.put_nowait((self.gen, _END))
                return True
            elif kind == "cancel":
                await self._cancel()
            else:
                raise bad("unknown message type; use start, text, flush, end_input, cancel, ping", "type")
        except ValueError:
            await self.error("invalid_request", "message is not valid JSON")
        except ApiError as exc:
            await self.error(exc.code, exc.message)
            if exc.code == "session_text_limit":
                return False
        return True

    async def _cancel(self):
        dropped = [self.chunker.clear()]
        self.gen += 1
        while not self.q.empty():
            g, item = self.q.get_nowait()
            if item is not _END and isinstance(item, str):
                dropped.append(item)
            elif item is _END:
                self.q.put_nowait((self.gen, _END))          # keep end_input semantics
        self._cancel_job()
        text = " ".join(d for d in dropped if d.strip())
        self.srv.metrics.inc("incremental_cancels")
        await self.send({"type": "cancelled", "dropped_chars": len(text), "dropped_text": text})

    # ---- synthesis worker
    async def _synth(self, text: str):
        srv, o = self.srv, self.opts
        gen = self.gen
        cid = self.chunk_id = self.chunk_id + 1
        lang = o["lang"] or parse_lang({}, text)
        prepared = await srv.prepare_text(text, lang, o["normalize"])
        await self.send({"type": "chunk_start", "chunk_id": cid, "text": text, "chars": len(text)})
        rid = f"i-{uuid.uuid4().hex[:12]}"
        job = srv.submit(prepared, lang, o["voice"], o["max_frames"], rid, srv.ws_max_lag_s,
                         on_overflow=lambda: self._cancel_job(), sampling=o["sampling"])
        self.job = job
        n_samples = 0
        try:
            try:
                kind, payload = await srv.first_event(job)
            except Overloaded as exc:
                raise ApiError(503, "overloaded", str(exc))
            while True:
                if gen != self.gen:                              # cancelled by barge-in
                    await self.send({"type": "chunk_end", "chunk_id": cid, "reason": "cancelled",
                                     "audio_s": round(n_samples / SR, 3), "frames": n_samples // FRAME, "ttfa_ms": None})
                    srv.stats["cancelled"] += 1
                    return
                if kind == "audio":
                    pcm, n = job.take_audio(payload, srv.coalesce)
                    off = self.audio_samples / SR
                    if o["binary"]:
                        msg = binary_frame(self.seq, cid, pcm)
                    else:
                        msg = {"type": "audio", "chunk_id": cid, "seq": self.seq, "t_offset_s": round(off, 3),
                               "audio": base64.b64encode(pcm).decode()}
                    self.seq += 1
                    if not await self.send(msg):
                        return
                    n_samples += n
                    self.audio_samples += n
                elif kind == "end":
                    ttfa = srv._ttfa_ms(job, payload)
                    await self.send({"type": "chunk_end", "chunk_id": cid, "reason": payload.get("reason"),
                                     "audio_s": payload.get("audio_s"), "frames": payload.get("frames"), "ttfa_ms": ttfa})
                    srv._log_end(rid, "inc", lang, o["voice"], len(text), job, payload)
                    self.chunks_done += 1
                    return
                elif kind == "interrupt":
                    # a barge-in cancel is caught at the top of the loop (gen changed); reaching here means the
                    # job was interrupted by the slow-consumer guard
                    raise ApiError(503, "slow_consumer", "client fell too far behind; synthesis cancelled")
                else:
                    raise ApiError(503, payload.get("code") or "internal", payload.get("error") or "generation failed")
                kind, payload = await job.get()
        finally:
            if job.overflowed:
                srv.stats["slow_consumer"] += 1
            job.close()
            self.job = None

    async def _worker(self):
        try:
            while True:
                gen, item = await self.q.get()
                if item is _END:
                    await self.send({"type": "done", "chunks": self.chunks_done, "audio_s": round(self.audio_samples / SR, 3),
                                     "chars_in": self.chunker.chars_in})
                    return
                if gen != self.gen:
                    continue                                     # dropped by cancel (already reported)
                try:
                    await self._synth(item)
                except ApiError as exc:
                    undelivered = [item] + [c for g, c in list(self.q._queue) if isinstance(c, str) and g == self.gen]
                    self.failed = True
                    await self.error(exc.code, exc.message, undelivered_text=" ".join(undelivered))
                    return
        except asyncio.CancelledError:
            raise

    async def _idle_watch(self):
        idle_s = self.srv.settings.inc_idle_flush_ms / 1000.0
        if idle_s <= 0:
            return
        while True:
            await asyncio.sleep(idle_s / 2)
            if self.started and not self.input_ended and self.chunker.has_pending() \
                    and time.monotonic() - self.last_text_t >= idle_s:
                self._enqueue(self.chunker.idle())
                self.last_text_t = time.monotonic()

    async def run(self):
        s = self.srv.settings
        await self.send({"type": "metadata", "protocol": "mme-incremental/1", "sample_rate": SR, "encoding": "pcm_s16le",
                         "channels": 1, "languages": list(LANGS), "granularity": "sentence/clause chunks",
                         "max_session_s": s.inc_max_session_s})
        worker = asyncio.create_task(self._worker())
        watcher = asyncio.create_task(self._idle_watch())
        deadline = self.t0 + s.inc_max_session_s
        try:
            while not worker.done() and not self.failed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    await self.error("session_timeout", "maximum session duration reached")
                    break
                try:
                    msg = await asyncio.wait_for(self.ws.receive(), min(remaining, 1.0))
                except asyncio.TimeoutError:
                    continue
                if msg.type == WSMsgType.TEXT:
                    if not await self._handle(msg.data):
                        break
                elif msg.type == WSMsgType.BINARY:
                    await self.error("invalid_request", "send JSON text messages")
                else:                                              # CLOSE / CLOSING / CLOSED / ERROR
                    break
            if worker.done() or self.input_ended:
                await asyncio.gather(worker, return_exceptions=True)
        finally:
            self._cancel_job()
            watcher.cancel()
            worker.cancel()
            await asyncio.gather(worker, watcher, return_exceptions=True)
