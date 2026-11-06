"""Incremental text -> synthesis-sized chunks (sentence/clause level). NOT token-level streaming.

Maya 2 Native only synthesises complete text submitted per request (docs/UPSTREAM_AUDIT.md section 8): there is no way
to append text to a running synthesis. This chunker lets an LLM token stream be cut into pieces that are submitted one
after another, so audio can start after the first sentence/clause instead of after the whole reply.

Guarantees (tested in tests/test_text_chunker.py):
  * nothing is ever discarded: whitespace-normalised concatenation of all emitted chunks == all text fed
    (flush() emits whatever is left, however short);
  * boundaries: . ? ! … newline, Devanagari danda U+0964 and double danda U+0965 (+ optional clause split at , ; :);
  * a '.' is NOT a boundary inside decimals (3.5), URLs/e-mails (a.com), initials ("A. Kumar"), common abbreviations
    (Dr. Mr. Mrs. Ms. Prof. vs. etc.), a.m./p.m., e.g./i.e., or when the next word starts lowercase;
  * a trailing '.' / '…' at the end of the buffer is held until the next character arrives (could be "3." + "5"),
    unless flush()/idle() is called. Danda, '?', '!' are unambiguous and emitted immediately;
  * tiny sentences are merged forward (very short prompts can drift in voice identity per upstream README), except on
    flush(); the first chunk uses a lower threshold to reduce time-to-first-audio.
Whitespace-only splitting means Latin words inside Devanagari sentences (code-switching) are never cut in half.
"""
from __future__ import annotations

from dataclasses import dataclass

STRONG = set("?!।॥\n")
DOTS = set(".…")
CLAUSE = set(",;:،，；")
CLOSERS = set("\"'”’)]}»")
ABBREVIATIONS = {"dr", "mr", "mrs", "ms", "prof", "sr", "jr", "st", "vs", "etc", "no", "mt", "ltd", "inc", "co", "approx",
                 "dept", "govt", "sri", "smt", "shri", "e.g", "i.e", "a.m", "p.m", "am", "pm"}


@dataclass
class ChunkerConfig:
    first_min_chars: int = 12
    min_chars: int = 24
    clause_min_chars: int = 60
    first_clause_min_chars: int = 28
    max_chars: int = 240
    split_on_clause: bool = True


class IncrementalChunker:
    def __init__(self, cfg: ChunkerConfig | None = None):
        self.cfg = cfg or ChunkerConfig()
        self.buf = ""
        self.emitted = 0
        self.chars_in = 0

    # ---- public
    @property
    def pending(self) -> str:
        return self.buf

    def has_pending(self) -> bool:
        return bool(self.buf.strip())

    def feed(self, text: str) -> list[str]:
        self.buf += text
        self.chars_in += len(text)
        return self._extract(final=False, idle=False)

    def idle(self) -> list[str]:
        """No new text for a while: confirm held terminators ('.' at end of buffer). Does not cut mid-sentence."""
        return self._extract(final=False, idle=True)

    def flush(self) -> list[str]:
        """End of utterance/input: emit EVERYTHING left (never discards)."""
        return self._extract(final=True, idle=False)

    def clear(self) -> str:
        """Drop the buffer (barge-in). Returns the dropped text so the caller can report it."""
        dropped, self.buf = self.buf, ""
        return dropped

    # ---- internals
    def _thresholds(self):
        c = self.cfg
        first = self.emitted == 0
        return (c.first_min_chars if first else c.min_chars,
                c.first_clause_min_chars if first else c.clause_min_chars)

    @staticmethod
    def _word_before(buf: str, i: int) -> str:
        j = i
        while j > 0 and (buf[j - 1].isalpha() or buf[j - 1] == "."):
            j -= 1
        return buf[j:i].lower().strip(".")

    def _dot_is_boundary(self, buf: str, i: int, final: bool, idle: bool):
        """buf[i] is '.' or '…'. Returns (is_boundary, end_index_exclusive) or (False, None)."""
        n = len(buf)
        j = i
        while j + 1 < n and buf[j + 1] in DOTS:          # run of dots
            j += 1
        k = j + 1
        while k < n and buf[k] in CLOSERS:
            k += 1
        if k >= n:                                         # terminator at end of buffer: ambiguous
            return (True, k) if (final or idle) and not self._is_abbrev(buf, i) else (False, None)
        if not buf[k].isspace():
            return False, None                             # 3.5 / a.com / "..." followed by text
        if self._is_abbrev(buf, i):
            return False, None
        m = k
        while m < n and buf[m].isspace():
            m += 1
        if m < n and buf[m].islower() and buf[m].isascii():
            return False, None                             # "e.g. something" / lowercase continuation
        if m >= n and not (final or idle):
            return False, None                             # whitespace only after it: wait for the next word
        return True, k

    def _is_abbrev(self, buf: str, i: int) -> bool:
        w = self._word_before(buf, i)
        if not w:
            return False          # dot after a digit/symbol ("paid 500.") ends the sentence; "3.5" is handled earlier
        if w in ABBREVIATIONS:
            return True
        if len(w) == 1 and w.isalpha() and buf[i - 1].isupper() and buf[i] == ".":
            return True                                    # initial: "A. Kumar"
        return False

    def _find_boundary(self, final: bool, idle: bool, strong_min: int, clause_min: int):
        buf, n = self.buf, len(self.buf)
        i = 0
        while i < n:
            ch = buf[i]
            end = None
            clause = False
            if ch in STRONG:
                end = i + 1
                while end < n and (buf[end] in STRONG or buf[end] in CLOSERS):
                    end += 1
            elif ch in DOTS:
                ok, e = self._dot_is_boundary(buf, i, final, idle)
                if ok:
                    end = e
            elif ch in CLAUSE and self.cfg.split_on_clause:
                if i + 1 < n and buf[i + 1].isspace():
                    end, clause = i + 1, True
            if end is not None:
                length = len(buf[:end].strip())
                if length >= (clause_min if clause else strong_min):
                    return end
                i = max(i + 1, end if not clause else i + 1)
                continue
            i += 1
        return None

    def _split_overlong(self):
        buf, mx = self.buf, self.cfg.max_chars
        if len(buf) <= mx:
            return None
        window = buf[:mx]
        cut = max(window.rfind(" "), window.rfind("\n"))
        if cut < mx * 0.4:
            nxt = buf.find(" ", mx)                       # no usable space: extend to the next space, never mid-word
            if nxt == -1:
                return None if len(buf) < mx * 2 else mx   # unbroken blob: hard split only if absurdly long
            cut = nxt
        return cut

    def _extract(self, final: bool, idle: bool) -> list[str]:
        out: list[str] = []
        while self.buf.strip():
            strong_min, clause_min = self._thresholds()
            end = self._find_boundary(final, idle, strong_min, clause_min)
            if end is None:
                end = self._split_overlong()
            if end is None:
                break
            chunk, self.buf = self.buf[:end].strip(), self.buf[end:].lstrip(" \t")
            if chunk:
                out.append(chunk)
                self.emitted += 1
        if final:
            rest = self.buf.strip()
            self.buf = ""
            if rest:
                out.append(rest)
                self.emitted += 1
        elif not self.buf.strip():
            self.buf = ""
        return out
