import random
import re

import pytest

from server.text_chunker import ChunkerConfig, IncrementalChunker


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


def run(text, step=None, cfg=None):
    c = IncrementalChunker(cfg)
    out = []
    if step is None:
        out += c.feed(text)
    else:
        for i in range(0, len(text), step):
            out += c.feed(text[i:i + step])
    out += c.flush()
    return out


def test_english_sentences_split_after_confirmation():
    c = IncrementalChunker()
    assert c.feed("Your appointment is confirmed for tomorrow.") == []        # '.' at end of buffer is held
    assert c.feed(" Please") == ["Your appointment is confirmed for tomorrow."]
    assert c.flush() == ["Please"]


def test_danda_is_immediate():
    c = IncrementalChunker()
    assert c.feed("आपका ऑर्डर कल पहुँच जाएगा।") == ["आपका ऑर्डर कल पहुँच जाएगा।"]
    assert c.feed("धन्यवाद, आपका दिन शुभ हो॥") == ["धन्यवाद, आपका दिन शुभ हो॥"]


def test_question_exclaim_immediate():
    c = IncrementalChunker()
    assert c.feed("क्या आप कल आ सकते हैं?") == ["क्या आप कल आ सकते हैं?"]


@pytest.mark.parametrize("text", [
    "The fee is 3.5 lakh rupees per year for the full course today",
    "Please ask Dr. Sharma about the admission process for next year",
    "Call us at 10 a.m. or 5 p.m. on any working day of the week",
    "Visit www.example.com for details about the scholarship program",
    "Meet A. Kumar at the counter for the registration of your documents",
    "Use e.g. a passport or i.e. a government photo identity card now",
])
def test_no_false_boundaries(text):
    chunks = run(text + ".")
    assert len(chunks) == 1, chunks


def test_decimal_split_across_deltas():
    c = IncrementalChunker()
    out = c.feed("The fee is 3.") + c.feed("5 lakh per year for everyone enrolled.") + c.flush()
    assert out == ["The fee is 3.5 lakh per year for everyone enrolled."]


def test_short_sentences_merge_but_first_is_quick():
    cfg = ChunkerConfig(first_min_chars=10, min_chars=24)
    out = run("Sure thing. Okay. Great. Let me check that for you now. Done.", cfg=cfg)
    assert out[0] == "Sure thing."                       # first chunk threshold is small
    assert all(len(c) >= 24 for c in out[1:-1]), out


def test_clause_split_only_when_long_enough():
    cfg = ChunkerConfig(first_clause_min_chars=28, first_min_chars=12)
    c = IncrementalChunker(cfg)
    out = c.feed("Hello, how are you doing today, I hope everything is fine with you ")
    assert out[0] == "Hello, how are you doing today,"
    assert run("Hi, ok", cfg=cfg) == ["Hi, ok"]


def test_long_text_without_punctuation_is_split_on_spaces_not_words():
    words = ["word%d" % i for i in range(200)]
    chunks = run(" ".join(words), cfg=ChunkerConfig(max_chars=100))
    assert all(len(c) <= 110 for c in chunks)
    assert norm(" ".join(chunks)) == " ".join(words)       # nothing lost, no word cut


def test_code_switching_not_cut_mid_word():
    text = "आपका admission confirm हो गया है। Please carry your ID card और original documents साथ लाएँ।"
    chunks = run(text, step=3)
    assert norm(" ".join(chunks)) == norm(text)
    assert any("admission" in c for c in chunks) and any("ID card" in c for c in chunks)


def test_idle_confirms_trailing_dot_but_not_midsentence():
    c = IncrementalChunker(ChunkerConfig(first_min_chars=5))
    assert c.feed("Hello there.") == []
    assert c.idle() == ["Hello there."]
    c.feed("And then we")
    assert c.idle() == [] and c.pending == "And then we"
    assert c.flush() == ["And then we"]


def test_clear_returns_dropped_text():
    c = IncrementalChunker()
    c.feed("partial sentence without end")
    assert c.clear() == "partial sentence without end" and not c.has_pending()


def test_flush_never_discards_tiny_text():
    assert run("ok") == ["ok"]
    assert run("।") == ["।"]
    assert run("   ") == []


def test_property_nothing_lost_random_chunking():
    rnd = random.Random(7)
    corpus = ("नमस्ते! आपका ऑर्डर नंबर 4521 है। Total is Rs. 1,250.50, paid on 3.5.2026 at 10 a.m. Dr. Rao will call. "
              "क्या आप तैयार हैं? Yes... maybe; okay: fine, great — see www.mme.in/apply or mail a.b@c.com. Thanks…")
    for _ in range(200):
        step = rnd.randint(1, 17)
        chunks = run(corpus, step=step, cfg=ChunkerConfig(max_chars=rnd.choice([40, 80, 240])))
        assert norm(" ".join(chunks)) == norm(corpus), (step, chunks)
        assert all(c.strip() for c in chunks)
