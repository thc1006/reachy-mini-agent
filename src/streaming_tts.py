"""Streaming TTS pipeline: SentenceChunker + TTSQueue.

Consumes a stream of text deltas, chunks into complete sentences, generates
TTS concurrently, plays audio back in submission order.

Goal: start playing sentence 1 while LLM still generating sentence 3,
halving perceived latency vs buffered synth.
"""
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Callable, Optional, List, Any


# Common English abbreviations that contain a trailing period but don't
# terminate a sentence. Short list — robot dialog doesn't need PhD NLP.
_ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "st", "jr", "sr", "prof", "vs", "etc"}

# Any run of sentence-terminating punctuation: . ! ? including repeated.
# Also supports full-width CJK terminators so Chinese replies stream sentence-
# by-sentence instead of landing as one giant chunk.
_TERM_RE = re.compile(r"[.!?。！？]+")


class SentenceChunker:
    """Incremental sentence segmenter for streaming LLM output.

    Usage:
        ch = SentenceChunker()
        for delta in token_stream:
            for sentence in ch.feed(delta):
                yield sentence        # full sentence with its terminator
        for sentence in ch.finalize():
            yield sentence            # flush any partial
    """

    def __init__(self):
        self._buf: str = ""

    def feed(self, delta: str) -> List[str]:
        if not delta:
            return []
        self._buf += delta
        return self._extract(require_boundary_after=True)

    def finalize(self) -> List[str]:
        out = self._extract(require_boundary_after=False)
        tail = self._buf.strip()
        if tail and any(c.isalpha() for c in tail):
            out.append(tail)
        self._buf = ""
        return out

    # --- internals --------------------------------------------------------

    def _is_real_terminator(self, buf: str, start: int, end: int) -> bool:
        # Decimal inside number: "3.14" — digit-dot-digit pattern
        if end - start == 1 and buf[start] == ".":
            if start > 0 and buf[start - 1].isdigit():
                if end < len(buf) and buf[end].isdigit():
                    return False
        # Abbreviation: the word immediately before (lowercase, stripped of
        # trailing dot) appears in the abbreviation list
        j = start - 1
        while j >= 0 and not buf[j].isspace():
            j -= 1
        word = buf[j + 1:start].lower().rstrip(".")
        if word in _ABBREVIATIONS:
            return False
        return True

    def _extract(self, require_boundary_after: bool) -> List[str]:
        out: List[str] = []
        buf = self._buf
        cursor = 0
        i = 0
        while i < len(buf):
            m = _TERM_RE.search(buf, i)
            if not m:
                break
            t_start, t_end = m.start(), m.end()
            if not self._is_real_terminator(buf, t_start, t_end):
                i = t_end
                continue
            if require_boundary_after:
                if t_end >= len(buf):
                    break  # partial — wait for more
                if not buf[t_end].isspace():
                    # e.g. "3.14" — already filtered above, but also for
                    # cases like "word.word" (no space) — treat as continuation
                    i = t_end
                    continue
            sentence = buf[cursor:t_end].strip()
            if sentence and any(c.isalpha() for c in sentence):
                out.append(sentence)
            cursor = t_end
            i = t_end
        self._buf = buf[cursor:].lstrip() if cursor > 0 else buf
        return out


class SpeechStreamExtractor:
    """Extract the value of the `speech` JSON field from a streaming raw string.

    The LLM emits tokens like:
        {"speech":"Hello there!","actions":["nod"]}
    Or less obediently:
        {"speech" : "Hello…"}        (spaces around colon / quote)
        {"actions":[], "speech":"…"}
    We watch for the `"speech":"` marker (whitespace-tolerant, may straddle
    deltas), then emit characters as they arrive until the unescaped closing
    `"`. Escape sequences (\\" \\n \\t \\\\) are decoded on the fly.

    Safe to feed even if the marker never appears (returns "" each call).
    If stream ends mid-speech (truncated output), call `finalize()` — no extra
    work, but you get any buffered chars already emitted via feed().
    """

    # Whitespace tolerance: "speech":" and "speech" : " and "speech"\n:\n" all hit.
    _MARKER_RE = re.compile(r'"\s*speech\s*"\s*:\s*"')

    # If after this many chars no JSON marker appears, fall back to plain-text.
    # qwen3.6 with enable_thinking=False tends to drop the JSON wrapper and just
    # speak directly, e.g. "Think! 🤖 Hello! ...\n\nactions: [...]". We pass
    # such text through (after the strippers below) instead of staying silent.
    _PLAIN_FALLBACK_THRESHOLD = 60

    # Strip preambles the model sometimes emits in non-thinking mode.
    _PREAMBLE_RE = re.compile(r"^(?:Think|Nod|Wave|Happy|Sad)!\s*[\U0001F300-\U0001FAFF]*\s*", re.UNICODE)
    # Strip trailing `actions: [...]` text-form (not real JSON).
    _ACTIONS_TAIL_RE = re.compile(r"\s*\n*\s*actions\s*:\s*\[.*?\]\s*$", re.DOTALL | re.IGNORECASE)

    def __init__(self) -> None:
        self._buf: str = ""
        self._state: str = "hunting"   # hunting → in_speech → done; or → plain
        self._pos: int = 0             # index in _buf we've processed up to
        self._plain_emitted: int = 0   # chars emitted in plain mode

    def _enter_plain_if_needed(self) -> None:
        """Switch to plain mode if we've buffered enough chars with no JSON marker."""
        if self._state != "hunting":
            return
        if len(self._buf) < self._PLAIN_FALLBACK_THRESHOLD:
            return
        # If buffer doesn't contain a `{` early on, the model isn't producing JSON.
        # Switch to plain pass-through mode.
        early = self._buf[: self._PLAIN_FALLBACK_THRESHOLD]
        if early.lstrip().startswith("{"):
            return  # JSON-like start; keep hunting for the marker
        self._state = "plain"

    def _plain_visible(self) -> str:
        """Return the cleaned-up buf for plain mode: strip preamble + trailing actions tail."""
        s = self._PREAMBLE_RE.sub("", self._buf, count=1)
        s = self._ACTIONS_TAIL_RE.sub("", s)
        return s

    def feed(self, delta: str) -> str:
        if not delta:
            return ""
        self._buf += delta
        out = ""
        if self._state == "hunting":
            m = self._MARKER_RE.search(self._buf)
            if m is not None:
                self._pos = m.end()
                self._state = "in_speech"
            else:
                self._enter_plain_if_needed()
                if self._state != "plain":
                    return ""
        if self._state == "plain":
            cleaned = self._plain_visible()
            if len(cleaned) > self._plain_emitted:
                out = cleaned[self._plain_emitted:]
                self._plain_emitted = len(cleaned)
            return out
        if self._state == "in_speech":
            i = self._pos
            buf = self._buf
            while i < len(buf):
                c = buf[i]
                if c == "\\":
                    # Need the next char to know the escape; if not arrived yet, wait.
                    if i + 1 >= len(buf):
                        break
                    esc = buf[i + 1]
                    if   esc == '"':  out += '"'
                    elif esc == "n":  out += "\n"
                    elif esc == "t":  out += "\t"
                    elif esc == "\\": out += "\\"
                    elif esc == "r":  out += "\r"
                    elif esc == "/":  out += "/"
                    else:             out += esc
                    i += 2
                elif c == '"':
                    self._state = "done"
                    self._pos = i + 1
                    return out
                else:
                    out += c
                    i += 1
            self._pos = i
        return out

    @property
    def done(self) -> bool:
        return self._state in ("done", "plain")


class TTSQueue:
    """Concurrent TTS synthesis with in-order playback.

    - `engine.synthesize(text) -> (samples, sr)` generates audio.
    - Multiple workers run in parallel (max_concurrent).
    - on_audio callback fires in submission order regardless of completion order.
    - Exceptions isolated: a failed slot triggers on_error, playback continues.
    """

    def __init__(self, engine: Any, max_concurrent: int = 2,
                 on_error: Optional[Callable[[BaseException], None]] = None):
        self.engine = engine
        self.max_concurrent = max_concurrent
        self.on_error = on_error or (lambda e: None)
        self.on_audio: Optional[Callable[[Any, int], None]] = None
        self._executor: Optional[ThreadPoolExecutor] = None
        self._futures: List[Future] = []
        self._lock = threading.Lock()
        self._player_thread: Optional[threading.Thread] = None
        self._started = False
        self._closing = threading.Event()

    def start(self, on_audio: Callable[[Any, int], None]) -> None:
        if self._started:
            return
        self.on_audio = on_audio
        self._executor = ThreadPoolExecutor(max_workers=self.max_concurrent,
                                             thread_name_prefix="tts-gen")
        self._started = True
        self._player_thread = threading.Thread(
            target=self._player_loop, name="tts-player", daemon=True)
        self._player_thread.start()

    def submit(self, text: str) -> None:
        if not self._started:
            raise RuntimeError("TTSQueue.submit() called before start()")
        fut = self._executor.submit(self._synth, text)
        with self._lock:
            self._futures.append(fut)

    def close(self, timeout: float = 10.0) -> None:
        self._closing.set()
        if self._player_thread is not None:
            self._player_thread.join(timeout=timeout)
        if self._executor is not None:
            self._executor.shutdown(wait=True)

    # --- internals --------------------------------------------------------

    def _synth(self, text: str):
        return self.engine.synthesize(text)

    def _player_loop(self) -> None:
        idx = 0
        while True:
            with self._lock:
                fut = self._futures[idx] if idx < len(self._futures) else None
            if fut is not None:
                try:
                    samples, sr = fut.result()
                    if self.on_audio is not None:
                        self.on_audio(samples, sr)
                except BaseException as e:
                    try:
                        self.on_error(e)
                    except Exception:
                        pass
                idx += 1
                continue
            # no pending future
            if self._closing.is_set():
                with self._lock:
                    if idx >= len(self._futures):
                        return
            time.sleep(0.005)
