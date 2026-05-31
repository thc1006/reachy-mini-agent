"""Elder-care features (P6/P7/P8) — gated behind ELDER_CARE_MODE=1.

Three additive features, all NO-OP unless ELDER_CARE_MODE is truthy:

  P6  is_emergency / handle_emergency
      Pre-LLM regex check on Whisper output. On match: log, speak
      acknowledgment, POST optional webhook, skip LLM entirely (sub-500ms
      response budget — research: ElliQ critique + arXiv elder-care 2024).

  P7  edge_tts_rate / extra_post_tts_pause_s
      Edge TTS rate=-10% + post-TTS pause +400ms. PMC10917141: elderly
      speech-perception studies show ~10% slower rate + longer inter-turn
      gap measurably improves intelligibility and turn-taking comfort.

  P8  antenna_cue_call
      Visible "I am listening" cue: gentle forward dip of both antennas
      when VAD detects speech onset. Reachy antennas are real motors;
      same `mini.goto_target(antennas=...)` infra used elsewhere in
      do_action(). Neutral on done/speaking.

Design: pure functions returning data; the robot_brain.py hooks call them
and apply the result. Keeps unit tests hardware-free (no SDK/audio import).
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.request
from typing import Any, Optional

# numpy is the production form (matches do_action() convention), but unit
# tests may run without it. Hoisted to module-top so we import once, not per
# antenna cue (per A4 code-review NIT).
try:
    import numpy as _np
    _HAS_NUMPY = True
except ImportError:
    _np = None
    _HAS_NUMPY = False

# Serialize concurrent appends to the JSONL emergency log so two audio
# threads racing on the same file can't interleave bytes (per A4 code-review
# F2). Cheap; only contended during a real emergency burst.
_emergency_log_lock = threading.Lock()


# ── Env helpers ───────────────────────────────────────────────────────────
# Track D-2 (2026-06-01): bool parsing moved to _env. Local alias kept for
# backward compatibility with any external importers of `elder_care._truthy`.
from _env import env_bool, is_truthy as _truthy  # noqa: E402,F401


def elder_mode_enabled() -> bool:
    """Master gate. False => all helpers no-op / return safe defaults."""
    return env_bool("ELDER_CARE_MODE", default=False)


# ── P6: Emergency phrase route ────────────────────────────────────────────

# Case-insensitive multi-lingual emergency regex. Covers:
#   - distress: 救命 / help me / emergency
#   - fall:     跌倒 / fall / fallen / falling
#   - medical:  胸痛 / chest pain / 不能呼吸 / can't breathe
#   - call X:   叫女兒/兒子/護士/爸/媽 / call (my) daughter/son/nurse/mom/dad/911
# English keywords carry explicit \b word boundaries — without them, "football
# fall game" matches "fall" and "yelphelp" matches "help". Chinese keywords
# stay bare (CJK char-boundaries don't play well with \b).
# Per A4 code-review F1: 2026-05-29 reviewers found 4+ false positives in the
# v1 pattern; this is v2 with word boundaries + post-match negation pre-check.
# English keywords carry explicit \b word boundaries. "fall" alone is too
# ambiguous (the season, "fall semester", "football fall game") — require
# the past/progressive form (fallen/falling), or the Chinese 跌倒 for the
# bare emergency. Chinese keywords stay bare (CJK char-boundaries don't
# play well with \b).
# Per A4 code-review F1 / fix-loop 2026-05-29: word boundaries + post-match
# wide-window negation scan, prefer false-NEGATIVES on literary phrasings
# over false-POSITIVES that trigger spurious family alerts.
EMERGENCY_PATTERN = re.compile(
    r"(救命|\bhelp\s+me\b|\bemergency\b|跌倒|\bfall(en|ing)\b|"
    r"胸痛|\bchest\s+pain\b|不能呼吸|\bcan'?t\s+breathe\b|"
    r"叫(我)?(的)?(女兒|兒子|護士|爸|媽|爸爸|媽媽)|"
    r"\bcall\s+(my\s+)?(daughter|son|nurse|mom|dad|mother|father|911)\b)",
    re.IGNORECASE,
)

# Negators scanned ACROSS the whole preceding ~40-char window (not just at
# end). This is intentionally aggressive — for elder care, a spurious 911
# from "I don't need help" is FAR worse than a missed "I'm not joking, help
# me" (elders won't phrase real emergencies sarcastically).
_NEG_EN = re.compile(r"\b(not|don'?t|do\s+not|won'?t|wouldn'?t|never|no\s+need)\b", re.IGNORECASE)
_NEG_ZH = re.compile(r"(不要|不需要|不可能|不會|無需|別|沒有?|不\s*會)")

EMERGENCY_LOG_PATH_DEFAULT = "/home/pollen/brain/logs/emergency.log"
# Renamed alias kept for backward-compat with v1 test imports.
EMERGENCY_LOG_PATH = EMERGENCY_LOG_PATH_DEFAULT
EMERGENCY_ACK_ZH = "我聽到了，正在通知你的家人。"
EMERGENCY_ACK_EN = "I heard you. I'm alerting your family now."


def _emergency_log_path() -> str:
    """Resolve the active log path: ELDER_EMERGENCY_LOG_PATH env wins, else
    EMERGENCY_LOG_PATH_DEFAULT. Per A4 code-review F2 — hardcoded POSIX path
    was breaking CI on Windows."""
    return os.getenv("ELDER_EMERGENCY_LOG_PATH", EMERGENCY_LOG_PATH_DEFAULT)


def _has_negation_prefix(text_before: str) -> bool:
    """True if the last ~40 chars of `text_before` contain ANY English or
    Chinese negator. Wider than strict-prefix because elder speech often
    inserts intermediate words ("I don't need help") — the negator and the
    keyword can be separated by 1-3 words but still mean refusal."""
    if not text_before:
        return False
    window = text_before[-40:]
    if _NEG_EN.search(window):
        return True
    if _NEG_ZH.search(window):
        return True
    return False


def is_emergency(text: str) -> Optional[str]:
    """Return the matched substring if `text` matches an emergency phrase
    AND is not preceded by a negator, else None.

    Empty/None input returns None. The negation pre-check rejects common
    refusal patterns ("I don't need help me" / "我不需要 help") that the
    bare regex would otherwise false-positive on."""
    if not text:
        return None
    m = EMERGENCY_PATTERN.search(text)
    if m is None:
        return None
    if _has_negation_prefix(text[:m.start()]):
        return None
    return m.group(0)


def _looks_chinese(text: str) -> bool:
    """Naive: any CJK Unified Ideograph in the string."""
    return any("一" <= ch <= "鿿" for ch in text or "")


def emergency_ack_text(stt_text: str) -> str:
    """Pick zh-TW or en acknowledgment based on STT language."""
    return EMERGENCY_ACK_ZH if _looks_chinese(stt_text) else EMERGENCY_ACK_EN


def log_emergency(stt_text: str, phrase: str, log_path: Optional[str] = None) -> bool:
    """Append a JSONL record. Returns True on success, False on any I/O
    error (caller should not abort emergency handling on log failure).

    log_path=None resolves via ELDER_EMERGENCY_LOG_PATH env (default the POSIX
    Pi path). Thread-safe: serialized via _emergency_log_lock so concurrent
    appends don't interleave bytes."""
    if log_path is None:
        log_path = _emergency_log_path()
    record = {
        "ts": time.time(),
        "iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "phrase": phrase,
        "text": stt_text,
        "robot": "reachy-mini",
    }
    try:
        with _emergency_log_lock:
            dirpath = os.path.dirname(log_path)
            if dirpath:
                os.makedirs(dirpath, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return True
    except Exception as e:
        print(f"  [elder/emergency log fail] {e}")
        return False


def post_webhook(stt_text: str, phrase: str,
                 url: Optional[str] = None,
                 timeout: float = 3.0) -> bool:
    """POST emergency payload to ELDER_EMERGENCY_WEBHOOK_URL.
    Returns True on 2xx, False on any error or when URL is empty/unset.
    Never raises — caller relies on this never aborting the alert.

    Per A4 code-review F4: only http(s) URLs are accepted; file:// / ftp://
    etc. would otherwise be honored by urllib.request.urlopen.
    Per A4 architect+code F3: caller should fire this in a daemon thread to
    avoid blocking the audio loop on a wedged webhook — see handle_emergency."""
    if url is None:
        url = os.getenv("ELDER_EMERGENCY_WEBHOOK_URL", "").strip()
    if not url:
        return False
    if not url.lower().startswith(("http://", "https://")):
        print(f"  [elder/webhook] refusing non-http(s) URL scheme")
        return False
    payload = {
        "ts": time.time(),
        "phrase": phrase,
        "text": stt_text,
        "robot": "reachy-mini",
    }
    try:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, method="POST",
            headers={"Content-Type": "application/json; charset=utf-8"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except Exception as e:
        print(f"  [elder/webhook fail] {type(e).__name__}: {e}")
        return False


def handle_emergency(stt_text: str, phrase: str,
                     speak_fn: Optional[Any] = None,
                     mini: Optional[Any] = None,
                     log_path: Optional[str] = None,
                     webhook_async: bool = True) -> dict:
    """Full emergency response: log → speak ack → POST webhook.
    Returns a dict with each step's result for caller logging/tests.

    `speak_fn` is called as speak_fn(mini, ack_text) — matches the existing
    speak(mini, text) signature in robot_brain.py. If None, ack is skipped
    (used in tests / when speech path itself is what's failing).

    `webhook_async=True` (default) fires the webhook in a daemon thread so a
    wedged endpoint can't block the audio/dialog loop for the urllib timeout
    (per A4 architect+code F3 — was reliably stalling next mic open by ≤3s).
    Tests can pass webhook_async=False for synchronous result inspection."""
    ack = emergency_ack_text(stt_text)
    result = {
        "phrase": phrase,
        "ack": ack,
        "logged": log_emergency(stt_text, phrase, log_path=log_path),
        "spoke": False,
        "webhook": None,
    }
    print(f"  [ELDER EMERGENCY] matched={phrase!r} text={stt_text!r}")
    if speak_fn is not None:
        try:
            speak_fn(mini, ack)
            result["spoke"] = True
        except Exception as e:
            print(f"  [elder/speak fail] {e}")
    if webhook_async:
        # Early bail when no URL configured: avoid spawning a thread that
        # would just return False — keeps result semantics honest.
        url = os.getenv("ELDER_EMERGENCY_WEBHOOK_URL", "").strip()
        if not url:
            result["webhook"] = False
        else:
            # Fire-and-forget so audio loop returns to mic immediately.
            threading.Thread(
                target=post_webhook, args=(stt_text, phrase),
                daemon=True, name="elder-webhook",
            ).start()
            result["webhook"] = "dispatched_async"
    else:
        result["webhook"] = post_webhook(stt_text, phrase)
    return result


# ── P7: TTS pacing for elderly ────────────────────────────────────────────

def edge_tts_rate() -> str:
    """Edge TTS rate parameter string. Default -10% under elder mode
    (PMC10917141). Empty string => use Edge default."""
    if not elder_mode_enabled():
        return ""
    # Reserved for future override; current research target is fixed -10%.
    return "-10%"


def extra_post_tts_pause_s() -> float:
    """Additional sleep AFTER existing TTS_TAIL_DRAIN_S, only when elder
    mode is on. Default 400 ms (PMC10917141 inter-turn gap recommendation)."""
    if not elder_mode_enabled():
        return 0.0
    try:
        ms = int(os.getenv("ELDER_PAUSE_MS", "400"))
    except ValueError:
        ms = 400
    if ms < 0:
        ms = 0
    return ms / 1000.0


# ── P8: Antenna visual listening cue ──────────────────────────────────────

def _antenna_dip_rad() -> float:
    if not elder_mode_enabled():
        return 0.0
    try:
        v = float(os.getenv("ELDER_ANTENNA_DIP_RAD", "0.15"))
    except ValueError:
        v = 0.15
    # Clamp to a safe, non-jerky range. Reachy antennas physically travel
    # well past 0.5 rad but we want a subtle social cue, not a wave.
    if v < 0.0:
        v = 0.0
    if v > 0.5:
        v = 0.5
    return v


def antenna_cue_call(state: str) -> Optional[dict]:
    """Build the kwargs dict to pass to mini.goto_target() for the listening
    cue. Returns None when elder mode is off or `state` is unknown — caller
    then skips the motor command entirely.

    state ∈ {"listening", "neutral"}:
      listening → both antennas dip forward by ELDER_ANTENNA_DIP_RAD
      neutral   → both antennas return to 0

    Existing brain code passes antennas as a 2-element array [right, left].
    We match that convention exactly to stay drop-in with do_action()."""
    if not elder_mode_enabled():
        return None
    if state == "listening":
        dip = _antenna_dip_rad()
        return {
            "antennas": [dip, dip],
            "duration": 0.2,
            "blocking": False,
        }
    if state == "neutral":
        return {
            "antennas": [0.0, 0.0],
            "duration": 0.2,
            "blocking": False,
        }
    return None


def fire_antenna_cue(mini: Any, state: str, motion_lock: Optional[Any] = None) -> bool:
    """Best-effort apply antenna_cue_call to the live `mini` SDK handle.
    Returns True if a motor command was attempted, False if skipped
    (elder mode off / unknown state / mini None / SDK raised / lock busy).

    Safe to call from VAD callback threads — never raises.

    Per A4 architect F3: when `motion_lock` is provided, acquire it
    NON-BLOCKING before issuing the SDK call. If the lock is busy (dance /
    emotion / other motor worker active) we skip the cue silently rather
    than race interleaved goto_target() calls and produce jerky motion."""
    call = antenna_cue_call(state)
    if call is None or mini is None:
        return False
    acquired = False
    if motion_lock is not None:
        acquired = motion_lock.acquire(blocking=False)
        if not acquired:
            # Active motion holding the lock — skip cue to keep the dance smooth.
            return False
    try:
        ants = call["antennas"]
        ants_arg = _np.array(ants, dtype=float) if _HAS_NUMPY else list(ants)
        mini.goto_target(
            antennas=ants_arg,
            duration=call["duration"],
            method="minjerk",
        )
        return True
    except Exception as e:
        print(f"  [elder/antenna cue fail] {type(e).__name__}: {e}")
        return False
    finally:
        if acquired and motion_lock is not None:
            try:
                motion_lock.release()
            except Exception:
                pass
