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
import time
import urllib.request
from typing import Any, Optional


# ── Env helpers ───────────────────────────────────────────────────────────

def _truthy(v: str) -> bool:
    return v.strip().lower() not in ("", "0", "false", "no", "off")


def elder_mode_enabled() -> bool:
    """Master gate. False => all helpers no-op / return safe defaults."""
    return _truthy(os.getenv("ELDER_CARE_MODE", "0"))


# ── P6: Emergency phrase route ────────────────────────────────────────────

# Case-insensitive multi-lingual emergency regex. Covers:
#   - distress: 救命 / help me / emergency
#   - fall:     跌倒 / fall / fallen / falling
#   - medical:  胸痛 / chest pain / 不能呼吸 / can't breathe
#   - call X:   叫女兒/兒子/護士/爸/媽 / call (my) daughter/son/nurse/mom/dad/911
EMERGENCY_PATTERN = re.compile(
    r"(救命|help\s*me|emergency|跌倒|fall(en|ing)?|"
    r"胸痛|chest\s*pain|不能呼吸|can'?t\s*breathe|"
    r"叫.*(女兒|兒子|護士|爸|媽)|call\s+(my\s+)?(daughter|son|nurse|mom|dad|911))",
    re.IGNORECASE,
)

EMERGENCY_LOG_PATH = "/home/pollen/brain/logs/emergency.log"
EMERGENCY_ACK_ZH = "我聽到了，正在通知你的家人。"
EMERGENCY_ACK_EN = "I heard you. I'm alerting your family now."


def is_emergency(text: str) -> Optional[str]:
    """Return the matched substring if `text` matches an emergency phrase,
    else None. Empty/None input returns None."""
    if not text:
        return None
    m = EMERGENCY_PATTERN.search(text)
    if m is None:
        return None
    return m.group(0)


def _looks_chinese(text: str) -> bool:
    """Naive: any CJK Unified Ideograph in the string."""
    return any("一" <= ch <= "鿿" for ch in text or "")


def emergency_ack_text(stt_text: str) -> str:
    """Pick zh-TW or en acknowledgment based on STT language."""
    return EMERGENCY_ACK_ZH if _looks_chinese(stt_text) else EMERGENCY_ACK_EN


def log_emergency(stt_text: str, phrase: str, log_path: str = EMERGENCY_LOG_PATH) -> bool:
    """Append a JSONL record. Returns True on success, False on any I/O
    error (caller should not abort emergency handling on log failure)."""
    record = {
        "ts": time.time(),
        "iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "phrase": phrase,
        "text": stt_text,
        "robot": "reachy-mini",
    }
    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
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
    Never raises — caller relies on this never aborting the alert."""
    if url is None:
        url = os.getenv("ELDER_EMERGENCY_WEBHOOK_URL", "").strip()
    if not url:
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
                     log_path: str = EMERGENCY_LOG_PATH) -> dict:
    """Full emergency response: log → speak ack → POST webhook.
    Returns a dict with each step's result for caller logging/tests.

    `speak_fn` is called as speak_fn(mini, ack_text) — matches the existing
    speak(mini, text) signature in robot_brain.py. If None, ack is skipped
    (used in tests / when speech path itself is what's failing)."""
    ack = emergency_ack_text(stt_text)
    result = {
        "phrase": phrase,
        "ack": ack,
        "logged": log_emergency(stt_text, phrase, log_path=log_path),
        "spoke": False,
        "webhook": False,
    }
    print(f"  [ELDER EMERGENCY] matched={phrase!r} text={stt_text!r}")
    if speak_fn is not None:
        try:
            speak_fn(mini, ack)
            result["spoke"] = True
        except Exception as e:
            print(f"  [elder/speak fail] {e}")
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


def fire_antenna_cue(mini: Any, state: str) -> bool:
    """Best-effort apply antenna_cue_call to the live `mini` SDK handle.
    Returns True if a motor command was attempted, False if skipped
    (elder mode off / unknown state / mini None / SDK raised).

    Safe to call from VAD callback threads — never raises."""
    call = antenna_cue_call(state)
    if call is None or mini is None:
        return False
    try:
        # numpy is the production form (matches do_action() convention), but
        # the SDK accepts plain sequences too — fall back to a list so this
        # function is unit-testable in CI environments without numpy.
        ants = call["antennas"]
        try:
            import numpy as np
            ants_arg = np.array(ants, dtype=float)
        except ImportError:
            ants_arg = list(ants)
        mini.goto_target(
            antennas=ants_arg,
            duration=call["duration"],
            method="minjerk",
        )
        return True
    except Exception as e:
        # Antennas dying is never worth blocking the conversation.
        print(f"  [elder/antenna cue fail] {type(e).__name__}: {e}")
        return False
