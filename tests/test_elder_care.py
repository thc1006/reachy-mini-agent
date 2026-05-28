"""Elder-care P6/P7/P8 tests.

P6 — emergency phrase regex + webhook + log
P7 — TTS rate string + extra inter-turn pause
P8 — antenna cue kwargs construction

Hardware-free: elder_care is a pure-Python module (only stdlib +
urllib). robot_brain hooks are spot-checked by source-text scan.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

# Make src/ importable (matches the convention used by other tests).
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import elder_care  # noqa: E402


# ── shared fixture: toggle ELDER_CARE_MODE per test ──────────────────────

@pytest.fixture
def elder_on(monkeypatch):
    monkeypatch.setenv("ELDER_CARE_MODE", "1")
    yield


@pytest.fixture
def elder_off(monkeypatch):
    monkeypatch.delenv("ELDER_CARE_MODE", raising=False)
    yield


# ── P6: emergency phrase route ───────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "救命",
    "我跌倒了",
    "help me",
    "Help Me!",
    "I have chest pain",
    "我胸痛",
    "I can't breathe",
    "can't breathe please",
    "叫我女兒過來",
    "請叫護士",
    "Call my daughter",
    "please call 911",
    "Emergency please",
    "I'm falling",
    "she has fallen",
])
def test_is_emergency_positive_matches(text):
    matched = elder_care.is_emergency(text)
    assert matched is not None, f"expected match for {text!r}"


@pytest.mark.parametrize("text", [
    "我跳支舞",
    "我想去散步",
    "Let's dance",
    "I love this song",
    "What's the time",
    "幫我訂披薩",
    "我跟我朋友聊天",
    "",
    None,
])
def test_is_emergency_negative_cases(text):
    assert elder_care.is_emergency(text) is None


def test_emergency_ack_text_chinese_vs_english():
    assert elder_care.emergency_ack_text("救命") == elder_care.EMERGENCY_ACK_ZH
    assert elder_care.emergency_ack_text("我跌倒了") == elder_care.EMERGENCY_ACK_ZH
    assert elder_care.emergency_ack_text("help me") == elder_care.EMERGENCY_ACK_EN
    assert elder_care.emergency_ack_text("chest pain") == elder_care.EMERGENCY_ACK_EN


def test_log_emergency_writes_jsonl(tmp_path):
    log_path = tmp_path / "emergency.log"
    ok = elder_care.log_emergency("救命啊", "救命", log_path=str(log_path))
    assert ok is True
    assert log_path.exists()
    line = log_path.read_text(encoding="utf-8").strip()
    rec = json.loads(line)
    assert rec["phrase"] == "救命"
    assert rec["text"] == "救命啊"
    assert rec["robot"] == "reachy-mini"
    assert "ts" in rec and isinstance(rec["ts"], (int, float))


def test_log_emergency_appends_multiple_records(tmp_path):
    log_path = tmp_path / "emergency.log"
    elder_care.log_emergency("救命", "救命", log_path=str(log_path))
    elder_care.log_emergency("help me", "help me", log_path=str(log_path))
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2


def test_log_emergency_safe_on_unwritable_path():
    """Bad path must return False, never raise — emergency path can't crash."""
    # Use an invalid path likely to fail on both Win and POSIX.
    ok = elder_care.log_emergency(
        "救命", "救命",
        log_path="\x00/cannot/write/here.log" if os.name != "nt"
        else "Z:\\nonexistent-drive\\elder.log",
    )
    assert ok is False


def test_post_webhook_empty_url_is_noop(elder_off):
    # No URL set, no env var → returns False, never raises
    assert elder_care.post_webhook("hi", "hi", url="") is False
    assert elder_care.post_webhook("hi", "hi", url=None) is False


def test_post_webhook_env_url_empty(monkeypatch):
    monkeypatch.setenv("ELDER_EMERGENCY_WEBHOOK_URL", "")
    assert elder_care.post_webhook("hi", "hi") is False


def test_post_webhook_bad_url_returns_false():
    # Definitely-unreachable URL; must swallow exception, return False
    assert elder_care.post_webhook(
        "救命", "救命",
        url="http://127.0.0.1:1/elder-test-no-listener",
        timeout=0.5,
    ) is False


def test_handle_emergency_full_flow_log_only(tmp_path, monkeypatch):
    """handle_emergency: speak_fn called, log written, webhook skipped."""
    monkeypatch.delenv("ELDER_EMERGENCY_WEBHOOK_URL", raising=False)
    log_path = tmp_path / "emergency.log"

    spoken = []
    def fake_speak(mini, text):
        spoken.append((mini, text))

    result = elder_care.handle_emergency(
        "救命啊", "救命",
        speak_fn=fake_speak, mini="MOCK_MINI",
        log_path=str(log_path),
    )
    assert result["phrase"] == "救命"
    assert result["ack"] == elder_care.EMERGENCY_ACK_ZH
    assert result["logged"] is True
    assert result["spoke"] is True
    assert result["webhook"] is False
    assert log_path.exists()
    assert spoken == [("MOCK_MINI", elder_care.EMERGENCY_ACK_ZH)]


def test_handle_emergency_continues_when_speak_raises(tmp_path, monkeypatch):
    """Speak failure must not abort log/webhook decisions."""
    monkeypatch.delenv("ELDER_EMERGENCY_WEBHOOK_URL", raising=False)
    log_path = tmp_path / "emergency.log"

    def boom(mini, text):
        raise RuntimeError("TTS pipeline dead")

    result = elder_care.handle_emergency(
        "help me", "help me",
        speak_fn=boom, mini=None,
        log_path=str(log_path),
    )
    assert result["logged"] is True
    assert result["spoke"] is False  # raised → marked False
    # Did not raise — that's the actual contract being tested.


# ── P7: TTS pacing ───────────────────────────────────────────────────────

def test_edge_tts_rate_off_when_mode_off(elder_off):
    assert elder_care.edge_tts_rate() == ""


def test_edge_tts_rate_minus_10_pct_when_mode_on(elder_on):
    assert elder_care.edge_tts_rate() == "-10%"


def test_extra_post_tts_pause_off_when_mode_off(elder_off):
    assert elder_care.extra_post_tts_pause_s() == 0.0


def test_extra_post_tts_pause_default_400ms(elder_on, monkeypatch):
    monkeypatch.delenv("ELDER_PAUSE_MS", raising=False)
    assert elder_care.extra_post_tts_pause_s() == pytest.approx(0.4)


def test_extra_post_tts_pause_respects_env(elder_on, monkeypatch):
    monkeypatch.setenv("ELDER_PAUSE_MS", "750")
    assert elder_care.extra_post_tts_pause_s() == pytest.approx(0.75)


def test_extra_post_tts_pause_handles_bad_env(elder_on, monkeypatch):
    monkeypatch.setenv("ELDER_PAUSE_MS", "not-an-int")
    assert elder_care.extra_post_tts_pause_s() == pytest.approx(0.4)


def test_extra_post_tts_pause_clamps_negative(elder_on, monkeypatch):
    monkeypatch.setenv("ELDER_PAUSE_MS", "-200")
    assert elder_care.extra_post_tts_pause_s() == 0.0


# ── P8: antenna cue ──────────────────────────────────────────────────────

def test_antenna_cue_call_returns_none_when_mode_off(elder_off):
    assert elder_care.antenna_cue_call("listening") is None
    assert elder_care.antenna_cue_call("neutral") is None


def test_antenna_cue_call_listening_kwargs(elder_on, monkeypatch):
    monkeypatch.delenv("ELDER_ANTENNA_DIP_RAD", raising=False)
    call = elder_care.antenna_cue_call("listening")
    assert call is not None
    assert call["antennas"] == [0.15, 0.15]
    assert call["duration"] == pytest.approx(0.2)
    assert call["blocking"] is False


def test_antenna_cue_call_neutral_kwargs(elder_on):
    call = elder_care.antenna_cue_call("neutral")
    assert call == {"antennas": [0.0, 0.0], "duration": 0.2, "blocking": False}


def test_antenna_cue_call_unknown_state(elder_on):
    assert elder_care.antenna_cue_call("dance") is None


def test_antenna_cue_call_respects_env_dip_rad(elder_on, monkeypatch):
    monkeypatch.setenv("ELDER_ANTENNA_DIP_RAD", "0.30")
    call = elder_care.antenna_cue_call("listening")
    assert call["antennas"] == [0.30, 0.30]


def test_antenna_cue_call_clamps_excessive_dip(elder_on, monkeypatch):
    monkeypatch.setenv("ELDER_ANTENNA_DIP_RAD", "9.99")
    call = elder_care.antenna_cue_call("listening")
    # Clamped to 0.5
    assert call["antennas"] == [0.5, 0.5]


def test_antenna_cue_call_bad_env_falls_back(elder_on, monkeypatch):
    monkeypatch.setenv("ELDER_ANTENNA_DIP_RAD", "garbage")
    call = elder_care.antenna_cue_call("listening")
    assert call["antennas"] == [0.15, 0.15]


def test_fire_antenna_cue_noop_when_mini_none(elder_on):
    # mini=None should short-circuit — no exception, returns False.
    assert elder_care.fire_antenna_cue(None, "listening") is False


def test_fire_antenna_cue_noop_when_mode_off(elder_off):
    class FakeMini:
        def goto_target(self, **kw):
            raise AssertionError("should not be called")
    assert elder_care.fire_antenna_cue(FakeMini(), "listening") is False


def test_fire_antenna_cue_calls_goto_target_with_correct_args(elder_on):
    seen = {}
    class FakeMini:
        def goto_target(self, **kw):
            seen.update(kw)
    ok = elder_care.fire_antenna_cue(FakeMini(), "listening")
    assert ok is True
    assert "antennas" in seen
    # numpy array — compare element-wise
    ants = list(seen["antennas"])
    assert ants == [pytest.approx(0.15), pytest.approx(0.15)]
    assert seen["duration"] == pytest.approx(0.2)
    assert seen["method"] == "minjerk"


def test_fire_antenna_cue_swallows_sdk_exceptions(elder_on):
    class FakeMini:
        def goto_target(self, **kw):
            raise RuntimeError("daemon connection lost")
    # Must return False, must NOT raise.
    assert elder_care.fire_antenna_cue(FakeMini(), "listening") is False


# ── Master gate behaviour ────────────────────────────────────────────────

@pytest.mark.parametrize("val,expected", [
    ("1", True),
    ("yes", True),
    ("true", True),
    ("on", True),
    ("0", False),
    ("false", False),
    ("no", False),
    ("off", False),
    ("", False),
])
def test_elder_mode_enabled_truthiness(monkeypatch, val, expected):
    monkeypatch.setenv("ELDER_CARE_MODE", val)
    assert elder_care.elder_mode_enabled() is expected


def test_elder_mode_disabled_when_unset(monkeypatch):
    monkeypatch.delenv("ELDER_CARE_MODE", raising=False)
    assert elder_care.elder_mode_enabled() is False


# ── robot_brain.py integration spot-check (source-text only) ─────────────

def test_robot_brain_imports_elder_care():
    """Catches accidental removal of the import line."""
    src = (SRC / "robot_brain.py").read_text(encoding="utf-8")
    assert "import elder_care" in src


def test_robot_brain_routes_emergency_before_llm():
    """Source must contain the P6 hook in do_conversation."""
    src = (SRC / "robot_brain.py").read_text(encoding="utf-8")
    assert "elder_care.is_emergency" in src
    assert "elder_care.handle_emergency" in src
    # The check must precede any LLM dispatch site — assert ordering.
    idx_check = src.index("elder_care.is_emergency")
    idx_ask = src.index("ask_llm(text)")
    assert idx_check < idx_ask, "P6 emergency check must precede LLM ask"


def test_robot_brain_uses_edge_tts_rate_hook():
    src = (SRC / "robot_brain.py").read_text(encoding="utf-8")
    assert "elder_care.edge_tts_rate()" in src


def test_robot_brain_extends_post_tts_pause():
    src = (SRC / "robot_brain.py").read_text(encoding="utf-8")
    assert "elder_care.extra_post_tts_pause_s()" in src


def test_robot_brain_fires_antenna_cue_on_vad_edge():
    src = (SRC / "robot_brain.py").read_text(encoding="utf-8")
    assert 'fire_antenna_cue(mini, "listening")' in src
    assert 'fire_antenna_cue(mini, "neutral")' in src
