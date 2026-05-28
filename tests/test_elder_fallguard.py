"""ElderFallGuard (Wave4-P4 / #74) unit tests.

Hardware-free: we mock MediaPipe pose landmark objects with simple
namespaces. The state-machine FallGuard class is fully testable without
opencv / mediapipe / a camera.

Scenarios covered (>=6):
  1. standing  — torso vertical, no transition out of NORMAL
  2. walking   — small frame-to-frame Y wobble, stays NORMAL
  3. no-person — empty pose result, stays NORMAL
  4. fall_then_rise   — rapid descent → SUSPECT → recovery within grace → NORMAL,
                        no alert fired
  5. sustained_lying  — rapid descent → SUSPECT → 3 s of prone → FALL_CONFIRMED,
                        alert callback receives torso_deg
  6. alert_cooldown   — second consecutive fall within cooldown does not
                        re-fire the alert callback
  7. classify_pose_geometry — pure helper: upright vs horizontal torso angle
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import elder_fallguard as fg  # noqa: E402


# ── helpers ────────────────────────────────────────────────────────────────


def _lm(x: float, y: float, visibility: float = 1.0):
    """Build a single fake mediapipe NormalizedLandmark."""
    o = types.SimpleNamespace()
    o.x = float(x)
    o.y = float(y)
    o.visibility = float(visibility)
    return o


def _pose_upright(hip_y: float = 0.75, sh_y: float = 0.30):
    """33-landmark fake pose, only the 4 indices we care about populated.
    Upright: shoulders high (small Y), hips low (large Y), Xs aligned."""
    lms = [_lm(0.5, 0.0, 0.0)] * 33
    lms[fg.LM_LEFT_SHOULDER] = _lm(0.45, sh_y)
    lms[fg.LM_RIGHT_SHOULDER] = _lm(0.55, sh_y)
    lms[fg.LM_LEFT_HIP] = _lm(0.46, hip_y)
    lms[fg.LM_RIGHT_HIP] = _lm(0.54, hip_y)
    return lms


def _pose_horizontal(centre_y: float = 0.70):
    """Lying down: shoulders LEFT, hips RIGHT, same Y (torso horizontal)."""
    lms = [_lm(0.5, 0.0, 0.0)] * 33
    lms[fg.LM_LEFT_SHOULDER] = _lm(0.20, centre_y)
    lms[fg.LM_RIGHT_SHOULDER] = _lm(0.25, centre_y - 0.01)
    lms[fg.LM_LEFT_HIP] = _lm(0.70, centre_y + 0.01)
    lms[fg.LM_RIGHT_HIP] = _lm(0.75, centre_y)
    return lms


def _features(landmarks):
    f = fg.classify_pose(landmarks)
    if f.get("present"):
        f["hip_y_raw"] = (
            landmarks[fg.LM_LEFT_HIP].y + landmarks[fg.LM_RIGHT_HIP].y
        ) / 2.0
    return f


# ── 1. standing ────────────────────────────────────────────────────────────


def test_standing_stays_normal():
    guard = fg.FallGuard(prone_s=3.0, grace_s=10.0, cooldown_s=60.0)
    t = 1000.0
    for _ in range(20):
        guard.step(_features(_pose_upright()), now=t)
        t += 0.33
    assert guard.state == fg.NORMAL


# ── 2. walking ──────────────────────────────────────────────────────────────


def test_walking_small_wobble_stays_normal():
    guard = fg.FallGuard(prone_s=3.0, grace_s=10.0, cooldown_s=60.0)
    t = 1000.0
    # Hip Y wobbles ±0.03 — well under the 0.15 descent threshold.
    for i in range(20):
        wob = 0.03 if (i % 2) else 0.0
        guard.step(_features(_pose_upright(hip_y=0.75 + wob)), now=t)
        t += 0.33
    assert guard.state == fg.NORMAL


# ── 3. no person in frame ───────────────────────────────────────────────────


def test_no_person_present_stays_normal():
    guard = fg.FallGuard()
    f = fg.classify_pose(None)
    assert f["present"] is False
    for _ in range(10):
        guard.step(f)
    assert guard.state == fg.NORMAL


# ── 4. fall then rise (grace recovery, no alert) ───────────────────────────


def test_fall_then_rise_no_alert():
    fired = []
    guard = fg.FallGuard(prone_s=3.0, grace_s=10.0, cooldown_s=60.0,
                         alert_cb=lambda **kw: fired.append(kw))
    t = 1000.0
    # 2 s upright baseline establishes the hip-Y rolling-min.
    for _ in range(6):
        guard.step(_features(_pose_upright(hip_y=0.30)), now=t)
        t += 0.33
    # Rapid descent: hip-Y jumps from 0.30 to 0.80 in one tick.
    guard.step(_features(_pose_upright(hip_y=0.80)), now=t)
    assert guard.state == fg.SUSPECT
    t += 0.33
    # Subject stands back up within grace window — should bounce to NORMAL
    # via the "recovered_upright" path without firing the alert.
    guard.step(_features(_pose_upright(hip_y=0.30)), now=t)
    assert guard.state == fg.NORMAL
    assert fired == []


# ── 5. sustained lying = FALL_CONFIRMED + alert ────────────────────────────


def test_sustained_lying_fires_alert():
    fired = []
    guard = fg.FallGuard(prone_s=3.0, grace_s=10.0, cooldown_s=60.0,
                         alert_cb=lambda **kw: fired.append(kw))
    t = 1000.0
    # Baseline upright then drop.
    for _ in range(6):
        guard.step(_features(_pose_upright(hip_y=0.30)), now=t)
        t += 0.33
    guard.step(_features(_pose_upright(hip_y=0.80)), now=t)
    assert guard.state == fg.SUSPECT
    t += 0.33
    # Now lying horizontal for >3 s (~12 ticks at 3 Hz).
    for _ in range(12):
        guard.step(_features(_pose_horizontal()), now=t)
        t += 0.33
    assert guard.state == fg.FALL_CONFIRMED
    assert len(fired) == 1
    # Alert payload carries torso_deg ≈ 90° (horizontal).
    assert "torso_deg" in fired[0]
    assert fired[0]["torso_deg"] is not None
    assert fired[0]["torso_deg"] > 60.0


# ── 6. cooldown suppresses duplicate alerts ─────────────────────────────────


def test_alert_cooldown_suppresses_duplicate():
    fired = []
    guard = fg.FallGuard(prone_s=1.0, grace_s=5.0, cooldown_s=60.0,
                         alert_cb=lambda **kw: fired.append(kw))
    t = 1000.0
    # First fall.
    for _ in range(3):
        guard.step(_features(_pose_upright(hip_y=0.30)), now=t)
        t += 0.33
    guard.step(_features(_pose_upright(hip_y=0.80)), now=t)
    t += 0.33
    for _ in range(6):
        guard.step(_features(_pose_horizontal()), now=t)
        t += 0.33
    assert guard.state == fg.FALL_CONFIRMED
    assert len(fired) == 1
    # Recover.
    for _ in range(3):
        guard.step(_features(_pose_upright(hip_y=0.30)), now=t)
        t += 0.33
    assert guard.state == fg.NORMAL
    # Second fall within the 60 s cooldown — should NOT alert again.
    guard.step(_features(_pose_upright(hip_y=0.80)), now=t)
    t += 0.33
    for _ in range(6):
        guard.step(_features(_pose_horizontal()), now=t)
        t += 0.33
    assert len(fired) == 1  # still just the one alert
    # State stays SUSPECT (alert throttled inside the cooldown window).
    assert guard.state == fg.SUSPECT


# ── 7. classify_pose geometry helper ───────────────────────────────────────


def test_classify_pose_geometry():
    up = fg.classify_pose(_pose_upright())
    assert up["present"] is True
    assert up["prone"] is False
    assert up["torso_deg"] is not None
    assert up["torso_deg"] < 30.0  # near-vertical

    flat = fg.classify_pose(_pose_horizontal())
    assert flat["present"] is True
    assert flat["prone"] is True
    assert flat["torso_deg"] > 60.0


def test_classify_pose_low_visibility_returns_absent():
    # Both shoulders visibility=0 → treated as absent.
    lms = _pose_upright()
    lms[fg.LM_LEFT_SHOULDER].visibility = 0.0
    lms[fg.LM_RIGHT_SHOULDER].visibility = 0.0
    f = fg.classify_pose(lms)
    assert f["present"] is False


# ── 8. env toggle ──────────────────────────────────────────────────────────


def test_fallguard_disabled_by_default(monkeypatch):
    monkeypatch.delenv("ELDER_FALLGUARD_ENABLED", raising=False)
    assert fg.fallguard_enabled() is False


def test_fallguard_enabled_truthy(monkeypatch):
    monkeypatch.setenv("ELDER_FALLGUARD_ENABLED", "1")
    assert fg.fallguard_enabled() is True
    monkeypatch.setenv("ELDER_FALLGUARD_ENABLED", "0")
    assert fg.fallguard_enabled() is False
