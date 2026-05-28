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

import json
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


# ── 9. H-F1: already-prone from t=0 must NOT enter SUSPECT (FP class) ──────


def test_already_prone_subject_from_t0_stays_normal():
    """Subject is already lying down when detection starts (sleeping on
    sofa, child on floor, tilted camera, short user). Without an observed
    upright baseline within grace_upright_s the prone pose alone must NOT
    trip SUSPECT — that was the H-F1 FP risk."""
    fired = []
    guard = fg.FallGuard(prone_s=3.0, grace_s=10.0, cooldown_s=60.0,
                         grace_upright_s=60.0,
                         alert_cb=lambda **kw: fired.append(kw))
    t = 1000.0
    # 20 ticks (~6.6s) of pure horizontal pose. No prior upright observation.
    for _ in range(20):
        guard.step(_features(_pose_horizontal()), now=t)
        t += 0.33
    assert guard.state == fg.NORMAL, "prone-only without recent upright must stay NORMAL"
    assert fired == [], "no alert should fire from ambient lying subject"


def test_upright_then_prone_within_grace_enters_suspect_and_confirms():
    """H-F1 path (b): upright observed within grace_upright_s, then prone
    without a clean descent sample. Must still enter SUSPECT and confirm
    after prone_s. Ensures we didn't kill legitimate fall detection."""
    fired = []
    guard = fg.FallGuard(prone_s=3.0, grace_s=10.0, cooldown_s=60.0,
                         grace_upright_s=60.0,
                         alert_cb=lambda **kw: fired.append(kw))
    t = 1000.0
    # 3 ticks upright (torso vertical) — establishes _last_upright_at.
    for _ in range(3):
        guard.step(_features(_pose_upright(hip_y=0.75, sh_y=0.30)), now=t)
        t += 0.33
    # Skip ahead 5 s — within grace_upright_s=60s — and observe prone
    # without an in-window descent sample.
    t += 5.0
    guard.step(_features(_pose_horizontal()), now=t)
    assert guard.state == fg.SUSPECT, "prone after recent upright must enter SUSPECT"
    # Hold prone for >3 s → FALL_CONFIRMED + alert.
    t += 0.33
    for _ in range(12):
        guard.step(_features(_pose_horizontal()), now=t)
        t += 0.33
    assert guard.state == fg.FALL_CONFIRMED
    assert len(fired) == 1


def test_upright_then_prone_beyond_grace_stays_normal():
    """If the last upright observation is older than grace_upright_s,
    prone-only is treated as ambient (subject moved away then we re-saw
    them lying). Stays NORMAL."""
    guard = fg.FallGuard(prone_s=3.0, grace_s=10.0, cooldown_s=60.0,
                         grace_upright_s=10.0)
    t = 1000.0
    guard.step(_features(_pose_upright(hip_y=0.75, sh_y=0.30)), now=t)
    # Jump past grace_upright_s.
    t += 120.0
    guard.step(_features(_pose_horizontal()), now=t)
    assert guard.state == fg.NORMAL


# ── 10. H-F2: webhook retry envelope + DLQ ─────────────────────────────────


def test_webhook_retries_then_succeeds_on_attempt_2(monkeypatch, tmp_path):
    """First POST returns HTTP 500 → second attempt returns 200. Final
    outcome should be retry_success; no DLQ entry written."""
    # Isolate DLQ path so this test cannot pollute $HOME/brain.
    dlq = tmp_path / "dlq.jsonl"
    monkeypatch.setenv("ELDER_FALLGUARD_WEBHOOK_DLQ", str(dlq))

    # Fake elder_care.post_webhook returning a status-dict (modern
    # signature). First call HTTP 500, second call HTTP 200.
    calls = []
    def fake_post_webhook(text, phrase):
        calls.append((text, phrase))
        if len(calls) == 1:
            return {"ok": False, "status_code": 500}
        return {"ok": True, "status_code": 200}

    import sys as _sys
    fake_module = types.SimpleNamespace(post_webhook=fake_post_webhook)
    monkeypatch.setitem(_sys.modules, "elder_care", fake_module)
    # Don't actually sleep through 1+4s exponential backoff.
    sleeps = []
    result = fg._deliver_webhook_with_retry(
        "hello", "fall_detected",
        sleep_fn=lambda s: sleeps.append(s),
    )

    assert result == "retry_success"
    assert len(calls) == 2
    assert sleeps == [1.0]  # only the first inter-attempt wait
    assert not dlq.exists(), "no DLQ entry expected on retry success"


def test_webhook_all_attempts_fail_writes_dlq(monkeypatch, tmp_path):
    """All 3 attempts return HTTP 500 → final outcome failed, DLQ line
    written, counter increments on result=failed."""
    dlq = tmp_path / "dlq.jsonl"
    monkeypatch.setenv("ELDER_FALLGUARD_WEBHOOK_DLQ", str(dlq))

    calls = []
    def fake_post_webhook(text, phrase):
        calls.append((text, phrase))
        return {"ok": False, "status_code": 500}

    import sys as _sys
    fake_module = types.SimpleNamespace(post_webhook=fake_post_webhook)
    monkeypatch.setitem(_sys.modules, "elder_care", fake_module)

    # Spy counter (no-op-compatible API).
    inc_calls = []
    class _SpyCounter:
        def labels(self, **kw):
            inc_calls.append(kw)
            return self
        def inc(self, *a, **k):
            pass
    fake_obs = types.SimpleNamespace(fall_webhook_outcome_total=_SpyCounter())
    monkeypatch.setitem(_sys.modules, "brain_observability", fake_obs)

    sleeps = []
    result = fg._deliver_webhook_with_retry(
        "hello", "fall_detected",
        sleep_fn=lambda s: sleeps.append(s),
    )

    assert result == "failed"
    assert len(calls) == 3
    assert sleeps == [1.0, 4.0]
    # DLQ line written exactly once.
    assert dlq.exists()
    line = dlq.read_text(encoding="utf-8").strip()
    assert json.loads(line)["phrase"] == "fall_detected"
    # Counter ticked exactly once on result=failed.
    assert {"result": "failed"} in inc_calls


# ── 11. M3: recreate-failure must not silently kill the worker ─────────────


def test_recreate_failure_clears_pulse_and_logs(monkeypatch):
    """When the periodic landmarker recreate raises, the loop must:
      - log fallguard_landmarker_recreate_failed (ERROR)
      - call clear_pulse_fn(name) so the watchdog forgets us
      - return (no silent zombie)
    """
    # Replace module-level _make_pose_landmarker so the first call
    # succeeds (initial build) and the second one (the recreate) raises.
    build_count = {"n": 0}
    class FakeLandmarker:
        def detect(self, _img):
            return types.SimpleNamespace(pose_landmarks=None)
        def close(self):
            pass
    def fake_make(_path):
        build_count["n"] += 1
        if build_count["n"] >= 2:
            raise RuntimeError("simulated recreate failure")
        return FakeLandmarker()
    monkeypatch.setattr(fg, "_make_pose_landmarker", fake_make)

    monkeypatch.setenv("ELDER_FALLGUARD_ENABLED", "1")
    monkeypatch.setenv("ELDER_FALLGUARD_RECREATE_S", "0.01")  # force recreate fast
    monkeypatch.setenv("ELDER_FALLGUARD_HZ", "10")            # quick ticks
    # Provide a fake model path that exists.
    fake_model = Path(__file__).resolve().parent / "_fake_pose.task"
    fake_model.write_bytes(b"")
    monkeypatch.setenv("ELDER_FALLGUARD_MODEL", str(fake_model))

    # Provide fake cv2 + mediapipe modules so the run_loop import-block
    # doesn't bail out before we hit the recreate path.
    import sys as _sys
    fake_cv2 = types.SimpleNamespace(cvtColor=lambda x, _c: x, COLOR_BGR2RGB=0)
    class _FakeImg:
        def __init__(self, image_format=None, data=None): pass
    fake_mp = types.SimpleNamespace(
        Image=_FakeImg,
        ImageFormat=types.SimpleNamespace(SRGB=0),
    )
    monkeypatch.setitem(_sys.modules, "cv2", fake_cv2)
    monkeypatch.setitem(_sys.modules, "mediapipe", fake_mp)

    class FakeLogger:
        def __init__(self): self.errors = []
        def info(self, ev, **kv): pass
        def warning(self, ev, **kv): pass
        def error(self, ev, **kv): self.errors.append(ev)
    logger = FakeLogger()
    cleared = []
    pulsed = []

    stop = __import__("threading").Event()
    # Drive run_loop in foreground; force exit after a few ticks via the
    # clear_pulse_fn side-channel (cleared name → set stop).
    def clear_pulse_fn(name):
        cleared.append(name)
        stop.set()

    fg.run_loop(
        frame_getter=lambda: object(),  # never None so we hit detect()
        stop_event=stop,
        logger=logger,
        pulse_fn=lambda n: pulsed.append(n),
        clear_pulse_fn=clear_pulse_fn,
    )

    assert "fallguard_landmarker_recreate_failed" in logger.errors
    assert cleared == ["fallguard"]
    # Cleanup the dummy model file.
    try: fake_model.unlink()
    except Exception: pass
