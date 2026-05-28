"""ElderFallGuard — MediaPipe Pose fall detector (Wave4-P4, task #74).

Off by default. Opt-in via `ELDER_FALLGUARD_ENABLED=1`. Reuses the existing
camera frame produced by robot_brain.tracking_loop (single-owner camera,
Wave3-P2 #64) — NO new VideoCapture / camera.open call. Subscribes by reading
the brain's `_latest_frame` global through an injected `frame_getter`
callable, which keeps this module hardware-free for unit tests.

Detection design (adapted from arXiv 2505.11845 "ElderFallGuard: Real-Time
IoT and Computer Vision-Based Fall Detection System for Elderly Safety"):

  Two-stage state machine
    NORMAL → SUSPECT          on a sudden downward shift of the body
                              centre-of-mass (hip Y crossing shoulder Y
                              within a short window) AKA rapid descent
    SUSPECT → FALL_CONFIRMED  when the prone / lying-down pose (torso angle
                              close to horizontal) has persisted for
                              PRONE_S seconds (paper: 3 s)
    SUSPECT → NORMAL          when the subject stands back up within
                              GRACE_S seconds (paper rationale: a stumble
                              that self-recovers is not a fall worth
                              alerting on; default 10 s)

  Sampling
    Runs at ~3 Hz (paper: lite model on CPU; we keep low rate to leave
    CPU+memory budget for hand_worker / vision_worker on the Pi 4 cgroup
    AllowedCPUs=2,3 + MemoryMax=1800M).

  Pose model leak workaround
    Same as HandLandmarker (mediapipe#5217 / #4785): periodically .close()
    and recreate the PoseLandmarker every POSE_RECREATE_S seconds (default
    300 s, matching HAND_LANDMARKER_RECREATE_S).

  Alert path
    On FALL_CONFIRMED, calls elder_care.log_emergency() and fires
    elder_care.post_webhook() in a daemon thread, reusing the existing
    Wave3-P6 webhook signature exactly (phrase="fall_detected"). No new
    wire format. Hard cooldown FALL_COOLDOWN_S prevents alert storms.

References
  - arXiv 2505.11845 (ElderFallGuard) — 3 s prone + motion-drop > 2 s
  - mediapipe Pose landmarks: 11/12 shoulders, 23/24 hips
"""
from __future__ import annotations

import gc
import os
import threading
import time
from typing import Any, Callable, Optional

# ── env helpers (mirror elder_care._truthy so behaviour is consistent) ──────


def _truthy(v: str) -> bool:
    return v.strip().lower() not in ("", "0", "false", "no", "off")


def fallguard_enabled() -> bool:
    """Master gate. Default OFF — production must explicitly opt in after
    human review of FP rate per the Wave4-P4 deploy plan."""
    return _truthy(os.getenv("ELDER_FALLGUARD_ENABLED", "0"))


def _model_path() -> str:
    return os.getenv(
        "ELDER_FALLGUARD_MODEL",
        "/home/pollen/brain/models/pose_landmarker_lite.task",
    )


def _hz() -> float:
    try:
        v = float(os.getenv("ELDER_FALLGUARD_HZ", "3"))
    except ValueError:
        v = 3.0
    # Clamp 1..10 Hz — below 1 Hz we lose the 2 s motion window;
    # above 10 Hz starves hand_worker on the Pi 4.
    return max(1.0, min(10.0, v))


def _prone_s() -> float:
    try:
        v = float(os.getenv("ELDER_FALLGUARD_PRONE_S", "3.0"))
    except ValueError:
        v = 3.0
    return max(1.0, v)


def _grace_s() -> float:
    """How long after SUSPECT we wait for self-recovery before alerting."""
    try:
        v = float(os.getenv("ELDER_FALLGUARD_GRACE_S", "10.0"))
    except ValueError:
        v = 10.0
    return max(2.0, v)


def _cooldown_s() -> float:
    try:
        v = float(os.getenv("ELDER_FALLGUARD_COOLDOWN_S", "60.0"))
    except ValueError:
        v = 60.0
    return max(10.0, v)


def _recreate_s() -> float:
    try:
        v = float(os.getenv("ELDER_FALLGUARD_RECREATE_S", "300"))
    except ValueError:
        v = 300.0
    return v


# ── State machine ──────────────────────────────────────────────────────────

NORMAL = "NORMAL"
SUSPECT = "SUSPECT"
FALL_CONFIRMED = "FALL_CONFIRMED"

_STATE_NUMERIC = {NORMAL: 0, SUSPECT: 1, FALL_CONFIRMED: 2}


# MediaPipe Pose landmark indices we care about.
LM_LEFT_SHOULDER = 11
LM_RIGHT_SHOULDER = 12
LM_LEFT_HIP = 23
LM_RIGHT_HIP = 24


# ── Geometry helpers (pure functions — fully unit-testable) ─────────────────


def _mid_y(lm_a, lm_b) -> Optional[float]:
    """Average normalized Y (image coords, top=0 bottom=1) of two landmarks,
    or None if either has low visibility."""
    # MediaPipe normalized landmarks expose .y and .visibility in [0, 1].
    try:
        if getattr(lm_a, "visibility", 1.0) < 0.5:
            return None
        if getattr(lm_b, "visibility", 1.0) < 0.5:
            return None
        return (lm_a.y + lm_b.y) / 2.0
    except Exception:
        return None


def _torso_angle_deg(shoulder_mid_y, hip_mid_y,
                     shoulder_mid_x, hip_mid_x) -> Optional[float]:
    """Angle of the shoulder→hip vector relative to the image vertical
    axis (degrees). 0° = perfectly upright, 90° = perfectly horizontal.

    Returns None if either midpoint is missing."""
    if None in (shoulder_mid_y, hip_mid_y, shoulder_mid_x, hip_mid_x):
        return None
    import math
    dy = hip_mid_y - shoulder_mid_y
    dx = hip_mid_x - shoulder_mid_x
    # atan2(|dx|, |dy|) → 0 when purely vertical, 90 when purely horizontal.
    return math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6))


def classify_pose(landmarks) -> dict:
    """Classify a single set of pose landmarks into geometric features.

    Returns a dict with keys:
        present       bool  — at least shoulders+hips visible
        torso_deg     float — angle vs vertical (None if absent)
        hip_below     bool  — hip Y > shoulder Y (subject upright in frame)
        prone         bool  — torso angle > 60° (close to horizontal)

    Pure function: no I/O, no global state. Used by tests with mocked
    landmark objects."""
    out = {"present": False, "torso_deg": None,
           "hip_below": None, "prone": False}
    if landmarks is None:
        return out
    try:
        ls = landmarks[LM_LEFT_SHOULDER]
        rs = landmarks[LM_RIGHT_SHOULDER]
        lh = landmarks[LM_LEFT_HIP]
        rh = landmarks[LM_RIGHT_HIP]
    except (IndexError, TypeError):
        return out
    sh_y = _mid_y(ls, rs)
    hp_y = _mid_y(lh, rh)
    if sh_y is None or hp_y is None:
        return out
    try:
        sh_x = (ls.x + rs.x) / 2.0
        hp_x = (lh.x + rh.x) / 2.0
    except Exception:
        return out
    out["present"] = True
    out["torso_deg"] = _torso_angle_deg(sh_y, hp_y, sh_x, hp_x)
    # Image coords: Y grows DOWNWARD. So hip below shoulder ⇒ hip_y > shoulder_y.
    out["hip_below"] = hp_y > sh_y
    out["prone"] = (out["torso_deg"] is not None and out["torso_deg"] > 60.0)
    return out


# ── FallGuard state machine ────────────────────────────────────────────────


class FallGuard:
    """Pure-state-machine fall detector. Decoupled from MediaPipe + camera
    so the state logic is unit-testable with synthesised classify_pose dicts.

    Thread-orchestration / model lifecycle live in `run_loop` below.
    """

    def __init__(self, logger: Any = None, alert_cb: Optional[Callable] = None,
                 metric_events: Any = None, metric_state: Any = None,
                 prone_s: Optional[float] = None,
                 grace_s: Optional[float] = None,
                 cooldown_s: Optional[float] = None):
        self.state = NORMAL
        self.logger = logger
        # Caller supplies the alert callback (defaults to fire_fall_alert
        # when used from run_loop). Keeps the class importable in tests
        # without an elder_care side-effect chain.
        self.alert_cb = alert_cb
        self.metric_events = metric_events
        self.metric_state = metric_state
        self.prone_s = prone_s if prone_s is not None else _prone_s()
        self.grace_s = grace_s if grace_s is not None else _grace_s()
        self.cooldown_s = cooldown_s if cooldown_s is not None else _cooldown_s()

        self._suspect_at: Optional[float] = None
        self._prone_since: Optional[float] = None
        self._last_alert_at: float = 0.0
        # Rolling-window of recent (t, hip_y) — used to detect rapid descent.
        # Keep ~2 s worth at our sample rate; default short list is fine.
        self._hip_history: list[tuple[float, float]] = []
        self._HIP_WINDOW_S = 2.0
        # Drop > 0.15 in normalised Y over 2 s counts as rapid descent.
        # Paper does not give a single number; this is conservative — small
        # height drops (sitting down) are < 0.10 in our manual probes.
        self._DESCENT_DELTA = 0.15

    def _log(self, event: str, **kv) -> None:
        if self.logger is None:
            return
        try:
            self.logger.info(event, **kv)
        except Exception:
            pass

    def _record_event(self, label: str) -> None:
        if self.metric_events is None:
            return
        try:
            self.metric_events.labels(state=label).inc()
        except Exception:
            pass

    def _set_state(self, new_state: str, now: float, reason: str = "") -> None:
        if new_state == self.state:
            return
        old = self.state
        self.state = new_state
        if self.metric_state is not None:
            try:
                self.metric_state.set(_STATE_NUMERIC.get(new_state, -1))
            except Exception:
                pass
        self._log("fallguard_state_transition",
                  from_state=old, to_state=new_state, reason=reason, t=now)

    def _track_descent(self, hip_y: float, now: float) -> bool:
        """Append the latest hip_y and return True if the subject has
        dropped > self._DESCENT_DELTA within the rolling window."""
        self._hip_history.append((now, hip_y))
        cutoff = now - self._HIP_WINDOW_S
        self._hip_history = [(t, y) for (t, y) in self._hip_history if t >= cutoff]
        if len(self._hip_history) < 2:
            return False
        min_y = min(y for _, y in self._hip_history)
        return (hip_y - min_y) >= self._DESCENT_DELTA

    def step(self, features: dict, now: Optional[float] = None) -> str:
        """Drive the state machine one tick. `features` is the dict returned
        by classify_pose(). Returns the new state."""
        if now is None:
            now = time.time()

        # No subject visible — reset history, stay in NORMAL.
        if not features.get("present"):
            self._hip_history.clear()
            if self.state == SUSPECT and self._suspect_at is not None and \
                    (now - self._suspect_at) > self.grace_s:
                # Lost the subject during a SUSPECT window: treat as recovery,
                # not a confirmed fall (could be off-frame). Conservative.
                self._set_state(NORMAL, now, reason="subject_lost")
                self._suspect_at = None
                self._prone_since = None
                self._record_event("normal")
            return self.state

        hip_y = features.get("hip_below")  # bool — needed for upright/below
        torso_deg = features.get("torso_deg")
        prone = features.get("prone", False)

        # Maintain rolling hip-Y history for descent detection. We need the
        # raw Y for the delta, so accept it on a second feature key.
        hip_y_value = features.get("hip_y_raw")
        rapid_descent = False
        if hip_y_value is not None:
            rapid_descent = self._track_descent(hip_y_value, now)

        if self.state == NORMAL:
            # Enter SUSPECT on either rapid descent OR an unambiguously
            # prone torso (someone already lying down when detection starts).
            if rapid_descent or prone:
                self._suspect_at = now
                self._prone_since = now if prone else None
                self._set_state(SUSPECT, now,
                                reason="rapid_descent" if rapid_descent else "prone_pose")
                self._record_event("suspect")
        elif self.state == SUSPECT:
            if prone:
                if self._prone_since is None:
                    self._prone_since = now
                if (now - self._prone_since) >= self.prone_s:
                    # Confirmed — but throttle alerts.
                    if (now - self._last_alert_at) >= self.cooldown_s:
                        self._set_state(FALL_CONFIRMED, now,
                                        reason="prone_held")
                        self._record_event("confirmed")
                        self._last_alert_at = now
                        self._fire_alert(now, torso_deg=torso_deg)
                    else:
                        # Stay in SUSPECT silently while inside cooldown.
                        pass
            else:
                # Not prone any more — but still inside grace window.
                self._prone_since = None
                # If the subject is fully upright again, recover early.
                # We treat torso_deg < 30 as upright; otherwise wait grace.
                if torso_deg is not None and torso_deg < 30.0:
                    self._set_state(NORMAL, now, reason="recovered_upright")
                    self._record_event("recovered")
                    self._suspect_at = None
                elif self._suspect_at is not None and \
                        (now - self._suspect_at) > self.grace_s:
                    # Long grace expired without prone or upright — bail.
                    self._set_state(NORMAL, now, reason="grace_expired")
                    self._record_event("normal")
                    self._suspect_at = None
        elif self.state == FALL_CONFIRMED:
            # Auto-clear back to NORMAL once the subject stands again, so we
            # are ready to detect a second fall. The cooldown protects from
            # alert spam in the meantime.
            if torso_deg is not None and torso_deg < 30.0:
                self._set_state(NORMAL, now, reason="post_fall_recovery")
                self._record_event("normal")
                self._suspect_at = None
                self._prone_since = None

        return self.state

    def _fire_alert(self, now: float, torso_deg: Optional[float] = None) -> None:
        if self.alert_cb is None:
            return
        try:
            self.alert_cb(now=now, torso_deg=torso_deg)
        except Exception as e:
            self._log("fallguard_alert_error", err=str(e))


# ── Default alert callback (production wiring to elder_care.py) ─────────────


def fire_fall_alert(now: float, torso_deg: Optional[float] = None,
                    logger: Any = None) -> dict:
    """Production alert path: log + (async) webhook via the SAME elder_care
    primitives used for the spoken-phrase emergency route. Reuses the
    existing JSON payload schema {ts, phrase, text, robot}.

    Safe to call from any thread; never raises."""
    try:
        import elder_care  # local import keeps tests free of the dependency
    except Exception as e:
        if logger is not None:
            try:
                logger.error("fallguard_elder_care_import_fail", err=str(e))
            except Exception:
                pass
        return {"logged": False, "webhook": False}

    phrase = "fall_detected"
    details = f"torso_deg={torso_deg:.1f}" if torso_deg is not None else "torso_deg=?"
    text = f"[fallguard] confirmed fall at t={now:.0f} {details}"

    result = {"logged": False, "webhook": False}
    try:
        result["logged"] = elder_care.log_emergency(text, phrase)
    except Exception as e:
        if logger is not None:
            try:
                logger.error("fall_log_error", err=str(e))
            except Exception:
                pass

    # Webhook fire-and-forget so the FallGuard tick loop is never blocked.
    url = os.getenv("ELDER_EMERGENCY_WEBHOOK_URL", "").strip()
    if url:
        try:
            threading.Thread(
                target=elder_care.post_webhook,
                args=(text, phrase),
                daemon=True,
                name="fallguard-webhook",
            ).start()
            result["webhook"] = "dispatched_async"
        except Exception as e:
            if logger is not None:
                try:
                    logger.error("fall_webhook_dispatch_error", err=str(e))
                except Exception:
                    pass

    if logger is not None:
        try:
            logger.error("fall_confirmed", torso_deg=torso_deg, t=now,
                         logged=result["logged"], webhook=result["webhook"])
        except Exception:
            pass
    return result


# ── Run loop (real-camera consumer) ─────────────────────────────────────────


def _make_pose_landmarker(model_path: str):
    """Build a MediaPipe PoseLandmarker in IMAGE mode (matches the existing
    HandLandmarker convention so leak mitigation applies the same way)."""
    import mediapipe as mp
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks.python.core.base_options import BaseOptions
    return mp_vision.PoseLandmarker.create_from_options(
        mp_vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )


def run_loop(
    frame_getter: Callable[[], Any],
    stop_event: threading.Event,
    logger: Any = None,
    pulse_fn: Optional[Callable[[str], None]] = None,
    metric_events: Any = None,
    metric_state: Any = None,
    alert_cb: Optional[Callable] = None,
) -> None:
    """Worker target. Loops at _hz() Hz pulling the latest frame from
    `frame_getter`, runs Pose, feeds classify_pose → FallGuard.step.

    Periodically recreates the PoseLandmarker every _recreate_s() seconds
    to flush the C++ packet-graph leak (mediapipe#5217 / #4785).
    """
    name = "fallguard"
    if not fallguard_enabled():
        if logger is not None:
            try:
                logger.info("fallguard_disabled_by_default",
                            hint="set ELDER_FALLGUARD_ENABLED=1 to enable")
            except Exception:
                pass
        return

    model_path = _model_path()
    if not os.path.exists(model_path):
        if logger is not None:
            try:
                logger.error("fallguard_model_missing", path=model_path)
            except Exception:
                pass
        return

    try:
        import cv2
        import mediapipe as mp
    except Exception as e:
        if logger is not None:
            try:
                logger.error("fallguard_deps_missing", err=str(e))
            except Exception:
                pass
        return

    try:
        landmarker = _make_pose_landmarker(model_path)
    except Exception as e:
        if logger is not None:
            try:
                logger.error("fallguard_landmarker_init_fail", err=str(e))
            except Exception:
                pass
        return

    period = 1.0 / _hz()
    recreate_s = _recreate_s()
    last_recreate = time.time()
    call_count = 0

    if alert_cb is None:
        alert_cb = lambda **kw: fire_fall_alert(logger=logger, **kw)

    guard = FallGuard(
        logger=logger,
        alert_cb=alert_cb,
        metric_events=metric_events,
        metric_state=metric_state,
    )

    if logger is not None:
        try:
            logger.info("fallguard_started",
                        hz=_hz(), prone_s=guard.prone_s,
                        grace_s=guard.grace_s, cooldown_s=guard.cooldown_s,
                        recreate_s=recreate_s, model=model_path)
        except Exception:
            pass

    lock = threading.Lock()  # mediapipe is not thread-safe; serialise our calls

    while not stop_event.is_set():
        if pulse_fn is not None:
            try:
                pulse_fn(name)
            except Exception:
                pass

        # Periodic landmarker recreate (leak mitigation).
        if recreate_s > 0 and (time.time() - last_recreate) >= recreate_s:
            with lock:
                try:
                    landmarker.close()
                except Exception as e:
                    if logger is not None:
                        try:
                            logger.warning("fallguard_landmarker_close_err",
                                           err=str(e))
                        except Exception:
                            pass
                try:
                    landmarker = _make_pose_landmarker(model_path)
                    last_recreate = time.time()
                    if logger is not None:
                        try:
                            logger.info("fallguard_landmarker_recreated")
                        except Exception:
                            pass
                except Exception as e:
                    if logger is not None:
                        try:
                            logger.error("fallguard_landmarker_recreate_fail",
                                         err=str(e))
                        except Exception:
                            pass
                    return
            gc.collect()

        frame = None
        try:
            frame = frame_getter()
        except Exception as e:
            if logger is not None:
                try:
                    logger.warning("fallguard_frame_getter_err", err=str(e))
                except Exception:
                    pass
        if frame is None:
            stop_event.wait(period)
            continue

        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            with lock:
                result = landmarker.detect(mp_img)
        except Exception as e:
            if metric_events is not None:
                try:
                    metric_events.labels(state="error").inc()
                except Exception:
                    pass
            if logger is not None:
                try:
                    logger.warning("fallguard_detect_err", err=str(e))
                except Exception:
                    pass
            stop_event.wait(period)
            continue

        # MediaPipe Pose: result.pose_landmarks is List[List[NormalizedLandmark]]
        landmarks = None
        try:
            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]
        except Exception:
            landmarks = None

        features = classify_pose(landmarks)
        # Inject raw hip Y for the rolling-descent window.
        if features.get("present") and landmarks is not None:
            try:
                features["hip_y_raw"] = (
                    landmarks[LM_LEFT_HIP].y + landmarks[LM_RIGHT_HIP].y
                ) / 2.0
            except Exception:
                pass

        try:
            guard.step(features)
        except Exception as e:
            if logger is not None:
                try:
                    logger.error("fallguard_step_err", err=str(e))
                except Exception:
                    pass

        call_count += 1
        # Cheap periodic gc — mirrors hand_worker pattern.
        if call_count % 50 == 0:
            try:
                del mp_img, rgb
            except Exception:
                pass
            gc.collect()

        stop_event.wait(period)

    try:
        landmarker.close()
    except Exception:
        pass
    if logger is not None:
        try:
            logger.info("fallguard_stopped")
        except Exception:
            pass
