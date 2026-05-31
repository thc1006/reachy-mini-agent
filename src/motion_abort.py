"""Cooperative motion abort signal — barge-in support (Wave4-P3 #73, Option B).

When set, long-running motion operations (goto, dance playback) should
abort at their next polling interval (~50ms). Caller is responsible for
clearing the event after they've handled the abort.

Why a module-global `threading.Event` (and not the original 3-layer
priority-queue spec):
  - Elder-care single-user scenario. The actual user value is barge-in —
    the user starts speaking mid-motion and the robot has to STOP and
    listen. Priority queue + layered POSTURE/GESTURE/BARGE_IN lanes are
    over-engineering for that one signal.
  - Verdict #92 RESTART_FRESH closed the earlier MotionQueue stash; this
    file is the minimal-viable replacement (~50 LOC + ~4 tests vs the
    446 LOC + 248 tests the queue path would have cost).
  - A module-level Event is cheap, lock-free, and importable from any
    thread (tracking_loop, do_action, dialog loop, stream TTS path).

Polling interval is hard-coded at 50 ms. That's fast enough that the
user does not perceive the lag between speaking and the motion stopping
(human reaction-time floor is ~200 ms), and slow enough that the busy-
wait cost on Pi 4 is negligible (<0.1% of one core).
"""
import threading
import time

_motion_abort = threading.Event()
_POLL_INTERVAL_S = 0.05  # 50 ms — fast enough for barge-in UX


def request_abort(reason: str = "user_speech") -> None:
    """Signal that any in-flight motion should stop ASAP. Idempotent.

    `reason` is accepted for logging / Prometheus labelling at the call
    site; this function itself does not log to avoid hard-coupling to
    structlog (which may not be imported in every caller's module).
    """
    _motion_abort.set()


def clear_abort() -> None:
    """Caller (motion executor) clears after handling. Idempotent."""
    _motion_abort.clear()


def is_aborted() -> bool:
    """Cheap check — call between motion sub-steps."""
    return _motion_abort.is_set()


def interruptible_sleep(duration_s: float) -> bool:
    """Sleep up to `duration_s` but wake immediately on abort.

    Returns True if aborted (caller should bail out of its motion loop),
    False if the full duration elapsed normally.
    """
    if duration_s <= 0:
        return _motion_abort.is_set()
    deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline:
        if _motion_abort.is_set():
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(_POLL_INTERVAL_S, remaining))
    return _motion_abort.is_set()
