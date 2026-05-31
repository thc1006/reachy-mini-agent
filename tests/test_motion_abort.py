"""Tests for src/motion_abort.py — Wave4-P3 #73 Option B barge-in.

The module exports a process-global Event; every test clears it on entry
and exit to keep cross-test isolation.
"""
import os
import sys
import threading
import time

import pytest

SRC = os.path.join(os.path.dirname(__file__), "..", "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import motion_abort  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_abort():
    motion_abort.clear_abort()
    yield
    motion_abort.clear_abort()


def test_request_abort_sets_flag():
    assert motion_abort.is_aborted() is False
    motion_abort.request_abort("user_speech")
    assert motion_abort.is_aborted() is True


def test_clear_abort_unsets_flag():
    motion_abort.request_abort()
    assert motion_abort.is_aborted() is True
    motion_abort.clear_abort()
    assert motion_abort.is_aborted() is False


def test_is_aborted_false_when_not_set():
    assert motion_abort.is_aborted() is False


def test_interruptible_sleep_aborts_within_100ms():
    """Set abort from another thread mid-sleep; the sleep must return True
    well before the requested 5 s duration. 100 ms ceiling = one POLL
    INTERVAL (50 ms) + plenty of OS-scheduling slack."""

    def _fire():
        time.sleep(0.02)
        motion_abort.request_abort("test_thread")

    t = threading.Thread(target=_fire, daemon=True)
    t.start()
    t0 = time.monotonic()
    aborted = motion_abort.interruptible_sleep(5.0)
    elapsed = time.monotonic() - t0
    t.join(timeout=1.0)

    assert aborted is True
    assert elapsed < 0.15, f"abort took {elapsed:.3f}s, expected <0.15s"


def test_interruptible_sleep_full_duration_if_not_aborted():
    """Without an abort, sleep should run for the full requested duration
    (within reasonable scheduler slack)."""
    t0 = time.monotonic()
    aborted = motion_abort.interruptible_sleep(0.2)
    elapsed = time.monotonic() - t0

    assert aborted is False
    assert elapsed >= 0.2, f"slept only {elapsed:.3f}s, expected >=0.2s"
    assert elapsed < 0.4, f"slept {elapsed:.3f}s, way over 0.2s budget"


def test_interruptible_sleep_zero_duration_returns_immediately():
    """duration_s<=0 must not busy-loop; returns current abort state."""
    t0 = time.monotonic()
    aborted = motion_abort.interruptible_sleep(0.0)
    elapsed = time.monotonic() - t0

    assert aborted is False
    assert elapsed < 0.05
