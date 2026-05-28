"""Wave4-P11 #85 Option (c) — unit tests for cap_gupnp_thread sidecar.

Covers:
  (a) find_gupnp_tids returns [] on a fake /proc tree with no matching comm,
      and returns the right TIDs when comm files match the truncated prefix.
  (b) apply_cap calls os.sched_setscheduler with SCHED_IDLE + os.sched_setaffinity
      with the configured core; both monkey-patched.
  (c) cap_loop pulses the heartbeat each iteration and exits promptly when
      the stop_event is set (must return within ~1 s without hanging).

All tests are pure-Python; no /proc dependency, no real syscalls, no robot.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import pytest

# Make src/ importable without conftest changes.
SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from brain_helpers import cap_gupnp_thread as gcap  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────
# (a) find_gupnp_tids
# ──────────────────────────────────────────────────────────────────────────
def test_find_gupnp_tids_returns_empty_when_no_thread(tmp_path):
    """Synthetic /proc/self/task with only non-matching comm files → []."""
    # 3 fake threads, none of them gupnp.
    for tid, comm in [(1001, "python"), (1002, "dialog_loop"), (1003, "kworker")]:
        td = tmp_path / str(tid)
        td.mkdir()
        (td / "comm").write_text(comm + "\n")
    assert gcap.find_gupnp_tids(task_dir=str(tmp_path)) == []


def test_find_gupnp_tids_picks_up_truncated_comm(tmp_path):
    """The kernel truncates comm to 15 chars; the matcher uses the
    truncated prefix so a literal `gupnp-igd-threa` matches and the
    siblings don't."""
    matches = []
    for tid, comm in [
        (2001, "gupnp-igd-threa"),      # exact truncated form — match
        (2002, "gupnp-igd-thread"),     # full name (some kernels) — match
        (2003, "gupnp-other"),          # different gupnp helper — no match
        (2004, "dialog_loop"),          # unrelated — no match
    ]:
        td = tmp_path / str(tid)
        td.mkdir()
        (td / "comm").write_text(comm + "\n")
        if comm.startswith(gcap.GUPNP_COMM_PREFIX):
            matches.append(tid)
    found = sorted(gcap.find_gupnp_tids(task_dir=str(tmp_path)))
    assert found == sorted(matches)
    assert 2001 in found
    assert 2002 in found
    assert 2003 not in found


def test_find_gupnp_tids_missing_dir_is_empty():
    """Non-Linux / missing /proc → returns [] (no crash)."""
    assert gcap.find_gupnp_tids(task_dir="/nope/definitely/not/a/path") == []


# ──────────────────────────────────────────────────────────────────────────
# (b) apply_cap
# ──────────────────────────────────────────────────────────────────────────
def test_apply_cap_calls_sched_and_affinity(monkeypatch):
    """apply_cap must call sched_setscheduler(tid, SCHED_IDLE, sched_param(0))
    AND sched_setaffinity(tid, {core}). Both are monkey-patched so we don't
    actually touch the running interpreter's threads."""
    calls = {"sched": None, "affinity": None}

    def fake_setscheduler(tid, policy, param):
        calls["sched"] = (tid, policy, getattr(param, "sched_priority", None))

    def fake_setaffinity(tid, mask):
        calls["affinity"] = (tid, set(mask))

    # Ensure SCHED_IDLE is present even on test platforms that lack it
    # natively (older macOS Python). On Linux this is always set already.
    if not hasattr(os, "SCHED_IDLE"):
        monkeypatch.setattr(os, "SCHED_IDLE", 5, raising=False)
    if not hasattr(os, "sched_param"):
        class _SP:
            def __init__(self, p): self.sched_priority = p
        monkeypatch.setattr(os, "sched_param", _SP, raising=False)

    monkeypatch.setattr(os, "sched_setscheduler", fake_setscheduler, raising=False)
    monkeypatch.setattr(os, "sched_setaffinity", fake_setaffinity, raising=False)

    ok, reason = gcap.apply_cap(tid=4242, core=3)
    assert ok is True
    assert reason == ""
    assert calls["sched"] is not None
    tid_seen, policy_seen, prio_seen = calls["sched"]
    assert tid_seen == 4242
    assert policy_seen == os.SCHED_IDLE
    assert prio_seen == 0
    assert calls["affinity"] == (4242, {3})


def test_apply_cap_unavailable_sched_returns_false(monkeypatch):
    """If os.sched_setscheduler is missing (non-Linux), apply_cap must
    return (False, reason) cleanly without raising."""
    monkeypatch.delattr(os, "sched_setscheduler", raising=False)
    ok, reason = gcap.apply_cap(tid=1, core=3)
    assert ok is False
    assert "unavailable" in reason


def test_apply_cap_oserror_returns_false(monkeypatch):
    """sched_setscheduler raising OSError (e.g. EPERM, ESRCH) is caught
    and surfaced as (False, "sched_setscheduler errno=...")."""
    def boom(tid, policy, param):
        raise OSError(1, "EPERM")
    if not hasattr(os, "SCHED_IDLE"):
        monkeypatch.setattr(os, "SCHED_IDLE", 5, raising=False)
    if not hasattr(os, "sched_param"):
        class _SP:
            def __init__(self, p): self.sched_priority = p
        monkeypatch.setattr(os, "sched_param", _SP, raising=False)
    monkeypatch.setattr(os, "sched_setscheduler", boom, raising=False)
    ok, reason = gcap.apply_cap(tid=1, core=3)
    assert ok is False
    assert "errno=1" in reason


# ──────────────────────────────────────────────────────────────────────────
# (c) cap_loop
# ──────────────────────────────────────────────────────────────────────────
def test_cap_loop_pulses_and_stops_on_event(monkeypatch):
    """cap_loop must:
      - Call pulse_fn at least once before exit (heartbeat for watchdog).
      - Return promptly (<1.5 s) when stop_event is set.
      - Not blow up if find_gupnp_tids returns [].
    """
    gcap._reset_capped_state_for_tests()
    monkeypatch.setattr(gcap, "find_gupnp_tids", lambda task_dir="/proc/self/task": [])
    pulses = []

    def fake_pulse():
        pulses.append(time.monotonic())

    stop_event = threading.Event()
    # Use a tiny interval so the loop wakes up quickly to observe stop_event.
    t = threading.Thread(
        target=gcap.cap_loop,
        kwargs=dict(stop_event=stop_event, interval_s=0.05, pulse_fn=fake_pulse),
        daemon=True,
    )
    started = time.monotonic()
    t.start()
    # Let it iterate a couple of times.
    time.sleep(0.2)
    stop_event.set()
    t.join(timeout=1.5)
    elapsed = time.monotonic() - started
    assert not t.is_alive(), "cap_loop did not exit within 1.5 s of stop_event.set()"
    assert elapsed < 2.0
    assert len(pulses) >= 1, "pulse_fn was never invoked"


def test_cap_loop_skips_already_capped_tids(monkeypatch):
    """The set of already-capped TIDs must prevent re-application on the
    next iteration — apply_cap should be called exactly once for the same
    TID across multiple loop iterations."""
    gcap._reset_capped_state_for_tests()
    monkeypatch.setattr(gcap, "find_gupnp_tids",
                        lambda task_dir="/proc/self/task": [9999])
    calls = []

    def fake_apply_cap(tid, core=gcap.CAP_CORE):
        calls.append(tid)
        return True, ""

    monkeypatch.setattr(gcap, "apply_cap", fake_apply_cap)
    stop_event = threading.Event()
    t = threading.Thread(
        target=gcap.cap_loop,
        kwargs=dict(stop_event=stop_event, interval_s=0.02, pulse_fn=None),
        daemon=True,
    )
    t.start()
    time.sleep(0.15)  # >= ~5 iterations at interval_s=0.02
    stop_event.set()
    t.join(timeout=1.0)
    assert calls == [9999], f"apply_cap called {len(calls)} times; expected exactly once"
