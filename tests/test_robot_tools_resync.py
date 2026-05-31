"""H6 _schedule_face_baseline_resync gen-counter race fix (2026-06-01).

Previously, Timer.cancel() was best-effort: if the timer already fired
and _do_sync was mid-execution (e.g. blocked on the daemon HTTP call)
when a newer clip scheduled a fresh timer, the stale _do_sync would
still write the baseline AFTER the cancel + AFTER the new timer was
created, producing out-of-order baseline writes when the user fired 3
clips in 200 ms.

The fix: each scheduling bumps a monotonic _resync_gen counter; _do_sync
snapshots the value at scheduling time and re-checks it under the lock
before writing. Stale generations abort.
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


@pytest.fixture
def fresh_tools():
    """Reload robot_tools to reset the module-level _resync_gen counter so
    tests don't share state. Returns the module."""
    if "robot_tools" in sys.modules:
        del sys.modules["robot_tools"]
    import robot_tools as rt
    yield rt
    # cleanup
    if rt._resync_timer is not None:
        try:
            rt._resync_timer.cancel()
        except Exception:
            pass


def test_resync_gen_counter_increments(fresh_tools):
    """Each call to _schedule_face_baseline_resync must bump _resync_gen."""
    rt = fresh_tools
    start = rt._resync_gen
    rt._schedule_face_baseline_resync(delay_s=0.5)
    rt._schedule_face_baseline_resync(delay_s=0.5)
    rt._schedule_face_baseline_resync(delay_s=0.5)
    assert rt._resync_gen == start + 3


def test_resync_out_of_order_drops_stale_write(fresh_tools, monkeypatch):
    """Simulate the bad race: timer-1 fires + enters _do_sync, blocks on
    daemon HTTP. While it's blocked, the user fires another clip → timer-2
    is scheduled (gen bumps). Then timer-1 unblocks. It must NOT write
    a baseline because its snapshot is now stale.
    """
    rt = fresh_tools
    writes: list[tuple[float, float, str]] = []
    barrier = threading.Event()
    inside_do_sync = threading.Event()

    # Patch the daemon HTTP call to block on the barrier so we can race a
    # second scheduling in between.
    class _FakeResp:
        def __init__(self, data: bytes):
            self._data = data
        def read(self):
            return self._data
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=3):
        inside_do_sync.set()
        # Block here so the test can schedule a second resync that bumps gen.
        barrier.wait(timeout=5)
        return _FakeResp(b'{"pitch": 1.23, "yaw": 4.56}')

    monkeypatch.setattr(rt._urlreq, "urlopen", fake_urlopen)

    # Stub robot_brain.note_head_command so we can see if a baseline write
    # would have happened.
    class _StubBrain:
        @staticmethod
        def note_head_command(pitch_deg, yaw_deg, body_yaw_rad, source):
            writes.append((pitch_deg, yaw_deg, source))
    sys.modules["robot_brain"] = _StubBrain

    # Schedule the first timer with the shortest legal delay (0.5s).
    rt._schedule_face_baseline_resync(delay_s=0.5, source="clip_1")
    gen_after_first = rt._resync_gen

    # Wait for timer-1 to fire and enter _do_sync (blocked on barrier).
    assert inside_do_sync.wait(timeout=3.0), "timer-1 never entered _do_sync"

    # Bump gen by scheduling timer-2 (the second clip the user fires).
    rt._schedule_face_baseline_resync(delay_s=0.5, source="clip_2")
    assert rt._resync_gen == gen_after_first + 1

    # Unblock timer-1 — it should detect its gen is stale and abort.
    barrier.set()
    time.sleep(0.2)   # let the stale worker finish its abort check

    # No write from the stale clip_1 timer.
    clip_1_writes = [w for w in writes if w[2] == "clip_1"]
    assert not clip_1_writes, (
        f"stale timer-1 wrote a baseline anyway: {clip_1_writes} — "
        "the gen-counter guard failed."
    )

    # Cleanup: cancel timer-2 before it fires.
    if rt._resync_timer is not None:
        rt._resync_timer.cancel()


def test_resync_latest_generation_writes_baseline(fresh_tools, monkeypatch):
    """Sanity: when no race happens, the latest scheduled timer DOES write
    the baseline. (Confirms we haven't broken the happy path with the
    gen-counter guard.)"""
    rt = fresh_tools
    writes: list[tuple[float, float, str]] = []

    class _FakeResp:
        def read(self): return b'{"pitch": 2.0, "yaw": 3.0}'
        def __enter__(self): return self
        def __exit__(self, *a): return False

    monkeypatch.setattr(rt._urlreq, "urlopen",
                        lambda req, timeout=3: _FakeResp())

    class _StubBrain:
        @staticmethod
        def note_head_command(pitch_deg, yaw_deg, body_yaw_rad, source):
            writes.append((pitch_deg, yaw_deg, source))
    sys.modules["robot_brain"] = _StubBrain

    rt._schedule_face_baseline_resync(delay_s=0.5, source="happy_path")
    # Wait for the 0.5s timer to fire + write.
    time.sleep(1.0)
    assert writes, "the happy-path timer should have written exactly one baseline"
    assert writes[-1][2] == "happy_path"
