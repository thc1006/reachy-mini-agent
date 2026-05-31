"""H8 RobotMemory.flush() race fix (2026-06-01).

Previously flush() did `executor.shutdown(wait=True)` followed by
`self._executor = ThreadPoolExecutor(...)` in two steps. A concurrent
add_turn() landing between the two could:
  - submit to the already-shut-down executor → RuntimeError, or
  - submit to a closed reference moments before the swap → silent loss.

The fix: flush() now clears self._executor under self._lock and lets
add_turn() lazy-create a fresh pool on the next call. Single invariant,
no swap window.

These tests stub out mem0 / Qdrant so they run in CI without Ollama.
"""
from __future__ import annotations

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


@pytest.fixture
def mem_stub(monkeypatch):
    """Build a RobotMemory instance with mem0 stubbed out — exercises the
    real executor / flush / add_turn plumbing without needing Ollama."""
    import robot_memory as rm

    class _FakeMem:
        def __init__(self):
            self.adds = []
            self.lock = threading.Lock()

        def add(self, text, **kw):
            # Simulate a slow Mem0 add so add_turn workers stay in-flight.
            time.sleep(0.02)
            with self.lock:
                self.adds.append(text)

        def search(self, *a, **k):
            return {"results": []}

    fake = _FakeMem()
    m = rm.RobotMemory.__new__(rm.RobotMemory)
    m.user_id = "test"
    m.enabled = True
    m._memory = fake
    m._max_workers = 1
    m._lock = threading.Lock()
    m._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mem0-writer")
    m._summary_executor = None
    m._summary_lock = threading.Lock()
    m._pending_summary_futures = []
    m._turns_since_last_summary = 0
    m.summary_every = 999_999      # never trigger summary in these tests
    m.summary_keep_recent = 20
    m.namespace = "test"
    m.qdrant_path = "/tmp/test_qdrant"
    m.conversation_log_path = "/tmp/test_conv.jsonl"
    m._write_own_log = False
    m.llm_provider = "ollama"
    m.embed_provider = "ollama"

    # Skip disk_usage check in _add_safe by stubbing shutil.disk_usage.
    class _DU:
        free = 10 * 1024**3
    monkeypatch.setattr(rm.shutil, "disk_usage", lambda p: _DU())
    return m, fake


def test_flush_clears_executor_atomically(mem_stub):
    """flush() must replace _executor with None under the lock so a follow-up
    add_turn() lazy-creates a fresh pool instead of racing the swap."""
    m, fake = mem_stub
    m.add_turn("hello", "world")
    m.flush(timeout=10)
    # After flush, executor is cleared (no live swap window).
    assert m._executor is None, "flush() should clear _executor, not re-init it"
    # First add was delivered.
    assert any("hello" in a for a in fake.adds)


def test_add_turn_lazy_recreates_after_flush(mem_stub):
    """After flush() clears the executor, the next add_turn() must lazy-create
    a fresh ThreadPoolExecutor and submit successfully."""
    m, fake = mem_stub
    m.add_turn("first", "reply")
    m.flush(timeout=10)
    assert m._executor is None
    # Lazy-create on next add_turn.
    m.add_turn("second", "reply")
    assert m._executor is not None, "add_turn must lazy-create executor after flush()"
    m.flush(timeout=10)
    texts = " ".join(fake.adds)
    assert "first" in texts
    assert "second" in texts


def test_flush_no_executor_swap_race(mem_stub):
    """Stress: hammer flush() concurrently with add_turn() and assert no
    submission raises RuntimeError ('cannot schedule new futures after
    shutdown'). All non-flushed adds must arrive at the fake mem0.

    With the old swap-based flush this test would either crash with
    RuntimeError or silently drop adds (depending on scheduling). With
    the lazy-create fix every add_turn() is either pre-flush (delivered)
    or post-flush (delivered to a fresh pool).
    """
    m, fake = mem_stub
    submit_errors = []
    n_adds = 50
    stop = threading.Event()

    def adder():
        for i in range(n_adds):
            if stop.is_set():
                return
            try:
                m.add_turn(f"user_{i}", f"bot_{i}")
            except Exception as e:
                submit_errors.append(e)
            time.sleep(0.001)

    def flusher():
        for _ in range(5):
            time.sleep(0.01)
            m.flush(timeout=10)

    t_add = threading.Thread(target=adder)
    t_flush = threading.Thread(target=flusher)
    t_add.start()
    t_flush.start()
    t_add.join(timeout=15)
    stop.set()
    t_flush.join(timeout=15)
    m.flush(timeout=10)   # final drain

    assert not submit_errors, (
        f"add_turn raised {len(submit_errors)} times during concurrent flush — "
        f"first error: {submit_errors[0]!r}"
    )
    # All submitted adds should have landed (none were silently lost in a swap window).
    assert len(fake.adds) == n_adds, (
        f"expected all {n_adds} adds to land, got {len(fake.adds)} — "
        "some were dropped during the flush race"
    )
