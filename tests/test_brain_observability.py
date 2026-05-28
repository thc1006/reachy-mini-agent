"""H-O2 — unit coverage for brain_observability.

Covers:
  (a) pulse / clear_pulse / heartbeat_min_age_s round-trip
  (b) watchdog "no heartbeats past grace → no notify" (fake sd_notify_fn,
      monkey-patched time.monotonic)
  (c) call_with_breaker returns CircuitBreakerError on open WITHOUT being
      retried by tenacity (H-O3 regression guard)
  (d) init_metrics idempotency

These are pure-Python unit tests; no Ollama / no Whisper / no robot SDK.
They run in the parent process (no subprocess) so we can introspect module
state directly.
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

# Make src/ importable without conftest changes
SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import brain_observability as obs  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────
# (a) pulse / clear_pulse / heartbeat_min_age_s
# ──────────────────────────────────────────────────────────────────────────
def _reset_heartbeats():
    """Help: heartbeats are module-global; tests must reset between cases."""
    with obs._heartbeat_lock:
        obs._thread_heartbeat.clear()


class TestHeartbeat:
    def setup_method(self):
        _reset_heartbeats()

    def test_pulse_records_monotonic_now(self):
        before = time.monotonic()
        obs.pulse("worker_a")
        snap = obs.heartbeat_snapshot()
        assert "worker_a" in snap
        assert snap["worker_a"] >= before

    def test_clear_pulse_removes_entry(self):
        obs.pulse("ephemeral")
        assert "ephemeral" in obs.heartbeat_snapshot()
        obs.clear_pulse("ephemeral")
        assert "ephemeral" not in obs.heartbeat_snapshot()
        # Idempotent — clearing a non-existent key must not raise
        obs.clear_pulse("ephemeral")

    def test_heartbeat_min_age_none_when_empty(self):
        assert obs.heartbeat_min_age_s() is None

    def test_heartbeat_min_age_picks_oldest(self):
        obs.pulse("old")
        time.sleep(0.05)
        obs.pulse("young")
        age = obs.heartbeat_min_age_s()
        assert age is not None
        # "old" was pulsed first, so min_age tracks its age (>= ~0.05 s)
        assert age >= 0.04


# ──────────────────────────────────────────────────────────────────────────
# (b) watchdog stale-post-grace gating
# ──────────────────────────────────────────────────────────────────────────
class _FakeLogger:
    def __init__(self):
        self.events: list[tuple[str, str, dict]] = []

    def info(self, event, **kw):    self.events.append(("info", event, kw))
    def warning(self, event, **kw): self.events.append(("warning", event, kw))
    def error(self, event, **kw):   self.events.append(("error", event, kw))
    def debug(self, event, **kw):   self.events.append(("debug", event, kw))


class TestWatchdogGating:
    def setup_method(self):
        _reset_heartbeats()
        obs._watchdog_armed_at = None

    def test_no_systemd_returns_none(self, monkeypatch):
        """Without NOTIFY_SOCKET the watchdog must be a no-op + return None."""
        monkeypatch.delenv("NOTIFY_SOCKET", raising=False)
        log = _FakeLogger()
        notifications: list[str] = []
        stop = threading.Event()
        t = obs.start_watchdog_thread(
            sd_notify_fn=lambda s: notifications.append(s),
            stop_event=stop,
            logger=log,
            threshold_s=5.0,
            interval_s=0.05,
        )
        assert t is None
        # Should have logged exactly one watchdog_disabled_no_systemd event
        assert any(e[1] == "watchdog_disabled_no_systemd" for e in log.events)
        # And nothing was notified
        assert notifications == []

    def test_stale_post_grace_skips_notify(self, monkeypatch):
        """When NOTIFY_SOCKET is set, watchdog is armed, and >grace seconds
        elapse with no heartbeats → must NOT send WATCHDOG=1 and must log
        watchdog_stale_post_grace."""
        monkeypatch.setenv("NOTIFY_SOCKET", "/dev/null")
        # Pretend we armed 1000 s ago — well past the 30 s grace window.
        obs._watchdog_armed_at = time.monotonic() - 1000.0
        log = _FakeLogger()
        notifications: list[str] = []
        stop = threading.Event()
        t = obs.start_watchdog_thread(
            sd_notify_fn=lambda s: notifications.append(s),
            stop_event=stop,
            logger=log,
            threshold_s=5.0,
            interval_s=0.05,
        )
        assert t is not None
        # Let the watchdog loop tick a few times, then stop it
        time.sleep(0.25)
        stop.set()
        t.join(timeout=1.0)
        # No WATCHDOG=1 sent (stale post-grace branch)
        assert "WATCHDOG=1" not in notifications, (
            f"watchdog notified despite stale post-grace; notifications={notifications}"
        )
        # And the stale event was logged at least once
        assert any(e[1] == "watchdog_stale_post_grace" for e in log.events), (
            f"missing stale-post-grace log event; events={[e[1] for e in log.events]}"
        )

    def test_pre_grace_no_heartbeat_still_notifies(self, monkeypatch):
        """During the 30 s arming grace window with no heartbeats, watchdog
        SHOULD still send WATCHDOG=1 so systemd doesn't kill us during startup."""
        monkeypatch.setenv("NOTIFY_SOCKET", "/dev/null")
        obs._watchdog_armed_at = time.monotonic()  # just armed
        log = _FakeLogger()
        notifications: list[str] = []
        stop = threading.Event()
        t = obs.start_watchdog_thread(
            sd_notify_fn=lambda s: notifications.append(s),
            stop_event=stop,
            logger=log,
            threshold_s=5.0,
            interval_s=0.05,
        )
        time.sleep(0.2)
        stop.set()
        t.join(timeout=1.0)
        assert "WATCHDOG=1" in notifications


# ──────────────────────────────────────────────────────────────────────────
# (c) call_with_breaker: CircuitBreakerError fast-fails (H-O3)
# ──────────────────────────────────────────────────────────────────────────
@pytest.mark.skipif(not obs._HAS_PYBREAKER or not obs._HAS_TENACITY,
                    reason="pybreaker + tenacity required for this test")
class TestBreakerRespectsTenacity:
    def setup_method(self):
        # Fresh breaker registry per test — avoid state bleed-through.
        with obs._breaker_lock:
            obs._breakers.clear()
            obs._breaker_cache.clear()

    def test_circuit_open_error_not_retried(self):
        """If the breaker opens, call_with_breaker must surface the
        CircuitBreakerError immediately instead of waiting for tenacity to
        burn its 3 attempts on already-open calls (H-O3)."""
        import pybreaker

        # Force the breaker open by tripping fail_max
        breaker = obs.get_breaker("vllm", fail_max=2, reset_timeout_s=60)
        # Hammer the (real) breaker until it opens
        for _ in range(3):
            try:
                breaker.call(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
            except Exception:
                pass

        attempts = {"n": 0}
        def _fn():
            attempts["n"] += 1
            return "should not be called"

        t0 = time.perf_counter()
        with pytest.raises(pybreaker.CircuitBreakerError):
            obs.call_with_breaker("vllm", _fn)
        dur = time.perf_counter() - t0

        # _fn must NOT have run (breaker rejected the call)
        assert attempts["n"] == 0
        # And we did NOT wait for 3 tenacity backoffs (~1+ s); should be near-instant
        assert dur < 0.5, f"CircuitBreakerError was retried by tenacity (took {dur:.2f}s)"


# ──────────────────────────────────────────────────────────────────────────
# (d) init_metrics idempotency
# ──────────────────────────────────────────────────────────────────────────
@pytest.mark.skipif(not obs._HAS_PROMETHEUS, reason="prometheus_client required")
class TestInitMetricsIdempotency:
    def test_init_twice_does_not_raise(self):
        """init_metrics must be safely callable multiple times — duplicate
        registration would crash prometheus_client otherwise."""
        ok1 = obs.init_metrics()
        ok2 = obs.init_metrics()
        ok3 = obs.init_metrics()
        assert ok1 is True
        assert ok2 is True
        assert ok3 is True
        # And the module-level handles are now real Counter/Histogram, not _NoOpMetric
        from prometheus_client import Counter as _C, Histogram as _H
        assert isinstance(obs.dialog_outcome, _C)
        assert isinstance(obs.llm_latency, _H)
        # And the new fallback counters added in M-O2 are wired
        assert isinstance(obs.llm_fallback_total, _C)
        assert isinstance(obs.stt_fallback_total, _C)
        assert isinstance(obs.tts_fallback_total, _C)
