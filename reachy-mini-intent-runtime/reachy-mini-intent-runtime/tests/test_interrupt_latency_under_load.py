"""Wall-clock latency test for critical interrupt under simulated load.

Companion to test_interrupt_dispatch_steps.py (which counts scheduler steps).
This file measures real elapsed time using time.perf_counter and enforces
SDD-04 acceptance criterion #1: critical stop dispatch within 500 ms.

Synthetic CPU pressure is applied via concurrent.futures.ProcessPoolExecutor
to validate that the synchronous preempt path completes well within budget
even when the Python interpreter is contended.
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor

from reachy_intent_runtime.models import ActionPriority, MotionCommand
from reachy_intent_runtime.motion_adapter import MockMotionAdapter
from reachy_intent_runtime.scheduler import ActionScheduler


def _busy_burn(deadline_s: float) -> None:
    """Simple CPU burner used by ProcessPoolExecutor to apply synthetic load."""
    end = time.perf_counter() + deadline_s
    x = 0
    while time.perf_counter() < end:
        x = (x + 1) % 1_000_003


def _dance(duration_ms: int = 30_000) -> MotionCommand:
    return MotionCommand(
        name="dance",
        tool="dance",
        priority=ActionPriority.BACKGROUND,
        interruptible=True,
        duration_ms=duration_ms,
        chunk_ms=250,
    )


def _stop() -> MotionCommand:
    return MotionCommand(
        name="stop_dance",
        tool="stop_dance",
        priority=ActionPriority.CRITICAL,
        interruptible=False,
        duration_ms=0,
    )


def test_critical_stop_dispatch_under_500ms_wallclock() -> None:
    """SDD-04 AC #1: critical stop dispatch completes in under 500 ms (no load)."""
    adapter = MockMotionAdapter()
    scheduler = ActionScheduler(adapter=adapter)
    scheduler.submit(_dance())

    t0 = time.perf_counter()
    scheduler.submit(_stop())
    elapsed_ms = (time.perf_counter() - t0) * 1000

    assert "stop_current" in adapter.log, "stop_current never issued"
    assert elapsed_ms < 500, f"critical stop dispatch took {elapsed_ms:.1f} ms > 500 ms budget"


def test_critical_stop_dispatch_remains_under_500ms_under_cpu_load() -> None:
    """SDD-04 AC #1 under CPU load: stop dispatch still completes in under 500 ms."""
    with ProcessPoolExecutor(max_workers=2) as pool:
        # Spin up burners then measure dispatch while they run
        _f1 = pool.submit(_busy_burn, 1.5)
        _f2 = pool.submit(_busy_burn, 1.5)
        time.sleep(0.05)  # allow burners to ramp up

        adapter = MockMotionAdapter()
        scheduler = ActionScheduler(adapter=adapter)
        scheduler.submit(_dance())

        t0 = time.perf_counter()
        scheduler.submit(_stop())
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # best-effort cancel — processes may already be finishing
        _f1.cancel()
        _f2.cancel()

    assert "stop_current" in adapter.log, "stop_current never issued under load"
    assert elapsed_ms < 500, f"under-load critical stop took {elapsed_ms:.1f} ms > 500 ms budget"


def test_critical_stop_token_cancelled_within_500ms_wallclock() -> None:
    """CancellationToken for the preempted dance is cancelled within 500 ms wall-clock."""
    adapter = MockMotionAdapter()
    scheduler = ActionScheduler(adapter=adapter)
    dance = _dance()
    scheduler.submit(dance)
    token = scheduler.token_for(dance)
    assert token is not None

    t0 = time.perf_counter()
    scheduler.submit(_stop())
    elapsed_ms = (time.perf_counter() - t0) * 1000

    assert token.is_cancelled, "dance token not cancelled after critical stop"
    assert elapsed_ms < 500, f"token cancellation took {elapsed_ms:.1f} ms > 500 ms budget"
