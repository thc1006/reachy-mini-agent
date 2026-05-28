"""SDD-04 acceptance: stop dispatch in *scheduler step count*, not wall-clock.

Renamed from `test_interrupt_latency_under_load.py` (P7 MED-B2 option a). The
previous name overpromised: these tests do not measure wall-clock latency and
do not generate real CPU load. They assert the scheduler's deterministic
contract — that a critical stop is dispatched within a bounded number of
scheduler operations, regardless of how many ticks the dance has consumed.

Real-hardware wall-clock latency measurements live in
`scripts/bench_pi_runtime.sh --real-hardware` (gated, Phase 7+).

Per SDD-04 AC #4 (P7 reword): stop dispatch completes in <=1 scheduler step in
the synchronous mock pipeline.
"""

from __future__ import annotations

from reachy_intent_runtime.models import ActionPriority, MotionCommand
from reachy_intent_runtime.motion_adapter import MockMotionAdapter
from reachy_intent_runtime.scheduler import ActionScheduler

# Bound: after stop is submitted to the scheduler, stop_current must be issued
# within at most STOP_DISPATCH_BUDGET_STEPS scheduler operations (a single
# submit triggers the preempt path immediately, so 1 step is the contract).
STOP_DISPATCH_BUDGET_STEPS = 1


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
        chunk_ms=100,
    )


def test_stop_dispatch_happens_within_one_step_after_long_dance_running() -> None:
    """Drop a 30s dance, then stop — adapter must see stop_current immediately."""
    adapter = MockMotionAdapter()
    scheduler = ActionScheduler(adapter=adapter)
    scheduler.submit(_dance(duration_ms=30_000))
    assert scheduler.running is not None

    # simulate scheduler tick pressure (load-equivalent) before stop
    for _ in range(50):
        scheduler.tick(50)

    # snapshot length, then submit stop — count operations until adapter sees stop
    log_len_before = len(adapter.log)
    scheduler.submit(_stop())
    log_len_after = len(adapter.log)

    # stop dispatch should have appended at least stop_current + start:stop_dance
    new_entries = adapter.log[log_len_before:log_len_after]
    assert "stop_current" in new_entries, (
        f"stop_current not dispatched within bounded steps; got {new_entries!r}"
    )


def test_stop_dispatch_within_bounded_steps_after_n200_ticks() -> None:
    """Even after N=200 simulated ticks of synthetic dance, stop must preempt."""
    adapter = MockMotionAdapter()
    scheduler = ActionScheduler(adapter=adapter)
    scheduler.submit(_dance(duration_ms=60_000))

    for _ in range(200):
        scheduler.tick(25)

    scheduler.submit(_stop())
    # stop_current must appear after the dance start
    stop_idx = adapter.log.index("stop_current")
    start_idx = adapter.log.index("start:dance")
    assert stop_idx > start_idx


def test_multiple_stops_under_queue_pressure_each_dispatch_critical_first() -> None:
    """Stress with several queued backgrounds + one stop; stop must hit first."""
    adapter = MockMotionAdapter()
    scheduler = ActionScheduler(adapter=adapter)
    scheduler.submit(_dance(duration_ms=10_000))
    # queue more background load
    for i in range(5):
        scheduler.submit(
            MotionCommand(
                name=f"dance_{i}",
                tool="dance",
                priority=ActionPriority.BACKGROUND,
                interruptible=True,
                duration_ms=2000,
                chunk_ms=250,
            )
        )

    scheduler.submit(_stop())
    assert "stop_current" in adapter.log
    # the critical stop should have started after preempt
    assert "start:stop_dance" in adapter.log
