from __future__ import annotations

import pytest

from reachy_intent_runtime.models import ActionPriority, MotionCommand
from reachy_intent_runtime.motion_adapter import MockMotionAdapter
from reachy_intent_runtime.scheduler import ActionScheduler


def cmd(
    name: str,
    priority: ActionPriority,
    duration_ms: int = 1000,
    interruptible: bool | None = None,
    chunk_ms: int = 100,
) -> MotionCommand:
    if interruptible is None:
        interruptible = priority != ActionPriority.CRITICAL
    return MotionCommand(
        name=name,
        tool=name,
        priority=priority,
        interruptible=interruptible,
        duration_ms=duration_ms,
        chunk_ms=chunk_ms,
    )


# ---------------------------------------------------------------------------
# A1 – existing baselines (regression guards)
# ---------------------------------------------------------------------------


def test_critical_stop_preempts_running_dance() -> None:
    adapter = MockMotionAdapter()
    scheduler = ActionScheduler(adapter=adapter)
    scheduler.submit(cmd("dance", ActionPriority.BACKGROUND, duration_ms=30_000))
    assert scheduler.running is not None
    scheduler.submit(cmd("stop_dance", ActionPriority.CRITICAL, duration_ms=0))
    assert adapter.log[:3] == ["start:dance", "stop_current", "start:stop_dance"]
    assert any(event.event == "preempt" for event in scheduler.events)


def test_background_actions_queue_fifo() -> None:
    adapter = MockMotionAdapter()
    scheduler = ActionScheduler(adapter=adapter)
    scheduler.submit(cmd("dance_a", ActionPriority.BACKGROUND, duration_ms=100))
    scheduler.submit(cmd("dance_b", ActionPriority.BACKGROUND, duration_ms=100))
    scheduler.tick(100)
    assert adapter.log == ["start:dance_a", "start:dance_b"]


def test_interactive_waits_until_boundary_for_background() -> None:
    adapter = MockMotionAdapter()
    scheduler = ActionScheduler(adapter=adapter)
    scheduler.submit(cmd("dance", ActionPriority.BACKGROUND, duration_ms=1000))
    scheduler.submit(cmd("call_nurse", ActionPriority.INTERACTIVE, duration_ms=100))
    assert adapter.log == ["start:dance"]
    scheduler.tick(100)
    assert "stop_current" in adapter.log
    assert "start:call_nurse" in adapter.log


# ---------------------------------------------------------------------------
# A2 – new edge-case tests
# ---------------------------------------------------------------------------


def test_critical_running_is_not_preempted_by_new_critical() -> None:
    """A running non-interruptible CRITICAL must not be preempted by another CRITICAL."""
    adapter = MockMotionAdapter()
    scheduler = ActionScheduler(adapter=adapter)
    first = cmd(
        "first_critical",
        ActionPriority.CRITICAL,
        duration_ms=500,
        interruptible=False,
    )
    scheduler.submit(first)
    assert scheduler.running is not None
    assert scheduler.running.name == "first_critical"

    second = cmd("second_critical", ActionPriority.CRITICAL, duration_ms=0)
    scheduler.submit(second)

    # first_critical must still be running — no stop_current yet
    assert "stop_current" not in adapter.log
    assert scheduler.running is not None
    assert scheduler.running.name == "first_critical"

    # Only one start logged for the original critical
    starts = [e for e in adapter.log if e.startswith("start:")]
    assert starts == ["start:first_critical"]


def test_zero_duration_critical_finishes_immediately() -> None:
    """A CRITICAL with duration_ms=0 starts, emits started+finished, leaves scheduler idle."""
    adapter = MockMotionAdapter()
    scheduler = ActionScheduler(adapter=adapter)
    scheduler.submit(cmd("stop_x", ActionPriority.CRITICAL, duration_ms=0))

    assert adapter.log == ["start:stop_x"]
    event_types = [e.event for e in scheduler.events]
    assert "started" in event_types
    assert "finished" in event_types
    assert scheduler.running is None


def test_zero_duration_critical_preempts_then_resumes_nothing() -> None:
    """CRITICAL(duration=0) preempts BACKGROUND dance; dance is NOT resumed after stop."""
    adapter = MockMotionAdapter()
    scheduler = ActionScheduler(adapter=adapter)
    scheduler.submit(
        cmd("dance", ActionPriority.BACKGROUND, duration_ms=10_000, interruptible=True)
    )
    assert scheduler.running is not None

    scheduler.submit(cmd("stop_dance", ActionPriority.CRITICAL, duration_ms=0))

    # preempt occurred
    assert "stop_current" in adapter.log
    assert "start:stop_dance" in adapter.log
    # scheduler ends idle — dance is discarded not paused
    assert scheduler.running is None
    # dance must not restart
    starts = [e for e in adapter.log if e.startswith("start:")]
    assert starts == ["start:dance", "start:stop_dance"]


def test_non_interruptible_background_blocks_interactive_at_boundary() -> None:
    """Non-interruptible BACKGROUND must not be preempted by INTERACTIVE at chunk boundary."""
    adapter = MockMotionAdapter()
    scheduler = ActionScheduler(adapter=adapter)
    bg = cmd(
        "bg_task",
        ActionPriority.BACKGROUND,
        duration_ms=1000,
        interruptible=False,
        chunk_ms=100,
    )
    scheduler.submit(bg)
    scheduler.submit(cmd("call_nurse", ActionPriority.INTERACTIVE, duration_ms=100))

    # After first boundary — bg_task must still be running (non-interruptible blocks preempt)
    scheduler.tick(100)
    assert "stop_current" not in adapter.log
    assert scheduler.running is not None
    assert scheduler.running.name == "bg_task"

    # Let bg_task finish — tick through remaining 900 ms.
    # _start_next_if_idle fires automatically so call_nurse starts inside this tick.
    scheduler.tick(900)
    assert "start:call_nurse" in adapter.log

    # Confirm boundary_preempt_blocked event was emitted for audit trail
    blocked = [e for e in scheduler.events if e.event == "boundary_preempt_blocked"]
    assert len(blocked) >= 1
    assert blocked[0].details.get("reason") == "non_interruptible"


def test_chunk_boundary_only_preempts_when_higher_priority_present() -> None:
    """At chunk boundary, same-priority BACKGROUND in queue must not preempt running BACKGROUND."""
    adapter = MockMotionAdapter()
    scheduler = ActionScheduler(adapter=adapter)
    scheduler.submit(
        cmd(
            "dance_a",
            ActionPriority.BACKGROUND,
            duration_ms=1000,
            interruptible=True,
            chunk_ms=100,
        )
    )
    scheduler.submit(
        cmd(
            "dance_b",
            ActionPriority.BACKGROUND,
            duration_ms=100,
            interruptible=True,
            chunk_ms=100,
        )
    )

    # At boundary, dance_b has same priority — no preempt
    scheduler.tick(100)
    assert "stop_current" not in adapter.log
    assert scheduler.running is not None
    assert scheduler.running.name == "dance_a"


def test_event_log_records_preempt_blocked_for_non_interruptible_running() -> None:
    """When non-interruptible runs and new CRITICAL is submitted, preempt_blocked event emitted."""
    adapter = MockMotionAdapter()
    scheduler = ActionScheduler(adapter=adapter)
    scheduler.submit(
        cmd("first_critical", ActionPriority.CRITICAL, duration_ms=500, interruptible=False)
    )
    scheduler.submit(cmd("second_critical", ActionPriority.CRITICAL, duration_ms=0))

    blocked_events = [e for e in scheduler.events if e.event == "preempt_blocked"]
    assert len(blocked_events) >= 1
    ev = blocked_events[0]
    assert ev.details.get("reason") == "non_interruptible"
    assert ev.details.get("requested") == "second_critical"


def test_preempt_event_includes_priority_in_details() -> None:
    """The 'preempt' event details must include from_priority and to_priority."""
    adapter = MockMotionAdapter()
    scheduler = ActionScheduler(adapter=adapter)
    scheduler.submit(
        cmd("dance", ActionPriority.BACKGROUND, duration_ms=30_000, interruptible=True)
    )
    scheduler.submit(cmd("stop_dance", ActionPriority.CRITICAL, duration_ms=0))

    preempt_events = [e for e in scheduler.events if e.event == "preempt"]
    assert len(preempt_events) >= 1
    ev = preempt_events[0]
    assert ev.details.get("from_priority") == "background"
    assert ev.details.get("to_priority") == "critical"


def test_tick_negative_raises() -> None:
    """tick() with a negative elapsed_ms must raise ValueError."""
    adapter = MockMotionAdapter()
    scheduler = ActionScheduler(adapter=adapter)
    with pytest.raises(ValueError):
        scheduler.tick(-1)


def test_tick_with_no_running_is_noop() -> None:
    """tick() on an idle scheduler with empty queue must not error or change state."""
    adapter = MockMotionAdapter()
    scheduler = ActionScheduler(adapter=adapter)
    scheduler.tick(50)
    assert scheduler.running is None
    assert adapter.log == []
    assert scheduler.events == []


def test_running_property_returns_command_or_none() -> None:
    """running property returns correct MotionCommand during lifecycle."""
    adapter = MockMotionAdapter()
    scheduler = ActionScheduler(adapter=adapter)

    # idle: None
    assert scheduler.running is None

    dance = cmd("dance", ActionPriority.BACKGROUND, duration_ms=500)
    scheduler.submit(dance)
    # after submit: running is dance
    assert scheduler.running is not None
    assert scheduler.running.name == "dance"

    # after it finishes: None
    scheduler.tick(500)
    assert scheduler.running is None


# ---------------------------------------------------------------------------
# MED-1: queued event priority string must be lowercase per SDD-02
# ---------------------------------------------------------------------------


def test_queued_event_priority_is_lowercase() -> None:
    """'queued' event details['priority'] must be lowercase per SDD-02 §42."""
    adapter = MockMotionAdapter()
    scheduler = ActionScheduler(adapter=adapter)
    scheduler.submit(cmd("dance", ActionPriority.BACKGROUND, duration_ms=100))
    queued = next(e for e in scheduler.events if e.event == "queued")
    assert queued.details["priority"] == "background", (
        f"Expected 'background' but got {queued.details['priority']!r}"
    )


# ---------------------------------------------------------------------------
# Phase-10: CancellationToken integration
# ---------------------------------------------------------------------------


def test_preempt_signals_cancellation_token_on_running_command() -> None:
    """When CRITICAL preempts running interruptible BACKGROUND, its token is cancelled."""
    adapter = MockMotionAdapter()
    scheduler = ActionScheduler(adapter=adapter)
    dance = cmd("dance", ActionPriority.BACKGROUND, duration_ms=10000)
    scheduler.submit(dance)
    token = scheduler.token_for(dance)
    assert token is not None
    assert not token.is_cancelled
    scheduler.submit(cmd("stop_dance", ActionPriority.CRITICAL, duration_ms=0))
    assert token.is_cancelled
    assert "preempt" in (token.reason or "")


def test_token_for_non_running_is_none() -> None:
    """token_for returns None for a command never submitted."""
    adapter = MockMotionAdapter()
    scheduler = ActionScheduler(adapter=adapter)
    fake = cmd("fake", ActionPriority.BACKGROUND, duration_ms=100)
    assert scheduler.token_for(fake) is None


def test_completed_command_token_is_not_cancelled() -> None:
    """A naturally-finished command token stays uncancelled (scheduler didn't preempt it)."""
    adapter = MockMotionAdapter()
    scheduler = ActionScheduler(adapter=adapter)
    short = cmd("short", ActionPriority.BACKGROUND, duration_ms=100)
    scheduler.submit(short)
    token = scheduler.token_for(short)
    scheduler.tick(100)  # natural finish
    assert token is not None
    assert not token.is_cancelled


def test_token_reason_contains_preempting_command_name() -> None:
    """Cancellation reason string must include the preempting command's name."""
    adapter = MockMotionAdapter()
    scheduler = ActionScheduler(adapter=adapter)
    dance = cmd("dance", ActionPriority.BACKGROUND, duration_ms=10000)
    scheduler.submit(dance)
    token = scheduler.token_for(dance)
    scheduler.submit(cmd("stop_dance", ActionPriority.CRITICAL, duration_ms=0))
    assert token is not None
    assert "stop_dance" in (token.reason or "")


def test_boundary_preempt_also_cancels_token() -> None:
    """Boundary preempt (tick-driven) must also cancel the running command's token."""
    adapter = MockMotionAdapter()
    scheduler = ActionScheduler(adapter=adapter)
    dance = cmd("dance", ActionPriority.BACKGROUND, duration_ms=1000, chunk_ms=100)
    scheduler.submit(dance)
    token = scheduler.token_for(dance)
    # Queue an interactive — it will preempt at the first boundary
    scheduler.submit(cmd("call_nurse", ActionPriority.INTERACTIVE, duration_ms=100))
    assert token is not None
    assert not token.is_cancelled  # not yet preempted
    scheduler.tick(100)  # boundary fires
    assert token.is_cancelled


# ---------------------------------------------------------------------------
# ITEM 2 — _tokens dict memory leak fix (Phase 11)
# ---------------------------------------------------------------------------


def test_tokens_dict_is_cleaned_on_natural_finish() -> None:
    """_tokens must be cleaned after a command finishes naturally."""
    adapter = MockMotionAdapter()
    scheduler = ActionScheduler(adapter=adapter)
    short = cmd("short", ActionPriority.BACKGROUND, duration_ms=100)
    scheduler.submit(short)
    assert scheduler._tokens  # token exists during run
    scheduler.tick(100)  # natural finish
    assert not scheduler._tokens, "tokens dict must be cleaned after natural finish"


def test_tokens_dict_is_cleaned_on_preempt() -> None:
    """_tokens must not grow without bound after multiple preempt cycles."""
    adapter = MockMotionAdapter()
    scheduler = ActionScheduler(adapter=adapter)
    dance = cmd("dance", ActionPriority.BACKGROUND, duration_ms=10000)
    scheduler.submit(dance)
    scheduler.submit(cmd("stop", ActionPriority.CRITICAL, duration_ms=0))
    initial_count = len(scheduler._tokens)
    for _ in range(20):
        more_dance = cmd("dance2", ActionPriority.BACKGROUND, duration_ms=10000)
        scheduler.submit(more_dance)
        scheduler.submit(cmd("stop2", ActionPriority.CRITICAL, duration_ms=0))
    final_count = len(scheduler._tokens)
    assert final_count <= initial_count + 5, f"tokens dict leaked: {initial_count} → {final_count}"
