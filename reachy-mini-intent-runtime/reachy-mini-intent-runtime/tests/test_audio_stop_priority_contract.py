"""SDD-04 acceptance: audio_listener stop event reaches scheduler within K steps.

Path: audio_listener -> orchestrator_worker -> ActionScheduler -> adapter.stop_current

K (bound) = 3 hop operations: listener.emit, orchestrator.route, scheduler.submit.
"""

from __future__ import annotations

from reachy_intent_runtime.audio_listener import MockAudioListener
from reachy_intent_runtime.motion_adapter import MockMotionAdapter
from reachy_intent_runtime.orchestrator_worker import OrchestratorWorker
from reachy_intent_runtime.scheduler import ActionScheduler

STOP_HOP_BUDGET = 3


def _bootstrap():
    adapter = MockMotionAdapter()
    scheduler = ActionScheduler(adapter=adapter)
    orchestrator = OrchestratorWorker(scheduler=scheduler)
    listener = MockAudioListener(orchestrator=orchestrator)
    return adapter, scheduler, orchestrator, listener


def test_audio_stop_event_reaches_adapter_within_hop_budget() -> None:
    adapter, scheduler, orch, listener = _bootstrap()
    # Drop a long-running dance into the scheduler first.
    from reachy_intent_runtime.models import ActionPriority, MotionCommand

    scheduler.submit(
        MotionCommand(
            name="dance",
            tool="dance",
            priority=ActionPriority.BACKGROUND,
            interruptible=True,
            duration_ms=30_000,
            chunk_ms=250,
        )
    )

    listener.emit_stop()  # synchronous mock; no threads
    # Hop count is tracked by the orchestrator's debug counter.
    assert orch.last_hop_count <= STOP_HOP_BUDGET, (
        f"stop reached scheduler in {orch.last_hop_count} hops, budget {STOP_HOP_BUDGET}"
    )
    assert "stop_current" in adapter.log


def test_audio_stop_event_when_idle_still_routes_to_scheduler() -> None:
    adapter, scheduler, orch, listener = _bootstrap()
    listener.emit_stop()
    # idle scheduler: critical-stop with duration 0 starts and finishes immediately
    assert any(name.startswith("start:stop_") for name in adapter.log)


def test_audio_listener_dispatches_only_stop_intent_in_mock_mode() -> None:
    adapter, scheduler, orch, listener = _bootstrap()
    # The mock listener only emits stops — never falsely fires a dance.
    listener.emit_stop()
    assert "start:dance" not in adapter.log
