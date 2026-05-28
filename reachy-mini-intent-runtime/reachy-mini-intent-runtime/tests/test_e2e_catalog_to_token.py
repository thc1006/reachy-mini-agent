"""End-to-end cross-track integration test (Phase 11 gap closure).

Exercises the full pipeline: ActionCatalog.load() -> OrchestratorWorker.route_utterance()
-> ActionScheduler.submit() -> MotionCommand with catalog-derived duration -> preempt by CRITICAL
-> scheduler.token_for() cancelled.

This guards against future regression where one track's change accidentally breaks
the cross-track contract.
"""

from __future__ import annotations

from reachy_intent_runtime.action_catalog import ActionCatalog
from reachy_intent_runtime.motion_adapter import MockMotionAdapter
from reachy_intent_runtime.orchestrator_worker import OrchestratorWorker
from reachy_intent_runtime.scheduler import ActionScheduler


def test_dance_then_stop_full_pipeline_with_catalog_and_token() -> None:
    catalog = ActionCatalog.load()
    adapter = MockMotionAdapter()
    scheduler = ActionScheduler(adapter=adapter)
    worker = OrchestratorWorker(scheduler=scheduler, catalog=catalog)

    # 1. User says "跳支舞" -> BACKGROUND dance enters scheduler
    worker.route_utterance("跳支舞")
    running_cmd = scheduler.running
    assert running_cmd is not None
    assert running_cmd.tool == "dance"
    # Duration drawn from catalog
    dance_durations = {e.estimated_duration_ms for e in catalog.all_dances()}
    assert running_cmd.duration_ms in dance_durations

    # 2. Cancellation token exists for the running dance
    token = scheduler.token_for(running_cmd)
    assert token is not None
    assert not token.is_cancelled

    # 3. User says "停止跳舞" -> CRITICAL stop_dance preempts
    worker.route_utterance("停止跳舞")

    # 4. Token cancelled with informative reason
    assert token.is_cancelled
    assert "preempt" in (token.reason or "")

    # 5. Adapter received stop_current + then start stop_dance
    assert "stop_current" in adapter.log


def test_natural_finish_does_not_leak_tokens() -> None:
    catalog = ActionCatalog.load()
    adapter = MockMotionAdapter()
    scheduler = ActionScheduler(adapter=adapter)
    worker = OrchestratorWorker(scheduler=scheduler, catalog=catalog)

    # Simulate 50 dance->stop cycles; tokens dict must remain bounded
    for _ in range(50):
        worker.route_utterance("跳支舞")
        worker.route_utterance("停止跳舞")

    assert len(scheduler._tokens) <= 5, (
        f"tokens dict leaked across 50 cycles: size={len(scheduler._tokens)}"
    )
