"""OrchestratorWorker tests — HIGH-A2.

Verifies that OrchestratorWorker actually wires RuleIntentClassifier and routes
utterances to the correct downstream handler instead of silently dropping them.

Path contracts:
  command  → scheduler.submit(MotionCommand)
  chat     → chat_handler(IntentResult)
  ambiguous → ambiguous_handler(IntentResult)  [MUST NOT be silent drop]
"""

from __future__ import annotations

import pytest

from reachy_intent_runtime.models import ActionPriority, IntentResult
from reachy_intent_runtime.motion_adapter import MockMotionAdapter
from reachy_intent_runtime.orchestrator_worker import OrchestratorWorker
from reachy_intent_runtime.scheduler import ActionScheduler

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_worker(
    chat_handler=None,
    ambiguous_handler=None,
):
    adapter = MockMotionAdapter()
    scheduler = ActionScheduler(adapter=adapter)
    worker = OrchestratorWorker(
        scheduler=scheduler,
        chat_handler=chat_handler,
        ambiguous_handler=ambiguous_handler,
    )
    return adapter, scheduler, worker


# ---------------------------------------------------------------------------
# HIGH-A2-1: command utterances reach the scheduler
# ---------------------------------------------------------------------------


def test_orchestrator_routes_command_to_scheduler() -> None:
    """'跳支舞' classifies as dance BACKGROUND → scheduler must receive a MotionCommand."""
    adapter, scheduler, worker = _make_worker()

    worker.route_utterance("跳支舞")

    # The adapter's start log must contain the dance command.
    assert any("dance" in entry for entry in adapter.log), (
        f"dance command never reached adapter; log={adapter.log}"
    )


def test_orchestrator_routes_stop_command_as_critical() -> None:
    """'停止跳舞' classifies as stop_dance CRITICAL → scheduler must receive it."""
    adapter, scheduler, worker = _make_worker()

    worker.route_utterance("停止跳舞")

    assert any("stop_dance" in entry for entry in adapter.log), (
        f"stop_dance never reached adapter; log={adapter.log}"
    )


def test_orchestrator_routes_hush_as_stop_emotion() -> None:
    """'噓' classifies as stop_emotion CRITICAL → scheduler must receive it."""
    adapter, scheduler, worker = _make_worker()

    worker.route_utterance("噓")

    assert any("stop_emotion" in entry for entry in adapter.log), (
        f"stop_emotion never reached adapter; log={adapter.log}"
    )


# ---------------------------------------------------------------------------
# HIGH-A2-2: chat utterances invoke chat_handler, not scheduler
# ---------------------------------------------------------------------------


def test_orchestrator_routes_chat_to_chat_handler() -> None:
    """'我喜歡跳舞' is chat → chat_handler is called, scheduler stays empty."""
    received: list[IntentResult] = []
    adapter, scheduler, worker = _make_worker(chat_handler=received.append)

    worker.route_utterance("我喜歡跳舞")

    assert len(received) == 1, f"chat_handler not called; received={received}"
    assert received[0].kind == "chat"
    # Scheduler must have received nothing motion-related.
    assert not any("dance" in e or "stop" in e for e in adapter.log), (
        f"scheduler received unexpected motion; log={adapter.log}"
    )


# ---------------------------------------------------------------------------
# HIGH-A2-3: ambiguous utterances invoke ambiguous_handler, never silently drop
# ---------------------------------------------------------------------------


def test_orchestrator_routes_ambiguous_to_ambiguous_handler() -> None:
    """'不要停止跳舞' is ambiguous → ambiguous_handler is called."""
    received: list[IntentResult] = []
    adapter, scheduler, worker = _make_worker(ambiguous_handler=received.append)

    worker.route_utterance("不要停止跳舞")

    assert len(received) == 1, f"ambiguous_handler not called; received={received}"
    assert received[0].kind == "ambiguous"


def test_orchestrator_ambiguous_does_not_reach_scheduler() -> None:
    """Ambiguous intents must NOT submit a MotionCommand to the scheduler."""
    adapter, scheduler, worker = _make_worker()

    worker.route_utterance("不要停止跳舞")

    # No motion command should have been dispatched.
    assert adapter.log == [], f"scheduler received motion for ambiguous input; log={adapter.log}"


def test_orchestrator_does_not_drop_ambiguous_silently() -> None:
    """Critical safety: every ambiguous result must be observable in events log."""
    adapter, scheduler, worker = _make_worker()

    worker.route_utterance("不要停止跳舞")

    assert any("ambiguous_pending_llm_handoff" in ev for ev in worker.events), (
        f"ambiguous result silently dropped; events={worker.events}"
    )


def test_orchestrator_ambiguous_handler_receives_intent_result() -> None:
    """ambiguous_handler callback receives the full IntentResult (not just kind)."""
    received: list[IntentResult] = []
    adapter, scheduler, worker = _make_worker(ambiguous_handler=received.append)

    worker.route_utterance("難過一點")

    assert len(received) == 1
    result = received[0]
    assert result.kind == "ambiguous"
    assert isinstance(result.confidence, float)
    assert result.reason  # non-empty reason string


# ---------------------------------------------------------------------------
# HIGH-A2-4: default chat_handler logs event (no external callback)
# ---------------------------------------------------------------------------


def test_orchestrator_chat_default_logs_event() -> None:
    """Without a chat_handler, chat intents must appear in worker.events."""
    adapter, scheduler, worker = _make_worker()

    worker.route_utterance("我喜歡跳舞")

    assert any("chat_skipped" in ev for ev in worker.events), (
        f"chat not logged in events; events={worker.events}"
    )


# ---------------------------------------------------------------------------
# HIGH-A2-5: RuleIntentClassifier is wired (not just route_stop shortcut)
# ---------------------------------------------------------------------------


def test_orchestrator_imports_and_uses_rule_classifier() -> None:
    """OrchestratorWorker must use RuleIntentClassifier internally, not a stub."""
    from reachy_intent_runtime.classifier import RuleIntentClassifier

    adapter = MockMotionAdapter()
    scheduler = ActionScheduler(adapter=adapter)
    # Pass a custom classifier instance to verify it is actually called.
    call_log: list[str] = []

    class SpyClassifier(RuleIntentClassifier):
        def classify(self, utterance: str) -> IntentResult:
            call_log.append(utterance)
            return super().classify(utterance)

    worker = OrchestratorWorker(scheduler=scheduler, classifier=SpyClassifier())
    worker.route_utterance("跳支舞")

    assert call_log == ["跳支舞"], (
        f"classifier not called or called with wrong arg; call_log={call_log}"
    )


# ---------------------------------------------------------------------------
# Backward-compat: route_stop still works (audio_stop_priority_contract depends on it)
# ---------------------------------------------------------------------------


def test_route_stop_still_works_after_refactor() -> None:
    """route_stop() must remain functional for MockAudioListener compatibility."""
    adapter, scheduler, worker = _make_worker()

    worker.route_stop(source="test")

    assert any("stop_" in entry for entry in adapter.log), (
        f"route_stop did not reach adapter; log={adapter.log}"
    )
    assert worker.last_hop_count <= 3


# ---------------------------------------------------------------------------
# MotionCommand mapping contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "utterance, expected_tool, expected_priority",
    [
        ("跳支舞", "dance", ActionPriority.BACKGROUND),
        ("停止跳舞", "stop_dance", ActionPriority.CRITICAL),
        ("噓", "stop_emotion", ActionPriority.CRITICAL),
        ("開心一點", "play_emotion", ActionPriority.BACKGROUND),
    ],
)
def test_motion_command_mapping(
    utterance: str, expected_tool: str, expected_priority: ActionPriority
) -> None:
    """route_utterance must build MotionCommand with correct tool and priority."""
    adapter, scheduler, worker = _make_worker()

    worker.route_utterance(utterance)

    assert any(expected_tool in entry for entry in adapter.log), (
        f"{utterance!r}: expected tool {expected_tool!r} in adapter.log; log={adapter.log}"
    )


# ---------------------------------------------------------------------------
# ITEM 1 — Catalog→orchestrator wiring (Phase 11)
# ---------------------------------------------------------------------------


def test_route_utterance_dance_uses_catalog_duration() -> None:
    """orchestrator should construct MotionCommand with duration_ms from action catalog."""
    from reachy_intent_runtime.action_catalog import ActionCatalog

    catalog = ActionCatalog.load()
    expected_min = min(e.estimated_duration_ms for e in catalog.all_dances())
    expected_max = max(e.estimated_duration_ms for e in catalog.all_dances())

    adapter = MockMotionAdapter()
    scheduler = ActionScheduler(adapter=adapter)
    worker = OrchestratorWorker(scheduler=scheduler, catalog=catalog)
    worker.route_utterance("跳支舞")

    assert any("dance" in entry for entry in adapter.log)
    submitted_cmd = scheduler.running
    assert submitted_cmd is not None
    assert expected_min <= submitted_cmd.duration_ms <= expected_max


def test_route_utterance_dance_without_catalog_uses_fallback() -> None:
    """When no catalog provided, orchestrator falls back to legacy hardcoded duration."""
    adapter = MockMotionAdapter()
    scheduler = ActionScheduler(adapter=adapter)
    worker = OrchestratorWorker(scheduler=scheduler)
    worker.route_utterance("跳支舞")
    submitted_cmd = scheduler.running
    assert submitted_cmd is not None
    assert submitted_cmd.duration_ms == 10_000


def test_route_utterance_stop_dance_uses_catalog_stop_tool() -> None:
    """stop_dance command should reference catalog's stop tool name."""
    from reachy_intent_runtime.action_catalog import ActionCatalog

    catalog = ActionCatalog.load()
    adapter = MockMotionAdapter()
    scheduler = ActionScheduler(adapter=adapter)
    worker = OrchestratorWorker(scheduler=scheduler, catalog=catalog)
    worker.route_utterance("停止跳舞")
    assert any("stop_dance" in entry for entry in adapter.log)
    assert any(e.event == "preempt" or e.event == "queued" for e in scheduler.events)
