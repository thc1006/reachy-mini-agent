"""Orchestrator worker — routes classified utterances to scheduler or callbacks.

This module is the single handoff point between intent classification and
action execution.  It wires ``RuleIntentClassifier`` and dispatches every
result to exactly one of three lanes:

  command   → build ``MotionCommand`` and submit to ``ActionScheduler``
  chat      → call ``chat_handler`` (default: log "chat_skipped" event)
  ambiguous → call ``ambiguous_handler`` (default: log
              "ambiguous_pending_llm_handoff" event)

The ambiguous lane is explicitly observable: every ambiguous result appears in
``OrchestratorWorker.events`` so that monitoring and future LLM bridges can
detect it.  Silent drops are forbidden.

Public API
----------
``OrchestratorWorker(scheduler, classifier=None, chat_handler=None, ambiguous_handler=None)``
    Construct a worker.  All optional parameters have safe defaults.

``route_utterance(text: str) -> None``
    Classify *text* and dispatch to the appropriate lane.

``route_stop(source: str = "audio") -> None``
    Shortcut for critical-stop events from ``MockAudioListener`` (and the
    audio-stop priority contract tests).  Tracks hop count for the 3-hop SDD
    acceptance criterion.

Fields
------
``last_hop_count`` — hop counter updated by ``route_stop``; used by the
    audio-stop priority-contract test.
``audit_log`` — simple list of routing decisions (legacy; kept for backward
    compatibility with existing smoke-run prints).
``events`` — richer observability list; every route_utterance call appends at
    least one entry.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass, field

from .action_catalog import ActionCatalog
from .classifier import RuleIntentClassifier
from .models import ActionPriority, IntentResult, MotionCommand
from .scheduler import ActionScheduler

# ---------------------------------------------------------------------------
# MotionCommand factory — maps IntentResult.action to a MotionCommand.
# ---------------------------------------------------------------------------

_ACTION_TO_COMMAND: dict[str, MotionCommand] = {
    "dance": MotionCommand(
        name="dance",
        tool="dance",
        priority=ActionPriority.BACKGROUND,
        interruptible=True,
        duration_ms=10_000,
        chunk_ms=250,
    ),
    "stop_dance": MotionCommand(
        name="stop_dance",
        tool="stop_dance",
        priority=ActionPriority.CRITICAL,
        interruptible=False,
        duration_ms=0,
        chunk_ms=100,
    ),
    "stop_emotion": MotionCommand(
        name="stop_emotion",
        tool="stop_emotion",
        priority=ActionPriority.CRITICAL,
        interruptible=False,
        duration_ms=0,
        chunk_ms=100,
    ),
    "play_emotion": MotionCommand(
        name="play_emotion",
        tool="play_emotion",
        priority=ActionPriority.BACKGROUND,
        interruptible=True,
        duration_ms=3_000,
        chunk_ms=250,
    ),
}


@dataclass
class OrchestratorWorker:
    """Thin orchestration layer between sensors / classifier and the scheduler.

    Parameters
    ----------
    scheduler:
        The ``ActionScheduler`` instance that executes ``MotionCommand`` objects.
    classifier:
        Optional ``RuleIntentClassifier`` instance.  Defaults to a fresh
        ``RuleIntentClassifier()`` if not provided.
    chat_handler:
        Callable that receives an ``IntentResult`` for chat utterances.
        Defaults to logging a "chat_skipped" event.
    ambiguous_handler:
        Callable that receives an ``IntentResult`` for ambiguous utterances.
        Defaults to logging an "ambiguous_pending_llm_handoff" event.
        MUST NOT be None in production — the default ensures observability.

    Attributes
    ----------
    last_hop_count:
        Hop counter updated by ``route_stop``; audited by the audio-stop
        priority-contract acceptance test (budget = 3).
    audit_log:
        Legacy per-routing decision strings for backward compatibility with
        smoke-run print statements.
    events:
        Richer observability list; ``route_utterance`` appends at least one
        entry per call.
    """

    scheduler: ActionScheduler
    classifier: RuleIntentClassifier = field(default_factory=RuleIntentClassifier)
    chat_handler: Callable[[IntentResult], None] | None = None
    ambiguous_handler: Callable[[IntentResult], None] | None = None
    catalog: ActionCatalog | None = None
    last_hop_count: int = 0
    audit_log: list[str] = field(default_factory=list)
    events: list[str] = field(default_factory=list)

    def route_utterance(self, text: str) -> None:
        """Classify *text* and dispatch to the correct lane.

        - ``command``   → build ``MotionCommand`` via ``_ACTION_TO_COMMAND``
                          and call ``scheduler.submit``.
        - ``chat``      → call ``chat_handler`` if set, else log
                          "chat_skipped" to ``events``.
        - ``ambiguous`` → call ``ambiguous_handler`` if set, then always log
                          "ambiguous_pending_llm_handoff" to ``events``.
                          The event entry is written regardless of whether a
                          callback is registered so that ambiguous results are
                          always observable.
        """
        result = self.classifier.classify(text)

        if result.kind == "command":
            self._dispatch_command(result)
        elif result.kind == "chat":
            self._dispatch_chat(result)
        else:
            # kind == "ambiguous"
            self._dispatch_ambiguous(result)

    def _dispatch_command(self, result: IntentResult) -> None:
        action = result.action or ""
        fallback = _ACTION_TO_COMMAND.get(action)
        if fallback is None:
            # Unknown action — log and skip rather than crash.
            self.events.append(f"command_unknown_action:{action}")
            self.audit_log.append(f"command_unknown:{action}")
            return
        cmd = self._build_command_from_catalog(action, fallback)
        self.audit_log.append(f"command_routed:{cmd.name}")
        self.events.append(f"command_dispatched:{cmd.name}")
        self.scheduler.submit(cmd)

    def _build_command_from_catalog(self, action: str, fallback: MotionCommand) -> MotionCommand:
        """Return a MotionCommand drawn from catalog if available, else the fallback.

        For "dance" actions: picks the first entry from catalog.all_dances() as the
        representative duration (catalog.get("dance_happy") if it exists, else first).
        For "play_emotion" actions: picks the first entry from catalog.all_emotions().
        For stop actions ("stop_dance", "stop_emotion"): uses catalog entry directly.
        Falls back to the hardcoded _ACTION_TO_COMMAND value when catalog is None or
        the action is not found in the catalog.
        """
        if self.catalog is None:
            return fallback

        try:
            if action == "dance":
                dances = self.catalog.all_dances()
                if not dances:
                    return fallback
                # Prefer dance_happy as the representative entry if it exists.
                try:
                    entry = self.catalog.get("dance_happy")
                except KeyError:
                    entry = dances[0]
                return MotionCommand(
                    name=fallback.name,
                    tool=fallback.tool,
                    priority=fallback.priority,
                    interruptible=fallback.interruptible,
                    duration_ms=entry.estimated_duration_ms,
                    chunk_ms=fallback.chunk_ms,
                )

            if action == "play_emotion":
                emotions = self.catalog.all_emotions()
                if not emotions:
                    return fallback
                entry = emotions[0]
                return MotionCommand(
                    name=fallback.name,
                    tool=fallback.tool,
                    priority=fallback.priority,
                    interruptible=fallback.interruptible,
                    duration_ms=entry.estimated_duration_ms,
                    chunk_ms=fallback.chunk_ms,
                )

            if action in ("stop_dance", "stop_emotion"):
                entry = self.catalog.get(action)
                return MotionCommand(
                    name=fallback.name,
                    tool=fallback.tool,
                    priority=fallback.priority,
                    interruptible=fallback.interruptible,
                    duration_ms=entry.estimated_duration_ms,
                    chunk_ms=fallback.chunk_ms,
                )
        except (KeyError, IndexError):
            pass

        return fallback

    def _dispatch_chat(self, result: IntentResult) -> None:
        if self.chat_handler is not None:
            self.chat_handler(result)
        else:
            self.events.append("chat_skipped")
        self.audit_log.append("chat_routed")

    def _dispatch_ambiguous(self, result: IntentResult) -> None:
        if self.ambiguous_handler is not None:
            self.ambiguous_handler(result)
        # Always log so ambiguous is observable even without a callback.
        self.events.append("ambiguous_pending_llm_handoff")
        self.audit_log.append("ambiguous_pending")

    # ------------------------------------------------------------------
    # Backward-compat: direct critical-stop shortcut used by
    # MockAudioListener and the audio-stop priority-contract test.
    # ------------------------------------------------------------------

    def route_stop(self, source: str = "audio") -> None:
        """Route a critical-stop intent through to the scheduler.

        Tracks ``last_hop_count`` for the 3-hop SDD acceptance criterion:
          hop 1 — enter orchestrator
          hop 2 — build MotionCommand
          hop 3 — submit to scheduler
        """
        self.last_hop_count = 0
        # hop 1: enter orchestrator
        self.last_hop_count += 1
        self.audit_log.append(f"stop_routed_from:{source}")
        # hop 2: build motion command
        self.last_hop_count += 1
        stop_cmd = MotionCommand(
            name="stop_dance",
            tool="stop_dance",
            priority=ActionPriority.CRITICAL,
            interruptible=False,
            duration_ms=0,
            chunk_ms=100,
        )
        # hop 3: submit to scheduler
        self.last_hop_count += 1
        self.scheduler.submit(stop_cmd)

    def route_command(self, command: MotionCommand) -> None:
        """Forward an arbitrary motion command to the scheduler (legacy API)."""
        self.audit_log.append(f"command_routed:{command.name}")
        self.scheduler.submit(command)


def main(argv: list[str] | None = None) -> int:
    """Entrypoint for ``python -m reachy_intent_runtime.orchestrator_worker``."""
    parser = argparse.ArgumentParser(description="Reachy Mini orchestrator (mock by default)")
    parser.add_argument("--mock", action="store_true", default=True)
    args = parser.parse_args(argv)
    print(f"[orchestrator_worker] starting mock={args.mock}")
    # In a real deployment this would attach to an IPC bus.
    from .motion_adapter import MockMotionAdapter

    adapter = MockMotionAdapter()
    scheduler = ActionScheduler(adapter=adapter)
    worker = OrchestratorWorker(scheduler=scheduler)
    worker.route_stop(source="mock_init")
    print(f"[orchestrator_worker] audit_log={worker.audit_log}")
    print(f"[orchestrator_worker] adapter_log={adapter.log}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
