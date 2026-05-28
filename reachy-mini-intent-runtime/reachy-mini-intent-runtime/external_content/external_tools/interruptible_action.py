"""External tool shim for reachy_mini_conversation_app.

This file is intentionally conservative because the exact Tool base class can change
upstream. Claude Code should inspect the installed official app and adapt this shim
while keeping the runtime logic in src/reachy_intent_runtime testable.
"""

from __future__ import annotations

from typing import Any

from reachy_intent_runtime.classifier import RuleIntentClassifier
from reachy_intent_runtime.models import ActionPriority, MotionCommand

_classifier = RuleIntentClassifier()


def classify_utterance(utterance: str) -> dict[str, Any]:
    """Classify an utterance into a command/chat decision.

    This plain function can be wrapped by the official conversation app's Tool base
    class once the exact installed version is inspected.
    """
    result = _classifier.classify(utterance)
    return {
        "kind": result.kind,
        "action": result.action,
        "priority": result.priority.value if result.priority else "none",
        "confidence": result.confidence,
        "reason": result.reason,
    }


def build_command(utterance: str) -> dict[str, Any] | None:
    result = _classifier.classify(utterance)
    if result.kind != "command" or result.action is None or result.priority is None:
        return None
    cmd = MotionCommand(
        name=result.action,
        tool=result.action,
        priority=result.priority,
        interruptible=result.priority != ActionPriority.CRITICAL,
        duration_ms=500 if result.priority == ActionPriority.CRITICAL else 5000,
        chunk_ms=100 if result.priority == ActionPriority.CRITICAL else 500,
        metadata={"utterance": utterance},
    )
    return cmd.to_dict()
