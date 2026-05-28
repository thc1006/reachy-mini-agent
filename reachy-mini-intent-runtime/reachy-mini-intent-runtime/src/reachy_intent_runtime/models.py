from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import IntEnum
from typing import Any, Literal

IntentKind = Literal["chat", "command", "ambiguous"]


class ActionPriority(IntEnum):
    """Lower numeric value means higher scheduling priority."""

    CRITICAL = 0
    INTERACTIVE = 10
    BACKGROUND = 20


@dataclass(frozen=True)
class IntentResult:
    kind: IntentKind
    action: str | None
    priority: ActionPriority | None
    confidence: float
    reason: str


@dataclass(frozen=True)
class MotionCommand:
    name: str
    tool: str
    priority: ActionPriority
    interruptible: bool = True
    duration_ms: int = 0
    chunk_ms: int = 250
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["priority"] = self.priority.name.lower()
        return data


@dataclass(frozen=True)
class SchedulerEvent:
    event: str
    command: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
