from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .models import MotionCommand


class MotionAdapter(Protocol):
    def start(self, command: MotionCommand) -> None: ...

    def stop_current(self) -> None: ...


@dataclass
class MockMotionAdapter:
    """Hardware-free adapter for tests and demos."""

    log: list[str] = field(default_factory=list)

    def start(self, command: MotionCommand) -> None:
        self.log.append(f"start:{command.name}")

    def stop_current(self) -> None:
        self.log.append("stop_current")
