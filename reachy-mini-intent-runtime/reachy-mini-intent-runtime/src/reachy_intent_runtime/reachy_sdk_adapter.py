from __future__ import annotations

from dataclasses import dataclass

from .models import MotionCommand


@dataclass
class ReachySdkAdapter:
    """Thin optional adapter around reachy_mini SDK.

    This adapter intentionally avoids import-time dependency on hardware SDK so tests
    can run on any machine. Claude Code should extend this after inspecting the exact
    installed SDK/conversation-app version.
    """

    connection_mode: str | None = None

    def __post_init__(self) -> None:
        try:
            from reachy_mini import ReachyMini  # type: ignore
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise RuntimeError(
                "reachy_mini SDK is not installed. "
                "Install the official SDK or use MockMotionAdapter."
            ) from exc
        self._reachy_cls = ReachyMini
        self._reachy = None

    def __enter__(self):  # pragma: no cover - hardware-dependent
        kwargs = {}
        if self.connection_mode:
            kwargs["connection_mode"] = self.connection_mode
        self._reachy = self._reachy_cls(**kwargs)
        self._reachy.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):  # pragma: no cover - hardware-dependent
        if self._reachy is not None:
            return self._reachy.__exit__(exc_type, exc, tb)
        return None

    def start(self, command: MotionCommand) -> None:  # pragma: no cover - hardware-dependent
        if self._reachy is None:
            raise RuntimeError("ReachySdkAdapter must be used as a context manager")
        raise NotImplementedError(
            f"Implement SDK call for {command.tool}. Prefer official app tools if available."
        )

    def stop_current(self) -> None:  # pragma: no cover - hardware-dependent
        if self._reachy is None:
            raise RuntimeError("ReachySdkAdapter must be used as a context manager")
        # TODO: Inspect official SDK/conversation app to call stop_dance/stop_emotion
        # or clear the queue once those semantics are verified.
        raise NotImplementedError("Implement stop_current after verifying SDK stop semantics.")
