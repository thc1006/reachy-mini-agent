"""Reachy Mini B2 multitask orchestrator.

Three actors (perception, dialog, motion) communicate over an event bus.
This package is a POC: actors run *simulated* workloads with realistic
latencies drawn from the production logs. The point is to compare a
serial baseline against the event-driven design head-to-head; hooking
the real STT/LLM/TTS is a follow-up PR.

See ``docs/multitask-arch.md`` for the design.
"""

try:
    from .dialog import Dialog, DialogConfig, DialogStats
    from .motion import Motion, MotionConfig, MotionStats
    from .perception import Perception, PerceptionConfig, PerceptionStats
    from .runner import ConcurrentRunner, RunResult, SerialRunner, TurnTiming
    _HAS_ACTORS = True
except Exception as _actor_exc:    # noqa: BLE001 — broad on purpose
    # On the production deploy host (s1) the bench-only optional deps
    # (e.g. matplotlib if we add it later) may be missing, but the
    # event bus + events should still be importable so robot_brain.py
    # can publish without the actor stubs being available.
    #
    # We catch broad Exception (not just ImportError) because an actor
    # module might raise at import time for a non-ImportError reason
    # (RuntimeError from an optional CUDA dep, ValueError from a config
    # parse, etc). Any such failure must not block the bus + events
    # imports below.
    import sys as _sys
    print(
        f"  [orchestrator] actor modules unavailable, "
        f"bus + events still usable: {_actor_exc!r}",
        file=_sys.stderr, flush=True,
    )
    Dialog = DialogConfig = DialogStats = None  # type: ignore
    Motion = MotionConfig = MotionStats = None  # type: ignore
    Perception = PerceptionConfig = PerceptionStats = None  # type: ignore
    ConcurrentRunner = RunResult = SerialRunner = TurnTiming = None  # type: ignore
    _HAS_ACTORS = False
from .event_bus import EventBus, DropPolicy
from .events import (
    Event,
    FaceSeen,
    FaceLost,
    HandGesture,
    UserSpeechStarted,
    UserSpeechPartial,
    UserSpeechFinal,
    SceneDescribed,
    DialogThinking,
    DialogSpeechChunk,
    DialogSpeechFinal,
    DialogTool,
    AudioSpeakStarted,
    AudioSpeakEnded,
    MotionDone,
)

__all__ = [
    "ConcurrentRunner",
    "Dialog",
    "DialogConfig",
    "DialogStats",
    "DropPolicy",
    "EventBus",
    "Motion",
    "MotionConfig",
    "MotionStats",
    "Perception",
    "PerceptionConfig",
    "PerceptionStats",
    "RunResult",
    "SerialRunner",
    "TurnTiming",
    "Event",
    "FaceSeen",
    "FaceLost",
    "HandGesture",
    "UserSpeechStarted",
    "UserSpeechPartial",
    "UserSpeechFinal",
    "SceneDescribed",
    "DialogThinking",
    "DialogSpeechChunk",
    "DialogSpeechFinal",
    "DialogTool",
    "AudioSpeakStarted",
    "AudioSpeakEnded",
    "MotionDone",
]
