"""Reachy Mini B2 multitask orchestrator.

Three actors (perception, dialog, motion) communicate over an event bus.
This package is a POC: actors run *simulated* workloads with realistic
latencies drawn from the production logs. The point is to compare a
serial baseline against the event-driven design head-to-head; hooking
the real STT/LLM/TTS is a follow-up PR.

See ``docs/multitask-arch.md`` for the design.
"""

from .dialog import Dialog, DialogConfig, DialogStats
from .event_bus import EventBus, DropPolicy
from .motion import Motion, MotionConfig, MotionStats
from .perception import Perception, PerceptionConfig, PerceptionStats
from .runner import ConcurrentRunner, RunResult, SerialRunner, TurnTiming
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
