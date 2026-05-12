"""Typed event payloads used on the orchestrator bus.

Each event carries the topic name as a class attribute so producers can
publish by passing only the dataclass instance.

Time fields are wall-clock seconds (``time.time()`` style). Use
``time.perf_counter()`` for benchmarking measurements; ``ts`` here is
informational, not for sub-millisecond bookkeeping.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar


@dataclass
class Event:
    """Base class. Subclasses override ``topic``."""
    topic: ClassVar[str] = "event"
    ts: float = 0.0


# --- perception → world -----------------------------------------------------

@dataclass
class FaceSeen(Event):
    topic: ClassVar[str] = "face.seen"
    bbox: tuple = (0.0, 0.0, 0.0, 0.0)
    dx: float = 0.0
    dy: float = 0.0
    conf: float = 0.0
    frame_ts: float = 0.0


@dataclass
class FaceLost(Event):
    topic: ClassVar[str] = "face.lost"
    last_seen_ts: float = 0.0


@dataclass
class HandGesture(Event):
    topic: ClassVar[str] = "hand.gesture"
    n_fingers: int = 0
    stable_ms: int = 0


@dataclass
class UserSpeechStarted(Event):
    """VAD detected the user began talking."""
    topic: ClassVar[str] = "user.speech.started"


@dataclass
class UserSpeechPartial(Event):
    """Streaming STT partial transcript (incremental)."""
    topic: ClassVar[str] = "user.speech.partial"
    text: str = ""


@dataclass
class UserSpeechFinal(Event):
    """User stopped talking; this is the final transcript."""
    topic: ClassVar[str] = "user.speech.final"
    text: str = ""
    audio_duration_s: float = 0.0


@dataclass
class SceneDescribed(Event):
    """One-shot vision caption."""
    topic: ClassVar[str] = "scene.described"
    text: str = ""


# --- dialog → world ---------------------------------------------------------

@dataclass
class DialogThinking(Event):
    """LLM call started (used to gate concurrent vision calls + idle motion)."""
    topic: ClassVar[str] = "dialog.thinking"
    user_text: str = ""


@dataclass
class DialogSpeechChunk(Event):
    """One sentence-sized chunk of streamed LLM speech."""
    topic: ClassVar[str] = "dialog.speech.chunk"
    text: str = ""
    idx: int = 0
    is_first: bool = False


@dataclass
class DialogSpeechFinal(Event):
    """LLM finished; full response + parsed actions."""
    topic: ClassVar[str] = "dialog.speech.final"
    text: str = ""
    actions: list = field(default_factory=list)


@dataclass
class DialogTool(Event):
    """Tool call extracted from streaming LLM output."""
    topic: ClassVar[str] = "dialog.tool"
    name: str = ""
    args: dict = field(default_factory=dict)


# --- audio output → world ---------------------------------------------------

@dataclass
class AudioSpeakStarted(Event):
    topic: ClassVar[str] = "audio.speak.started"
    chunk_idx: int = 0


@dataclass
class AudioSpeakEnded(Event):
    topic: ClassVar[str] = "audio.speak.ended"
    chunk_idx: int = 0
    last: bool = False


# --- motion → world ---------------------------------------------------------

@dataclass
class MotionDone(Event):
    topic: ClassVar[str] = "motion.done"
    action: str = ""
    duration_ms: float = 0.0
