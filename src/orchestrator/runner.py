"""POC wiring: assemble the bus + three actors, drive turns, collect stats.

Two flavours:

* :class:`SerialRunner` — single-threaded, mirrors ``do_conversation`` in
  ``robot_brain.py``. No event bus is used; we still publish events for
  observability but everything runs inline.
* :class:`ConcurrentRunner` — wires the actors onto an :class:`EventBus`
  and uses the streaming-STT path. Multiple turns run sequentially (one
  conversation) but within a turn STT → LLM → TTS overlap properly.

The benchmark (``bench_multitask.py``) instantiates both, runs the same
synthetic turn N times against each, and reports the percentile
latencies + drop counters.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

from .dialog import Dialog, DialogConfig, DialogStats
from .event_bus import EventBus
from .events import (
    AudioSpeakEnded,
    AudioSpeakStarted,
    DialogSpeechFinal,
    UserSpeechFinal,
)
from .motion import Motion, MotionConfig, MotionStats
from .perception import Perception, PerceptionConfig, PerceptionStats


@dataclass
class TurnTiming:
    """Per-turn timing in milliseconds, all measured from the
    end-of-user-utterance (the moment the user actually stops talking)."""
    ttfb_audio_ms: float = 0.0
    turn_total_ms: float = 0.0
    mic_blocked_ms: float = 0.0
    scene_age_at_llm_ms: float = 0.0
    prefill_warm: bool = False


@dataclass
class RunResult:
    label: str
    timings: List[TurnTiming] = field(default_factory=list)
    perception_stats: Optional[PerceptionStats] = None
    dialog_stats: Optional[DialogStats] = None
    motion_stats: Optional[MotionStats] = None
    bus_stats: Optional[list] = None


# ---------------------------------------------------------------------------
# Concurrent runner — uses the event bus, mirrors the target design.
# ---------------------------------------------------------------------------

class ConcurrentRunner:
    def __init__(
        self,
        *,
        perception_cfg: Optional[PerceptionConfig] = None,
        dialog_cfg: Optional[DialogConfig] = None,
        motion_cfg: Optional[MotionConfig] = None,
        streaming_stt: bool = True,
        gate_mic_during_speak: bool = False,
    ) -> None:
        self.bus = EventBus()
        # Concurrent path: opt INTO on-demand vision (the architectural
        # win) and disable the periodic worker by default (caller may
        # override via ``perception_cfg.periodic_vision_interval_ms``).
        # If the caller doesn't supply a config at all, build one with
        # on-demand vision enabled — otherwise Perception's own default
        # would land us at enable_on_demand_vision=False, silently
        # disabling the feature this runner is named after.
        if perception_cfg is None:
            cfg = PerceptionConfig(enable_on_demand_vision=True,
                                   periodic_vision_interval_ms=0.0)
        elif not perception_cfg.enable_on_demand_vision:
            cfg = PerceptionConfig(**{
                **perception_cfg.__dict__, "enable_on_demand_vision": True,
            })
        else:
            cfg = perception_cfg
        self.perception = Perception(
            self.bus, cfg,
            streaming_stt=streaming_stt,
            gate_mic_during_speak=gate_mic_during_speak,
        )
        self.dialog = Dialog(self.bus, dialog_cfg)
        self.motion = Motion(self.bus, motion_cfg)
        self._final_event = None
        self._first_audio_at: Optional[float] = None
        self._last_audio_at: Optional[float] = None

        self.bus.subscribe(
            DialogSpeechFinal.topic,
            self._on_final,
            name="runner.dialog_final",
            queue_size=2,
        )
        self.bus.subscribe(
            AudioSpeakStarted.topic,
            self._on_audio_start,
            name="runner.audio_start",
            queue_size=8,
        )
        self.bus.subscribe(
            AudioSpeakEnded.topic,
            self._on_audio_end,
            name="runner.audio_end",
            queue_size=8,
        )

    # ------------------------------------------------------------------

    def start(self) -> None:
        self.bus.start()

    def stop(self) -> None:
        self.perception.stop_periodic_vision()
        self.dialog.close()
        self.bus.stop(timeout=2.0)

    # ------------------------------------------------------------------

    def _on_audio_start(self, ev: AudioSpeakStarted) -> None:
        if self._first_audio_at is None:
            self._first_audio_at = time.perf_counter()

    def _on_audio_end(self, ev: AudioSpeakEnded) -> None:
        self._last_audio_at = time.perf_counter()

    def _on_final(self, ev: DialogSpeechFinal) -> None:
        self._final_event = ev

    # ------------------------------------------------------------------

    def run_turn(self, utterance: str, duration_ms: float) -> TurnTiming:
        # reset per-turn state
        self._final_event = None
        self._first_audio_at = None
        self._last_audio_at = None
        # snapshot stats so we can diff them at end
        snap_mic = self.perception.stats.mic_blocked_ms_total
        snap_warm = self.dialog.stats.prefill_warm

        # The user simulator blocks for the talking duration + endpoint
        # latency. We mark "user done" at the end of that call so the
        # measured TTFB excludes the user's own speaking time.
        # Production note: in ``_record_via_robot_mic`` the user-talking
        # window is the time between first energy-above-threshold and
        # last energy-above-threshold; ``simulate_user_speech`` collapses
        # the same into ``duration_ms``.
        t_user_start = time.perf_counter()
        self.perception.simulate_user_speech(utterance, duration_ms)
        # For the *concurrent* (streaming) path, the user-end-of-speech
        # moment is when the talking-duration elapses, regardless of how
        # long endpointing/STT takes after that.
        t_user_done = t_user_start + duration_ms / 1000.0

        # Wait for the dialog turn to fully finish.
        deadline = time.perf_counter() + 30.0
        while self._final_event is None and time.perf_counter() < deadline:
            time.sleep(0.005)

        if self._first_audio_at is None or self._last_audio_at is None:
            return TurnTiming(
                ttfb_audio_ms=float("inf"),
                turn_total_ms=float("inf"),
                mic_blocked_ms=0.0,
                scene_age_at_llm_ms=self.dialog.stats.last_scene_age_ms_at_llm_start,
                prefill_warm=self.dialog.stats.prefill_warm and not snap_warm,
            )
        return TurnTiming(
            ttfb_audio_ms=(self._first_audio_at - t_user_done) * 1000.0,
            turn_total_ms=(self._last_audio_at - t_user_done) * 1000.0,
            mic_blocked_ms=(
                self.perception.stats.mic_blocked_ms_total - snap_mic
            ),
            scene_age_at_llm_ms=self.dialog.stats.last_scene_age_ms_at_llm_start,
            prefill_warm=self.dialog.stats.prefill_warm and not snap_warm,
        )

    def snapshot(self, label: str, timings: List[TurnTiming]) -> RunResult:
        return RunResult(
            label=label,
            timings=timings,
            perception_stats=self.perception.stats,
            dialog_stats=self.dialog.stats,
            motion_stats=self.motion.stats,
            bus_stats=self.bus.stats(),
        )


# ---------------------------------------------------------------------------
# Serial runner — single-threaded, no overlap. Mirrors do_conversation().
# ---------------------------------------------------------------------------

class SerialRunner:
    """Closer-to-the-baseline path: same actor classes, but configured
    so that perception uses batch STT + mic gating, and there is no
    early prefill. This is what ``do_conversation`` looks like today.

    The event bus is still used so that the comparison is *fair* — both
    paths pay the same bus overhead. The difference is in **what the
    perception actor does during user-talking time** and **whether the
    mic gating window blocks the next utterance.**
    """

    def __init__(
        self,
        *,
        perception_cfg: Optional[PerceptionConfig] = None,
        dialog_cfg: Optional[DialogConfig] = None,
        motion_cfg: Optional[MotionConfig] = None,
    ) -> None:
        self.bus = EventBus()
        # Serial baseline: no on-demand vision, only the periodic worker
        # (matches production ``vision_worker`` at 30s). If no config is
        # supplied, build one with periodic vision enabled — otherwise
        # the baseline would silently run without any vision at all and
        # bench's scene-age comparison would be meaningless.
        if perception_cfg is None:
            cfg = PerceptionConfig(enable_on_demand_vision=False,
                                   periodic_vision_interval_ms=30000.0)
        elif perception_cfg.enable_on_demand_vision:
            cfg = PerceptionConfig(**{
                **perception_cfg.__dict__, "enable_on_demand_vision": False,
            })
        else:
            cfg = perception_cfg
        self.perception = Perception(
            self.bus, cfg,
            streaming_stt=False,
            gate_mic_during_speak=True,
        )
        self.dialog = Dialog(self.bus, dialog_cfg)
        self.motion = Motion(self.bus, motion_cfg)
        self._final_event = None
        self._first_audio_at: Optional[float] = None
        self._last_audio_at: Optional[float] = None

        self.bus.subscribe(
            DialogSpeechFinal.topic, self._on_final,
            name="runner.dialog_final", queue_size=2,
        )
        self.bus.subscribe(
            AudioSpeakStarted.topic, self._on_audio_start,
            name="runner.audio_start", queue_size=8,
        )
        self.bus.subscribe(
            AudioSpeakEnded.topic, self._on_audio_end,
            name="runner.audio_end", queue_size=8,
        )

    def start(self) -> None:
        self.bus.start()

    def stop(self) -> None:
        self.perception.stop_periodic_vision()
        self.dialog.close()
        self.bus.stop(timeout=2.0)

    def _on_audio_start(self, ev) -> None:
        if self._first_audio_at is None:
            self._first_audio_at = time.perf_counter()

    def _on_audio_end(self, ev) -> None:
        self._last_audio_at = time.perf_counter()

    def _on_final(self, ev) -> None:
        self._final_event = ev

    def run_turn(self, utterance: str, duration_ms: float) -> TurnTiming:
        self._final_event = None
        self._first_audio_at = None
        self._last_audio_at = None
        snap_mic = self.perception.stats.mic_blocked_ms_total

        t_user_start = time.perf_counter()
        self.perception.simulate_user_speech(utterance, duration_ms)
        # In the serial path, after the user stops talking, perception
        # still waits SILENCE_DURATION + STT_BATCH before emitting
        # ``user.speech.final``. The "user-done" reference is still the
        # moment the talking ended — those waits are *cost*.
        t_user_done = t_user_start + duration_ms / 1000.0

        deadline = time.perf_counter() + 30.0
        while self._final_event is None and time.perf_counter() < deadline:
            time.sleep(0.005)

        if self._first_audio_at is None or self._last_audio_at is None:
            return TurnTiming(
                ttfb_audio_ms=float("inf"),
                turn_total_ms=float("inf"),
                mic_blocked_ms=0.0,
                scene_age_at_llm_ms=self.dialog.stats.last_scene_age_ms_at_llm_start,
                prefill_warm=False,
            )
        return TurnTiming(
            ttfb_audio_ms=(self._first_audio_at - t_user_done) * 1000.0,
            turn_total_ms=(self._last_audio_at - t_user_done) * 1000.0,
            mic_blocked_ms=(
                self.perception.stats.mic_blocked_ms_total - snap_mic
            ),
            scene_age_at_llm_ms=self.dialog.stats.last_scene_age_ms_at_llm_start,
            prefill_warm=False,
        )

    def snapshot(self, label: str, timings: List[TurnTiming]) -> RunResult:
        return RunResult(
            label=label,
            timings=timings,
            perception_stats=self.perception.stats,
            dialog_stats=self.dialog.stats,
            motion_stats=self.motion.stats,
            bus_stats=self.bus.stats(),
        )
