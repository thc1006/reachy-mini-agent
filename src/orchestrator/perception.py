"""Perception actor.

Owns (in production): camera frames, mic samples, face detection,
hand detection, VAD, streaming STT, vision caption scheduler.

This POC version simulates those workloads with timed sleeps so the
benchmark can compare the serial baseline against the event-driven
concurrent design with controllable latencies. Real STT/camera hooks
land in a follow-up PR.

The actor is a passive event-handler plus a small set of stimuli the
benchmark can fire (``simulate_user_speech``, ``set_face_present``).
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from .event_bus import EventBus
from .events import (
    AudioSpeakEnded,
    AudioSpeakStarted,
    DialogThinking,
    FaceLost,
    FaceSeen,
    SceneDescribed,
    UserSpeechFinal,
    UserSpeechPartial,
    UserSpeechStarted,
)


@dataclass
class PerceptionConfig:
    """Latencies, all in milliseconds. Defaults chosen from real
    production logs (see memory/project_v916_session_complete and
    project_phase_a_b_bench)."""
    # Batch STT (serial baseline): runs after user stops + silence-wait.
    silence_wait_ms: float = 1400.0
    stt_batch_ms: float = 200.0

    # Streaming STT (concurrent target): partial transcripts at cadence;
    # final emitted shortly after user stops (VAD endpointing).
    #
    # NB: 150ms is the optimistic end of WhisperLiveKit endpointing. Real
    # streaming STT endpointing typically delivers final in 200-500ms.
    # Production-grade comparison should bench at 350ms.
    stt_partial_cadence_ms: float = 400.0
    stt_final_after_user_stop_ms: float = 150.0

    # Vision caption: triggered either on-demand (concurrent design) or
    # periodically by a background worker (mirrors production ``vision_worker``
    # at 30s wall-clock). Modes are exclusive — pick one.
    vision_caption_ms: float = 1000.0
    enable_on_demand_vision: bool = False
    periodic_vision_interval_ms: float = 0.0   # 0 = disabled, e.g. 30000 = every 30s

    # Mic suppression window after TTS ends, to flush speaker → mic echo.
    mic_drain_after_speak_ms: float = 400.0

    # Face tracking publish rate (Hz). Only used by ``set_face_present``.
    face_publish_hz: float = 20.0


@dataclass
class PerceptionStats:
    """Counters the bench inspects after a run."""
    partials_emitted: int = 0
    partials_dropped_during_speak: int = 0  # would-be partials inside speak window
    mic_blocked_ms_total: float = 0.0       # serial only
    speak_windows: list = field(default_factory=list)  # (start, end+drain)


class Perception:
    """Simulated perception subsystem.

    Pubishes:
      - face.seen / face.lost
      - user.speech.started / partial / final
      - scene.described

    Subscribes:
      - audio.speak.started / ended (to know when mic should be ignored)
      - dialog.thinking (to trigger a vision caption)

    The benchmark drives stimuli via :py:meth:`simulate_user_speech`
    and the static face state.
    """

    def __init__(
        self,
        bus: EventBus,
        cfg: Optional[PerceptionConfig] = None,
        *,
        streaming_stt: bool = False,
        gate_mic_during_speak: bool = True,
    ) -> None:
        self.bus = bus
        self.cfg = cfg or PerceptionConfig()
        self.streaming_stt = streaming_stt
        self.gate_mic_during_speak = gate_mic_during_speak
        self.stats = PerceptionStats()

        self._speaking = threading.Event()
        self._speak_started_at: Optional[float] = None
        self._speak_drain_until: float = 0.0

        bus.subscribe(
            (AudioSpeakStarted.topic, AudioSpeakEnded.topic),
            self._on_audio_event,
            name="perception.audio",
            queue_size=8,
        )
        if self.cfg.enable_on_demand_vision:
            bus.subscribe(
                DialogThinking.topic,
                self._on_dialog_thinking,
                name="perception.vision_trigger",
                queue_size=4,
            )

        # Background periodic vision worker (mirrors production behaviour).
        # Only active if interval > 0.
        self._periodic_stop = threading.Event()
        self._periodic_thread: Optional[threading.Thread] = None
        if self.cfg.periodic_vision_interval_ms > 0:
            self._periodic_thread = threading.Thread(
                target=self._periodic_vision_loop,
                name="perception-vision-periodic",
                daemon=True,
            )
            self._periodic_thread.start()

    # ------------------------------------------------------------------
    # event handlers
    # ------------------------------------------------------------------

    def _on_audio_event(self, ev) -> None:
        if isinstance(ev, AudioSpeakStarted):
            if not self._speaking.is_set():
                self._speak_started_at = time.perf_counter()
                self._speaking.set()
        elif isinstance(ev, AudioSpeakEnded) and ev.last:
            now = time.perf_counter()
            drain = self.cfg.mic_drain_after_speak_ms / 1000.0
            self._speak_drain_until = now + drain
            if self.gate_mic_during_speak and self._speak_started_at is not None:
                window = (self._speak_started_at, self._speak_drain_until)
                self.stats.speak_windows.append(window)
                self.stats.mic_blocked_ms_total += (window[1] - window[0]) * 1000.0
            self._speaking.clear()
            self._speak_started_at = None

    def _on_dialog_thinking(self, ev: DialogThinking) -> None:
        # On-demand vision caption — replaces the 30-second wall-clock
        # interval from the baseline. Only wired up when
        # ``enable_on_demand_vision=True``.
        def _run():
            try:
                time.sleep(self.cfg.vision_caption_ms / 1000.0)
                self.bus.publish(SceneDescribed(
                    text="(simulated) the user is sitting in front of a laptop"
                ))
            except Exception:
                # The thread can wake after bus.stop() — publish would
                # raise. Swallow: missing one caption during shutdown is
                # harmless, and a noisy stack trace would obscure the
                # real shutdown sequence.
                pass
        threading.Thread(target=_run, daemon=True,
                         name="perception-vision-ondemand").start()

    def _periodic_vision_loop(self) -> None:
        """Mirrors production ``vision_worker``: every N ms, fire one
        vision-caption call and publish ``scene.described``."""
        interval_s = self.cfg.periodic_vision_interval_ms / 1000.0
        caption_s = self.cfg.vision_caption_ms / 1000.0
        while not self._periodic_stop.is_set():
            if self._periodic_stop.wait(timeout=interval_s):
                return
            time.sleep(caption_s)
            if self._periodic_stop.is_set():
                return
            self.bus.publish(SceneDescribed(
                text="(simulated/periodic) scene snapshot"
            ))

    def stop_periodic_vision(self) -> None:
        self._periodic_stop.set()
        if self._periodic_thread is not None:
            self._periodic_thread.join(timeout=2.0)

    # ------------------------------------------------------------------
    # public stimuli (driven by the benchmark)
    # ------------------------------------------------------------------

    def simulate_user_speech(self, utterance_text: str, duration_ms: float) -> None:
        """Blocks for the duration of the utterance and emits partial/final.

        Behaviour differs by ``streaming_stt`` flag:

        * ``streaming_stt=False`` (serial baseline):
          - emit ``user.speech.started`` immediately
          - sleep ``duration_ms`` (user talking)
          - sleep ``silence_wait_ms`` (waiting for VAD-batch silence)
          - sleep ``stt_batch_ms`` (Whisper batch transcribe)
          - emit ``user.speech.final``

        * ``streaming_stt=True`` (concurrent target):
          - emit ``user.speech.started`` immediately
          - emit ``user.speech.partial`` every ``stt_partial_cadence_ms``
          - sleep ``duration_ms``
          - sleep ``stt_final_after_user_stop_ms`` (endpointing)
          - emit ``user.speech.final``

        If a TTS speak window is active and ``gate_mic_during_speak`` is
        set, partial emissions inside that window are *dropped* (counted
        in stats). The user is assumed not to actually talk during this
        window in the serial baseline. The benchmark schedules around
        speak windows, so this counter is a safety net.
        """
        words = utterance_text.split()
        n_partials = max(1, int(duration_ms / self.cfg.stt_partial_cadence_ms))

        # If the mic is gated (serial baseline) and we're inside a
        # current speak window, block until it ends + drain. This is the
        # honest model of the production behaviour.
        if self.gate_mic_during_speak:
            self._wait_for_mic_open()

        self.bus.publish(UserSpeechStarted())

        if self.streaming_stt:
            partial_dt = self.cfg.stt_partial_cadence_ms / 1000.0
            user_done_at = time.perf_counter() + duration_ms / 1000.0
            for i in range(1, n_partials + 1):
                deadline = time.perf_counter() + partial_dt
                _sleep_until(min(deadline, user_done_at))
                if self._inside_speak_window():
                    self.stats.partials_dropped_during_speak += 1
                    continue
                text_so_far = " ".join(words[: max(1, int(len(words) * i / n_partials))])
                self.bus.publish(UserSpeechPartial(text=text_so_far))
                self.stats.partials_emitted += 1
                if time.perf_counter() >= user_done_at:
                    break
            _sleep_until(user_done_at)
            time.sleep(self.cfg.stt_final_after_user_stop_ms / 1000.0)
            self.bus.publish(UserSpeechFinal(
                text=utterance_text,
                audio_duration_s=duration_ms / 1000.0,
            ))
        else:
            time.sleep(duration_ms / 1000.0)
            # silence-detection wait (real code: SILENCE_DURATION inside
            # _record_via_robot_mic, between last-speech-energy and exit)
            time.sleep(self.cfg.silence_wait_ms / 1000.0)
            # batch Whisper
            time.sleep(self.cfg.stt_batch_ms / 1000.0)
            self.bus.publish(UserSpeechFinal(
                text=utterance_text,
                audio_duration_s=duration_ms / 1000.0,
            ))

    def set_face_present(self, present: bool) -> None:
        """Fire one face.seen or face.lost (used by bench for context)."""
        if present:
            self.bus.publish(FaceSeen(dx=0.05, dy=-0.02, conf=0.92))
        else:
            self.bus.publish(FaceLost(last_seen_ts=time.time()))

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _inside_speak_window(self) -> bool:
        if self._speaking.is_set():
            return True
        return time.perf_counter() < self._speak_drain_until

    def _wait_for_mic_open(self) -> None:
        while self._inside_speak_window():
            time.sleep(0.005)


def _sleep_until(deadline: float) -> None:
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 0.01))
