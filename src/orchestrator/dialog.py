"""Dialog actor.

Owns (in production): the vLLM HTTP client, Mem0 long-term memory,
turn history, ``SYSTEM_PROMPT`` assembly, and the streaming
``SentenceChunker`` + ``TTSQueue`` pipeline.

This POC simulates the LLM as a stream of sentence-sized chunks
spaced ``llm_chunk_gen_ms`` apart, with a first-chunk TTFB of
``llm_ttfb_ms``. Tool calls are not modelled in detail; one fixed
``move_head`` tool is emitted halfway through to exercise the
motion subscription path.

The TTS path mirrors ``streaming_tts.TTSQueue``: parallel synth with
in-order playback.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import List, Optional

from .event_bus import EventBus
from .events import (
    AudioSpeakEnded,
    AudioSpeakStarted,
    DialogSpeechChunk,
    DialogSpeechFinal,
    DialogThinking,
    DialogTool,
    SceneDescribed,
    UserSpeechFinal,
    UserSpeechPartial,
)


@dataclass
class DialogConfig:
    llm_ttfb_ms: float = 350.0
    llm_chunk_count: int = 4
    llm_chunk_gen_ms: float = 120.0
    tts_synth_ms_per_chunk: float = 300.0
    tts_play_ms_per_chunk: float = 350.0
    tts_max_concurrent: int = 2
    # Emit one tool call at chunk index N (or -1 to skip).
    tool_at_chunk: int = 1
    # If True, record that the first ``user.speech.partial`` warmed
    # prefill. With vLLM prefix cache OFF in production this is mostly
    # bookkeeping; the actual TTFB save is folded into ``llm_ttfb_ms``.
    use_partial_prefill: bool = False


@dataclass
class DialogStats:
    n_turns: int = 0
    last_ttfb_audio_ms: float = 0.0
    last_full_speech_ms: float = 0.0
    prefill_warm: bool = False
    last_scene_age_ms_at_llm_start: float = 0.0


class Dialog:
    """Simulated dialog subsystem."""

    _SENTENCES = (
        "Sure, let me think about that.",
        "Based on what I see, you are at the laptop.",
        "Reachy Mini supports tool calls now.",
        "Tell me if you want a different demo.",
    )

    def __init__(
        self,
        bus: EventBus,
        cfg: Optional[DialogConfig] = None,
    ) -> None:
        self.bus = bus
        self.cfg = cfg or DialogConfig()
        self.stats = DialogStats()
        self._last_scene_ts: Optional[float] = None

        # Per-turn synth pool + in-order playback (mirrors prod TTSQueue).
        self._synth_pool = ThreadPoolExecutor(
            max_workers=self.cfg.tts_max_concurrent,
            thread_name_prefix="dialog-synth",
        )

        bus.subscribe(
            UserSpeechFinal.topic,
            self._on_user_final,
            name="dialog.final",
            queue_size=4,
        )
        bus.subscribe(
            UserSpeechPartial.topic,
            self._on_user_partial,
            name="dialog.partial",
            queue_size=16,
        )
        bus.subscribe(
            SceneDescribed.topic,
            self._on_scene,
            name="dialog.scene",
            queue_size=4,
        )

    # ------------------------------------------------------------------
    # event handlers
    # ------------------------------------------------------------------

    def _on_scene(self, ev: SceneDescribed) -> None:
        self._last_scene_ts = time.perf_counter()

    def _on_user_partial(self, ev: UserSpeechPartial) -> None:
        if not self.cfg.use_partial_prefill:
            return
        if not self.stats.prefill_warm:
            self.stats.prefill_warm = True

    def _on_user_final(self, ev: UserSpeechFinal) -> None:
        # Runs on this subscription's own thread; one turn at a time
        # matches production.
        self._run_turn(ev)

    # ------------------------------------------------------------------
    # turn execution
    # ------------------------------------------------------------------

    def _run_turn(self, user: UserSpeechFinal) -> None:
        self.stats.n_turns += 1
        t_final = time.perf_counter()
        self.bus.publish(DialogThinking(user_text=user.text))

        # Scene freshness measured at LLM-start.
        if self._last_scene_ts is None:
            self.stats.last_scene_age_ms_at_llm_start = float("inf")
        else:
            self.stats.last_scene_age_ms_at_llm_start = (
                t_final - self._last_scene_ts
            ) * 1000.0

        # ---- LLM streaming simulation -----------------------------------
        # ttfb_ms is "time until first chunk text becomes available".
        # Subsequent chunks every llm_chunk_gen_ms.
        chunk_futures: List[Future] = []

        def _produce_and_submit():
            time.sleep(self.cfg.llm_ttfb_ms / 1000.0)
            for idx in range(self.cfg.llm_chunk_count):
                if idx > 0:
                    time.sleep(self.cfg.llm_chunk_gen_ms / 1000.0)
                text = self._SENTENCES[idx % len(self._SENTENCES)]
                self.bus.publish(DialogSpeechChunk(
                    text=text, idx=idx, is_first=(idx == 0),
                ))
                # synth future (parallel up to tts_max_concurrent)
                fut = self._synth_pool.submit(self._synth_chunk, text)
                chunk_futures.append(fut)
                if idx == self.cfg.tool_at_chunk:
                    self.bus.publish(DialogTool(
                        name="move_head",
                        args={"yaw_deg": 12.0, "pitch_deg": 0.0, "duration_s": 0.8},
                    ))

        prod_thread = threading.Thread(
            target=_produce_and_submit, daemon=True, name="dialog-llm",
        )
        prod_thread.start()

        # ---- in-order playback ------------------------------------------
        # Wait for each future in submission order, play it, then move on.
        next_idx = 0
        first_audio_at: Optional[float] = None
        last_idx = self.cfg.llm_chunk_count - 1
        while next_idx <= last_idx:
            # Wait until producer has submitted the future for next_idx.
            while next_idx >= len(chunk_futures):
                if not prod_thread.is_alive():
                    break
                time.sleep(0.002)
            if next_idx >= len(chunk_futures):
                # producer exited without emitting all chunks; bail cleanly
                break
            fut = chunk_futures[next_idx]
            samples = fut.result()
            self.bus.publish(AudioSpeakStarted(chunk_idx=next_idx))
            if first_audio_at is None:
                first_audio_at = time.perf_counter()
                self.stats.last_ttfb_audio_ms = (first_audio_at - t_final) * 1000.0
            time.sleep(self.cfg.tts_play_ms_per_chunk / 1000.0)
            self.bus.publish(AudioSpeakEnded(
                chunk_idx=next_idx, last=(next_idx == last_idx),
            ))
            next_idx += 1

        prod_thread.join(timeout=2.0)
        speech = " ".join(self._SENTENCES[: self.cfg.llm_chunk_count])
        self.bus.publish(DialogSpeechFinal(text=speech, actions=[]))
        self.stats.last_full_speech_ms = (time.perf_counter() - t_final) * 1000.0

    # ------------------------------------------------------------------
    # synth (stand-in for edge/kokoro/hagen)
    # ------------------------------------------------------------------

    def _synth_chunk(self, text: str) -> bytes:
        time.sleep(self.cfg.tts_synth_ms_per_chunk / 1000.0)
        return b"audio"

    def close(self) -> None:
        self._synth_pool.shutdown(wait=False)
