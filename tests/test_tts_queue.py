"""S2 TDD - Part 2: TTS queue — generate concurrently, play in order.

Design:
    q = TTSQueue(engine)     # engine.synthesize(text) → (samples, sr)
    q.start(on_audio=lambda samples, sr: player.push(samples, sr))
    q.submit("First sentence.")   # gen starts immediately
    q.submit("Second sentence.")  # gen starts concurrently
    q.close()                     # wait for all done
    # on_audio callback fires in submission order, even if sentence 2 finishes first

Critical property: playback order == submission order, even with out-of-order generation.
"""
import threading
import time
import pytest

from streaming_tts import TTSQueue


class MockEngine:
    """Simulates an engine whose synth time depends on text."""
    def __init__(self, delays_by_len=None):
        self.delays_by_len = delays_by_len or {}
        self.synth_calls = []
        self._lock = threading.Lock()

    def synthesize(self, text):
        with self._lock:
            self.synth_calls.append((text, time.perf_counter()))
        delay = self.delays_by_len.get(len(text), 0.05)
        time.sleep(delay)
        # return fake "audio": a list of chars representing the text
        return list(text), 16000


class TestTTSQueueBasics:
    def test_single_submit_single_playback(self):
        engine = MockEngine()
        played = []
        q = TTSQueue(engine, max_concurrent=2)
        q.start(on_audio=lambda s, sr: played.append(("".join(s), sr)))
        q.submit("Hello.")
        q.close(timeout=5)
        assert played == [("Hello.", 16000)]

    def test_multiple_play_in_submission_order(self):
        engine = MockEngine()
        played = []
        q = TTSQueue(engine, max_concurrent=3)
        q.start(on_audio=lambda s, sr: played.append("".join(s)))
        for t in ["One.", "Two.", "Three.", "Four."]:
            q.submit(t)
        q.close(timeout=10)
        assert played == ["One.", "Two.", "Three.", "Four."]


class TestTTSQueueConcurrency:
    def test_out_of_order_gen_keeps_playback_order(self):
        # sentence 1 is slow (0.3s), sentence 2 fast (0.05s)
        # playback must still go 1 → 2
        engine = MockEngine(delays_by_len={3: 0.30, 5: 0.05})   # len 3 slow, len 5 fast
        played = []
        q = TTSQueue(engine, max_concurrent=2)
        q.start(on_audio=lambda s, sr: played.append("".join(s)))
        q.submit("Aaa")          # slow (len 3)
        q.submit("Bbbbb")        # fast (len 5)
        q.close(timeout=5)
        assert played == ["Aaa", "Bbbbb"]


class TestTTSQueueTiming:
    def test_concurrent_generation_is_faster_than_serial(self):
        """If we submit 3 sentences each taking 0.2s, total wall should be
        much less than 3×0.2=0.6s thanks to concurrency."""
        engine = MockEngine(delays_by_len={3: 0.20})
        q = TTSQueue(engine, max_concurrent=3)
        q.start(on_audio=lambda s, sr: None)
        t0 = time.perf_counter()
        for t in ["Aaa", "Bbb", "Ccc"]:
            q.submit(t)
        q.close(timeout=5)
        elapsed = time.perf_counter() - t0
        # serial would be 0.6s; parallel should be ~0.2s (+ overhead)
        assert elapsed < 0.45, f"parallelism broken: {elapsed:.2f}s (expected <0.45s)"


class TestTTSQueueRobustness:
    def test_engine_error_doesnt_crash_queue(self):
        class FlakyEngine:
            def synthesize(self, text):
                if "BOOM" in text:
                    raise RuntimeError("simulated failure")
                time.sleep(0.02)
                return list(text), 16000

        played = []
        errors = []
        q = TTSQueue(FlakyEngine(), max_concurrent=2, on_error=lambda e: errors.append(e))
        q.start(on_audio=lambda s, sr: played.append("".join(s)))
        q.submit("First.")
        q.submit("BOOM")
        q.submit("Last.")
        q.close(timeout=5)
        assert played == ["First.", "Last."], f"expected failed slot skipped, got {played}"
        assert len(errors) == 1

    def test_close_without_start_does_not_hang(self):
        q = TTSQueue(MockEngine(), max_concurrent=2)
        q.close(timeout=2)   # no start, no submit — should be instant
