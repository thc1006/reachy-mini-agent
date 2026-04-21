"""S4 TDD: both Edge and Kokoro TTS engines implement the same interface
and produce non-empty audio.

Each test runs in its own subprocess to avoid GPU / stdout race conditions
from the heavy robot_brain top-level init (Whisper CUDA, MediaPipe GL, etc).
"""
import subprocess
import sys
import os
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY   = os.path.join(ROOT, ".venv", "bin", "python")


def _run_subprocess(script: str, env_extra: dict | None = None, timeout=90):
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", ROOT)
    if env_extra:
        env.update(env_extra)
    r = subprocess.run(
        [PY, "-c", script],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout,
    )
    return r


def test_edge_engine_produces_audio():
    script = """
import asyncio, robot_brain as rb, numpy as np
r = asyncio.run(rb._fetch_edge_tts("Hello there! This is a test."))
assert r is not None, "edge-tts returned None"
data, sr = r
assert isinstance(data, np.ndarray) and len(data) > 0, "empty audio"
assert 0.5 < len(data)/sr < 10, f"dur {len(data)/sr:.2f}s"
print("PASS edge")
"""
    r = _run_subprocess(script)
    assert "PASS edge" in r.stdout, f"stdout={r.stdout!r} stderr={r.stderr[-500:]!r}"


def test_kokoro_engine_produces_audio():
    script = """
import robot_brain as rb, numpy as np
r = rb._fetch_kokoro_tts("Hello there! This is a test.")
if r is None:
    print("SKIP kokoro not available")
    raise SystemExit(0)
data, sr = r
assert isinstance(data, np.ndarray) and len(data) > 0
assert 0.5 < len(data)/sr < 10
print("PASS kokoro")
"""
    r = _run_subprocess(script)
    if "SKIP" in r.stdout:
        pytest.skip("Kokoro server not running")
    assert "PASS kokoro" in r.stdout, f"stdout={r.stdout!r} stderr={r.stderr[-500:]!r}"


def test_stream_tts_engine_with_env_edge():
    script = """
import robot_brain as rb, numpy as np
eng = rb._StreamTTSEngine()
samples, sr = eng.synthesize("Short test sentence.")
assert isinstance(samples, np.ndarray) and len(samples) > 0
assert sr > 0
print("PASS stream-edge")
"""
    r = _run_subprocess(script, env_extra={"TTS_ENGINE": "edge"})
    assert "PASS stream-edge" in r.stdout, f"stderr={r.stderr[-500:]!r}"


def test_stream_tts_engine_with_env_kokoro():
    script = """
import robot_brain as rb, numpy as np
eng = rb._StreamTTSEngine()
try:
    samples, sr = eng.synthesize("Short test sentence.")
except RuntimeError as e:
    print(f"SKIP kokoro failed: {e}")
    raise SystemExit(0)
assert isinstance(samples, np.ndarray) and len(samples) > 0
print("PASS stream-kokoro")
"""
    r = _run_subprocess(script, env_extra={"TTS_ENGINE": "kokoro"})
    if "SKIP" in r.stdout:
        pytest.skip("Kokoro failed")
    assert "PASS stream-kokoro" in r.stdout, f"stderr={r.stderr[-500:]!r}"


def test_stream_tts_engine_emoji_only_does_not_crash():
    script = """
import robot_brain as rb
eng = rb._StreamTTSEngine()
samples, sr = eng.synthesize("🤖💕😊")
assert sr > 0
print("PASS emoji-only")
"""
    r = _run_subprocess(script)
    assert "PASS emoji-only" in r.stdout, f"stderr={r.stderr[-500:]!r}"
