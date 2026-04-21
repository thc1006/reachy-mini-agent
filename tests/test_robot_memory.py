"""T2 TDD: robot_memory module wraps Mem0 with:
  - local-only backends (Ollama LLM, Ollama bge-m3 embedder, embedded Qdrant)
  - simple `add_turn(user_text, bot_text)` / `search(query, limit)` API
  - silent no-op if Mem0/Ollama unavailable (feature can be disabled)
  - thread-safe (called from LLM worker threads)

All heavy tests run in subprocess to avoid Whisper/CUDA load conflicts with
other test files.
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from tests._helpers import PY, ROOT, ollama_running

pytestmark = pytest.mark.skipif(
    not ollama_running(),
    reason="Ollama not reachable — skipping Mem0 e2e tests",
)


def _run(script: str, env_extra: dict | None = None, timeout: int = 180):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    if env_extra:
        env.update(env_extra)
    return subprocess.run([PY, "-c", script], cwd=str(ROOT), env=env,
                          capture_output=True, text=True, timeout=timeout)


class TestRobotMemoryModule:
    def test_import(self):
        r = _run("from robot_memory import RobotMemory; print('OK')")
        assert "OK" in r.stdout, f"{r.stderr}"

    def test_disabled_when_env_off(self):
        """ROBOT_MEMORY=0 → all ops are silent no-op."""
        r = _run(
            "import robot_memory as m; x = m.RobotMemory(); "
            "x.add_turn('hi','hello'); "
            "res = x.search('hi'); "
            "print(f'ENABLED={x.enabled}'); print(f'RES_LEN={len(res)}')",
            env_extra={"ROBOT_MEMORY": "0"},
        )
        assert "ENABLED=False" in r.stdout, f"{r.stderr}"
        assert "RES_LEN=0" in r.stdout

    def test_search_format(self):
        """When enabled, search() must return list[str] of relevant memory texts."""
        tmp = tempfile.mkdtemp(prefix="mem0_fmt_")
        r = _run(
            f"""
import robot_memory as m
mem = m.RobotMemory(qdrant_path={tmp!r}, user_id='fmt_test')
if not mem.enabled:
    print('SKIP')
else:
    mem.add_turn('My name is Hctsai and I love tofu', 'Nice to meet you')
    mem.flush(timeout=60)
    out = mem.search('user food preference', limit=3)
    assert isinstance(out, list), f'expected list got {{type(out)}}'
    for x in out:
        assert isinstance(x, str), f'items must be str got {{type(x)}}'
    print(f'N={{len(out)}}')
    for i, x in enumerate(out):
        print(f'[{{i}}] {{x[:80]}}')
""",
            timeout=240,
        )
        if "SKIP" in r.stdout:
            pytest.skip("Mem0 / Ollama not available")
        assert "N=" in r.stdout, f"stderr: {r.stderr[-600:]}"


class TestAddTurnFactExtraction:
    """Mem0 should automatically distill facts from raw dialog text."""
    def test_name_extracted(self):
        tmp = tempfile.mkdtemp(prefix="mem0_name_")
        r = _run(
            f"""
import robot_memory as m
mem = m.RobotMemory(qdrant_path={tmp!r}, user_id='name_test')
if not mem.enabled:
    print('SKIP')
else:
    mem.add_turn('Hi there! My name is Hctsai.', 'Nice to meet you, Hctsai!')
    mem.flush(timeout=60)       # wait for async fact extraction to finish
    out = mem.search('what is the user name', limit=5)
    joined = ' '.join(out).lower()
    print('HIT' if 'hctsai' in joined else 'MISS')
    print(f'DEBUG: {{out}}')
""",
            timeout=240,
        )
        if "SKIP" in r.stdout:
            pytest.skip("Mem0 not available")
        assert "HIT" in r.stdout, f"name not extracted: {r.stdout} | stderr: {r.stderr[-400:]}"


class TestThreadSafety:
    def test_concurrent_adds_do_not_crash(self):
        tmp = tempfile.mkdtemp(prefix="mem0_thread_")
        r = _run(
            f"""
import robot_memory as m, threading
mem = m.RobotMemory(qdrant_path={tmp!r}, user_id='thread_test')
if not mem.enabled:
    print('SKIP')
else:
    errs = []
    def worker(i):
        try:
            mem.add_turn(f'I like fruit {{i}}', f'Cool, fruit {{i}} is nice.')
        except Exception as e:
            errs.append(e)
    ts = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in ts: t.start()
    for t in ts: t.join(timeout=120)
    print(f'ERRS={{len(errs)}}')
""",
            timeout=240,
        )
        if "SKIP" in r.stdout:
            pytest.skip("Mem0 not available")
        assert "ERRS=0" in r.stdout, f"{r.stdout} {r.stderr[-400:]}"


class TestGracefulDegradation:
    def test_bad_ollama_url_does_not_crash_init(self):
        r = _run(
            "import robot_memory as m; "
            "x = m.RobotMemory(ollama_base_url='http://127.0.0.1:1'); "
            "print(f'ENABLED={x.enabled}')",
            timeout=30,
        )
        # Either enabled=False (gracefully degraded) or a clean error message
        # — must NOT hang or raise
        assert "ENABLED" in r.stdout, f"init crashed: {r.stderr[-500:]}"
