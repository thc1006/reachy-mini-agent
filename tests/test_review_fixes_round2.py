"""Tests for round-2 review fixes (M3, M6, M7, C4)."""
import os
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

from tests._helpers import PY, ROOT, ollama_running
# No module-level skipif: pure-logic classes (TestCountParserQueryAware) must
# still run on CI without Ollama; subprocess-based classes below are skipped
# individually.

def _run(script, env_extra=None, timeout=120):
    env = os.environ.copy(); env["PYTHONPATH"] = str(ROOT)
    if env_extra: env.update(env_extra)
    return subprocess.run([PY, "-c", script], cwd=str(ROOT), env=env,
                          capture_output=True, text=True, timeout=timeout)


# ── M3: query-aware count parser ──────────────────────────────────────────

class TestCountParserQueryAware:
    def test_picks_number_for_query_noun(self):
        """'0 cups but 3 plates' queried as cups → 0, as plates → 3."""
        from robot_tools import _parse_count_response as p
        txt = "I see 0 cups but 3 plates in view."
        assert p(txt, query="cups")["count"] == 0
        assert p(txt, query="plates")["count"] == 3

    def test_picks_nearby_number_with_adjectives(self):
        from robot_tools import _parse_count_response as p
        # "3 red cups" — 1-3 words between int and query still matches
        assert p("I see 3 red cups.", query="cups")["count"] == 3
        assert p("There are 5 small blue books.", query="books")["count"] == 5

    def test_colon_form(self):
        from robot_tools import _parse_count_response as p
        assert p("cups: 4, plates: 9", query="cups")["count"] == 4
        assert p("cups: 4, plates: 9", query="plates")["count"] == 9

    def test_json_still_preferred(self):
        from robot_tools import _parse_count_response as p
        # JSON wins even if inline numbers are ambiguous
        assert p('{"count": 7, "note": "ignore 99"}', query="whatever")["count"] == 7

    def test_no_query_falls_back_to_first_integer(self):
        from robot_tools import _parse_count_response as p
        assert p("I see 3 cups and 7 plates", query="")["count"] == 3


# ── M6: memory search TTL cache ──────────────────────────────────────────

@pytest.mark.skipif(not ollama_running(), reason="subprocess imports robot_brain (requires Ollama)")
class TestMemorySearchCache:
    """All 3 scenarios in one subprocess — robot_brain import is heavy (CUDA +
    Mediapipe) and triggers rare init races when repeated across many subprocs."""
    def test_cache_hit_miss_and_eviction(self):
        script = """
import robot_brain as rb

# Scenario 1: same query twice → cache hit, backing search once
calls = {'n': 0}
class FakeMem:
    enabled = True
    def get_rolling_summary(self): return ''
    def search(self, query, limit=3):
        calls['n'] += 1
        return [f'fact for {query}']
rb._get_robot_memory = lambda: FakeMem()
rb._MEM_SEARCH_CACHE.clear()
ctx1 = rb._memory_context_for('what is my name')
ctx2 = rb._memory_context_for('what is my name')
print(f'S1_CALLS={calls[\"n\"]}')
print(f'S1_SAME={ctx1 == ctx2}')

# Scenario 2: different queries → miss, two backing calls
rb._MEM_SEARCH_CACHE.clear()
calls['n'] = 0
rb._memory_context_for('what is my name')
rb._memory_context_for('what do I like to eat')
print(f'S2_CALLS={calls[\"n\"]}')

# Scenario 3: push past _MEM_SEARCH_MAX → size-bounded
rb._MEM_SEARCH_CACHE.clear()
for i in range(80):
    rb._memory_context_for(f'query-{i}')
print(f'S3_CACHE_SIZE={len(rb._MEM_SEARCH_CACHE)}')
"""
        r = _run(script, timeout=90)
        if r.returncode == -11:
            pytest.skip("subprocess SIGSEGV (CUDA/mediapipe init race) — rare, retry")
        assert "S1_CALLS=1" in r.stdout, f"S1 cache miss: {r.stdout[-300:]}"
        assert "S1_SAME=True" in r.stdout
        assert "S2_CALLS=2" in r.stdout, f"S2 cache hit on different query: {r.stdout[-300:]}"
        size_line = [l for l in r.stdout.splitlines() if l.startswith("S3_CACHE_SIZE=")][0]
        assert int(size_line.split("=")[1]) <= 64, f"S3 cache not bounded: {size_line}"


# ── M7: VISION_OLLAMA_URL / VISION_MODEL runtime env ─────────────────────

@pytest.mark.skipif(not ollama_running(), reason="subprocess imports robot_tools (needs reachy_mini SDK)")
class TestVisionEndpointRuntime:
    def test_env_change_after_import_wins(self):
        """_vision_endpoint reads env at call time, not import time."""
        script = """
import os, robot_tools as rt
os.environ['OLLAMA_HOST']  = 'http://foo.invalid:9999'
os.environ['VISION_MODEL'] = 'bar:latest'
url, model = rt._vision_endpoint()
print(f'URL={url}')
print(f'MODEL={model}')
"""
        r = _run(script, timeout=20)
        assert "URL=http://foo.invalid:9999" in r.stdout, f"env not picked up: {r.stdout}"
        assert "MODEL=bar:latest" in r.stdout

    def test_monkeypatch_wins_when_no_env(self):
        """tests that monkey-patch module attributes wins if env is unset."""
        script = """
import os, robot_tools as rt
# make sure env is clean
os.environ.pop('OLLAMA_HOST', None)
os.environ.pop('VISION_MODEL', None)
rt.VISION_OLLAMA_URL = 'http://monkey:1'
rt.VISION_MODEL      = 'monkey-model'
url, model = rt._vision_endpoint()
print(f'URL={url}')
print(f'MODEL={model}')
"""
        r = _run(script, timeout=20)
        assert "URL=http://monkey:1" in r.stdout


# ── C4: flush_summary real timeout ───────────────────────────────────────

@pytest.mark.skipif(not ollama_running(), reason="subprocess imports robot_memory (Mem0 requires Ollama)")
class TestFlushSummaryTimeout:
    def test_returns_true_when_nothing_pending(self):
        script = """
import robot_memory as m
x = m.RobotMemory.__new__(m.RobotMemory)
x.__init__()
r = x.flush_summary(timeout=0.5)
print(f'RET={r}')
"""
        r = _run(script, timeout=20)
        assert "RET=True" in r.stdout

    def test_returns_false_when_work_exceeds_timeout(self):
        """If we schedule a slow job, flush_summary(timeout<duration) must return False."""
        script = """
import robot_memory as m
x = m.RobotMemory.__new__(m.RobotMemory)
x.__init__()
if not x.enabled:
    print('SKIP')
else:
    import time
    def slow(): time.sleep(3)
    fut = x._summary_executor.submit(slow)
    x._pending_summary_futures.append(fut)
    t0 = time.perf_counter()
    ret = x.flush_summary(timeout=0.5)
    dt = time.perf_counter() - t0
    print(f'RET={ret}')
    print(f'DT={dt:.2f}')
"""
        r = _run(script, timeout=30)
        if "SKIP" in r.stdout: pytest.skip("Mem0 not up")
        assert "RET=False" in r.stdout, r.stdout
        # The flush must have returned within ~1s, not waited for 3s slow work
        dt_line = [l for l in r.stdout.splitlines() if l.startswith("DT=")][0]
        dt = float(dt_line.split("=")[1])
        assert dt < 1.5, f"flush_summary ignored timeout: waited {dt:.2f}s"
