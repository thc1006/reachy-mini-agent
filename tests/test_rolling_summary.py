"""Track D TDD: rolling summary in RobotMemory.

Design:
  - Every N turns (default 20), trigger async summary regeneration.
  - Summary is a single 200-300 word paragraph covering turns
    [0 .. len(history)-keep], where `keep` (default 20) leaves recent turns to
    the FIFO window.
  - Stored as plain text file `conversation_summary.txt` next to the jsonl log.
  - `get_rolling_summary() -> str` returns the current summary (empty if none).
  - All LLM work runs in the executor — never blocks add_turn().
  - Safe when RobotMemory is disabled (all no-ops).
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PY   = str(ROOT / ".venv" / "bin" / "python")


def _run(script: str, env_extra=None, timeout=240):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    if env_extra: env.update(env_extra)
    return subprocess.run([PY, "-c", script], cwd=str(ROOT), env=env,
                          capture_output=True, text=True, timeout=timeout)


class TestSummaryAPI:
    def test_get_summary_returns_empty_when_no_summary(self):
        r = _run(
            "import robot_memory as m; "
            "x = m.RobotMemory(summary_path='/tmp/does_not_exist_summary.txt'); "
            "print(f'SUM={x.get_rolling_summary()!r}')"
        )
        assert "SUM=''" in r.stdout, f"{r.stderr}"

    def test_get_summary_returns_file_contents_if_exists(self):
        tmp = tempfile.mktemp(suffix="_sum.txt", dir="/tmp")
        Path(tmp).write_text("The user likes tofu. Previously asked about weather.\n")
        r = _run(
            f"import robot_memory as m; "
            f"x = m.RobotMemory(summary_path={tmp!r}); "
            f"s = x.get_rolling_summary(); "
            f"print(f'LEN={{len(s)}}'); print(f'HIT={{\"tofu\" in s}}')"
        )
        assert "HIT=True" in r.stdout, f"{r.stderr}"
        Path(tmp).unlink(missing_ok=True)

    def test_disabled_memory_returns_empty(self):
        r = _run(
            "import robot_memory as m; "
            "x = m.RobotMemory(); "
            "print(f'SUM={x.get_rolling_summary()!r}')",
            env_extra={"ROBOT_MEMORY": "0"},
        )
        assert "SUM=''" in r.stdout, r.stderr


class TestSummaryGenerationTrigger:
    def test_summary_generated_after_threshold_turns(self):
        """After N add_turn calls above threshold, summary file appears."""
        tmp_qd  = tempfile.mkdtemp(prefix="mem0_sumgen_")
        tmp_sum = tempfile.mktemp(suffix="_sum.txt", dir="/tmp")
        # use a low threshold (6) so test is fast
        script = f"""
import robot_memory as m
import tempfile, os
_log = tempfile.mktemp(suffix='_log.jsonl', dir='/tmp')
mem = m.RobotMemory(
    qdrant_path={tmp_qd!r},
    user_id='sum_gen_test',
    summary_path={tmp_sum!r},
    conversation_log_path=_log,
    summary_every=6,
    summary_keep_recent=2,
    write_own_log=True,
)
if not mem.enabled:
    print('SKIP')
else:
    # 10 turns — crosses threshold once at turn 6
    for i in range(10):
        mem.add_turn(f'User utterance {{i}} about food and weather', f'Robot reply {{i}} acknowledging.')
    mem.flush(timeout=180)   # wait for writes
    # Also block on summary task if flush doesn't cover it
    mem.flush_summary(timeout=180)
    import os
    exists = os.path.exists({tmp_sum!r})
    size = os.path.getsize({tmp_sum!r}) if exists else 0
    print(f'EXISTS={{exists}}')
    print(f'SIZE={{size}}')
    if exists:
        with open({tmp_sum!r}) as f: content = f.read()
        print(f'FIRST_200={{content[:200]!r}}')
"""
        r = _run(script, timeout=360)
        if "SKIP" in r.stdout: pytest.skip("Mem0 / Ollama not up")
        assert "EXISTS=True" in r.stdout, f"stdout={r.stdout[-400:]} err={r.stderr[-400:]}"
        # file must have meaningful content
        assert "SIZE=" in r.stdout
        size_line = [l for l in r.stdout.splitlines() if l.startswith("SIZE=")][0]
        assert int(size_line.split("=")[1]) > 40, "summary too short"


class TestSummaryContent:
    def test_summary_mentions_topics_from_dialog(self):
        tmp_qd  = tempfile.mkdtemp(prefix="mem0_sumtop_")
        tmp_sum = tempfile.mktemp(suffix="_sum.txt", dir="/tmp")
        script = f"""
import robot_memory as m
import tempfile
_log = tempfile.mktemp(suffix='_log.jsonl', dir='/tmp')
mem = m.RobotMemory(
    qdrant_path={tmp_qd!r}, user_id='sum_topic',
    summary_path={tmp_sum!r}, conversation_log_path=_log,
    summary_every=4, summary_keep_recent=1, write_own_log=True,
)
if not mem.enabled:
    print('SKIP')
else:
    # 6 distinctive turns
    turns = [
        ('I am planning a trip to Kyoto next spring.',
         'Kyoto in spring is lovely — cherry blossoms peak early April.'),
        ('I love tofu, especially silken in miso soup.',
         'Great choice, Japan has amazing tofu.'),
        ('My dog is called Momo and she is a corgi.',
         'Momo the corgi sounds adorable.'),
        ('I play violin every weekend.',
         'Do you play any particular composer?'),
        ('Mostly Bach partitas.',
         'Ambitious — the chaconne is incredible.'),
        ('Next week I also have a medical exam.',
         'Good luck with it.'),
    ]
    for u, b in turns:
        mem.add_turn(u, b)
    mem.flush(timeout=240)
    mem.flush_summary(timeout=240)
    s = mem.get_rolling_summary().lower()
    hits = sum([
        any(w in s for w in ['kyoto','japan','cherry']),
        any(w in s for w in ['tofu','miso']),
        any(w in s for w in ['momo','dog','corgi']),
        any(w in s for w in ['violin','bach','music','partita']),
    ])
    print(f'HITS={{hits}}/4')
    print(f'SUMMARY_LEN={{len(s)}}')
    print(f'SUMMARY_HEAD={{s[:300]!r}}')
"""
        r = _run(script, timeout=480)
        if "SKIP" in r.stdout: pytest.skip("Mem0 down")
        # Expect at least 3 of 4 topics to survive summarization
        hits_line = [l for l in r.stdout.splitlines() if l.startswith("HITS=")][0]
        hits = int(hits_line.split("=")[1].split("/")[0])
        assert hits >= 3, f"summary dropped too many topics: {r.stdout[-500:]}"


class TestSummaryIsAsync:
    def test_add_turn_returns_quickly_even_at_summary_trigger(self):
        tmp_qd  = tempfile.mkdtemp(prefix="mem0_async_")
        tmp_sum = tempfile.mktemp(suffix="_sum.txt", dir="/tmp")
        script = f"""
import robot_memory as m, time
import tempfile
_log = tempfile.mktemp(suffix='_log.jsonl', dir='/tmp')
mem = m.RobotMemory(
    qdrant_path={tmp_qd!r}, user_id='async_test',
    summary_path={tmp_sum!r}, conversation_log_path=_log,
    summary_every=4, summary_keep_recent=1, write_own_log=True,
)
if not mem.enabled:
    print('SKIP')
else:
    # fill 3 turns fast (no trigger)
    for i in range(3):
        mem.add_turn(f'q {{i}}', f'a {{i}}')
    # trigger turn
    t0 = time.perf_counter()
    mem.add_turn('q 4', 'a 4')   # this should SCHEDULE summary, not block on it
    dt_ms = (time.perf_counter() - t0) * 1000
    print(f'ADD_MS={{dt_ms:.0f}}')
    mem.flush_summary(timeout=240)
"""
        r = _run(script, timeout=360)
        if "SKIP" in r.stdout: pytest.skip("Mem0 down")
        ms_line = [l for l in r.stdout.splitlines() if l.startswith("ADD_MS=")][0]
        add_ms = float(ms_line.split("=")[1])
        # add_turn triggers summary async → should return under 500ms even on slow day
        assert add_ms < 800, f"add_turn blocked: {add_ms:.0f}ms"
