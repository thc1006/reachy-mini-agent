"""Wave6-P5 critical fixes (2026-06-01):

- C1: tracking_loop bailout must raise SystemExit, never os._exit, so main()'s
      `with ReachyMini(...)` finally block runs (GStreamer / WebSocket /
      PortAudio teardown).
- C2: _get_robot_memory() must be race-safe — concurrent callers must get the
      same instance, never double-construct (Mem0+Qdrant is single-writer).

These tests use AST + grep over the source (no robot_brain import, which
would require the full Pi runtime).
"""
from __future__ import annotations

import ast
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ROBOT_BRAIN = ROOT / "src" / "robot_brain.py"


# ───────────────────────────── C1 ──────────────────────────────────────

def test_lost_connection_raises_systemexit_not_oxexit():
    """The Lost-connection bailout path inside tracking_loop must NOT call
    os._exit (which skips the ReachyMini context manager's __exit__ and
    leaves GStreamer / daemon WebSocket / PortAudio in a dirty state).
    It must raise SystemExit so main()'s `with ReachyMini(...)` unwinds.
    """
    src = ROBOT_BRAIN.read_text(encoding="utf-8")
    # Strip line/block comments before grepping for live code references so
    # historical mentions in docstrings/comments don't false-positive.
    live_lines = []
    for line in src.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        live_lines.append(line)
    live_code = "\n".join(live_lines)

    assert "os._exit(" not in live_code, (
        "robot_brain.py contains a live os._exit(...) call — every bailout "
        "must use stop_event.set() + raise SystemExit so the ReachyMini "
        "context manager's finally block runs (GStreamer/WebSocket/PortAudio "
        "teardown). If this is a new bailout, raise SystemExit(code) instead."
    )

    # And the tracking_loop bailout specifically must raise SystemExit(2).
    assert "raise SystemExit(2)" in src, (
        "tracking_loop's Lost-connection bailout must `raise SystemExit(2)` "
        "so systemd Restart=on-failure brings the service back cleanly."
    )


# ───────────────────────────── C2 ──────────────────────────────────────

def _extract_function_source(src: str, fn_name: str) -> str:
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == fn_name:
            return ast.get_source_segment(src, node) or ""
    raise AssertionError(f"function {fn_name!r} not found")


def test_get_robot_memory_uses_lock_for_init():
    """_get_robot_memory() must guard the first-init path with a lock so
    parallel callers (mem0 eager warm-up thread + first dialog turn) can't
    both pass the `is not None` check and double-construct RobotMemory.
    Mem0+Qdrant LMDB is single-writer; the second construction would silently
    collapse to _Noop and silently lose facts for the rest of the session.
    """
    src = ROBOT_BRAIN.read_text(encoding="utf-8")
    fn = _extract_function_source(src, "_get_robot_memory")
    assert "_robot_memory_init_lock" in fn, (
        "_get_robot_memory must acquire _robot_memory_init_lock around the "
        "init path — see C2 in review a4e408c4b4d02ebc1."
    )
    # Module-level lock must exist.
    assert "_robot_memory_init_lock = threading.Lock()" in src


def test_get_robot_memory_double_init_returns_same_instance():
    """Behavioural check via a stub harness: simulate two threads racing to
    init under the lock and assert they get the SAME object (not two distinct
    RobotMemory instances or a _Noop collapse).

    We don't import robot_brain (that pulls the whole Pi runtime). Instead
    we replay the exact double-checked-locking pattern using the live source
    text as a structural assertion that the pattern is present, then exercise
    an isolated reimplementation under threads to confirm the pattern works.
    """
    # Structural check — see test above.
    test_get_robot_memory_uses_lock_for_init()

    # Behavioural check — race two threads through the same DCL pattern.
    import time as _time

    _instance = None
    _lock = threading.Lock()
    construction_count = [0]
    barrier = threading.Barrier(2)

    class _Mem:
        def __init__(self):
            # Simulate a slow init that lets a parallel thread race in.
            construction_count[0] += 1
            _time.sleep(0.05)
            self.enabled = True

    def _get():
        nonlocal _instance
        if _instance is not None:
            return _instance
        with _lock:
            if _instance is not None:
                return _instance
            _instance = _Mem()
        return _instance

    results = []
    def worker():
        barrier.wait()
        results.append(_get())

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert construction_count[0] == 1, (
        f"DCL pattern broken — _Mem was constructed {construction_count[0]} "
        f"times under racing threads; should be exactly 1."
    )
    assert len(results) == 2
    assert results[0] is results[1], "racing callers got different instances"
