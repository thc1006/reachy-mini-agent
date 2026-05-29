"""Wave6-P4 (2026-05-29): query_vision LLM tool.

Distinct from see_what:
  * REQUIRES `question` (not optional)
  * Returns {"answer": "..."} for direct LLM synthesis (vs {"description": ...})
  * No-frame returns a canned answer instead of {"error": ...} so the dialog
    LLM can gracefully synthesize a "camera not ready" reply.

All tests are pure-Python and never hit a real VLM — _ask_vision and
_get_frame_b64 are monkey-patched at the module level (matching the pattern
in test_vision_tools.py).
"""
import json

import pytest


def test_query_vision_tool_definition_well_formed():
    """Spec must be JSON-serializable + match the OpenAI tools schema."""
    from robot_tools import TOOLS, get_tool_specs

    assert "query_vision" in TOOLS, "query_vision must be registered in TOOLS"
    spec, handler = TOOLS["query_vision"]

    # Serializable (vLLM POSTs this verbatim)
    blob = json.dumps(spec)
    assert "query_vision" in blob

    assert spec["type"] == "function"
    fn = spec["function"]
    assert fn["name"] == "query_vision"
    assert isinstance(fn.get("description"), str) and len(fn["description"]) > 30

    params = fn["parameters"]
    assert params["type"] == "object"
    assert "question" in params["properties"]
    assert params["properties"]["question"]["type"] == "string"
    assert params["required"] == ["question"], "question must be required"

    # And the spec must appear in the list shipped to vLLM
    specs = get_tool_specs()
    names = {s["function"]["name"] for s in specs}
    assert "query_vision" in names
    # Sanity: existing tools still registered (tool-registration math).
    for legacy in ("see_what", "play_emotion", "move_head", "find_in_view",
                   "count_items", "recall_memory"):
        assert legacy in names, f"existing tool {legacy} disappeared"


def test_query_vision_dispatch_calls_ask_vision_with_question(monkeypatch):
    """Handler must (a) require question, (b) grab the current frame,
    (c) embed the question in the VL prompt, (d) return {'answer': ...}."""
    import robot_tools as rt

    captured = {}

    def fake_ask_vision(prompt, b64, num_predict=200, temperature=0.2, timeout=30):
        captured["prompt"] = prompt
        captured["b64"] = b64
        captured["num_predict"] = num_predict
        captured["timeout"] = timeout
        return "The user is holding a blue ceramic mug in their right hand."

    monkeypatch.setattr(rt, "_get_frame_b64", lambda: "FAKEB64==")
    monkeypatch.setattr(rt, "_ask_vision", fake_ask_vision)

    out = rt.execute_tool("query_vision", {"question": "what is in the user's right hand?"})

    assert isinstance(out, dict)
    assert "answer" in out, f"expected answer key, got {out}"
    assert out["answer"].startswith("The user is holding")
    assert out.get("question") == "what is in the user's right hand?"

    # Question must be embedded in the VL prompt
    assert "right hand" in captured["prompt"]
    assert captured["b64"] == "FAKEB64=="
    # 20 s timeout cap (task spec) so a hung VLM cannot freeze the brain
    assert captured["timeout"] <= 20, "timeout must be capped at 20s"
    # Bounded num_predict keeps Pi RAM + TTS latency in check
    assert captured["num_predict"] <= 200


def test_query_vision_no_frame_returns_canned_answer(monkeypatch):
    """When the camera worker isn't running, return a graceful canned answer
    instead of an opaque {'error': ...} so the LLM can synthesize a sensible
    user-facing reply. Must NOT call _ask_vision (saves a wasted VLM RTT)."""
    import robot_tools as rt

    monkeypatch.setattr(rt, "_get_frame_b64", lambda: None)

    def boom(*a, **kw):
        raise AssertionError("_ask_vision must NOT be called when there's no frame")

    monkeypatch.setattr(rt, "_ask_vision", boom)

    out = rt.execute_tool("query_vision", {"question": "what color is my shirt?"})
    assert isinstance(out, dict)
    assert "answer" in out, f"no-frame path must return canned answer, got {out}"
    # Heuristic: the canned answer should explain the limitation
    msg = out["answer"].lower()
    assert "can't" in msg or "cannot" in msg or "camera" in msg


def test_query_vision_missing_question_rejected(monkeypatch):
    """Bonus: empty question is rejected before any VLM call."""
    import robot_tools as rt

    def boom(*a, **kw):
        raise AssertionError("_ask_vision must NOT be called for empty question")

    monkeypatch.setattr(rt, "_ask_vision", boom)
    monkeypatch.setattr(rt, "_get_frame_b64", lambda: "FAKE==")

    out = rt.execute_tool("query_vision", {"question": "   "})
    assert "error" in out
    assert "required" in out["error"].lower()


def test_query_vision_vlm_exception_returns_error(monkeypatch):
    """VLM timeout / network error → {'error': ...}, not a raise.
    Brain dispatcher then increments the timeout-result counter."""
    import robot_tools as rt

    monkeypatch.setattr(rt, "_get_frame_b64", lambda: "FAKE==")

    def hung(*a, **kw):
        raise TimeoutError("VLM did not respond in 20s")

    monkeypatch.setattr(rt, "_ask_vision", hung)

    out = rt.execute_tool("query_vision", {"question": "is there a cat?"})
    assert "error" in out
    assert "timeout" in out["error"].lower() or "timeouterror" in out["error"].lower()


# ───────────────────────── Wave6-P4 YELLOW fix-loop tests ────────────────────

def test_get_frame_b64_rejects_stale_frame(monkeypatch):
    """M1 fix: `_get_frame_b64()` must mirror vision_worker's recency guard.
    A frame older than _STALE_FRAME_MAX_AGE_S (2 s) → None so the caller falls
    back to the canned "no view" answer instead of feeding a stale image to
    the VLM and reporting a confidently-wrong answer."""
    import time
    import sys
    import types
    import robot_tools as rt

    # Build a minimal fake robot_brain module that exposes _latest_frame +
    # _latest_frame_t. This works whether or not robot_brain is importable in
    # the test env (it requires a heavy cv2 backend on Pi).
    fake_brain = types.ModuleType("robot_brain")
    fake_brain._latest_frame = object()  # truthy placeholder; cv2.imencode is mocked below
    fake_brain._latest_frame_t = time.time() - 5.0  # 5 s old → stale
    monkeypatch.setitem(sys.modules, "robot_brain", fake_brain)

    # Ensure cv2.imencode is NOT reached when the frame is rejected.
    import cv2  # cv2-headless is a test dep
    def boom_imencode(*a, **kw):
        raise AssertionError("cv2.imencode must NOT run for a stale frame")
    monkeypatch.setattr(cv2, "imencode", boom_imencode)

    assert rt._get_frame_b64() is None, "stale frame (>2s) must return None"

    # Sanity: a fresh frame (within window) is NOT rejected by the recency
    # check — imencode should be called and yield a base64 string.
    fake_brain._latest_frame_t = time.time()
    class _FakeJpg:
        def tobytes(self):
            return b"\xff\xd8\xff"  # 3-byte JPEG magic
    def fake_imencode(*a, **kw):
        return True, _FakeJpg()
    monkeypatch.setattr(cv2, "imencode", fake_imencode)
    out = rt._get_frame_b64()
    assert out is not None, "fresh frame should pass recency guard"
    assert isinstance(out, str)


def test_outcome_label_timeout_for_timed_out_message(monkeypatch):
    """M2 fix: brain `_instrumented_exec_tool` outcome classifier must label
    socket.timeout / urlopen-style 'timed out' errors as outcome='timeout'
    (not 'error'). Without this, real production timeouts get bucketed as
    generic errors and the timeout SLO dashboard misses them."""
    import sys
    import importlib

    # robot_brain has heavy import-time side effects on Pi; skip if it can't
    # be loaded in the test env.
    rb = pytest.importorskip("robot_brain")

    # Patch the underlying execute_tool to return a 'timed out' error (the
    # exact substring socket / urllib surface, NOT 'timeout').
    import robot_tools as rt
    def fake_exec(name, args):
        return {"error": "vision call failed: timeout: The read operation timed out"}
    monkeypatch.setattr(rt, "execute_tool", fake_exec)

    # Mute structlog to avoid test-env noise; the Prom counter is what matters.
    calls = {"labels": [], "inc": 0}
    class _FakeMetric:
        def labels(self, **kw):
            calls["labels"].append(kw)
            class _Inc:
                def inc(self_inner):
                    calls["inc"] += 1
            return _Inc()
    monkeypatch.setattr(rb.obs, "vision_tool_call_total", _FakeMetric())

    rb._instrumented_exec_tool("query_vision", {"question": "anything?"})

    assert calls["inc"] == 1, "counter must increment exactly once"
    assert calls["labels"], "labels() must have been called"
    assert calls["labels"][0].get("result") == "timeout", (
        f"'timed out' substring should map to result='timeout', "
        f"got {calls['labels'][0]}"
    )
