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
