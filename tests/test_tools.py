"""S5 TDD: tool registry + individual tool handler tests.

Tools are plain Python functions. Each tool:
  - has a JSON-schema spec (for Ollama / OpenAI tool_call)
  - executes without raising even on bad args (returns error dict)
  - returns a JSON-serializable result dict

Registry: `robot_tools.TOOLS = {"name": (spec, handler), ...}`
"""
import json
import pytest


def test_registry_exists_and_has_core_tools():
    from robot_tools import TOOLS
    # Must have at least these 3 core tools
    for name in ("get_current_time", "stop_motion", "move_head"):
        assert name in TOOLS, f"missing tool {name}"


def test_every_tool_has_valid_schema():
    """Each tool spec must be OpenAI-compatible dict."""
    from robot_tools import TOOLS
    for name, (spec, handler) in TOOLS.items():
        assert spec["type"] == "function"
        fn = spec["function"]
        assert fn["name"] == name
        assert "description" in fn and fn["description"]
        assert "parameters" in fn
        params = fn["parameters"]
        assert params["type"] == "object"
        assert "properties" in params   # may be empty dict {} for zero-arg
        assert callable(handler)


def test_every_tool_spec_is_json_serializable():
    """Schema goes over HTTP to Ollama — must serialize."""
    from robot_tools import TOOLS
    for name, (spec, _) in TOOLS.items():
        json.dumps(spec)  # should not raise


class TestGetCurrentTime:
    def test_returns_iso_time(self):
        from robot_tools import TOOLS
        _, handler = TOOLS["get_current_time"]
        result = handler()
        assert isinstance(result, dict)
        assert "now" in result or "time" in result or "iso" in result
        # any reasonable time-ish string
        val = result.get("now") or result.get("time") or result.get("iso")
        assert isinstance(val, str) and len(val) > 5

    def test_ignores_extra_args(self):
        from robot_tools import TOOLS
        _, handler = TOOLS["get_current_time"]
        # LLM might pass empty dict or junk — should not crash
        result = handler(**{})
        assert "now" in result or "time" in result or "iso" in result


class TestStopMotion:
    def test_calls_daemon_disable(self, monkeypatch):
        """stop_motion must hit reachy daemon motor disable endpoint."""
        called = []
        def fake_post(url, **kw):
            called.append(url)
            class _R:
                status = 200
                def read(self): return b'{"status":"ok"}'
                def __enter__(self): return self
                def __exit__(self, *a): pass
            return _R()
        import urllib.request
        monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: fake_post(a[0].full_url if hasattr(a[0], 'full_url') else a[0]))
        from robot_tools import TOOLS
        _, handler = TOOLS["stop_motion"]
        result = handler()
        assert result.get("ok") is True or "success" in str(result).lower() or "stop" in str(result).lower()
        assert any("set_mode" in u or "motor" in u.lower() or "stop" in u.lower() for u in called), f"no motor endpoint called: {called}"


class TestMoveHead:
    def test_accepts_pitch_yaw_args(self, monkeypatch):
        """move_head must accept {pitch: float, yaw: float} and invoke move API."""
        called = []
        def fake_post(url, data=None, **kw):
            called.append((url if isinstance(url,str) else url.full_url, data))
            class _R:
                def read(self): return b'{"uuid":"abc"}'
                def __enter__(self): return self
                def __exit__(self, *a): pass
            return _R()
        import urllib.request
        monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: fake_post(a[0]))
        from robot_tools import TOOLS
        _, handler = TOOLS["move_head"]
        result = handler(pitch=10.0, yaw=-5.0)
        # may return ok/uuid/error — must not raise
        assert isinstance(result, dict)

    def test_rejects_out_of_range(self):
        from robot_tools import TOOLS
        _, handler = TOOLS["move_head"]
        # pitch out of safe range (> 30 deg) — should clip or return error
        result = handler(pitch=999.0, yaw=0.0)
        assert isinstance(result, dict)
        # either clamped silently or error key present
        assert "error" in result or "clipped" in result or result.get("ok") is True or "uuid" in result


class TestToolParsing:
    """Ollama returns tool_calls in message — helper must parse them."""
    def test_parse_ollama_tool_call_format(self):
        from robot_tools import parse_tool_calls
        ollama_message = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "function": {
                    "name": "get_current_time",
                    "arguments": {},
                }
            }]
        }
        calls = parse_tool_calls(ollama_message)
        assert len(calls) == 1
        assert calls[0]["name"] == "get_current_time"
        assert calls[0]["arguments"] == {}

    def test_parse_no_tool_calls(self):
        from robot_tools import parse_tool_calls
        assert parse_tool_calls({"role": "assistant", "content": "hi"}) == []

    def test_arguments_may_be_string(self):
        """Some llama.cpp versions serialize args as JSON string."""
        from robot_tools import parse_tool_calls
        msg = {"role": "assistant", "tool_calls": [{
            "function": {"name": "move_head", "arguments": '{"pitch": 5, "yaw": 0}'}
        }]}
        calls = parse_tool_calls(msg)
        assert calls[0]["arguments"] == {"pitch": 5, "yaw": 0}


class TestExecuteTool:
    def test_execute_calls_handler(self):
        from robot_tools import execute_tool
        result = execute_tool("get_current_time", {})
        assert "now" in result or "time" in result or "iso" in result

    def test_execute_unknown_tool_returns_error(self):
        from robot_tools import execute_tool
        result = execute_tool("nonexistent_fake_tool", {})
        assert "error" in result

    def test_execute_bad_args_does_not_crash(self):
        from robot_tools import execute_tool
        # pass wrong type — handler should catch and return error
        result = execute_tool("move_head", {"pitch": "not-a-number", "yaw": "also-not"})
        assert isinstance(result, dict)
        assert "error" in result or "ok" in result
