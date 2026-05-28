"""H-M6 — coverage for the openai (vLLM) / fastembed branch of RobotMemory.

The original test_robot_memory.py is gated on Ollama being reachable. On the
Pi (and on CI) Ollama is not installed but the production path is still
openai+fastembed — so these tests use mocks and run unconditionally.

Tested:
  (a) RobotMemory(llm_provider="openai", embed_provider="fastembed") builds
      the right mem0 config (provider names, base_url, dims), with the
      mem0.Memory.from_config mocked. Asserts the dict shape includes
      "openai" + "fastembed" wiring.
  (b) _regenerate_summary openai branch with urllib.request.urlopen mocked —
      verifies POST to /v1/chat/completions with bearer auth, JSON payload
      shape, and that the response is parsed via choices[0].message.content.
"""
from __future__ import annotations

import io
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Make src/ importable
SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture
def fake_mem0():
    """Inject a fake `mem0` module that captures the from_config call."""
    fake_mod = MagicMock()
    fake_memory_instance = MagicMock()
    fake_mod.Memory.from_config = MagicMock(return_value=fake_memory_instance)
    # Save + restore so other tests in the same process aren't affected.
    saved = sys.modules.get("mem0")
    sys.modules["mem0"] = fake_mod
    try:
        yield fake_mod, fake_memory_instance
    finally:
        if saved is not None:
            sys.modules["mem0"] = saved
        else:
            sys.modules.pop("mem0", None)


def _fresh_import():
    """Drop the cached robot_memory module so each test sees a fresh import
    that picks up our mocked mem0."""
    sys.modules.pop("robot_memory", None)
    import robot_memory as rm  # noqa: F401
    return rm


class TestOpenAiFastembedConfig:
    """(a) Construct RobotMemory with openai+fastembed → from_config shape."""

    def test_config_includes_openai_and_fastembed(self, fake_mem0, monkeypatch):
        monkeypatch.setenv("ROBOT_MEMORY", "1")
        # Force defaults to make the test deterministic regardless of dev .env
        monkeypatch.delenv("MEM0_LLM_PROVIDER", raising=False)
        monkeypatch.delenv("MEM0_EMBED_PROVIDER", raising=False)
        fake_mod, fake_inst = fake_mem0

        tmp = tempfile.mkdtemp(prefix="mem0_cfg_")
        rm = _fresh_import()
        mem = rm.RobotMemory(
            qdrant_path=tmp,
            llm_provider="openai",
            embed_provider="fastembed",
            llm_model="qwen36-awq",
            embed_model="BAAI/bge-small-zh-v1.5",
            llm_base_url="http://vllm0528:8000/v1",
            llm_api_key="secret-not-logged",
            embed_dims=512,  # matches known dim — should NOT trigger override
        )
        assert mem.enabled, "RobotMemory should be enabled when mem0 is mocked"

        # Inspect the dict passed to from_config
        fake_mod.Memory.from_config.assert_called_once()
        config_arg = fake_mod.Memory.from_config.call_args[0][0]
        assert config_arg["llm"]["provider"] == "openai"
        assert config_arg["llm"]["config"]["model"] == "qwen36-awq"
        assert config_arg["llm"]["config"]["openai_base_url"] == "http://vllm0528:8000/v1"
        assert config_arg["llm"]["config"]["api_key"] == "secret-not-logged"
        assert config_arg["embedder"]["provider"] == "fastembed"
        assert config_arg["embedder"]["config"]["model"] == "BAAI/bge-small-zh-v1.5"
        assert config_arg["vector_store"]["provider"] == "qdrant"
        assert config_arg["vector_store"]["config"]["embedding_model_dims"] == 512

    def test_dim_mismatch_warns_and_overrides(self, fake_mem0, monkeypatch, capsys):
        """M-M3: passing wrong embed_dims for a known model logs a warning
        and overrides to the correct value before building the qdrant config."""
        monkeypatch.setenv("ROBOT_MEMORY", "1")
        fake_mod, _ = fake_mem0
        tmp = tempfile.mkdtemp(prefix="mem0_dim_")
        rm = _fresh_import()
        rm.RobotMemory(
            qdrant_path=tmp,
            llm_provider="openai",
            embed_provider="fastembed",
            embed_model="BAAI/bge-small-zh-v1.5",
            embed_dims=1024,  # wrong — should override to 512
        )
        out = capsys.readouterr().out
        assert "MEM0_EMBED_DIMS=1024" in out and "expected 512" in out
        config_arg = fake_mod.Memory.from_config.call_args[0][0]
        assert config_arg["vector_store"]["config"]["embedding_model_dims"] == 512

    def test_api_key_redacted_in_repr(self, fake_mem0, monkeypatch):
        """H-M4: repr() of the api_key attribute must NOT leak the value."""
        monkeypatch.setenv("ROBOT_MEMORY", "1")
        tmp = tempfile.mkdtemp(prefix="mem0_key_")
        rm = _fresh_import()
        mem = rm.RobotMemory(
            qdrant_path=tmp,
            llm_provider="openai",
            llm_api_key="sk-very-secret-12345",
        )
        r = repr(mem.llm_api_key)
        assert "sk-very-secret-12345" not in r, f"api key leaked in repr: {r!r}"
        assert "redacted" in r, f"expected redacted marker in {r!r}"
        # But the value is still usable (it's a str subclass)
        assert mem.llm_api_key.startswith("sk-")


class TestOpenAiSummaryHttp:
    """(b) _regenerate_summary openai branch — mock urlopen, verify wire shape."""

    def _make_mem(self, fake_mem0, tmp_dir, log_lines):
        rm = _fresh_import()
        log_path = Path(tmp_dir) / "conversation_log.jsonl"
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        mem = rm.RobotMemory(
            qdrant_path=str(Path(tmp_dir) / "qdrant"),
            conversation_log_path=str(log_path),
            llm_provider="openai",
            llm_base_url="http://vllm0528:8000/v1",
            llm_api_key="dummy",
            llm_model="qwen36-awq",
            summary_path=str(Path(tmp_dir) / "summary.txt"),
            summary_keep_recent=2,
        )
        return mem

    def test_post_payload_and_parse(self, fake_mem0, monkeypatch):
        monkeypatch.setenv("ROBOT_MEMORY", "1")
        tmp = tempfile.mkdtemp(prefix="mem0_sum_")

        # 5 turns — keep_recent=2 → 3 are eligible for summarization
        lines = [
            json.dumps({"ts": "t0", "user": f"u{i}", "robot": f"r{i}"})
            for i in range(5)
        ]
        mem = self._make_mem(fake_mem0, tmp, lines)
        assert mem.enabled

        # Build a fake urlopen response with an OpenAI-shaped reply
        fake_payload = json.dumps({
            "choices": [{"message": {"content": "User likes tofu and lives in Hsinchu."}}],
        }).encode("utf-8")

        class _FakeResp:
            def __init__(self, body):
                self._body = io.BytesIO(body)
            def read(self): return self._body.read()
            def __enter__(self): return self
            def __exit__(self, *a): return False

        captured = {}
        def _fake_urlopen(req, timeout=None):
            captured["url"] = req.full_url
            captured["headers"] = dict(req.headers)
            captured["body"] = json.loads(req.data.decode("utf-8"))
            captured["timeout"] = timeout
            return _FakeResp(fake_payload)

        with patch("urllib.request.urlopen", _fake_urlopen):
            mem._regenerate_summary()

        # Wire shape
        assert captured["url"].endswith("/chat/completions")
        # urllib normalizes headers to Title-Case — find Authorization case-insensitively
        auth = next((v for k, v in captured["headers"].items()
                     if k.lower() == "authorization"), None)
        assert auth == "Bearer dummy", f"missing/incorrect auth header: {captured['headers']}"
        assert captured["body"]["model"] == "qwen36-awq"
        assert captured["body"]["messages"][0]["role"] == "user"
        assert "UPDATED SUMMARY" in captured["body"]["messages"][0]["content"] \
            or "SUMMARY:" in captured["body"]["messages"][0]["content"]
        # M-M1: timeout tightened to 30 s
        assert captured["timeout"] == 30

        # Summary file was written
        summary_file = Path(tmp) / "summary.txt"
        assert summary_file.exists()
        assert "tofu" in summary_file.read_text(encoding="utf-8")

    def test_empty_response_logs_raw(self, fake_mem0, monkeypatch, capsys):
        """M-M5: when vLLM returns empty content, raw payload is logged."""
        monkeypatch.setenv("ROBOT_MEMORY", "1")
        tmp = tempfile.mkdtemp(prefix="mem0_empty_")
        lines = [
            json.dumps({"ts": "t0", "user": f"u{i}", "robot": f"r{i}"})
            for i in range(5)
        ]
        mem = self._make_mem(fake_mem0, tmp, lines)

        empty_body = json.dumps({
            "choices": [{"message": {"content": ""}}],
            "error": "rate_limited",
        }).encode("utf-8")

        class _R:
            def read(self): return empty_body
            def __enter__(self): return self
            def __exit__(self, *a): return False

        with patch("urllib.request.urlopen", lambda req, timeout=None: _R()):
            mem._regenerate_summary()

        out = capsys.readouterr().out
        assert "empty summary" in out
        assert "rate_limited" in out, f"raw data not surfaced: {out!r}"
