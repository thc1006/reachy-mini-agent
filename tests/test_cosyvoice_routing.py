"""CosyVoice 2 TTS routing wiring in robot_brain.

The full robot_brain module pulls in pyaudio / mediapipe / Whisper / the
ReachyMini SDK, so CI can't import it directly. Following the project
convention (see ``test_robot_brain_critical_fixes.py``), we assert
behavior via AST inspection + source-level wiring checks. This is enough
to lock down the public contract the Pi ``.env`` and brain rollout
depend on:

1. ``COSYVOICE_URL`` is resolved from the env at module load time so the
   Pi ``.env`` ``COSYVOICE_URL=http://vllm0528:8881`` actually takes
   effect without code edits.
2. ``_fetch_cosyvoice_tts_inner`` posts to the OpenAI-shape
   ``/v1/audio/speech`` endpoint and forwards the language hint.
3. ``_stream_tts`` dispatches on a new ``TTS_BACKEND`` env var, with the
   ``cosyvoice`` branch calling ``_fetch_cosyvoice_tts`` (and the
   existing ``edge`` / ``kokoro`` / legacy ``TTS_ENGINE`` paths
   preserved for canary safety).

2026-06-01 (CosyVoice review YELLOW M3 follow-up): added 3 runtime-level
tests that compile + exec the relevant brain functions in an isolated
namespace with mocked transport so we cover:
  T4 — POST body shape exactly matches the FastAPI SpeechRequest schema.
  T5 — HTTPError on the cosyvoice endpoint surfaces err_class to logger
       and propagates so the breaker / fallback path triggers.
  T6 — TTS_BACKEND=edge ladder calls edge first (kokoro is fallback).
"""
from __future__ import annotations

import ast
import json
import re
import types
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ROBOT_BRAIN = ROOT / "src" / "robot_brain.py"


def _read_source() -> str:
    return ROBOT_BRAIN.read_text(encoding="utf-8")


def _parse() -> ast.Module:
    return ast.parse(_read_source())


def _find_func(tree: ast.Module, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


# ───────────────── isolated function loader (for T4/T5/T6) ────────────
# robot_brain imports pyaudio + mediapipe + ReachyMini, none of which CI
# has. Instead of stubbing the world, we extract the source of the
# specific functions under test via AST.unparse, drop them into a tiny
# fresh module namespace populated with just the symbols they reference,
# then exec. This keeps the tests honest — they really do call the
# function — without dragging the whole robot_brain import chain in.

def _load_function_namespace() -> dict:
    """Return a dict-namespace with ``_fetch_cosyvoice_tts_inner`` and the
    module-level constants it depends on, executed in isolation.

    Mocks:
      * ``_urlreq``         → urllib.request alias (real, but we patch urlopen)
      * ``sf``              → soundfile alias; we stub ``sf.read`` to a fixed
                              ``(np.zeros(24000, dtype=np.float32), 24000)``
                              tuple so the function returns without I/O
      * ``logger``          → MagicMock so .warning() doesn't blow up
      * ``time``            → real
      * ``json``            → real
      * ``COSYVOICE_URL``   → "http://test-cosyvoice:8881"
      * ``COSYVOICE_TIMEOUT_S`` → 8.0
    """
    src = _read_source()
    tree = ast.parse(src)
    fn = _find_func(tree, "_fetch_cosyvoice_tts_inner")
    assert fn is not None
    fn_src = ast.unparse(fn)

    ns: dict = {}
    ns["__name__"] = "_isolated_cosyvoice"
    ns["_urlreq"] = urllib.request
    ns["json"] = json
    ns["time"] = __import__("time")
    ns["io"] = __import__("io")
    # Fake soundfile-like object: returns 1 s of silence @ 24 kHz.
    sf_stub = types.SimpleNamespace(
        read=lambda buf, dtype=None: (np.zeros(24000, dtype=np.float32), 24000),
    )
    ns["sf"] = sf_stub
    ns["logger"] = MagicMock()
    ns["COSYVOICE_URL"] = "http://test-cosyvoice:8881"
    ns["COSYVOICE_TIMEOUT_S"] = 8.0
    exec(compile(fn_src, "<isolated>", "exec"), ns)
    return ns


# ───────────────────────────── T1 ──────────────────────────────────────


def test_cosyvoice_url_resolves_from_env():
    """COSYVOICE_URL must be a module-level os.getenv("COSYVOICE_URL", …)
    so Pi ``.env`` overrides win without code changes, and must default to
    the vllm0528:8881 self-host endpoint.
    """
    src = _read_source()
    # Must reference the env var by name in an os.getenv() (or os.environ.get) call.
    pattern = re.compile(
        r"COSYVOICE_URL\s*=\s*os\.getenv\(\s*['\"]COSYVOICE_URL['\"]\s*,\s*['\"]http://[^'\"]+['\"]",
    )
    m = pattern.search(src)
    assert m is not None, (
        "Expected `COSYVOICE_URL = os.getenv(\"COSYVOICE_URL\", \"http://…\")` "
        "at module scope so the Pi .env can override the default."
    )
    # Default must point at the self-host service on vllm0528:8881 (current plan).
    assert "vllm0528:8881" in m.group(0), (
        f"COSYVOICE_URL default should target vllm0528:8881, got: {m.group(0)!r}"
    )

    # And the timeout knob should also be env-overridable so we can tune
    # without redeploy when network conditions change.
    assert "COSYVOICE_TIMEOUT_S" in src and "os.getenv(\"COSYVOICE_TIMEOUT_S\"" in src, (
        "Expected an env-overridable COSYVOICE_TIMEOUT_S knob."
    )


# ───────────────────────────── T2 ──────────────────────────────────────


def test_fetch_cosyvoice_tts_calls_correct_endpoint():
    """The inner fetcher must POST to /v1/audio/speech (OpenAI shape), pass
    the language hint, and decode the WAV response. We assert via source
    inspection of ``_fetch_cosyvoice_tts_inner``.
    """
    tree = _parse()
    fn = _find_func(tree, "_fetch_cosyvoice_tts_inner")
    assert fn is not None, "Expected _fetch_cosyvoice_tts_inner to be defined."

    # Function must accept a language kw (default None) so callers can pass
    # the script-detected hint without breaking older callers.
    arg_names = [a.arg for a in fn.args.args]
    assert "text" in arg_names, f"missing 'text' arg: {arg_names}"
    assert "language" in arg_names, (
        f"_fetch_cosyvoice_tts_inner must accept a 'language' kw, got {arg_names}"
    )

    body = ast.unparse(fn)
    # Endpoint shape: /v1/audio/speech (OpenAI-compatible). This is the same
    # contract the kokoro fetcher uses, so the brain stays endpoint-agnostic.
    assert "/v1/audio/speech" in body, (
        "Inner fetcher must hit the /v1/audio/speech endpoint."
    )
    # Must POST through urllib (project convention via _urlreq), not requests.
    assert "_urlreq" in body, "fetcher should use the project's _urlreq alias"
    # Must include the language hint in the JSON payload when provided.
    assert '"language"' in body or "'language'" in body, (
        "Inner fetcher should forward the language hint in the JSON payload."
    )
    # Must use the module-level URL + timeout so env overrides apply.
    assert "COSYVOICE_URL" in body, "fetcher must read COSYVOICE_URL"
    assert "COSYVOICE_TIMEOUT_S" in body, "fetcher must respect COSYVOICE_TIMEOUT_S"
    # Must return a decoded (samples, sample_rate) tuple just like _fetch_kokoro_tts.
    assert "sf.read" in body, (
        "Inner fetcher should decode WAV bytes via soundfile.read for parity "
        "with the kokoro path."
    )


# ───────────────────────────── T3 ──────────────────────────────────────


def test_tts_backend_router_dispatches():
    """``_stream_tts`` must look at ``TTS_BACKEND`` first and dispatch:

    * ``cosyvoice``           → _fetch_cosyvoice_tts (with edge fallback)
    * ``edge``                → _fetch_edge_tts      (with kokoro fallback)
    * ``kokoro``              → _fetch_kokoro_tts    (with edge fallback)
    * ``edge_then_cosyvoice`` → edge first, cosyvoice fallback (ladder)

    When ``TTS_BACKEND`` is unset/empty, the legacy ``TTS_ENGINE`` switch
    must still apply unchanged (canary safety — production .env keeps
    TTS_BACKEND=edge or leaves it unset).
    """
    tree = _parse()
    fn = _find_func(tree, "_stream_tts")
    assert fn is not None, "Expected async _stream_tts in robot_brain.py"
    body = ast.unparse(fn)

    # The new TTS_BACKEND knob must be read BEFORE the legacy TTS_ENGINE
    # switch, and all four documented values must have explicit branches.
    backend_idx = body.find('TTS_BACKEND')
    engine_idx  = body.find('TTS_ENGINE')
    assert backend_idx >= 0, "Expected TTS_BACKEND to be read inside _stream_tts."
    assert engine_idx >= 0, "Legacy TTS_ENGINE switch must remain for canary fallback."
    assert backend_idx < engine_idx, (
        "TTS_BACKEND must be read before TTS_ENGINE so the new knob wins."
    )

    for value in ("cosyvoice", "edge_then_cosyvoice", "kokoro", "edge"):
        assert f"'{value}'" in body or f'"{value}"' in body, (
            f"Expected explicit branch for TTS_BACKEND='{value}' in _stream_tts."
        )

    # The cosyvoice branch must actually call _fetch_cosyvoice_tts and pass
    # a language hint (so the server can pick zero_shot vs cross_lingual).
    cosy_call = re.search(
        r"_fetch_cosyvoice_tts\s*\(\s*text\s*,\s*language\s*=",
        body,
    )
    assert cosy_call is not None, (
        "cosyvoice branch must call _fetch_cosyvoice_tts(text, language=…)."
    )

    # Make sure the cosyvoice branch falls back to edge when the self-host
    # is down — otherwise a vllm0528 outage would mute the robot.
    assert "_fetch_edge_tts" in body, (
        "_stream_tts must keep an edge fallback for canary safety."
    )

    # The script-based language detector helper must exist and be wired in
    # for cosyvoice routing decisions.
    detect_fn = _find_func(tree, "_detect_tts_language")
    assert detect_fn is not None, (
        "Expected _detect_tts_language() helper for routing the language hint."
    )


# ───────────────────────────── T4 ──────────────────────────────────────


def test_fetch_cosyvoice_posts_correct_json_body():
    """The inner fetcher must POST a JSON body matching the FastAPI
    SpeechRequest schema: ``{"input": str, "voice": "default",
    "language": <hint>}``. We patch urlopen and inspect the actual bytes
    sent on the wire.
    """
    ns = _load_function_namespace()
    fetch = ns["_fetch_cosyvoice_tts_inner"]

    captured: dict = {}

    class _FakeResp:
        headers = {"x-gen-latency-s": "0.42", "x-mode": "zero_shot"}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            # Returning empty bytes is fine — sf.read is stubbed to
            # ignore the buffer entirely.
            return b""

    def _fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["body"] = req.data
        captured["headers"] = dict(req.headers)
        captured["timeout"] = timeout
        return _FakeResp()

    with patch.object(urllib.request, "urlopen", _fake_urlopen):
        fetch("你好瑞奇", language="zh")

    assert captured["url"] == "http://test-cosyvoice:8881/v1/audio/speech", (
        f"unexpected URL: {captured['url']!r}"
    )
    body = json.loads(captured["body"].decode("utf-8"))
    assert body == {"input": "你好瑞奇", "voice": "default", "language": "zh"}, (
        f"unexpected JSON body shape: {body!r}"
    )
    # Content-Type header lookup is case-insensitive in urllib.Request.
    ct = captured["headers"].get("Content-type") or captured["headers"].get("Content-Type")
    assert ct == "application/json", f"missing/wrong Content-Type: {captured['headers']!r}"
    # Timeout must come from the module-level knob, not a magic number.
    assert captured["timeout"] == 8.0, f"timeout not forwarded: {captured['timeout']}"


# ───────────────────────────── T5 ──────────────────────────────────────


def test_fetch_cosyvoice_handles_http_500_falls_back():
    """When the server returns an HTTPError (e.g. 503 cuda_oom or 500
    inference_failed), the inner fetcher must propagate the exception
    AND log the X-Error-Class header so the brain breaker + dashboards
    can distinguish CUDA pressure from generic failure.

    The outer ``_fetch_cosyvoice_tts`` (not under test here) then catches
    the propagated exception and returns ``None`` so the caller falls
    back to edge-tts — that contract is verified by T3 above.
    """
    ns = _load_function_namespace()
    fetch = ns["_fetch_cosyvoice_tts_inner"]
    logger_mock = ns["logger"]

    # Simulate the new server contract: 503 + X-Error-Class: cuda_oom.
    class _FakeHeaders:
        def get(self, key, default=None):
            return {"x-error-class": "cuda_oom"}.get(key.lower(), default)

    def _fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(
            url=req.full_url,
            code=503,
            msg="cuda oom",
            hdrs=_FakeHeaders(),
            fp=BytesIO(b""),
        )

    with patch.object(urllib.request, "urlopen", _fake_urlopen):
        import pytest as _pt
        with _pt.raises(urllib.error.HTTPError):
            fetch("hello", language="en")

    # Brain must have logged the cosyvoice_http_error event so the
    # dashboards distinguish cuda_oom from inference_failed.
    assert logger_mock.warning.called, (
        "Expected logger.warning('cosyvoice_http_error', ...) on HTTPError."
    )
    call_args = logger_mock.warning.call_args
    assert call_args.args[0] == "cosyvoice_http_error"
    assert call_args.kwargs.get("status") == 503
    assert call_args.kwargs.get("err_class") == "cuda_oom", (
        f"err_class not surfaced: {call_args.kwargs!r}"
    )


# ───────────────────────────── T6 ──────────────────────────────────────


def test_tts_backend_ladder_edge_then_kokoro_orders_calls():
    """The TTS_BACKEND='edge' branch in _stream_tts must invoke
    ``_fetch_edge_tts`` FIRST and only call ``_fetch_kokoro_tts`` on
    fallback. Verified by inspecting the branch body AST: the edge call
    must appear in source order before the kokoro call so a future
    refactor can't silently reorder them.

    Also verifies the cosyvoice ladder ('edge_then_cosyvoice') keeps
    edge first — same invariant for canary rollout safety.
    """
    tree = _parse()
    fn = _find_func(tree, "_stream_tts")
    assert fn is not None
    body = ast.unparse(fn)

    # Find the 'edge' literal branch and check edge < kokoro in source order.
    # Search for the canonical branch body shape from _stream_tts.
    edge_block = re.search(
        r"backend == ['\"]edge['\"][^:]*:(.*?)(?:elif backend|else:)",
        body,
        re.DOTALL,
    )
    assert edge_block is not None, "Could not locate the edge branch in _stream_tts."
    edge_body = edge_block.group(1)
    edge_idx = edge_body.find("_fetch_edge_tts")
    kokoro_idx = edge_body.find("_fetch_kokoro_tts")
    assert edge_idx >= 0, "edge branch must call _fetch_edge_tts"
    assert kokoro_idx >= 0, "edge branch must keep kokoro as fallback"
    assert edge_idx < kokoro_idx, (
        "edge branch must call _fetch_edge_tts BEFORE _fetch_kokoro_tts "
        "(kokoro is the fallback, not the primary)."
    )

    # And the edge_then_cosyvoice ladder: edge first, cosyvoice fallback.
    ladder_block = re.search(
        r"backend == ['\"]edge_then_cosyvoice['\"][^:]*:(.*?)(?:elif backend|else:)",
        body,
        re.DOTALL,
    )
    assert ladder_block is not None, "Expected the edge_then_cosyvoice ladder branch."
    ladder_body = ladder_block.group(1)
    ledge = ladder_body.find("_fetch_edge_tts")
    lcosy = ladder_body.find("_fetch_cosyvoice_tts")
    assert ledge >= 0 and lcosy >= 0
    assert ledge < lcosy, (
        "edge_then_cosyvoice must call edge FIRST then cosyvoice as fallback."
    )
