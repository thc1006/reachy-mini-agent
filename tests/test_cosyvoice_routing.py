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
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

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
