"""Regression tests for the _vllm_base() env-resolution helper.

The helper exists in both src/robot_brain.py and src/llm_backend.py; this
module imports the llm_backend copy (which is import-safe — robot_brain has
top-level side effects). Code review 2026-05-28 flagged two parser footguns:

  - bare ``host:port`` form (e.g. ``VLLM_HOST=vllm0528:8001``) was being
    appended with ``VLLM_PORT`` → ``http://vllm0528:8001:8000`` (unreachable).
  - trailing slash on legacy URL form (e.g. ``http://x:8000/``) yielded
    ``http://x:8000//v1/chat/completions`` (rejected by some proxies).

Both are fixed; these tests prevent regression.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_backend import _vllm_base  # noqa: E402


def _clear(*names: str) -> None:
    for n in names:
        os.environ.pop(n, None)


def test_default_when_unset() -> None:
    _clear("VLLM_HOST", "VLLM_PORT")
    assert _vllm_base("VLLM_HOST", "VLLM_PORT", "vllm0528", "8000") == "http://vllm0528:8000"


def test_bare_host_uses_port_env() -> None:
    os.environ["VLLM_HOST"] = "s1"
    os.environ["VLLM_PORT"] = "9001"
    try:
        assert _vllm_base("VLLM_HOST", "VLLM_PORT", "d", "8000") == "http://s1:9001"
    finally:
        _clear("VLLM_HOST", "VLLM_PORT")


def test_bare_host_port_shortcircuits_port_env() -> None:
    os.environ["VLLM_HOST"] = "vllm0528:8001"
    os.environ["VLLM_PORT"] = "8000"  # should be ignored — host already specifies a port
    try:
        assert _vllm_base("VLLM_HOST", "VLLM_PORT", "d", "9000") == "http://vllm0528:8001"
    finally:
        _clear("VLLM_HOST", "VLLM_PORT")


def test_legacy_url_form_passthrough() -> None:
    os.environ["VLLM_HOST"] = "http://legacy:8002"
    try:
        assert _vllm_base("VLLM_HOST", "VLLM_PORT", "d", "9000") == "http://legacy:8002"
    finally:
        _clear("VLLM_HOST")


def test_legacy_url_trailing_slash_stripped() -> None:
    os.environ["VLLM_HOST"] = "http://x:8000/"
    try:
        # Must not produce // when concatenated with /v1/...
        assert _vllm_base("VLLM_HOST", "VLLM_PORT", "d", "9000") == "http://x:8000"
    finally:
        _clear("VLLM_HOST")


def test_https_form_preserved() -> None:
    os.environ["VLLM_HOST"] = "https://secure.example/"
    try:
        assert _vllm_base("VLLM_HOST", "VLLM_PORT", "d", "9000") == "https://secure.example"
    finally:
        _clear("VLLM_HOST")


def test_whitespace_stripped_on_host_and_port() -> None:
    os.environ["VLLM_HOST"] = "  s1  "
    os.environ["VLLM_PORT"] = "  8000\n"
    try:
        assert _vllm_base("VLLM_HOST", "VLLM_PORT", "d", "9000") == "http://s1:8000"
    finally:
        _clear("VLLM_HOST", "VLLM_PORT")


def test_ipv6_bracketed_form_treated_as_url_segment() -> None:
    # Bracketed IPv6 contains ':' but starts with '['; must NOT short-circuit
    # the bare host:port branch — fall through to bare-host + port path.
    os.environ["VLLM_HOST"] = "[::1]"
    os.environ["VLLM_PORT"] = "8000"
    try:
        assert _vllm_base("VLLM_HOST", "VLLM_PORT", "d", "9000") == "http://[::1]:8000"
    finally:
        _clear("VLLM_HOST", "VLLM_PORT")
