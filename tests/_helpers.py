"""Shared test helpers.

- `PY`: the Python interpreter subprocess-isolated tests should spawn. Uses
  `sys.executable` so CI (GitHub Actions, no `.venv`) and local dev
  (which has a `.venv`) both work.
- `ROOT`: the repo root; tests that read source files for text assertions
  use `ROOT / "src" / "<file>.py"`.
- `SRC`:  `ROOT / "src"` — where the agent modules live.
- `ollama_running()`: tiny HTTP probe used by `@pytest.mark.skipif` on E2E
  tests so CI (no Ollama) skips cleanly instead of failing.
- `pillow_available()`: gate for tests that generate synthetic JPEGs via
  PIL; CI installs the minimal layer by default and skips those.
"""
from __future__ import annotations

import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC  = ROOT / "src"
PY   = sys.executable


def ollama_running(url: str | None = None, timeout: float = 0.8) -> bool:
    url = url or os.getenv("OLLAMA_HOST", "http://localhost:11434")
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/api/tags", timeout=timeout):
            return True
    except Exception:
        return False


def pillow_available() -> bool:
    try:
        import PIL  # noqa: F401
        return True
    except Exception:
        return False
