"""Tests for the shared `_env` boolean helpers (Track D-2, 2026-06-01).

Covers:
  - env_bool returns the supplied default when the env var is unset
  - env_bool parses every documented falsy form (0/false/no/off/empty + F/N)
  - env_bool parses every documented truthy form (1/true/yes/on + arbitrary
    non-empty values)
  - Regression: ELDER_CARE_MODE=0 evaluates to bool False, not the
    truthy-string "0", so consumers can `if env_bool(...)` safely. Replaces
    the prior pattern where `os.getenv("ELDER_CARE_MODE", "0")` was passed
    raw to truthiness checks and the string "0" silently turned ON.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Match the convention used by other tests / root conftest: ensure src/ is on
# sys.path so `import _env` resolves to src/_env.py.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from _env import env_bool, is_truthy  # noqa: E402


# ── env_bool default ─────────────────────────────────────────────────────

def test_env_bool_default_false_when_unset(monkeypatch):
    monkeypatch.delenv("TRACK_D2_FLAG", raising=False)
    assert env_bool("TRACK_D2_FLAG") is False


def test_env_bool_default_true_when_unset(monkeypatch):
    monkeypatch.delenv("TRACK_D2_FLAG", raising=False)
    assert env_bool("TRACK_D2_FLAG", default=True) is True


# ── env_bool falsy variants ──────────────────────────────────────────────

@pytest.mark.parametrize("raw", [
    "0", "false", "FALSE", "False",
    "no", "NO", "No",
    "off", "OFF", "Off",
    "",
    "f", "F", "n", "N",
    "  0  ",         # whitespace stripped
    "  false  ",
])
def test_env_bool_parses_all_falsy_variants(monkeypatch, raw):
    monkeypatch.setenv("TRACK_D2_FLAG", raw)
    assert env_bool("TRACK_D2_FLAG", default=True) is False, \
        f"{raw!r} should be falsy even when default=True"


# ── env_bool truthy variants ─────────────────────────────────────────────

@pytest.mark.parametrize("raw", [
    "1", "true", "TRUE", "True",
    "yes", "YES", "Yes",
    "on", "ON", "On",
    "anything-else",   # any non-empty non-falsy is truthy
    "2", "y", "t",
])
def test_env_bool_parses_all_truthy_variants(monkeypatch, raw):
    monkeypatch.setenv("TRACK_D2_FLAG", raw)
    assert env_bool("TRACK_D2_FLAG", default=False) is True, \
        f"{raw!r} should be truthy even when default=False"


# ── is_truthy (value-form) ───────────────────────────────────────────────

@pytest.mark.parametrize("value,expected", [
    (None, False),
    ("", False),
    ("0", False),
    ("false", False),
    ("no", False),
    ("off", False),
    ("1", True),
    ("true", True),
    ("on", True),
    ("yes", True),
    ("anything", True),
])
def test_is_truthy_value_form(value, expected):
    assert is_truthy(value) is expected


# ── regression: ELDER_CARE_MODE=0 is OFF, not ON ─────────────────────────

def test_elder_care_mode_zero_is_off(monkeypatch):
    """Regression: prior to Track D-2 some sites read ELDER_CARE_MODE via
    `os.getenv("ELDER_CARE_MODE", "0")` and used the result in places that
    treated the raw STRING as truthiness — meaning `ELDER_CARE_MODE=0`
    silently evaluated to True (any non-empty string is truthy in Python).

    Now env_bool returns a real bool: False for "0".
    """
    monkeypatch.setenv("ELDER_CARE_MODE", "0")
    assert env_bool("ELDER_CARE_MODE") is False
    # Sanity: the raw-string pattern (what we replaced) WOULD have been True.
    import os
    assert bool(os.getenv("ELDER_CARE_MODE", "0")) is True


def test_elder_care_mode_one_is_on(monkeypatch):
    monkeypatch.setenv("ELDER_CARE_MODE", "1")
    assert env_bool("ELDER_CARE_MODE") is True


def test_elder_care_mode_unset_is_off(monkeypatch):
    monkeypatch.delenv("ELDER_CARE_MODE", raising=False)
    assert env_bool("ELDER_CARE_MODE") is False
