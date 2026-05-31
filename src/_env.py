"""Shared env helpers — single source for boolean semantics.

Track D-2 (2026-06-01): consolidates three drifting bool conventions —
`getenv("X", "0") == "1"` strict, ad-hoc `_truthy(v)` value-parsers, and
mixed pass-string-as-bool sites — into one tiny module.

Two complementary entry points:

  env_bool(name, default=False)
      Read env var by name and parse as bool. Use at module top / startup
      gates (the common case). Falsy set: 0, false, no, off, "", f, n
      (case-insensitive). Unset variable => `default`. Empty string is
      treated as falsy (NOT default) — this matches the existing
      elder_care._truthy convention used in production for ~12 months.

  is_truthy(value)
      Parse an already-read value (e.g. from a config dict, a CLI flag
      string, or a logged env snapshot). Same falsy set as env_bool.

Keep this module dependency-free (stdlib only) so brain_helpers /
elder_care imports stay cheap at startup.
"""
from __future__ import annotations

import os
from typing import Optional

# Same falsy set as the prior elder_care._truthy + elder_fallguard._truthy
# implementations so the migration is observably behaviour-preserving.
_FALSY = frozenset(("0", "false", "no", "off", "", "f", "n"))


def is_truthy(value: Optional[str]) -> bool:
    """Parse a string value as bool. None => False. Empty/whitespace => False.

    Truthy: anything not in {0, false, no, off, "", f, n} (case-insensitive,
    leading/trailing whitespace stripped).
    """
    if value is None:
        return False
    return value.strip().lower() not in _FALSY


def env_bool(name: str, default: bool = False) -> bool:
    """Read env var `name` and parse as bool.

    Replaces ad-hoc `os.getenv("X", "0") == "1"` patterns across the
    codebase. Accepts 1/0, true/false, yes/no, on/off (case-insensitive).

    Args:
        name: env var name.
        default: returned when the env var is UNSET (not when it is empty).
                 Empty string is treated as falsy to match existing
                 elder_care._truthy semantics.
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in _FALSY
