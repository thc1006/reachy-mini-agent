"""Reachy Mini official action catalog loader.

This module is the foundational data layer for Phase 10 Track A.  It loads
``data/reachy_official_actions.yaml`` (a hand-curated, source-grounded list of
official SDK actions) into typed :class:`ActionEntry` records and exposes a
small lookup / filter API via :class:`ActionCatalog`.

Design notes
------------
* The catalog is **data-only**: it knows nothing about scheduling, the
  ``MotionCommand`` dataclass, or the orchestrator.  Consumers compose those
  themselves (see ``orchestrator_worker._ACTION_TO_COMMAND`` for the eventual
  consumer).
* ``interruptible`` accepts ``True``, ``False`` or the string ``"unknown"``.
  YAML naturally yields the first two from ``true``/``false`` literals and the
  string from the unquoted token ``unknown``.  We deliberately keep the
  tri-state because hardware behavior for most dances/emotions is not yet
  documented; per CLAUDE.md we refuse to guess.
* YAML parser choice: **PyYAML**.  It is listed in
  ``[project.optional-dependencies].dev`` AND in the runtime ``dependencies``
  block of ``pyproject.toml`` because the catalog is a runtime artifact (the
  orchestrator will consume it at startup).  Stdlib has no YAML parser and
  hand-rolling one would invite drift; PyYAML is tiny (~150 KB wheel) and the
  schema is list-of-dicts (no anchors, no flow style, safe to ``yaml.safe_load``).

See also
--------
* ``docs/research/reachy-official-action-catalog.md`` — provenance, methodology,
  hardware-verification TODOs.
* ``docs/adr/0001-cooperative-priority-scheduler.md`` — why long
  non-interruptible actions must be segmented.
* ``docs/research/2026-05-28-official-stack.md`` — upstream tool list.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

# Path to the canonical bundled catalog file.  Resolved relative to the repo
# root: ``src/reachy_intent_runtime/action_catalog.py`` -> ``../../data/...``.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CATALOG_PATH: Path = _REPO_ROOT / "data" / "reachy_official_actions.yaml"

VALID_TYPES: frozenset[str] = frozenset({"dance", "emotion", "pose", "speech", "camera", "stop"})
VALID_CPU_RISKS: frozenset[str] = frozenset({"low", "medium", "high"})

# Type alias for the interruptibility tri-state.  Callers should treat the
# string literal ``"unknown"`` as "behave conservatively" (treat as
# non-interruptible) per ADR-0001.
InterruptibleField = bool | Literal["unknown"]


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActionEntry:
    """One catalog entry — frozen for safe sharing across workers."""

    name: str
    type: str
    official_tool: str
    estimated_duration_ms: int
    interruptible: InterruptibleField
    has_stop_tool: bool
    stop_tool: str | None = None
    requires_segmentation: bool = False
    cpu_risk: str = "low"
    source: str = ""
    notes: str = ""


# ---------------------------------------------------------------------------
# Loader / catalog
# ---------------------------------------------------------------------------


def _coerce_interruptible(value: Any, *, entry_name: str) -> InterruptibleField:
    """Accept ``True``/``False``/``"true"``/``"false"``/``"unknown"``/``None``.

    ``None`` and missing values are treated as ``"unknown"`` to enforce the
    "don't guess" rule.
    """
    if value is None:
        return "unknown"
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v == "true":
            return True
        if v == "false":
            return False
        if v == "unknown":
            return "unknown"
    raise ValueError(
        f"entry {entry_name!r}: invalid interruptible value {value!r}; "
        "expected true | false | unknown"
    )


def _build_entry(raw: dict[str, Any]) -> ActionEntry:
    """Validate one raw dict and return a frozen :class:`ActionEntry`."""
    if not isinstance(raw, dict):
        raise ValueError(f"catalog entry is not a mapping: {raw!r}")

    name = raw.get("name")
    if not name or not isinstance(name, str):
        raise ValueError(f"catalog entry missing/invalid name: {raw!r}")

    type_ = raw.get("type")
    if type_ not in VALID_TYPES:
        raise ValueError(f"entry {name!r}: invalid type {type_!r}; allowed: {sorted(VALID_TYPES)}")

    official_tool = raw.get("official_tool")
    if not official_tool or not isinstance(official_tool, str):
        raise ValueError(f"entry {name!r}: missing/invalid official_tool")

    duration = raw.get("estimated_duration_ms")
    if not isinstance(duration, int) or duration < 0:
        raise ValueError(
            f"entry {name!r}: estimated_duration_ms must be a non-negative int, got {duration!r}"
        )

    interruptible = _coerce_interruptible(raw.get("interruptible"), entry_name=name)

    has_stop_tool = raw.get("has_stop_tool", False)
    if not isinstance(has_stop_tool, bool):
        raise ValueError(f"entry {name!r}: has_stop_tool must be bool, got {has_stop_tool!r}")

    stop_tool = raw.get("stop_tool")
    if stop_tool is not None and not isinstance(stop_tool, str):
        raise ValueError(f"entry {name!r}: stop_tool must be str or null")

    requires_segmentation = raw.get("requires_segmentation", False)
    if not isinstance(requires_segmentation, bool):
        raise ValueError(
            f"entry {name!r}: requires_segmentation must be bool, got {requires_segmentation!r}"
        )

    cpu_risk = raw.get("cpu_risk", "low")
    if cpu_risk not in VALID_CPU_RISKS:
        raise ValueError(
            f"entry {name!r}: invalid cpu_risk {cpu_risk!r}; allowed: {sorted(VALID_CPU_RISKS)}"
        )

    source = raw.get("source", "") or ""
    notes = raw.get("notes", "") or ""
    if not isinstance(source, str) or not isinstance(notes, str):
        raise ValueError(f"entry {name!r}: source/notes must be strings")

    return ActionEntry(
        name=name,
        type=type_,
        official_tool=official_tool,
        estimated_duration_ms=duration,
        interruptible=interruptible,
        has_stop_tool=has_stop_tool,
        stop_tool=stop_tool,
        requires_segmentation=requires_segmentation,
        cpu_risk=cpu_risk,
        source=source,
        notes=notes,
    )


class ActionCatalog:
    """Read-only lookup over the bundled YAML catalog.

    Use :meth:`load` to construct.  Entries are immutable; the catalog itself
    is also effectively immutable after construction (no public mutators).
    """

    def __init__(self, entries: list[ActionEntry]) -> None:
        # Detect duplicate names early — they would silently break ``get``.
        seen: set[str] = set()
        for e in entries:
            if e.name in seen:
                raise ValueError(f"duplicate catalog entry name: {e.name!r}")
            seen.add(e.name)
        self._entries: list[ActionEntry] = list(entries)
        self._by_name: dict[str, ActionEntry] = {e.name: e for e in entries}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CATALOG_PATH) -> ActionCatalog:
        """Load and validate a catalog YAML file.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        ValueError
            If the YAML structure is wrong or any entry fails validation.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"catalog file not found: {p}")
        with p.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        if raw is None:
            raise ValueError(f"catalog file is empty: {p}")
        if not isinstance(raw, list):
            raise ValueError(
                f"catalog file must be a YAML list of mappings, got {type(raw).__name__}"
            )
        entries = [_build_entry(item) for item in raw]
        return cls(entries)

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    @property
    def entries(self) -> list[ActionEntry]:
        """Return a shallow copy of all entries (order preserved)."""
        return list(self._entries)

    def get(self, name: str) -> ActionEntry:
        """Lookup by exact name.  Raises ``KeyError`` if unknown."""
        try:
            return self._by_name[name]
        except KeyError as exc:
            raise KeyError(f"unknown action: {name!r}") from exc

    def by_type(self, type_: str) -> list[ActionEntry]:
        """All entries of a given ``type`` (e.g. ``"dance"``)."""
        return [e for e in self._entries if e.type == type_]

    def by_tool(self, tool: str) -> list[ActionEntry]:
        """All entries that map to a given ``official_tool``."""
        return [e for e in self._entries if e.official_tool == tool]

    def stop_tool_for(self, action_name: str) -> str | None:
        """Return the discrete stop tool for an action, or ``None`` if there
        is no stop primitive (the action itself, the action is trivially
        interruptible, or the action is itself a stop primitive).
        """
        entry = self.get(action_name)
        if not entry.has_stop_tool:
            return None
        return entry.stop_tool

    def all_dances(self) -> list[ActionEntry]:
        """Sugar for ``by_type('dance')``."""
        return self.by_type("dance")

    def all_emotions(self) -> list[ActionEntry]:
        """Sugar for ``by_type('emotion')``."""
        return self.by_type("emotion")

    def needs_segmentation(self, name: str) -> bool:
        """Whether the named action must be chunked per ADR-0001."""
        return self.get(name).requires_segmentation

    def cpu_high_risk(self) -> list[ActionEntry]:
        """Entries flagged as ``cpu_risk: high`` — these need CM4 budget care."""
        return [e for e in self._entries if e.cpu_risk == "high"]

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._by_name


__all__ = [
    "DEFAULT_CATALOG_PATH",
    "VALID_TYPES",
    "VALID_CPU_RISKS",
    "InterruptibleField",
    "ActionEntry",
    "ActionCatalog",
]
