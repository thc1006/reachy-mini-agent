"""Tests for ``reachy_intent_runtime.action_catalog``.

These tests load the real ``data/reachy_official_actions.yaml`` fixture (no
mocking) and assert structural invariants documented in
``docs/research/reachy-official-action-catalog.md``.

The catalog is the foundational data layer behind Track A of Phase 10 and is
consumed (eventually) by ``orchestrator_worker._ACTION_TO_COMMAND``.  These
tests therefore act as a contract: schema regressions or accidental data drops
will fail here before they reach the orchestrator.
"""

from __future__ import annotations

import pytest

from reachy_intent_runtime.action_catalog import (
    DEFAULT_CATALOG_PATH,
    VALID_CPU_RISKS,
    VALID_TYPES,
    ActionCatalog,
    ActionEntry,
)

# ---------------------------------------------------------------------------
# Fixture: load once per session.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def catalog() -> ActionCatalog:
    return ActionCatalog.load()


# ---------------------------------------------------------------------------
# Basic load + schema invariants
# ---------------------------------------------------------------------------


def test_loads_default_catalog() -> None:
    cat = ActionCatalog.load()
    assert cat is not None
    assert isinstance(cat, ActionCatalog)
    # Default path is resolvable and points at a real file.
    assert DEFAULT_CATALOG_PATH.exists(), (
        f"default catalog file is missing at {DEFAULT_CATALOG_PATH}"
    )


def test_catalog_has_at_least_15_entries(catalog: ActionCatalog) -> None:
    assert len(catalog.entries) >= 15, (
        f"catalog has {len(catalog.entries)} entries; expected >= 15 per Phase 10 Track A spec"
    )


def test_every_entry_has_required_fields(catalog: ActionCatalog) -> None:
    for entry in catalog.entries:
        assert isinstance(entry, ActionEntry)
        assert entry.name, "entry missing name"
        assert entry.type in VALID_TYPES, f"entry {entry.name!r} has invalid type {entry.type!r}"
        assert entry.official_tool, f"entry {entry.name!r} missing official_tool"
        assert isinstance(entry.estimated_duration_ms, int)
        assert entry.estimated_duration_ms >= 0
        assert isinstance(entry.has_stop_tool, bool)
        assert entry.cpu_risk in VALID_CPU_RISKS, (
            f"entry {entry.name!r} has invalid cpu_risk {entry.cpu_risk!r}"
        )


# ---------------------------------------------------------------------------
# Stop tool invariants
# ---------------------------------------------------------------------------


def test_stop_dance_has_zero_duration(catalog: ActionCatalog) -> None:
    entry = catalog.get("stop_dance")
    assert entry.estimated_duration_ms == 0
    assert entry.has_stop_tool is False
    assert entry.type == "stop"


def test_stop_emotion_has_zero_duration(catalog: ActionCatalog) -> None:
    entry = catalog.get("stop_emotion")
    assert entry.estimated_duration_ms == 0
    assert entry.has_stop_tool is False
    assert entry.type == "stop"


def test_dance_entries_have_stop_tool_dance_pattern(catalog: ActionCatalog) -> None:
    dances = catalog.all_dances()
    assert dances, "catalog has no dance entries"
    for entry in dances:
        assert entry.has_stop_tool is True, (
            f"dance entry {entry.name!r} must declare has_stop_tool=true"
        )
        assert entry.stop_tool == "stop_dance", (
            f"dance entry {entry.name!r} must map to stop_dance tool, got {entry.stop_tool!r}"
        )


def test_emotion_entries_have_stop_tool_emotion_pattern(catalog: ActionCatalog) -> None:
    emotions = catalog.all_emotions()
    assert emotions, "catalog has no emotion entries"
    for entry in emotions:
        assert entry.has_stop_tool is True, (
            f"emotion entry {entry.name!r} must declare has_stop_tool=true"
        )
        assert entry.stop_tool == "stop_emotion", (
            f"emotion entry {entry.name!r} must map to stop_emotion tool, got {entry.stop_tool!r}"
        )


# ---------------------------------------------------------------------------
# Filtering / lookup
# ---------------------------------------------------------------------------


def test_by_type_dance_returns_only_dances(catalog: ActionCatalog) -> None:
    result = catalog.by_type("dance")
    assert result, "by_type('dance') returned empty list"
    for entry in result:
        assert entry.type == "dance"


def test_get_unknown_raises_keyerror(catalog: ActionCatalog) -> None:
    """Documented behavior: unknown lookups raise KeyError (not return None).

    Callers that prefer Optional semantics can wrap with try/except or use
    ``catalog.entries`` directly.
    """
    with pytest.raises(KeyError):
        catalog.get("does_not_exist_action_xyz")


# ---------------------------------------------------------------------------
# Segmentation + CPU policy
# ---------------------------------------------------------------------------


def test_needs_segmentation_for_long_non_interruptible(catalog: ActionCatalog) -> None:
    """Any non-interruptible, long-duration action must be flagged for
    segmentation per ADR-0001 (cooperative priority scheduler) — chunking is
    the only way the scheduler can preempt at boundary points.
    """
    long_threshold_ms = 1_000
    for entry in catalog.entries:
        if entry.interruptible is False and entry.estimated_duration_ms >= long_threshold_ms:
            assert catalog.needs_segmentation(entry.name) is True, (
                f"entry {entry.name!r}: non-interruptible and long, "
                f"must have requires_segmentation=true"
            )


def test_cpu_high_risk_filter(catalog: ActionCatalog) -> None:
    high = catalog.cpu_high_risk()
    # The filter must be consistent with the underlying field; emptiness is
    # acceptable but if any entry is high, it must appear here.
    expected = [e for e in catalog.entries if e.cpu_risk == "high"]
    assert {e.name for e in high} == {e.name for e in expected}


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def test_catalog_validates_type_field(tmp_path) -> None:
    """Loader rejects entries whose ``type`` is outside the allowed enum."""
    bad = tmp_path / "bad_types.yaml"
    bad.write_text(
        """
- name: weird_thing
  type: not_a_real_type
  official_tool: dance
  estimated_duration_ms: 1000
  interruptible: unknown
  has_stop_tool: false
  cpu_risk: low
""".strip(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="type"):
        ActionCatalog.load(bad)


def test_catalog_validates_cpu_risk_enum(tmp_path) -> None:
    """Loader rejects entries whose ``cpu_risk`` is outside the allowed enum."""
    bad = tmp_path / "bad_cpu.yaml"
    bad.write_text(
        """
- name: weird_thing
  type: dance
  official_tool: dance
  estimated_duration_ms: 1000
  interruptible: unknown
  has_stop_tool: true
  stop_tool: stop_dance
  cpu_risk: insane
""".strip(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="cpu_risk"):
        ActionCatalog.load(bad)


# ---------------------------------------------------------------------------
# Bonus sanity: by_tool index sanity
# ---------------------------------------------------------------------------


def test_by_tool_returns_consistent_entries(catalog: ActionCatalog) -> None:
    dance_tool_entries = catalog.by_tool("dance")
    assert dance_tool_entries
    for e in dance_tool_entries:
        assert e.official_tool == "dance"


def test_stop_tool_for_known_dance(catalog: ActionCatalog) -> None:
    dances = catalog.all_dances()
    assert dances
    assert catalog.stop_tool_for(dances[0].name) == "stop_dance"


def test_stop_tool_for_stop_action_is_none(catalog: ActionCatalog) -> None:
    # The stop_* actions themselves do not have a further stop tool.
    assert catalog.stop_tool_for("stop_dance") is None
    assert catalog.stop_tool_for("stop_emotion") is None
