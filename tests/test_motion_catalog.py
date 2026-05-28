"""Tests for src/motion_catalog.py — Wave4-P6 (#76).

Covers:
  * Fresh-cache short-circuit (no HF call).
  * Stale cache → HF refresh → cache rewritten.
  * HF failure → bundled fallback.
  * Catalog build (clips + aliases + generic dance synonyms).
  * Alias whose target clip is missing is dropped (no orphan trigger).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from unittest import mock

import pytest

# tests/_helpers already adds src/ to sys.path; mirror that locally for safety.
_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


@pytest.fixture
def fresh_catalog_env(tmp_path, monkeypatch):
    """Isolate cache + fallback + module-level state per test."""
    cache = tmp_path / "cache.json"
    fallback = tmp_path / "fallback.json"
    fallback.write_text(json.dumps({
        "emotions": ["cheerful1", "sad1", "no1", "yes1"],
        "dances": ["yeah_nod", "jackson_square"],
        "aliases": {
            "happy": "cheerful1",
            "sad": "sad1",
            "ghost": "nonexistent_clip",  # should be dropped
        },
        "generic_dance_synonyms": ["dance", "boogie"],
        "default_dance": "yeah_nod",
    }), encoding="utf-8")

    monkeypatch.setenv("REACHY_MOTION_CATALOG_CACHE", str(cache))
    monkeypatch.setenv("REACHY_MOTION_CATALOG_FALLBACK", str(fallback))
    monkeypatch.setenv("REACHY_MOTION_CATALOG_TTL_S", "3600")

    # Reload module so env vars take effect.
    if "motion_catalog" in sys.modules:
        del sys.modules["motion_catalog"]
    import motion_catalog  # noqa: WPS433
    motion_catalog.reset_for_tests()
    return motion_catalog, cache, fallback


def test_fallback_used_when_hf_unreachable(fresh_catalog_env):
    """HF down + no cache → catalog built from bundled fallback."""
    mc, cache, _ = fresh_catalog_env
    with mock.patch.object(mc, "_hf_list_clips", return_value=[]):
        catalog = mc.load_catalog(force_refresh=True)
    # Direct clip
    assert "cheerful1" in catalog
    assert catalog["cheerful1"].kind == "emotion"
    # Dance
    assert "jackson_square" in catalog
    assert catalog["jackson_square"].kind == "dance"
    # Alias mapping preserved
    assert catalog["happy"].clip == "cheerful1"
    assert catalog["happy"].kind == "alias"
    # Generic synonym → default dance
    assert catalog["dance"].clip == "yeah_nod"
    assert catalog["dance"].kind == "generic_dance"
    # Orphan alias dropped (target clip missing from fallback)
    assert "ghost" not in catalog
    # Cache written
    assert cache.exists()


def test_hf_success_populates_catalog_and_writes_cache(fresh_catalog_env):
    """When HF returns clip lists, they win over fallback emotion/dance sets."""
    mc, cache, _ = fresh_catalog_env

    def fake_list(dataset, timeout=mc.HF_API_TIMEOUT_S):
        if "emotions" in dataset:
            return ["cheerful1", "sad1", "no1", "yes1", "amazed1"]  # extra clip
        if "dances" in dataset:
            return ["yeah_nod", "jackson_square", "dizzy_spin"]
        return []

    with mock.patch.object(mc, "_hf_list_clips", side_effect=fake_list):
        catalog = mc.load_catalog(force_refresh=True)

    assert "amazed1" in catalog and catalog["amazed1"].kind == "emotion"
    assert "dizzy_spin" in catalog and catalog["dizzy_spin"].kind == "dance"
    # Cache JSON on disk reflects HF-sourced clips + timestamp.
    cached = json.loads(cache.read_text(encoding="utf-8"))
    assert "amazed1" in cached["emotions"]
    assert "dizzy_spin" in cached["dances"]
    assert isinstance(cached["_fetched_at"], (int, float))


def test_fresh_cache_short_circuits_hf(fresh_catalog_env):
    """A non-stale cache file means zero HF calls."""
    mc, cache, _ = fresh_catalog_env
    cache.write_text(json.dumps({
        "emotions": ["cheerful1"],
        "dances": ["yeah_nod"],
        "aliases": {"happy": "cheerful1"},
        "generic_dance_synonyms": ["dance"],
        "default_dance": "yeah_nod",
        "_fetched_at": time.time(),  # fresh
    }), encoding="utf-8")
    mc.reset_for_tests()

    with mock.patch.object(mc, "_hf_list_clips") as m:
        catalog = mc.load_catalog()
        assert m.call_count == 0, "fresh cache must not trigger HF fetch"
    assert "cheerful1" in catalog
    assert catalog["happy"].clip == "cheerful1"


def test_stale_cache_triggers_hf_refresh(fresh_catalog_env):
    """A cache older than TTL is refreshed from HF and rewritten."""
    mc, cache, _ = fresh_catalog_env
    # 100 days ago — definitely stale vs 3600s TTL.
    stale_ts = time.time() - (100 * 86400)
    cache.write_text(json.dumps({
        "emotions": ["old_emotion"],
        "dances": ["old_dance"],
        "aliases": {},
        "generic_dance_synonyms": ["dance"],
        "default_dance": "old_dance",
        "_fetched_at": stale_ts,
    }), encoding="utf-8")
    mc.reset_for_tests()

    with mock.patch.object(mc, "_hf_list_clips", return_value=["cheerful1"]) as m:
        catalog = mc.load_catalog()
        assert m.call_count == 2, "expected one HF call per dataset (emotions + dances)"

    # Stale entry gone; fresh entry present.
    assert "old_emotion" not in catalog
    assert "cheerful1" in catalog
    # Cache rewritten with a new timestamp.
    cached = json.loads(cache.read_text(encoding="utf-8"))
    assert cached["_fetched_at"] > stale_ts + 86400


# ── H-M1: HF failure must NOT poison the 24h cache ─────────────────────────


def test_hf_failure_does_not_stamp_cache_fresh(fresh_catalog_env):
    """When both HF datasets return [], we must NOT write
    _fetched_at = now to disk — otherwise the next 24h of calls would
    short-circuit to fallback even after HF recovers. The cache file
    should remain absent (no prior cache existed)."""
    mc, cache, _ = fresh_catalog_env
    # The module's import-time load_catalog() may have written cache (if
    # the test host has network and real HF is reachable). Wipe it so
    # this test exercises the no-prior-cache branch deterministically.
    if cache.exists():
        cache.unlink()
    mc.reset_for_tests()
    with mock.patch.object(mc, "_hf_list_clips", return_value=[]) as m:
        catalog = mc.load_catalog(force_refresh=True)
        assert m.call_count == 2  # both datasets attempted
    # Fallback was used (cheerful1 from bundled fallback).
    assert "cheerful1" in catalog
    # Cache file must NOT have been written with a fresh timestamp on failure.
    assert not cache.exists(), \
        "HF failure must not write a poisoned fresh cache to disk"


def test_hf_failure_then_success_writes_cache(fresh_catalog_env):
    """First call: HF down → no cache stamped. Second call (after backoff
    reset): HF recovers → cache properly written with fresh timestamp."""
    mc, cache, _ = fresh_catalog_env
    if cache.exists():
        cache.unlink()
    mc.reset_for_tests()

    with mock.patch.object(mc, "_hf_list_clips", return_value=[]):
        mc.load_catalog(force_refresh=True)
    assert not cache.exists()

    # Reset module + backoff so the next call actually hits HF.
    mc.reset_for_tests()

    def fake_list(dataset, timeout=mc.HF_API_TIMEOUT_S):
        if "emotions" in dataset:
            return ["cheerful1", "extra1"]
        return ["yeah_nod"]
    with mock.patch.object(mc, "_hf_list_clips", side_effect=fake_list):
        catalog = mc.load_catalog(force_refresh=True)

    assert "extra1" in catalog
    assert cache.exists()
    cached = json.loads(cache.read_text(encoding="utf-8"))
    assert "_fetched_at" in cached
    assert "extra1" in cached["emotions"]


def test_hf_backoff_prevents_hammering(fresh_catalog_env, monkeypatch):
    """After an HF failure, the next call within the backoff window
    (>=60 s by default) must NOT re-hit HF — it serves stale cache or
    fallback. Once the window passes (or force_refresh), HF is retried."""
    mc, cache, _ = fresh_catalog_env
    # Clear any eager-import cache so the disk-cache short-circuit
    # doesn't mask the backoff path under test.
    if cache.exists():
        cache.unlink()
    # Shorten min-backoff so the test stays fast; we'll patch time.time
    # to simulate the wait.
    monkeypatch.setattr(mc, "HF_FAILURE_BACKOFF_MIN_S", 30.0)
    mc.reset_for_tests()

    call_count = {"n": 0}
    def failing_hf(dataset, timeout=mc.HF_API_TIMEOUT_S):
        call_count["n"] += 1
        return []
    monkeypatch.setattr(mc, "_hf_list_clips", failing_hf)

    # First call → HF tried for both datasets → both fail → backoff set.
    mc.load_catalog(force_refresh=True)
    assert call_count["n"] == 2

    # Reset only the in-process catalog cache (but NOT the backoff state),
    # then load again immediately. Backoff should suppress HF entirely.
    with mc._lock:
        mc._loaded_raw = None
    mc.load_catalog()  # not force_refresh — must honour backoff
    assert call_count["n"] == 2, "second call within backoff window must not hit HF"

    # Advance the in-module failure timestamp into the past so the backoff
    # window has elapsed, then verify HF IS retried.
    with mc._lock:
        mc._hf_failed_at = time.time() - 999.0
        mc._loaded_raw = None
    mc.load_catalog()
    assert call_count["n"] == 4, "after backoff window expires HF must be retried"


def test_motion_spec_play_path_urlencodes_dataset(fresh_catalog_env):
    """MotionSpec.play_path encodes slashes in dataset name (daemon expects that)."""
    mc, _, _ = fresh_catalog_env
    spec = mc.MotionSpec(
        name="happy", clip="cheerful1",
        dataset="pollen-robotics/reachy-mini-emotions-library", kind="alias",
    )
    p = spec.play_path
    assert p.startswith("/api/move/play/recorded-move-dataset/")
    assert "pollen-robotics%2Freachy-mini-emotions-library" in p
    assert p.endswith("/cheerful1")
